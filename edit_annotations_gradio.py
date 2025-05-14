import os
import random
import numpy as np
import h5py
import torch
from PIL import Image, ImageDraw
import gradio as gr
from segment_anything import sam_model_registry, SamPredictor
import argparse
import requests
from tqdm import tqdm

# Session initializer
def init_session():
    return {
        "id": None,
        "image": None,
        "image_copy": None,
        "points": [],
        "masks": None,
        "annotations": []
    }

# Load existing annotation
def load_annotation(image_path, annotation_path, session):
    if not os.path.exists(image_path) or not os.path.exists(annotation_path):
        return None, "Image or annotation file not found.", session
    
    try:
        # Load image
        img = np.array(Image.open(image_path))
        session['image'] = img
        session['image_copy'] = img.copy()
        
        # Extract image ID from filename
        session['id'] = os.path.basename(image_path).split('_')[0]
        
        # Load annotations from HDF5
        with h5py.File(annotation_path, 'r') as hf:
            # Load segments
            if 'segments' in hf:
                for key in hf['segments'].keys():
                    mask = hf[f'segments/{key}'][:]
                    points = hf[f'points/{key}'][:] if f'points/{key}' in hf else []
                    session['annotations'].append({
                        'segment_id': int(key),
                        'mask': mask,
                        'points': points.tolist() if isinstance(points, np.ndarray) else points
                    })
        
        # Update gallery with loaded segments
        gallery = [[(ann['mask'] * 255).astype(np.uint8), f"Segment {ann['segment_id']}"] for ann in session['annotations']]
        
        return session['image_copy'], f"Loaded {len(session['annotations'])} segments for image {session['id']}", session, gallery
    
    except Exception as e:
        return None, f"Error loading annotation: {str(e)}", session, []

# Add a clicked point and redraw with size relative to image dims
def add_point(img, session, evt: gr.SelectData):
    x, y = int(evt.index[0]), int(evt.index[1])
    session['points'].append([x, y])
    h, w = session['image_copy'].shape[:2]
    # Maintain same ratio: r=5 at 600px
    r = max(int(max(h, w) * 5 / 600), 2)
    pil = Image.fromarray(session['image_copy'])
    draw = ImageDraw.Draw(pil)
    draw.ellipse((x-r, y-r, x+r, y+r), fill=(255, 0, 0))
    session['image_copy'] = np.array(pil)
    return session['image_copy'], session

# Reset only points (and visual markers)
def reset_points(session):
    if session['image'] is None:
        return None, session, "No image to reset points on."
    session['points'] = []
    session['image_copy'] = session['image'].copy()
    return session['image_copy'], session, "Points reset."

# Reset entire session (including input image)
def reset_all(session, history):
    new_session = init_session()
    return None, None, None, new_session, []

# Run SAM segmentation
def run_sam(session):
    if session['image'] is None or len(session['points']) == 0:
        return [None, None, None, "Upload image and add at least one point.", session]
    rgb_numpy = session['image'].astype(np.uint8)
    segmenter.set_image(rgb_numpy)
    coords = np.array(session['points'])
    labels = np.ones(len(coords), dtype=int)
    masks, scores, logits = segmenter.predict(
        point_coords=coords,
        point_labels=labels,
        multimask_output=True
    )
    masks = masks[:3]
    session['masks'] = masks
    mask_imgs = [(mask * 255).astype(np.uint8) for mask in masks]
    outputs = [mask_imgs[i] if i < len(mask_imgs) else None for i in range(3)]
    return [*outputs, f"Generated {len(masks)} masks for image ID {session['id']}", session]

# Save mask via click and reset points, store both mask and points
def make_save_fn(mask_idx):
    def save_fn(img, session):
        if session['masks'] is None:
            return "Run SAM before saving.", session, []
        try:
            mask = session['masks'][mask_idx]
        except:
            return "Invalid mask index.", session, []
        seg_id = len(session['annotations'])
        # capture current points
        pts = session['points'].copy()
        session['annotations'].append({'segment_id': seg_id, 'mask': mask, 'points': pts})
        # reset points
        session['points'] = []
        session['image_copy'] = session['image'].copy()
        gallery = [[(ann['mask'] * 255).astype(np.uint8), f"Segment {ann['segment_id']}"] for ann in session['annotations']]
        return f"Saved segment {seg_id}.", session, gallery
    return save_fn

# Delete a saved mask segment
def delete_segment(segment_id, session):
    if not session['annotations']:
        return "No segments to delete.", session, []
    
    # Find and remove the segment with matching ID
    session['annotations'] = [ann for ann in session['annotations'] if ann['segment_id'] != segment_id]
    
    # Update the gallery with remaining segments
    gallery = [[(ann['mask'] * 255).astype(np.uint8), f"Segment {ann['segment_id']}"] for ann in session['annotations']]
    
    return f"Deleted segment {segment_id}.", session, gallery

# Save edited annotations
def save_annotations(annotation_dir, difficulty, session):
    if session['id'] is None or not session['annotations']:
        return "No segments to save.", session
    
    try:
        # Create the difficulty subdirectory if it doesn't exist
        save_dir = os.path.join(annotation_dir, difficulty)
        os.makedirs(save_dir, exist_ok=True)
        h5_path = os.path.join(save_dir, f"{session['id']}_annotation.hdf5")
        
        with h5py.File(h5_path, 'w') as hf:
            # Save image dimensions
            height, width = session['image'].shape[:2]
            hf.attrs['height'] = height
            hf.attrs['width'] = width
            
            # Save segments and points
            seg_grp = hf.create_group('segments')
            pts_grp = hf.create_group('points')
            for ann in session['annotations']:
                sid = str(ann['segment_id'])
                seg_grp.create_dataset(sid, data=ann['mask'])
                pts_grp.create_dataset(sid, data=np.array(ann['points']))
        
        return f"Saved {len(session['annotations'])} segments to {h5_path}", session
    
    except Exception as e:
        return f"Error saving annotations: {str(e)}", session

# Add this function after the other utility functions
def view_annotation_file(image_id, difficulty, annotation_dir):
    if not image_id:
        return None, "Please enter an image ID."
    
    annotation_path = os.path.join(annotation_dir, difficulty, f"{image_id}_annotation.hdf5")
    if not os.path.exists(annotation_path):
        return None, f"No annotation file found at {annotation_path}"
    
    try:
        with h5py.File(annotation_path, 'r') as hf:
            masks = []
            if 'segments' in hf:
                for key in hf['segments'].keys():
                    mask = hf[f'segments/{key}'][:]
                    masks.append([(mask * 255).astype(np.uint8), f"Segment {key}"])
            return masks, f"Loaded {len(masks)} segments from annotation file"
    except Exception as e:
        return None, f"Error reading annotation file: {str(e)}"

# Build Gradio UI
def build_ui(rgb_dir, annotation_dir):
    with gr.Blocks(title="SAM Segment Annotation Editor") as demo:
        gr.Markdown(
            """
            <div style='text-align:center'>
              <h1 style='font-size:2.5em'>SAM Segment Annotation Editor</h1>
              <p style='font-size:1.2em'>Load an existing annotation, edit segments, and save changes.</p>
            </div>
            """
        )
        with gr.Row():
            with gr.Column():
                # Create a list of all image files with their full paths
                image_files = []
                for difficulty in ["Easy", "Medium", "Hard"]:
                    diff_dir = os.path.join(rgb_dir, difficulty)
                    if os.path.exists(diff_dir):
                        for file in os.listdir(diff_dir):
                            if file.endswith('_rgb.png'):
                                image_files.append(os.path.join(diff_dir, file))
                
                file_browser = gr.Dropdown(
                    label="Select Image",
                    choices=image_files,
                    type="value"
                )
                image_id = gr.Textbox(label="Image ID (auto-filled)", interactive=False)
                current_difficulty = gr.Textbox(label="Difficulty (auto-filled)", interactive=False)
                load_button = gr.Button("Load Annotation")
                view_annotation_button = gr.Button("View Annotation File")
                image_display = gr.Image(label="Image (click to add points)", type="numpy", interactive=True)
                run_button = gr.Button("Run SAM")
                reset_points_button = gr.Button("Reset Points")
                reset_all_button = gr.Button("Reset All")
                save_button = gr.Button("Save Annotations")
            with gr.Column():
                gr.Markdown('**Click on a mask below to save it**')
                with gr.Row():
                    mask1 = gr.Image(label="Mask 1", type="numpy")
                    mask2 = gr.Image(label="Mask 2", type="numpy")
                    mask3 = gr.Image(label="Mask 3", type="numpy")
                with gr.Row():
                    segment_id_input = gr.Number(label="Segment ID to Delete", precision=0)
                    delete_button = gr.Button("Delete Segment")
                status = gr.Textbox(label="Status", interactive=False)
                annotation_gallery = gr.Gallery(label="Annotation File Masks", columns=4)
                saved_gallery = gr.Gallery(label="Saved Masks", columns=4)

        session = gr.State(init_session())

        def extract_info(filepath):
            if not filepath:
                return "", ""
            # Extract the image ID and difficulty from the filepath
            filename = os.path.basename(filepath)
            image_id = filename.split('_')[0]
            # Get the parent directory name (difficulty level)
            difficulty = os.path.basename(os.path.dirname(filepath))
            return image_id, difficulty

        file_browser.change(
            fn=extract_info,
            inputs=[file_browser],
            outputs=[image_id, current_difficulty]
        )

        def load_annotation_wrapper(image_id, difficulty):
            if not image_id:
                return None, "Please enter an image ID.", None, []
            # Reset session annotations before loading new ones
            session.value['annotations'] = []
            image_path = os.path.join(rgb_dir, difficulty, f"{image_id}_rgb.png")
            annotation_path = os.path.join(annotation_dir, difficulty, f"{image_id}_annotation.hdf5")
            if not os.path.exists(image_path) or not os.path.exists(annotation_path):
                return None, "Image or annotation file not found.", None, []
            return load_annotation(image_path, annotation_path, session.value)

        load_button.click(
            fn=load_annotation_wrapper,
            inputs=[image_id, current_difficulty],
            outputs=[image_display, status, session, saved_gallery]
        )
        
        image_display.select(fn=add_point, inputs=[image_display, session], outputs=[image_display, session])
        reset_points_button.click(fn=reset_points, inputs=[session], outputs=[image_display, session, status])
        reset_all_button.click(fn=reset_all, inputs=[session, gr.State([])], outputs=[image_display, status, session])
        run_button.click(fn=run_sam, inputs=[session], outputs=[mask1, mask2, mask3, status, session])
        mask1.select(fn=make_save_fn(0), inputs=[mask1, session], outputs=[status, session, saved_gallery])
        mask2.select(fn=make_save_fn(1), inputs=[mask2, session], outputs=[status, session, saved_gallery])
        mask3.select(fn=make_save_fn(2), inputs=[mask3, session], outputs=[status, session, saved_gallery])
        delete_button.click(fn=delete_segment, inputs=[segment_id_input, session], outputs=[status, session, saved_gallery])
        
        # Create a hidden textbox to store the annotation directory
        annotation_dir_state = gr.State(annotation_dir)
        
        def save_with_difficulty(annotation_dir, session):
            return save_annotations(annotation_dir, current_difficulty.value, session)
            
        save_button.click(fn=save_with_difficulty, inputs=[annotation_dir_state, session], outputs=[status, session])
        
        view_annotation_button.click(
            fn=view_annotation_file,
            inputs=[image_id, current_difficulty, gr.State(annotation_dir)],
            outputs=[annotation_gallery, status]
        )
        
        return demo

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            size = file.write(data)
            progress_bar.update(size)
    progress_bar.close()

def ensure_checkpoint(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}. Downloading...")
        url = "https://huggingface.co/spaces/abhishek/StableSAM/resolve/main/sam_vit_h_4b8939.pth"
        download_file(url, checkpoint_path)
        print(f"Checkpoint downloaded to {checkpoint_path}")
    else:
        print(f"Checkpoint found at {checkpoint_path}")
    return checkpoint_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM Segment Annotation Editor")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to SAM Checkpoint File")
    parser.add_argument("--device", type=str, default='mps' if torch.backends.mps.is_available() else 'cpu',
                       help="Device to run SAM on (mps for Apple Silicon, cpu for Intel Macs, cuda:0 for NVIDIA GPUs)")
    parser.add_argument("--rgb_dir", type=str, required=True, help="Directory containing RGB images")
    parser.add_argument("--annotation_dir", type=str, required=True, help="Directory containing annotations")
    args = parser.parse_args()

    device = args.device
    rgb_dir = args.rgb_dir
    annotation_dir = args.annotation_dir

    checkpoint_path = ensure_checkpoint(args.checkpoint)
    global sam
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
    sam.to(device)
    sam.eval()
    global segmenter
    segmenter = SamPredictor(sam)

    build_ui(rgb_dir, annotation_dir).launch(share=True) 
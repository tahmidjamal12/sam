import os
import numpy as np
import h5py
import gradio as gr
from PIL import Image
import re

def init_session():
    return {
        "id": None,
        "image": None,
        "masks": [],
        "combined_mask": None
    }

def load_annotation(image_path, annotation_path, session):
    if not os.path.exists(image_path) or not os.path.exists(annotation_path):
        return None, "Image or annotation file not found.", None, []
    
    try:
        # Load image
        img = np.array(Image.open(image_path))
        session['image'] = img
        
        # Extract image ID from filename
        session['id'] = os.path.basename(image_path).split('_')[0]
        
        # Load annotations from HDF5
        masks = []
        with h5py.File(annotation_path, 'r') as hf:
            if 'segments' in hf:
                for key in hf['segments'].keys():
                    mask = hf[f'segments/{key}'][:]
                    masks.append(((mask * 255).astype(np.uint8), key))
        
        session['masks'] = masks
        return session['image'], f"Loaded {len(masks)} segments for image {session['id']}", session, masks
    
    except Exception as e:
        return None, f"Error loading annotation: {str(e)}", session, []

def combine_masks(mask1_idx, mask2_idx, operation, session):
    if not session['masks'] or mask1_idx >= len(session['masks']) or mask2_idx >= len(session['masks']):
        return None, "Please select valid masks to combine."
    
    # Get the binary masks (0 or 1)
    mask1 = session['masks'][mask1_idx][0] > 127
    mask2 = session['masks'][mask2_idx][0] > 127
    
    if operation == "Add":
        combined = np.logical_or(mask1, mask2)
    else:  # Subtract
        combined = np.logical_and(mask1, np.logical_not(mask2))
    
    # Convert to uint8 for display
    combined_display = (combined * 255).astype(np.uint8)
    session['combined_mask'] = combined
    
    return combined_display, f"Combined masks {mask1_idx} and {mask2_idx} using {operation}"

def save_combined_mask(annotation_dir, difficulty, session):
    if session['id'] is None or session['combined_mask'] is None:
        return "No combined mask to save."
    
    try:
        save_dir = os.path.join(annotation_dir, difficulty)
        os.makedirs(save_dir, exist_ok=True)
        h5_path = os.path.join(save_dir, f"{session['id']}_annotation.hdf5")
        
        with h5py.File(h5_path, 'a') as hf:
            if 'segments' not in hf:
                seg_grp = hf.create_group('segments')
            else:
                seg_grp = hf['segments']
            existing_segments = set(seg_grp.keys())
            i = 0
            while str(i) in existing_segments:
                i += 1
            new_segment_name = str(i)
            seg_grp.create_dataset(new_segment_name, data=session['combined_mask'])
            after_count = len(seg_grp.keys())
        
        return f"Saved combined mask as '{new_segment_name}' in {h5_path}. Segments before: {len(existing_segments)}, after: {after_count}"
    
    except Exception as e:
        return f"Error saving combined mask: {str(e)}"

def delete_segment(annotation_dir, difficulty, image_id, segment_number):
    try:
        h5_path = os.path.join(annotation_dir, difficulty, f"{image_id}_annotation.hdf5")
        with h5py.File(h5_path, 'a') as hf:
            if 'segments' not in hf:
                return None, f"No segments found in {h5_path}"
            seg_grp = hf['segments']
            seg_name = str(int(segment_number))
            if seg_name not in seg_grp:
                return None, f"Segment {seg_name} not found."
            del seg_grp[seg_name]
            # Prepare updated gallery
            masks = []
            for key in seg_grp.keys():
                mask = seg_grp[key][:]
                masks.append(((mask * 255).astype(np.uint8), key))
        return masks, f"Deleted segment {seg_name} from {h5_path}"
    except Exception as e:
        return None, f"Error deleting segment: {str(e)}"

def build_ui(rgb_dir, annotation_dir):
    with gr.Blocks(title="Mask Combination Tool") as demo:
        gr.Markdown(
            """
            <div style='text-align:center'>
              <h1 style='font-size:2.5em'>Mask Combination Tool</h1>
              <p style='font-size:1.2em'>Load masks and combine them through addition or subtraction.</p>
            </div>
            """
        )
        
        # Create a list of all image files with their full paths
        image_files = []
        for difficulty in ["Easy", "Medium", "Hard"]:
            diff_dir = os.path.join(rgb_dir, difficulty)
            if os.path.exists(diff_dir):
                for file in os.listdir(diff_dir):
                    if file.endswith('_rgb.png'):
                        image_files.append(os.path.join(diff_dir, file))
        
        with gr.Row():
            with gr.Column():
                file_browser = gr.Dropdown(
                    label="Select Image",
                    choices=image_files,
                    type="value"
                )
                image_id = gr.Textbox(label="Image ID (auto-filled)", interactive=False)
                current_difficulty = gr.Textbox(label="Difficulty (auto-filled)", interactive=False)
                load_button = gr.Button("Load Image and Masks")
                image_display = gr.Image(label="Image", type="numpy")
                delete_number = gr.Number(label="Segment Number to Delete", precision=0)
                delete_button = gr.Button("Delete Segment")
            
            with gr.Column():
                mask1_number = gr.Number(label="First Mask Number (e.g. 0)", precision=0)
                mask2_number = gr.Number(label="Second Mask Number (e.g. 1)", precision=0)
                operation = gr.Radio(
                    choices=["Add", "Subtract"],
                    label="Operation",
                    value="Add"
                )
                combine_button = gr.Button("Combine Masks")
                combined_mask_display = gr.Image(label="Combined Mask", type="numpy")
                save_button = gr.Button("Save Combined Mask")
                status = gr.Textbox(label="Status", interactive=False)
        
        mask_gallery = gr.Gallery(label="Annotation File Masks", columns=4)
        
        session = gr.State(init_session())
        
        def extract_info(filepath):
            if not filepath:
                return "", ""
            filename = os.path.basename(filepath)
            image_id = filename.split('_')[0]
            difficulty = os.path.basename(os.path.dirname(filepath))
            return image_id, difficulty
        
        def gallery_masks(masks):
            # Each mask is (mask_img, label)
            return masks
        
        file_browser.change(
            fn=extract_info,
            inputs=[file_browser],
            outputs=[image_id, current_difficulty]
        )
        
        def load_wrapper(filepath, current_session):
            if not filepath:
                return None, "Please select an image.", current_session, [], []
            
            image_path = filepath
            image_id = os.path.basename(filepath).split('_')[0]
            difficulty = os.path.basename(os.path.dirname(filepath))
            annotation_path = os.path.join(annotation_dir, difficulty, f"{image_id}_annotation.hdf5")
            
            image, status_msg, current_session, masks = load_annotation(image_path, annotation_path, current_session)
            gallery = gallery_masks(masks)
            
            return image, status_msg, current_session, masks, gallery
        
        load_button.click(
            fn=load_wrapper,
            inputs=[file_browser, session],
            outputs=[image_display, status, session, gr.State([]), mask_gallery]
        )
        
        combine_button.click(
            fn=combine_masks,
            inputs=[mask1_number, mask2_number, operation, session],
            outputs=[combined_mask_display, status]
        )
        
        save_button.click(
            fn=save_combined_mask,
            inputs=[gr.State(annotation_dir), current_difficulty, session],
            outputs=[status]
        )
        
        delete_button.click(
            fn=delete_segment,
            inputs=[gr.State(annotation_dir), current_difficulty, image_id, delete_number],
            outputs=[mask_gallery, status]
        )
        
        return demo

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Mask Combination Tool")
    parser.add_argument("--rgb_dir", type=str, required=True, help="Directory containing RGB images")
    parser.add_argument("--annotation_dir", type=str, required=True, help="Directory containing annotations")
    args = parser.parse_args()

    build_ui(args.rgb_dir, args.annotation_dir).launch(share=True) 
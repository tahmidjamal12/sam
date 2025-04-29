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

# Generate a random 7-digit ID (for saving image) -- def can do something different though
def generate_id():
    return str(random.randint(10**6, 10**7 - 1))


# Initialize session and display image on upload
def init_display(img, session):
    if img is None:
        return None, session
    session = init_session()
    session['id'] = generate_id()
    session['image'] = img
    session['image_copy'] = img.copy()
    return session['image_copy'], session

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

# Finalize and reset entire session, update history, write segments+points
def finalize_and_reset(rgb_dir, ann_dir, session, history):
    if session['id'] is None or not session['annotations']:
        return "No segments to save.", None, None, None, None, [], session, history, history
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)
    img_id = session['id']
    filename = f"{img_id}_rgb.png"
    Image.fromarray(session['image']).save(os.path.join(rgb_dir, filename))
    h5_path = os.path.join(ann_dir, f"{img_id}_annotation.hdf5")
    height, width = session['image'].shape[:2]
    with h5py.File(h5_path, 'w') as hf:
        hf.attrs['filename'] = filename
        hf.attrs['height'] = height
        hf.attrs['width'] = width
        seg_grp = hf.create_group('segments')
        pts_grp = hf.create_group('points')
        for ann in session['annotations']:
            sid = str(ann['segment_id'])
            seg_grp.create_dataset(sid, data=ann['mask'])
            pts_grp.create_dataset(sid, data=np.array(ann['points']))
    # update session history
    entry = [[session['image'], f"Image {img_id}"]]
    entry += [[(ann['mask'] * 255).astype(np.uint8), f"Segment {ann['segment_id']}"] for ann in session['annotations']]
    new_history = history + [entry]
    new_session = init_session()
    return (
        f"Saved {len(entry)-1} segments for image {filename}",
        None, None, None, None,
        [], new_session,
        new_history,
        sum(new_history, [])
    )

# Build Gradio UI
def build_ui(rgb_save_path, annotation_save_path):
    with gr.Blocks(title="SAM Segment Manual Dataset") as demo:
        gr.Markdown(
            """
            <div style='text-align:center'>
              <h1 style='font-size:2.5em'>SAM Segment Manual Dataset</h1>
              <p style='font-size:1.2em'>Upload an image, click to add prompt points, then run SAM.<br>
              Click on a mask to save it. Use the buttons to reset points, reset all, or finalize annotations.</p>
            </div>
            """
        )
        with gr.Row():
            with gr.Column():
                rgb_dir = gr.Textbox(label="RGB Save Directory", value=rgb_save_path, interactive=True)
                ann_dir = gr.Textbox(label="Annotation Save Directory", value=annotation_save_path, interactive=True)
                image_input = gr.Image(sources="upload", label="Image (click to add points)", type="numpy", interactive=True)
                run_button = gr.Button("Run SAM")
                reset_points_button = gr.Button("Reset Points")
                reset_all_button = gr.Button("Reset All")
            with gr.Column():
                gr.Markdown('**Click on a mask below to save it**')
                with gr.Row():
                    mask1 = gr.Image(label="Mask 1", type="numpy")
                    mask2 = gr.Image(label="Mask 2", type="numpy")
                    mask3 = gr.Image(label="Mask 3", type="numpy")
                status = gr.Textbox(label="Status", interactive=False)
                saved_gallery = gr.Gallery(label="Saved Masks", columns=4)
                finalize_button = gr.Button("Finalize & Reset")
        with gr.Row():
            session_gallery = gr.Gallery(label="Session History", columns=4)

        session = gr.State(init_session())
        history_state = gr.State([])

        image_input.upload(fn=init_display, inputs=[image_input, session], outputs=[image_input, session])
        image_input.select(fn=add_point, inputs=[image_input, session], outputs=[image_input, session])
        reset_points_button.click(fn=reset_points, inputs=[session], outputs=[image_input, session, status])
        reset_all_button.click(fn=reset_all, inputs=[session, history_state], outputs=[image_input, session, status, session, history_state])
        run_button.click(fn=run_sam, inputs=[session], outputs=[mask1, mask2, mask3, status, session])
        mask1.select(fn=make_save_fn(0), inputs=[mask1, session], outputs=[status, session, saved_gallery])
        mask2.select(fn=make_save_fn(1), inputs=[mask2, session], outputs=[status, session, saved_gallery])
        mask3.select(fn=make_save_fn(2), inputs=[mask3, session], outputs=[status, session, saved_gallery])
        finalize_button.click(
            fn=finalize_and_reset,
            inputs=[rgb_dir, ann_dir, session, history_state],
            outputs=[status, image_input, mask1, mask2, mask3, saved_gallery, session, history_state, session_gallery]
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
    parser = argparse.ArgumentParser(description="SAM Segment Manual Dataset")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to SAM Checkpoint File")
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--rgb_save_path", type=str, required=True, help="Path to save RGB image")
    parser.add_argument("--annotation_save_path", type=str, required=True, help="Path to save annotations")
    args = parser.parse_args()

    device = args.device
    rgb_save_path = args.rgb_save_path
    annotation_save_path = args.annotation_save_path # "/ccn2/u/lilianch/data/segment/annotations/"

    checkpoint_path = ensure_checkpoint(args.checkpoint)
    global sam
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
    sam.to(device)
    sam.eval()
    global segmenter
    segmenter = SamPredictor(sam)

    build_ui(rgb_save_path, annotation_save_path).launch(share=True)


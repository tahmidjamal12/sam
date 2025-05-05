import os
import json
import numpy as np
import math
import matplotlib.pyplot as plt
import glob
import cv2
import random
from pycocotools import mask as mask_util
from PIL import Image, ImageOps
import h5py

def segments_to_rgb(segments):
    """
    Takes in list of segments, returns rgb image of segments in one image
    """
    # Assume all segment maps have the same shape
    height, width = segments[0].shape
    rgb_image = np.zeros((height, width, 3), dtype=np.float32)

    # Assign a unique color to each segment
    unique_colors = [tuple(np.random.randint(0, 256, 3)) for _ in range(len(segments))]

    for idx, segment in enumerate(segments):
        color = unique_colors[idx]
        # Apply the color to the RGB image where the binary map is 1
        for channel in range(3):
            rgb_image[:, :, channel] += segment.astype(np.float32) * color[channel]

    # Clip values to valid range and convert back to uint8
    rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)

    return rgb_image

def crop(image, annotations_image_id, size=256):
    """
    Apply square crop on images and associated segments
    """
    cropped = False
    h, w, _ = image.shape

    # Resize and crop
    scale = max(size / h, size / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    start_h = (new_h - size) // 2
    start_w = (new_w - size) // 2
    image_cropped = image_resized[start_h:start_h + size, start_w:start_w + size]

    cropped_segs = {}
    all_segs = []

    for annot in annotations_image_id:
        seg = annot['segmentation']
        id = (annot['iscrowd'], annot['area'], annot['image_id'], annot['category_id'], annot['attribute'], annot['id'])
        seg_mask = mask_util.decode(seg)
        seg_mask_resized = cv2.resize(seg_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        seg_mask_cropped = seg_mask_resized[start_h:start_h + size, start_w:start_w + size]

        # Skip if cropped segment mask is completely out of frame
        if seg_mask_cropped.sum() == 0:
            continue

        # if passes filtering, add it
        cropped_segs[id] = seg_mask_cropped
        all_segs.append(annot)

    return image_cropped, cropped_segs, all_segs

def get_seg_map(images_dir, filename, image_id, annotation, categories, size=256):
    """
    Applies semantic filtering, returns relevant segments
    """
    image_path = os.path.join(images_dir, filename)
    image_og = cv2.imread(image_path)[:, :, ::-1]  # BGR to RGB
    h, w, _ = image_og.shape
    all_segs = []
    all_ids = []
    exclude_segs = []

    annotations_image_id = [x for x in annotation if x['image_id'] == image_id]
    image, cropped_segs, annotations = crop(image_og, annotations_image_id)

    # TODO: FILTER annotations, can change annotations
    for annot in annotations:
        id = (annot['iscrowd'], annot['area'], annot['image_id'], annot['category_id'], annot['attribute'], annot['id'])
        seg_mask = cropped_segs[id]
        category_id = annot['category_id']

        for category in categories:
            if category['id'] == category_id:
                if category['type'] == 'thing' and category['supercategory'] not in ['table', 'cabinets', 'kitchen_appliances', 'traffic_facility', 'stove', 'light'] and category['name'] not in ['kitchen_sink', 'billboard', 'blackboard', 'signboard', 'guideboard', 'rug', 'curtain']:
                    # edge = seg_mask[0, :].sum() +  seg_mask[-1, :].sum() + seg_mask[:, 0].sum() + seg_mask[:, -1].sum()
                    if (seg_mask.sum() / (size * size)) * 100 > 0.5: # and edge < 256:
                        all_segs.append(seg_mask)
                        all_ids.append(annot['id'])
                    else:
                        exclude_segs.append(seg_mask)

    exclude_rgb = None
    if len(exclude_segs) != 0:
        exclude_rgb = segments_to_rgb(exclude_segs)
    seg_map = None
    if len(all_segs) != 0:
        seg_map = segments_to_rgb(all_segs)


    return image, seg_map, len(all_segs), exclude_rgb, len(exclude_segs), all_ids, all_segs


def process_and_merge_visonly(images_dir, image_data, annotations, vis_dir, index, categories):
    # global vis_info
    np.random.seed(24)

    filename = image_data['file_name']
    image_id = image_data['id']

    # Generate segmentation map and related data
    rgb_img, seg_map, seg_len, exclude_map, exclude_len, all_ids, _ = get_seg_map(images_dir, filename, image_id, annotations, categories)

    if seg_len == 0:
        return

    if seg_len < 2: # filter out too simple scenes
        return

    # set up visual!
    if exclude_len != 0:
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    else:
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    # Show original image8ax[0].imshow(rgb_img)
    ax[0].imshow(rgb_img)
    ax[0].set_title(f'RGB: {filename}')
    ax[1].imshow(seg_map)
    ax[1].set_title(f'Segment Map: {seg_len}')
    if exclude_len != 0:
        ax[2].set_title(f'Non-Stuff Excluded Segment: {exclude_len}')
        ax[2].imshow(exclude_map)

    merged_path = os.path.join(vis_dir, "result_" + str(index) + ".png")
    fig.savefig(merged_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return

def process_and_merge_h5(images_dir, image_data, annotations, vis_dir, index, categories, h5_path):
    np.random.seed(24)

    filename = image_data['file_name']
    image_id = image_data['id']

    # Generate segmentation map and related data - now also returning individual segment masks (all_segs)
    rgb_img, seg_map, seg_len, exclude_map, exclude_len, all_ids, all_segs = get_seg_map(
        images_dir, filename, image_id, annotations, categories
    )

    # Create visualization
    if vis_dir:
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))

        ax[0].imshow(rgb_img)
        ax[0].set_title(f'RGB: {filename}')
        ax[1].imshow(seg_map)
        ax[1].set_title(f'Segment Map: {seg_len}')

        merged_path = os.path.join(vis_dir, f"result_{index}.png")
        fig.savefig(merged_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    # Save data to H5 file (if provided)
    # Stack the segmentation masks to create an N×H×W array
    stacked_segments = np.stack(all_segs)
    seg_map = np.transpose(seg_map, (2, 0, 1))
    if stacked_segments is not None:
        image_key = f'image_{index}'

        with h5py.File(h5_path, 'a') as f:
            # Create a group for this image if it doesn't exist
            if image_key in f:
                del f[image_key]  # Remove existing data if any

            g = f.create_group(image_key)

            # Store filename as attribute
            g.attrs['filename'] = filename
            # Store RGB image
            g.create_dataset('rgb', data=rgb_img, compression='gzip')

            # Store stacked segment masks (N, H, W)
            g.create_dataset('segment', data=stacked_segments, compression='gzip')

            # Store segment IDs
            g.create_dataset('segment_ids', data=np.array(all_ids), compression='gzip')

            g.create_dataset('seg_map', data=np.array(seg_map), compression='gzip')


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    # dataset
    parser.add_argument("--ann_path", type=str,
                        default="/ccn2/u/lilianch/data/entityseg/entityseg_train_01.json",
                        help="path to annotations json")
    parser.add_argument("--images_dir", type=str,
                        default="/ccn2/u/rmvenkat/data/entity_seg_dataset/entity_seg_images/entity_01_11580",
                        help="path to directory of entityseg images")

    # save paths
    parser.add_argument("--vis_dir", type=str,
                        required=True,
                        help="path to save visualizations")
    parser.add_argument("--h5_path", type=str,
                        required=False,
                        default=None,
                        # default="/ccn2/u/lilianch/data/entityseg_100.h5",
                        help="path to save h5 file, if none then h5 is not saved")

    args = parser.parse_args()
    np.random.seed(24)

    # Paths
    ann_path = args.ann_path  # Path to JSON
    images_dir = args.images_dir  # Directory containing the images
    vis_dir = args.vis_dir
    # json_file_path = args.json_file_path

    os.makedirs(vis_dir, exist_ok=True)

    with open(ann_path, 'r') as f:
        data = json.load(f)

    entity_seg = json.load(open(ann_path))
    annotation = entity_seg['annotations']
    categories = entity_seg['categories']

    for ct, images in enumerate(data['images'][:500]):
        if (ct % 100 == 0): print(f"Result {ct} Fin")
        if args.h5_path:
            process_and_merge_h5(images_dir, images, annotation, vis_dir, ct, categories, args.h5_path)
        else:
            process_and_merge_visonly(images_dir, images, annotation, vis_dir, ct, categories)

    print(f"Entityseg visualized saved to {vis_output_path}")

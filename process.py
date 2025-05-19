import numpy as np
import cv2
import h5py
import os

def crop_image(image, size=256):
    """
    Crops and resizes an image to a square of given size.
    """
    h, w, _ = image.shape
    scale = max(size / h, size / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    start_h = (new_h - size) // 2
    start_w = (new_w - size) // 2
    return image_resized[start_h:start_h + size, start_w:start_w + size]

def crop_annotation(h5_path, size=256):
    """
    Crops an annotation HDF5 file and removes it if all segments are out of bounds.
    """
    with h5py.File(h5_path, 'r+') as hf:
        segments = hf['segments']
        points = hf['points']
        valid_segments = {}
        valid_points = {}

        for sid in list(segments.keys()):
            mask = segments[sid][()]
            h, w = mask.shape
            scale = max(size / h, size / w)
            new_h, new_w = int(round(h * scale)), int(round(w * scale))
            mask_resized = cv2.resize(mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            start_h = (new_h - size) // 2
            start_w = (new_w - size) // 2
            mask_cropped = mask_resized[start_h:start_h + size, start_w:start_w + size]

            if np.any(mask_cropped):
                valid_segments[sid] = mask_cropped
                if sid in points:
                    valid_points[sid] = points[sid][()]

        if not valid_segments:
            hf.close()
            os.remove(h5_path)
            return False  # Indicates deletion

        # Clear and rewrite datasets
        for group in ['segments', 'points']:
            if group in hf:
                del hf[group]
        seg_grp = hf.create_group('segments')
        pts_grp = hf.create_group('points')

        for sid in valid_segments:
            seg_grp.create_dataset(sid, data=valid_segments[sid])
            if sid in valid_points:
                pts_grp.create_dataset(sid, data=valid_points[sid])

    return True

import os
import numpy as np
from tqdm import tqdm
current_path = os.getcwd()
# Define the relative paths to your data and .venv folders
venv_path = os.path.join(current_path, '.venv')

# Use the relative paths in your code
OPENSLIDE_PATH = os.path.join(venv_path, 'Lib', 'site-packages', 'openslide-win64-20230414', 'bin')

if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

from src import preprocessing as pre

def main():
    data_root = os.path.join(current_path, 'data')
    slide_root = os.path.join(data_root, 'raw')
    annotations_root = os.path.join(data_root, 'annotations')
    save_path = os.path.join(data_root, 'masks')
    test_path = os.path.join(slide_root, 'test')
    test_slides = os.listdir(test_path)

    tqdm.write("Processing test set...")

    for slide_filename in tqdm(test_slides):
        slide_id = os.path.splitext(slide_filename)[0]
        annotation_path = os.path.join(annotations_root, 'test', f"{slide_id}.xml")
        if os.path.exists(annotation_path):
            mask_save_path = os.path.join(save_path, 'test', f"{slide_id}.npz")
            slide = openslide.open_slide(os.path.join(test_path, slide_filename))
            slide_dims = slide.dimensions
            polygon_coords = pre.annotations_to_coordinates_flipped(annotation_path)
            mask = pre.coordinates_to_mask_cv2(polygon_coords, slide_dims)
            np.savez_compressed(mask_save_path, mask=mask)
            slide.close()


if __name__ == "__main__":
    main()
import os
import numpy as np
from tqdm import tqdm

# Get the current working directory to base all other paths off of it
current_path = os.getcwd()

# Define the relative paths to your data and the virtual environment for openslide
venv_path = os.path.join(current_path, '.venv')
OPENSLIDE_PATH = os.path.join(venv_path, 'Lib', 'site-packages', 'openslide-win64-20230414', 'bin')

# Ensure the OpenSlide binary is available in the path on Windows systems
if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

# Import custom preprocessing module
from src import preprocessing as pre

def main():
    # Define the paths for data storage, raw slides, annotations, and mask save location
    data_root = os.path.join(current_path, 'data')
    slide_root = os.path.join(data_root, 'raw')
    annotations_root = os.path.join(data_root, 'annotations')
    save_path = os.path.join(data_root, 'masks')

    # Define paths to tumor slides and test slides
    tumor_path = os.path.join(slide_root, 'train', 'tumor')
    test_path = os.path.join(slide_root, 'test')

    # List the filenames of the tumor slides
    tumor_slides = os.listdir(tumor_path)

    tqdm.write("Processing training set...")
    # Process each slide in the training set
    for slide_filename in tqdm(tumor_slides):
        slide_id = os.path.splitext(slide_filename)[0]
        annotation_path = os.path.join(annotations_root, 'train', f"{slide_id}.xml")
        mask_save_path = os.path.join(save_path, 'train', f"{slide_id}.npz")

        # Open the whole slide image using OpenSlide
        slide = openslide.open_slide(os.path.join(tumor_path, slide_filename))
        slide_dims = slide.dimensions

        # Extract coordinates from XML and create a binary mask
        polygon_coords = pre.annotations_to_coordinates(annotation_path)
        mask = pre.coordinates_to_mask(polygon_coords, slide_dims)

        # Save the mask in a compressed format
        np.savez_compressed(mask_save_path, mask=mask)

        # Close the slide file to free resources
        slide.close()

    # Process each slide in the test set
    tqdm.write("Processing test set...")
    test_slides = os.listdir(test_path)

    for slide_filename in tqdm(test_slides):
        slide_id = os.path.splitext(slide_filename)[0]
        annotation_path = os.path.join(annotations_root, 'test', f"{slide_id}.xml")

        # Check if there is an associated annotation file
        if os.path.exists(annotation_path):
            mask_save_path = os.path.join(save_path, 'test', f"{slide_id}.npz")

            # Repeat the process as done with the training set
            slide = openslide.open_slide(os.path.join(test_path, slide_filename))
            slide_dims = slide.dimensions
            polygon_coords = pre.annotations_to_coordinates(annotation_path)
            mask = pre.coordinates_to_mask(polygon_coords, slide_dims)
            np.savez_compressed(mask_save_path, mask=mask)
            slide.close()


if __name__ == "__main__":
    main()


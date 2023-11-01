import os
import numpy as np
from src import preprocessing as pre
from tqdm import tqdm
# Get the current working directory
current_path = os.getcwd()

# Define the relative paths to your data and .venv folders
venv_path = os.path.join(current_path, '.venv')

# Set the path for OpenSlide
OPENSLIDE_PATH = os.path.join(venv_path, 'Lib', 'site-packages', 'openslide-win64-20230414', 'bin')

# Import OpenSlide, accounting for Windows-specific DLL loading
if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

def main():
    # Define directories for slides, annotations, and masks
    data_root = os.path.join(current_path, 'data')
    slide_root = os.path.join(data_root, 'raw')
    annotations_root = os.path.join(data_root, 'annotations')
    save_path = os.path.join(data_root, 'masks')

    # Directory for tumor slides in the training set
    tumor_path = os.path.join(slide_root, 'train', 'tumor')
    # Directory for slides in the test set
    test_path = os.path.join(slide_root, 'test')

    # List all tumor slides
    tumor_slides = os.listdir(tumor_path)

    tqdm.write("Processing training set...")

    # Loop over each slide in the training set
    for slide_path in tqdm(tumor_slides):
        slide_id = os.path.splitext(slide_path)[0]
        annotation_path = os.path.join(annotations_root, 'train', f"{slide_id}.xml")
        mask_save_path = os.path.join(save_path, 'train', f"{slide_id}.npy")

        # Open the slide using OpenSlide
        slide = openslide.open_slide(os.path.join(tumor_path, slide_path))
        slide_dims = slide.dimensions

        # Generate the binary mask
        polygon_coords = pre.annotations_to_coordinates(annotation_path)
        mask = pre.coordinates_to_mask(polygon_coords, slide_dims)

        # Save the mask to disk
        np.savez_compressed(mask_save_path, mask=mask)

        # Close the slide to free resources
        slide.close()

    # List all slides in the test set
    test_slides = os.listdir(test_path)

    tqdm.write("Processing test set...")

    # Loop over each slide in the test set
    for slide_path in tqdm(test_slides):
        slide_id = os.path.splitext(slide_path)[0]
        annotation_path = os.path.join(annotations_root, 'test', f"{slide_id}.xml")

        # Check if annotation exists for the test slide
        if os.path.exists(annotation_path):
            mask_save_path = os.path.join(save_path, 'test', f"{slide_id}.npy")

            # Open the slide using OpenSlide
            slide = openslide.open_slide(os.path.join(test_path, slide_path))
            slide_dims = slide.dimensions

            # Generate the binary mask
            polygon_coords = pre.annotations_to_coordinates(annotation_path)
            mask = pre.coordinates_to_mask(polygon_coords, slide_dims)

            # Save the compmressedmask to disk
            np.savez_compressed(mask_save_path, mask=mask)

            # Close the slide to free resources
            slide.close()
        else:
            tqdm.write(f"Annotation for {slide_id} not found. Skipping.")
if __name__ == "__main__":
    main()

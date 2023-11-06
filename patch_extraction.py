import os
from tqdm import tqdm
# Get the current working directory
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
from src.preprocessing import extract_and_save_patches_and_labels

def main():
    data_root = os.path.join(current_path, 'data')
    raw_root = os.path.join(data_root, 'raw')
    mask_root = os.path.join(data_root, 'masks')
    preprocessed_patches_root = os.path.join(data_root, 'preprocessed_patches')
    
    # Handle training set
    tqdm.write("Processing training set...")
    for subset in ['train']:
        for slide_type in ['normal', 'tumor']:
            slide_folder = os.path.join(raw_root, subset, slide_type)
            tqdm.write(f"Processing {slide_type} slides...")
            if os.path.exists(slide_folder):
                process_slides(slide_folder, subset, slide_type, mask_root, preprocessed_patches_root)
    # Handle test set
    tqdm.write("Processing test set...")
    test_folder = os.path.join(raw_root, 'test')
    if os.path.exists(test_folder):
        process_test_slides(test_folder, 'test', mask_root, preprocessed_patches_root)

def process_slides(slide_folder, subset, slide_type, mask_root, preprocessed_patches_root):
    for slide in tqdm(os.listdir(slide_folder)):
        slide_path = os.path.join(slide_folder, slide)
        
        # Determine if there's an associated annotation
        mask_path = None
        if slide_type == 'tumor':
            mask_file = slide.replace('.tif', '.npz')  # Assuming .tif and .npz extensions
            mask_path = os.path.join(mask_root, subset, mask_file)
            
            if not os.path.exists(mask_path):
                tqdm.write(f"Annotation for {slide} not found. Skipping.")
                continue
        
        # Define the save path
        save_folder = os.path.join(preprocessed_patches_root, subset)
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, slide.replace('.tif', '.h5'))
        
        # Call your patch extraction function
        extract_and_save_patches_and_labels(
            slide_path, save_path, tissue_threshold=0.3,  # other parameters here
            mask_path=mask_path, patch_size=(512, 512), stride=(512, 512),
            enable_logging=True
        )

def process_test_slides(test_folder, subset, mask_root, preprocessed_patches_root):
    for slide in tqdm(os.listdir(test_folder)):
        slide_path = os.path.join(test_folder, slide)
        
        # Determine if there's an associated annotation
        mask_path = None
        mask_file = slide.replace('.tif', '.npz')  # Assuming .tif and .npz extensions
        mask_path_candidate = os.path.join(mask_root, subset, mask_file)
        
        if os.path.exists(mask_path_candidate):
            mask_path = mask_path_candidate
        
        # Define the save path
        save_folder = os.path.join(preprocessed_patches_root, subset)
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, slide.replace('.tif', '.h5'))
        
        # Call your patch extraction function
        extract_and_save_patches_and_labels(
            slide_path, save_path, tissue_threshold=0.3,  # other parameters here
            mask_path=mask_path, patch_size=(512, 512), stride=(512, 512),
            enable_logging=True
        )

if __name__ == '__main__':
    main()

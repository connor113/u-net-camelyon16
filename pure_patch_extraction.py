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
from src.preprocessing import extract_pure_patches

def main():
    data_root = os.path.join(current_path, 'data')
    test_root = os.path.join(data_root, 'test')
    file_list = [os.path.join(test_root, file) for file in os.listdir(test_root)]

    new_file_path = os.path.join(data_root, 'new_test.h5')
    # Handle test set
    tqdm.write("Processing test...")

    extract_pure_patches(file_list, new_file_path, True)

if __name__ == '__main__':
    main()
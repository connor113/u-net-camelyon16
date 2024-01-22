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
from src.preprocessing import modify_and_extract_patches

def main():
    data_root = os.path.join(current_path, 'data/preprocessed_patches')
    train_root = os.path.join(data_root, 'train/tumor')
    test_root = os.path.join(data_root, 'test/test')
    train_list = [os.path.join(train_root, file) for file in os.listdir(train_root)]
    test_list = [os.path.join(test_root, file) for file in os.listdir(test_root)]

    train_file_path = os.path.join(data_root, 'pure_train.h5')
    test_file_path = os.path.join(data_root, 'pure_test.h5')
    # Handle test set
    tqdm.write("Processing train...")

    modify_and_extract_patches(train_list, train_file_path, True)

    tqdm.write("Processing test...")

    modify_and_extract_patches(test_list, test_file_path, True)

if __name__ == '__main__':
    main()
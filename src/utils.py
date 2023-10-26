import os
from openslide import open_slide
import numpy as np
from matplotlib import pyplot as plt
import cv2
import h5py
from src.preprocessing import annotations_to_coordinates, coordinates_to_mask

def visualize_mask_on_slide(slide, mask, level=0):
    """
    Visualize a binary mask overlaid on the original image from an OpenSlide object.
    
    Parameters:
    - slide: The OpenSlide slide object.
    - mask: Binary mask where 1 indicates foreground and 0 indicates background.
    - level: Desired level or magnification.
    
    """
    # Extract the image at the desired level
    img = slide.read_region((0, 0), level, slide.level_dimensions[level])
    img = np.array(img)[:, :, :3]  # Exclude the alpha channel

    # Create a colored mask (e.g., red for foreground)
    upsampled_mask = cv2.resize(mask, slide.level_dimensions[level], interpolation=cv2.INTER_LINEAR)
    colored_mask = np.zeros((upsampled_mask.shape[0], upsampled_mask.shape[1], 3), dtype=np.uint8)
    colored_mask[upsampled_mask == 1] = [255, 0, 0]  # Red color for foreground

    # Plotting
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.imshow(colored_mask, alpha=0.4)  # Overlay with transparency
    plt.axis('off')
    plt.show()


def visualize_mask_on_slide_side_by_side(slide, mask, level=0):
    """
    Visualize a binary mask alongside and overlaid on the original image from an OpenSlide object.
    
    Parameters:
    - slide: The OpenSlide slide object.
    - mask: Binary mask where 1 indicates foreground and 0 indicates background.
    - level: Desired level or magnification.
    
    """
    # Extract the image at the desired level
    img = slide.read_region((0, 0), level, slide.level_dimensions[level])
    img = np.array(img)[:, :, :3]  # Exclude the alpha channel

    # Create a colored mask (e.g., red for foreground)
    upsampled_mask = cv2.resize(mask, slide.level_dimensions[level], interpolation=cv2.INTER_LINEAR)
    colored_mask = np.zeros((upsampled_mask.shape[0], upsampled_mask.shape[1], 3), dtype=np.uint8)
    colored_mask[upsampled_mask == 1] = [255, 0, 0]  # Red color for foreground

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    
    # Original image with overlaid mask
    ax[0].imshow(img)
    ax[0].imshow(colored_mask, alpha=0.4)  # Overlay with transparency
    ax[0].axis('off')
    ax[0].set_title('Image with Mask Overlay')
    
    # Binary mask
    ax[1].imshow(upsampled_mask, cmap='gray')
    ax[1].axis('off')
    ax[1].set_title('Binary Mask')
    
    plt.tight_layout()
    plt.show()


def visualise_patch_and_label(patch, label):
    """
    Visualise a patch and its corresponding label.

    Parameters:
    - patch (np.array): A 3D numpy array representing the patch.
    - label (np.array): A 2D numpy array representing the label for each pixel in the patch.

    """
    coloured_label = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    coloured_label[label == 1] = [255, 0, 0]  # Red color for foreground
    
    # Plotting
    plt.figure(figsize=(5, 5))
    plt.imshow(patch)
    plt.imshow(coloured_label, alpha=0.4)  # Overlay with transparency
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_patches_from_hdf5(hdf5_path, num_patches=5):
    """
    Visualize patches and their corresponding labels from an HDF5 file.

    Parameters:
        hdf5_path (str): Path to the HDF5 file.
        num_patches (int): Number of random patches to visualize.
    """
    with h5py.File(hdf5_path, 'r') as f:
        # Assuming the file structure is WSI/Level/patches and WSI/Level/labels
        wsi_names = list(f.keys())
        level_names = list(f[wsi_names[0]].keys())
        
        patch_group = f[f"{wsi_names[0]}/{level_names[0]}/patches"]
        label_group = f[f"{wsi_names[0]}/{level_names[0]}/labels"]
        
        patch_names = list(patch_group.keys())
        label_names = list(label_group.keys())
        
        # Randomly select patches to visualize
        selected_indices = np.random.choice(len(patch_names), num_patches, replace=False)
        
        for idx in selected_indices:
            patch_name = patch_names[idx]
            label_name = label_names[idx]  # Assuming patch and label names correspond
            
            patch_data = np.array(patch_group[patch_name])
            label_data = np.array(label_group[label_name])
            
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            
            axes[0].imshow(patch_data)
            axes[0].set_title(f"Patch: {patch_name}")
            axes[0].axis("off")
            
            axes[1].imshow(label_data, cmap='gray')
            axes[1].set_title(f"Label: {label_name}")
            axes[1].axis("off")
            
            plt.show()


def visualize_tumor_patches_from_hdf5_and_annotations(hdf5_path, annotation_path, num_patches=5):
    """
    Visualize patches that overlap with tumor regions in an HDF5 file using XML annotations.

    Parameters:
        hdf5_path (str): Path to the HDF5 file.
        annotation_path (str): Path to the XML annotation file.
        num_patches (int): Number of random patches to visualize.
    """
    # Convert annotations to a binary mask
    polygon_coords = annotations_to_coordinates(annotation_path)

    slide_name = os.path.basename(hdf5_path)
    
    with h5py.File(hdf5_path, 'r') as f:
        wsi_names = list(f.keys())
        print(f"Slide name: {wsi_names[0]}")
        slide_dims = f[wsi_names[0]].attrs['slide_dimensions']
        tumor_mask = coordinates_to_mask(polygon_coords, slide_dims)

        # Find coordinates where the tumor mask is 1
        tumor_coords = np.column_stack(np.where(tumor_mask == 1))

        # Randomly select tumor coordinates
        selected_tumor_coords = tumor_coords[np.random.choice(tumor_coords.shape[0], num_patches, replace=False), :]

    with h5py.File(hdf5_path, 'r') as f:
        # Assuming the file structure is WSI/Level/patches and WSI/Level/labels
        wsi_names = list(f.keys())
        level_names = list(f[wsi_names[0]].keys())
        
        patch_group = f[f"{wsi_names[0]}/{level_names[0]}/patches"]
        label_group = f[f"{wsi_names[0]}/{level_names[0]}/labels"]

        for y, x in selected_tumor_coords:
        
            patch_name = f"patch_{slide_name}_{x}_{y}"
            label_name = f"label_{slide_name}_{x}_{y}"

            if patch_name in patch_group and label_name in label_group:
                patch_data = np.array(patch_group[patch_name])
                label_data = np.array(label_group[label_name])

                fig, axes = plt.subplots(1, 2, figsize=(10, 5))

                axes[0].imshow(patch_data)
                axes[0].set_title(f"Patch: {patch_name}")
                axes[0].axis("off")

                axes[1].imshow(label_data, cmap='gray')
                axes[1].set_title(f"Label: {label_name}")
                axes[1].axis("off")

                plt.show()    
from openslide import open_slide
import numpy as np
from matplotlib import pyplot as plt
import cv2

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
import os
from openslide import open_slide
import numpy as np
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET
import random
import cv2
from skimage.draw import polygon


def foreground_background_segmentation(slide_path, input_level=3, output_level=0):
    """
    Segment the foreground and background of a given slide using Otsu thresholding on its HSV representation.

    Parameters:
    - slide_path (str): The path to the slide to be segmented.
    - input_level (int, optional): The level of the slide to be used for segmentation. Defaults to 3.
    - output_level (int, optional): The level of the output mask resolution. Defaults to 0.

    Returns:
    - np.array: A binary mask where 1 indicates foreground and 0 indicates background.

    Approach:
    - Open the slide using the OpenSlide library.
    - Convert the RGB slide image to the HSV color space.
    - Apply Otsu thresholding on the H (hue) and S (saturation) channels.
    - Combine the two binary masks to get the final segmentation.
    - If the desired output level is different from the input level, upsample the mask to the desired level.

    Note:
    The function uses the hue and saturation channels from the HSV space as they can provide good distinction 
    between tissue and background, especially when there are variations in staining and lighting.
    """

    slide = open_slide(slide_path)
    
    # Extract the slide image at the desired input level
    slide_image = slide.read_region((0, 0), input_level, slide.level_dimensions[input_level])
    slide_image = np.array(slide_image)[:, :, :3]  # Convert PIL to numpy and remove any alpha channel

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(slide_image, cv2.COLOR_RGB2HSV)
    
    # Apply Otsu thresholding on the H and S channels of the HSV image
    _, h_thresh = cv2.threshold(hsv_image[:, :, 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, s_thresh = cv2.threshold(hsv_image[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Convert thresholded images to binary masks
    h_thresh = h_thresh / 255
    s_thresh = s_thresh / 255

    # Combine the two binary masks using a bitwise AND operation
    combined_mask = cv2.bitwise_and(h_thresh, s_thresh)
    combined_mask = combined_mask.astype(np.uint8)

    # If the desired output level is different from the input level, upsample the mask to the desired resolution
    if output_level != input_level:
        upsampled_mask = cv2.resize(combined_mask, slide.level_dimensions[output_level], interpolation=cv2.INTER_LINEAR)
        return upsampled_mask

    return combined_mask


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
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    colored_mask[mask == 1] = [255, 0, 0]  # Red color for foreground

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
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    colored_mask[mask == 1] = [255, 0, 0]  # Red color for foreground

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    
    # Original image with overlaid mask
    ax[0].imshow(img)
    ax[0].imshow(colored_mask, alpha=0.4)  # Overlay with transparency
    ax[0].axis('off')
    ax[0].set_title('Image with Mask Overlay')
    
    # Binary mask
    ax[1].imshow(mask, cmap='gray')
    ax[1].axis('off')
    ax[1].set_title('Binary Mask')
    
    plt.tight_layout()
    plt.show()


def annotations_to_coordinates(annotation_path):
    """
    Parses the given XML file to extract coordinates of annotated regions.

    Parameters:
    - annotation_path (str): Path to the XML file.

    Returns:
    - polygons (list): List of polygons where each polygon is represented as a list of 
                       (x, y) coordinate tuples.

    Note:
    The coordinates are provided in the form (y, x) which corresponds to (row, column) in 
    image matrices. This is the standard convention for image processing tasks.
    
    """

    # Parse the XML file
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    # Extracting the polygons from the annotations
    polygons = []
    for annotation in root.findall('.//Annotation'):
        # Extracting (y, x) coordinates for each annotated point
        points = [(float(coord.attrib['Y']), float(coord.attrib['X'])) for coord in annotation.findall('.//Coordinate')]
        polygons.append(points)

    return polygons


def coordinates_to_mask(polygon_coords, slide_dims):
    """
    Convert a list of polygon coordinates to a binary mask.
    
    Args:
    - polygon_coords (list): List of polygons where each polygon is a list of (x, y) coordinates.
    - slide_dims (tuple): Dimensions of the slide (width, height).
    
    Returns:
    - numpy.ndarray: Binary mask with ones where the annotations are and zeros elsewhere.
    """
    mask = np.zeros((slide_dims[1], slide_dims[0]), dtype=np.uint8)

    for coords in polygon_coords:
        x_coords, y_coords = zip(*coords)
        rr, cc = polygon(x_coords, y_coords)
        mask[rr, cc] = 1

    return mask


def sample_positive_patches(slide_path, polygons, patch_size, num_patches, level=0):
    """
    Samples patches completely within the polygons.

    Parameters:
    - slide_path (str): The path to the slide to be segmented.
    - polygons (list): List of polygons where each polygon is represented as a list of (x, y) coordinate tuples.
    - patch_size (int): Size of the patch to be sampled.
    - num_patches (int): Number of patches to be sampled.
    - level (int): Level at which the patches are to be sampled. Defaults to 0.

    Returns:
    - patches (list): List of sampled patches.
    - patch_origins (list): List of top-left coordinates for each patch in the slide.
    """
    patches = []
    patch_origins = []
    half_patch_size = patch_size // 2
    slide = open_slide(slide_path)
    # Convert polygons to a binary mask
    width, height = slide.dimensions
    polygon_mask = coordinates_to_mask(polygons, (width, height))
    indices = np.nonzero(polygon_mask)

    # Sampling patches
    for _ in range(num_patches):
        while True:
            # Randomly select a point in the polygon mask
            idx = np.random.choice(len(indices[0]))
            center_x, center_y = indices[0][idx], indices[1][idx]

            # Check if a patch centered on this point would be entirely contained within the mask
            x_start, x_end = center_x - half_patch_size, center_x + (half_patch_size - 1)
            y_start, y_end = center_y - half_patch_size, center_y + (half_patch_size - 1)

            if (x_start >= 0 and x_end < height and y_start >= 0 and y_end < width):
                if np.all(polygon_mask[x_start:x_end, y_start:y_end] == 1):
                    break

        # Extract the patch from the slide
        img_patch = slide.read_region((y_start, x_start), level, (patch_size, patch_size))
        patch = np.array(img_patch)[:, :, :3]
        patches.append(patch)
        patch_origins.append((x_start, y_start))

    return patches, patch_origins


def sample_negative_patches(slide_path, foreground_mask, polygons, patch_size, num_patches, level=0):
    """
    Samples patches completely outside the polygons(ROI) but within the foreground.

    Parameters:
    - slide_path (str): The path to the slide to be segmented.
    - foreground_mask (np.array): Binary mask indicating the foreground region.
    - polygons (list): List of polygons where each polygon is represented as a list of (x, y) coordinate tuples.
    - patch_size (int): Size of the patch to be sampled.
    - num_patches (int): Number of patches to be sampled.
    - level (int): Level at which the patches are to be sampled. Defaults to 0.

    Returns:
    - patches (list): List of sampled patches.
    - patch_origins (list): List of top-left coordinates for each patch in the slide.
    """
    patches = []
    patch_origins = []
    half_patch_size = patch_size // 2
    slide = open_slide(slide_path)

    # Convert polygons to a binary mask
    width, height = slide.dimensions
    polygon_mask = coordinates_to_mask(polygons, (width, height))

    # Determine the mask for allowable sampling region
    sampling_mask = np.bitwise_and(foreground_mask, 1 - polygon_mask)
    indices = np.nonzero(sampling_mask)

    # Sampling patches
    for _ in range(num_patches):
        while True:
            # Randomly select a point in the sampling mask
            idx = np.random.choice(len(indices[0]))
            center_x, center_y = indices[0][idx], indices[1][idx]

            # Check if a patch centered on this point would be entirely contained within the mask
            x_start, x_end = center_x - half_patch_size, center_x + (half_patch_size - 1)
            y_start, y_end = center_y - half_patch_size, center_y + (half_patch_size - 1)

            if (x_start >= 0 and x_end < height and y_start >= 0 and y_end < width):
                if np.all(polygon_mask[x_start:x_end, y_start:y_end] == 0):
                    break

        # Extract the patch from the slide
        img_patch = slide.read_region((y_start, x_start), level, (patch_size, patch_size))
        patch = np.array(img_patch)[:, :, :3]
        patches.append(patch)
        patch_origins.append((x_start, y_start))

    return patches, patch_origins


def boundary_centered_sample_patches(slide_path, polygons, patch_size, num_patches, level=0):
    """
    Samples patches from the slide where each patch is centered on a boundary point of the given polygons.
    
    Parameters:
    - slide_path (str): The path to the slide to be segmented.
    - polygons (list): List of polygons where each polygon is represented as a list of (x, y) coordinate tuples.
    - patch_size (int): Size of the patch to be sampled.
    - num_patches (int): Number of patches to be sampled.
    - level (int): Level at which the patches are to be sampled. Defaults to 0.
    
    Returns:
    - patches (list): List of sampled patches.
    - patch_origins (list): List of top-left coordinates for each patch in the slide.
    """
    
    patches = []  # To store the extracted patches
    patch_origins = []  # To store the top-left coordinates of each patch in the slide
    slide = open_slide(slide_path)
    half_patch_size = patch_size // 2  # Half the size of the patch for calculating the origin

    for i in range(num_patches):
        # Sample using boundary points from the ROI
        polygon = random.choice(polygons)  # Choose a random polygon from the provided list

        center_x, center_y = random.choice(polygon)  # Randomly select a boundary point from the chosen polygon

        # Calculate the top-left x and y coordinates of the patch such that the randomly selected point is the center
        x = int(center_x - half_patch_size)
        y = int(center_y - half_patch_size)

        # Extract the patch from the slide using the calculated top-left coordinates
        img_patch = slide.read_region((y, x), level, (patch_size, patch_size))
        patch = np.array(img_patch)[:, :, :3]  # Convert PIL Image patch to numpy array and remove alpha channel if present
        patches.append(patch)
        patch_origins.append((x, y))
    
    return patches, patch_origins


def sample_boundary_patches(slide_path, polygons, patch_size, num_patches, level=0):
    """
    Samples patches from the slide such that each patch contains a boundary point of the given polygons with
    a random offset to ensure variability in the position of the boundary within the patches.
    
    Parameters:
    - slide_path (str): The path to the slide to be segmented.
    - polygons (list): List of polygons where each polygon is represented as a list of (x, y) coordinate tuples.
    - patch_size (int): Size of the patch to be sampled.
    - num_patches (int): Number of patches to be sampled.
    - level (int): Level at which the patches are to be sampled. Defaults to 0.
    
    Returns:
    - patches (list): List of sampled patches.
    - patch_origins (list): List of top-left coordinates for each patch in the slide.
    """
    
    patches = []  # To store the extracted patches
    patch_origins = []  # To store the top-left coordinates of each patch in the slide
    slide = open_slide(slide_path)
    half_patch_size = patch_size // 2  # Half the size of the patch for calculating the origin

    for i in range(num_patches):
        # Sample using boundary points from the ROI
        polygon = random.choice(polygons)  # Choose a random polygon from the provided list

        center_x, center_y = random.choice(polygon)  # Randomly select a boundary point from the chosen polygon

        # Introduce a random offset to the x and y coordinates of the boundary point to ensure variability
        # in the position of the boundary within the patches
        center_x = int(center_x) + np.random.randint(-half_patch_size, half_patch_size)
        center_y = int(center_y) + np.random.randint(-half_patch_size, half_patch_size)

        # Calculate the top-left x and y coordinates of the patch using the offsetted center coordinates
        x = center_x - half_patch_size
        y = center_y - half_patch_size

        # Extract the patch from the slide using the calculated top-left coordinates
        img_patch = slide.read_region((y, x), level, (patch_size, patch_size))
        patch = np.array(img_patch)[:, :, :3]  # Convert PIL Image patch to numpy array and remove alpha channel if present
        patches.append(patch)
        patch_origins.append((x, y))
    
    return patches, patch_origins


def sample_patches(slide_path, patch_size, threshold=0.5, pos_emb=False):
    """
    Sample patches from a given slide using a foreground/background segmentation and optional positional encoding.

    Parameters:
    - slide_path (str): The path to the slide to be sampled.
    - patch_size (int): The size of the patches to be sampled.
    - threshold (float, optional): The threshold for the percentage of foreground pixels in a patch. Defaults to 0.5.
    - pos_emb (bool, optional): Whether to include a positional embedding for each patch. Defaults to False.

    Returns:
    - tuple: A tuple containing a list of PIL Image objects representing the sampled patches and a list of their corresponding positional embeddings (if positional_embedding is True).

    Approach:
    - Use the foreground_background_segmentation function to produce a foreground mask.
    - Loop through all possible patch positions.
    - Check if the percentage of foreground pixels in the patch is greater than or equal to the threshold.
    - If so, read in the patch from the slide and append it to a list of patches.
    - If pos_emb is True, compute the positional embedding for the patch and append it to a list of positional embeddings.
    """

    # Use the foreground_background_segmentation function to produce a foreground mask
    mask = foreground_background_segmentation(slide_path)

    # Open the slide using the OpenSlide library
    slide = open_slide(slide_path)

    # Get the dimensions of the slide and mask
    slide_height, slide_width = slide.dimensions
    mask_width, mask_height = mask.shape

    # Check that the dimensions of the slide and mask match
    assert slide_width == mask_width and slide_height == mask_height, "Dimensions do not match"

    # Initialize lists to store the sampled patches and their corresponding positional embeddings (if positional_embedding is True)
    patches = []
    embeddings = []

    # Loop through all possible patch positions
    for i in range(0, slide_width, patch_size):
        for j in range(0, slide_height, patch_size):
            # Check if the percentage of foreground pixels in the patch is greater than or equal to the threshold
            patch_mask = mask[i:i+patch_size, j:j+patch_size]
            foreground_pixels = np.sum(patch_mask > 0)
            if foreground_pixels / (patch_size * patch_size) >= threshold:
                # Read in the patch from the slide and append it to the list of patches
                patch = slide.read_region((j, i), 0, (patch_size, patch_size)).convert('RGB')
                patches.append(patch)

                # If positional_embedding is True, compute the positional embedding for the patch and append it to the list of positional embeddings
                if pos_emb:
                    embedding = [i, j]
                    embeddings.append(embedding)

    # If positional_embedding is True, return a tuple containing the list of patches and the list of positional embeddings
    if pos_emb:
        return patches, embeddings
    # Otherwise, return a tuple containing only the list of patches
    else:
        return patches,


def assign_labels(patch_size, patch_origin, gt_mask):
    """
    Assign dense labels to a patch based on the ground truth mask.

    Parameters:
    - patch_size (int): The size of the patch (both width and height) for which labels are to be assigned.
    - patch_origin (tuple): A tuple containing the y (row) and x (column) coordinates of the top-left corner 
                            of the patch in the ground truth mask.
    - gt_mask (np.array): A 2D numpy array representing the ground truth mask where each pixel value indicates 
                          its label.

    Returns:
    - labels (np.array): A 2D numpy array of shape (patch_size, patch_size) containing the labels for each pixel 
                         in the patch.

    """

    # Initialize a zero matrix to hold the labels
    labels = np.zeros((patch_size, patch_size), dtype=np.uint8)

    # Extract and assign the labels for the patch from the ground truth mask
    labels = gt_mask[patch_origin[0]:patch_origin[0]+patch_size,
                     patch_origin[1]:patch_origin[1]+patch_size]

    return labels


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
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    # Original image
    ax[0].imshow(patch)
    ax[0].axis('off')
    ax[0].set_title('Patch')

    # Label
    ax[1].imshow(coloured_label, alpha=0.4)  # Overlay with transparency
    ax[1].axis('off')
    ax[1].set_title('Label')

    plt.tight_layout()
    plt.show()
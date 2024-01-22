import os
import logging
from tqdm import tqdm
from openslide import open_slide
import numpy as np
import xml.etree.ElementTree as ET
import random
import cv2
import h5py
from skimage.draw import polygon
from time import time

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

def annotations_to_coordinates_flipped(annotation_path):
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
        points = [(float(coord.attrib['X']), float(coord.attrib['Y'])) for coord in annotation.findall('.//Coordinate')]
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


def coordinates_to_mask_cv2(polygon_coords, slide_dims):
    """
    Convert a list of polygon coordinates to a binary mask using OpenCV's fillPoly.

    Args:
    - polygon_coords (list): List of polygons where each polygon is a list of (x, y) coordinates.
    - slide_dims (tuple): Dimensions of the slide (width, height).

    Returns:
    - numpy.ndarray: Binary mask with ones where the annotations are and zeros elsewhere.
    """
    # Initialize the mask as a zero array with the same dimensions as the slide.
    mask = np.zeros((slide_dims[1], slide_dims[0]), dtype=np.uint8)

    # Convert polygon coordinates to integer and to the appropriate shape for fillPoly.
    int_polygons = [np.array(coords, dtype=np.int32).reshape((-1, 1, 2)) for coords in polygon_coords]

    # Fill the polygons with 1's.
    cv2.fillPoly(mask, int_polygons, color=1)

    return mask


def extract_and_save_patches_and_labels(slide_path: str, save_path: str, tissue_threshold: float, 
                                  mask_path: str = None, input_level: int = 3,
                                  output_level: int = 0, patch_size: tuple = (256, 256),
                                  stride: tuple = (256, 256), enable_logging: bool = False):
    """
    Extract and save patches and labels from a whole slide image based on a tissue threshold.
    Patches and Lables are saved in an HDF5 file.
    
    Parameters:
        slide_path (str): Path to the whole slide image.
        save_path (str): Path to the HDF5 file where patches will be saved.
        tissue_threshold (float): Minimum percentage of tissue required to save a patch.
        mask_path (str, optional): Path to the binary mask file. Defaults to None.
        input_level (int, optional): The level for foreground-background segmentation. Defaults to 3.
        output_level (int, optional): The level for output mask and patch resolution. Defaults to 0.
        patch_size (tuple, optional): Size of the patches to be extracted. Defaults to (256, 256).
        stride (tuple, optional): Stride for patch extraction. Defaults to (256, 256).
        enable_logging (bool, optional): Enable or disable logging. Defaults to False.
    
    Returns:
        None: Patches and labels are saved to the HDF5 file at the given save_path.
    """
    
    # Preliminary Checks
    if not os.path.exists(slide_path):
        error_message = f"Slide file {slide_path} not found."
        raise FileNotFoundError(error_message)
    
    # Initialize logging if enabled
    if enable_logging:
        # Extract the base name (file name + extension)
        slide_name_w_ext = os.path.basename(slide_path)
        # Remove the file extension to get only the file name
        name_without_ext = os.path.splitext(slide_name_w_ext)[0]
        save_dir = os.path.dirname(save_path)
        log_dir = os.path.join(save_dir, 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logging.basicConfig(filename=os.path.join(log_dir, f"{name_without_ext}.log"), level=logging.INFO)
        logging.info(f'Starting patch extraction for slide {name_without_ext}')

    # Initialize tumor mask if annotations are provided
    tumor_mask = None
    is_tumor = False
    # Open the slide using OpenSlide
    slide = open_slide(slide_path)
    
    if mask_path is not None:
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Binary Mask file {mask_path} not found.")
        
        # Load the binary mask
        tumor_mask = np.load(mask_path)['mask']
        is_tumor = True
        if enable_logging:
            logging.info(f"Initialized tumor mask from binary mask file.")
    # Foreground-Background Segmentation
    mask = foreground_background_segmentation(slide_path, input_level, output_level)

    if enable_logging:
        logging.info(f"Completed foreground-background segmentation, mask shape: {mask.shape}")
        logging.info(f"Slide dimensions: {slide.dimensions}")
    

    # Open or create HDF5 file
    with h5py.File(save_path, 'a') as f:
        # Create a group for this slide, resolution level and patch size if it doesn't exist
        wsi_group = f.require_group(f"{name_without_ext}")

        level_group = wsi_group.require_group(f"Level_{output_level}")
        level_group.attrs['slide_dimensions'] = slide.dimensions

        size_group = level_group.require_group(f"Patch_Size_{patch_size[0]}")
        size_group.attrs['patch_size'] = patch_size
        size_group.attrs['is_tumor'] = is_tumor
        # Create sub-groups for patches and labels
        patch_group = size_group.require_group("patches")
        label_group = size_group.require_group("labels")
        
        # Patch Extraction and Quality Control
        patch_counter = 0
        for y in range(0, mask.shape[0] - patch_size[0], stride[0]):
            for x in range(0, mask.shape[1] - patch_size[1], stride[1]):
                patch_mask = mask[y:y + patch_size[0], x:x + patch_size[1]]
                
                # Check the tissue ratio in the mask patch
                tissue_ratio = np.sum(patch_mask) / (patch_size[0] * patch_size[1])
                
                if tissue_ratio >= tissue_threshold:
                    # Extract the patch from the slide
                    patch = np.array(slide.read_region((x, y), output_level, patch_size))[:, :, :3]
                    
                    # Create a dataset in the HDF5 file to save this patch
                    patch_name = f"patch_{patch_counter:05d}"
                    if patch_name not in patch_group:
                        patch_group.create_dataset(patch_name, data=patch)
                    
                    # Add attributes like tissue_ratio for additional metadata if needed
                    patch_group[patch_name].attrs['tissue_ratio'] = tissue_ratio
                    patch_group[patch_name].attrs['patch_size'] = patch_size
                    patch_group[patch_name].attrs['patch_origin'] = (x, y)
        
                    
                    if enable_logging:
                        logging.info(f"Saved patch {patch_name} with tissue ratio {tissue_ratio:.2f}")

                    # Save associated label if a tumor mask is available
                    if tumor_mask is not None:
                        patch_tumor_mask = tumor_mask[y:y + patch_size[0], x:x + patch_size[1]]
                        contains_tumor = np.any(patch_tumor_mask == 1)
                    else:
                        patch_tumor_mask = np.zeros((patch_size[0], patch_size[1]), dtype=np.uint8)
                        contains_tumor = False

                    label_name = f"label_{patch_counter:05d}"
                    if label_name not in label_group:
                        label_group.create_dataset(label_name, data=patch_tumor_mask)
                        
                    if enable_logging:
                        logging.info(f"Saved label {label_name}.")
                    label_group[label_name].attrs['associated_patch'] = patch_name
                    label_group[label_name].attrs['patch_size'] = patch_size
                    label_group[label_name].attrs['patch_origin'] = (x, y)
                    label_group[label_name].attrs['contains_positive'] = contains_tumor
                    patch_group[patch_name].attrs['associated_label'] = label_name
                    patch_group[patch_name].attrs['contains_positive'] = contains_tumor

                    patch_counter += 1
        
        size_group.attrs['total_patches'] = patch_counter
    if enable_logging:
        logging.info(f"Completed patch extraction for slide {name_without_ext}")
        handlers = logging.root.handlers[:]
        for handler in handlers:
            handler.close()
            logging.root.removeHandler(handler)

def modify_and_extract_patches(input_files: list, output_file: str, enable_logging: bool = False):
    """
    Modifies original HDF5 files by adding a positive percentage attribute and 
    extracts completely positive or negative patches into a new HDF5 file.

    Parameters:
        input_files (list): List of file paths to existing HDF5 files.
        output_file (str): Path to the new HDF5 file where selected patches will be saved.
        enable_logging (bool, optional): Enable or disable logging. Defaults to False.
    
    Returns:
        None
    """
    # Initialize logging if enabled
    if enable_logging:
        logging.basicConfig(filename=f"{output_file}_extraction.log", level=logging.INFO)
    tqdm.write(f"Starting patch extraction to {output_file}...")
    # Initialize counters for positive and negative patches
    positive_patch_count = 0
    negative_patch_count = 0

    with h5py.File(output_file, 'w') as output_h5:
        # Create groups for positive and negative patches
        positive_group = output_h5.create_group("positives")
        negative_group = output_h5.create_group("negatives")

        pos_patch_group = positive_group.create_group("patches")
        pos_label_group = positive_group.create_group("labels")

        neg_patch_group = negative_group.create_group("patches")
        neg_label_group = negative_group.create_group("labels")

        for file_path in tqdm(input_files):
            with h5py.File(file_path, 'r') as input_h5:  # Open in read mode
                for wsi_name in input_h5.keys():
                    for level_name in input_h5[wsi_name].keys():
                        for size_name in input_h5[wsi_name][level_name].keys():
                            patch_group = input_h5[wsi_name][level_name][size_name]['patches']
                            label_group = input_h5[wsi_name][level_name][size_name]['labels']

                            for patch_name, patch_dataset in patch_group.items():
                                label_dataset = label_group[patch_dataset.attrs['associated_label']]
                                label_data = label_dataset[()]

                                positive_percentage = label_dataset.attrs['positive_percentage']

                                # Extract completely positive or negative patches
                                if positive_percentage == 1.0:
                                    new_patch_name = f"patch_{positive_patch_count:05d}"
                                    new_label_name = f"label_{positive_patch_count:05d}"

                                    # Save in new HDF5 file
                                    pos_patch_group.create_dataset(new_patch_name, data=patch_dataset[()])
                                    pos_label_group.create_dataset(new_label_name, data=label_data)

                                    # Add metadata about original file and patch/label name
                                    pos_patch_group[new_patch_name].attrs['original_file'] = file_path
                                    pos_patch_group[new_patch_name].attrs['original_patch_name'] = patch_name
                                    pos_label_group[new_label_name].attrs['original_file'] = file_path
                                    pos_label_group[new_label_name].attrs['original_label_name'] = patch_dataset.attrs['associated_label']

                                    # Update counters
                                    positive_patch_count += 1

                                    if enable_logging:
                                        logging.info(f"Saved pos_{new_patch_name} with positive percentage {positive_percentage}")

                                elif positive_percentage == 0.0:
                                    new_patch_name = f"patch_{negative_patch_count:05d}"
                                    new_label_name = f"label_{negative_patch_count:05d}"

                                    # Save in new HDF5 file
                                    neg_patch_group.create_dataset(new_patch_name, data=patch_dataset[()])
                                    neg_label_group.create_dataset(new_label_name, data=label_data)

                                    # Add metadata about original file and patch/label name
                                    neg_patch_group[new_patch_name].attrs['original_file'] = file_path
                                    neg_patch_group[new_patch_name].attrs['original_patch_name'] = patch_name
                                    neg_label_group[new_label_name].attrs['original_file'] = file_path
                                    neg_label_group[new_label_name].attrs['original_label_name'] = patch_dataset.attrs['associated_label']

                                    # Update counters
                                    negative_patch_count += 1

                                    if enable_logging:
                                        logging.info(f"Saved neg_{new_patch_name} with positive percentage {positive_percentage}")

        # Set attributes for the number of patches in each group
        positive_group.attrs['total_patches'] = positive_patch_count
        negative_group.attrs['total_patches'] = negative_patch_count

    if enable_logging:
        logging.info("Completed processing of all files.")
        handlers = logging.root.handlers[:]
        for handler in handlers:
            handler.close()
            logging.root.removeHandler(handler)


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


def assign_dense_labels(patch_size, patch_origin, gt_mask):
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


def assign_patch_labels(patch_size, patch_origins, gt_mask):
    """
    Assign labels to a list of patches based on the ground truth mask.

    Parameters:
    - patch_size (int): The size of the patch (both width and height) for which labels are to be assigned.
    - patch_origins (list): A list of tuples containing the y (row) and x (column) coordinates of the top-left 
                            corner of each patch in the ground truth mask.
    - gt_mask (np.array): A 2D numpy array representing the ground truth mask where each pixel value indicates 
                          its label.

    Returns:
    - labels (list): A list of 2D numpy arrays of shape (patch_size, patch_size) containing the labels for each 
                     pixel in the patch.

    """

    # Initialize a list to hold the labels
    labels = []

    # Loop through each patch and assign the labels
    for patch_origin in patch_origins:
        labels.append(assign_dense_labels(patch_size, patch_origin, gt_mask))

    return labels

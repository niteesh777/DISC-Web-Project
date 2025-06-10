# processing_script.py

import matplotlib
matplotlib.use('Agg')  # Must be called before importing pyplot

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from PIL import Image
from scipy.ndimage import zoom
from tqdm import tqdm
import importlib.util
from scipy.interpolate import griddata
import face_recognition  
import torch
from segment_anything import sam_model_registry, SamPredictor
# Import pydic directly since it's in the same directory
from . import pydic


def your_processing_function(image1_path, image2_path):
    # Define directories
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of this script
    media_root = os.path.abspath(os.path.join(base_dir, '..', 'media'))  # Adjust the path to point to media/
    output_dir = os.path.join(media_root, 'results')  # Save results in media/results/

    os.makedirs(output_dir, exist_ok=True)

    # Prepare output subdirectories
    aligned_dir = os.path.join(output_dir, 'Aligned')
    cropped_dir = os.path.join(output_dir, 'Cropped')
    pydic_result_dir = os.path.join(cropped_dir, 'pydic', 'result')
    csv_dir = os.path.join(cropped_dir, 'csv')
    overlay_dir = os.path.join(cropped_dir, 'OVERLAY')
    symmetry_dir = os.path.join(overlay_dir, 'Symmetry_Scores')

    for directory in [aligned_dir, cropped_dir, pydic_result_dir, csv_dir, overlay_dir, symmetry_dir]:
        os.makedirs(directory, exist_ok=True)

    # Settings for processing
    settings = {
        'OverLay': True,
        'Contour': True,
        'Vector': True,  # Set to True to generate vector plots
        'FixedScaleValue': [0, 90]
    }

    # Process images
    prepare_images(image1_path, image2_path, output_dir)
    auto_align_images_sift(output_dir, aligned_dir)
    auto_crop_images(aligned_dir, cropped_dir)
    run_pydic(cropped_dir)
    auto_create_mask_and_process_csv(cropped_dir)
    generate_heatmaps_and_vectors(cropped_dir, csv_dir, overlay_dir, settings)
    symmetry_score = calculate_symmetry_scores(cropped_dir, csv_dir)

    # The heatmap is saved in overlay_dir
    # Assuming the heatmap file is named after the second image with '_heatmap.png' suffix
    heatmap_filename = os.path.splitext(os.path.basename(image2_path))[0] + '_heatmap.png'
    heatmap_path = os.path.join(overlay_dir, 'Heatmap', heatmap_filename)

    # Calculate the relative path to the media directory
    result_heatmap_path = os.path.relpath(heatmap_path, media_root)

    return result_heatmap_path, symmetry_score

def prepare_images(image1_path, image2_path, output_dir):
    """Copy the two images to the output directory and rename them sequentially."""
    images = [image1_path, image2_path]
    for idx, img_path in enumerate(images):
        img = cv2.imread(img_path)
        if img is None:
            raise Exception(f"Error loading image: {img_path}")
        filename = os.path.join(output_dir, f'{1000000 + idx:07d}.png')
        cv2.imwrite(filename, img)
    print("Images prepared for processing.")

def auto_align_images_sift(input_dir, aligned_dir):
    """Automatically align images using SIFT and save them to the aligned directory."""
    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
    if len(image_files) < 2:
        raise Exception(f"Not enough images found in directory: {input_dir}")

    os.makedirs(aligned_dir, exist_ok=True)
    print("Automatic alignment process initiated using SIFT...")

    # Load images
    images = []
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        image = cv2.imread(image_path)
        if image is not None:
            images.append((image_file, image))
        else:
            raise Exception(f"Error loading image: {image_path}")

    # Use the first image as the reference
    ref_image_file, ref_image = images[0]
    ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)

    aligned_images = [(ref_image_file, ref_image)]  # Start with the reference image

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors in the reference image
    ref_keypoints, ref_descriptors = sift.detectAndCompute(ref_gray, None)

    for idx in range(1, len(images)):
        image_file, image = images[idx]
        curr_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find keypoints and descriptors in the current image
        curr_keypoints, curr_descriptors = sift.detectAndCompute(curr_gray, None)

        # Match descriptors using FLANN matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(ref_descriptors, curr_descriptors, k=2)

        # Store good matches using Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        MIN_MATCH_COUNT = 10
        if len(good_matches) >= MIN_MATCH_COUNT:
            # Extract location of good matches
            ref_pts = np.float32([ref_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            curr_pts = np.float32([curr_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Compute homography
            H, mask = cv2.findHomography(curr_pts, ref_pts, cv2.RANSAC, 5.0)

            # Warp current image to align with reference image
            height, width = ref_image.shape[:2]
            aligned_image = cv2.warpPerspective(image, H, (width, height))

            aligned_images.append((image_file, aligned_image))
        else:
            print(f"Not enough matches are found - {len(good_matches)}/{MIN_MATCH_COUNT}")
            print(f"Cannot align image: {image_file}")
            aligned_images.append((image_file, image))  # Use the original image

    # Save aligned images to the aligned directory
    for image_file, aligned_image in aligned_images:
        cv2.imwrite(os.path.join(aligned_dir, image_file), aligned_image)

    print("Images aligned using SIFT.")

def auto_crop_images(input_dir, output_dir):
    """Automatically detect faces and crop images around the face, including more of the head."""
    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
    if len(image_files) < 1:
        raise Exception(f"No images found in directory: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)
    print("Automatic cropping process initiated...")

    face_coords = []

    # First pass: Detect face locations in all images
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        image = cv2.imread(image_path)
        if image is not None:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_image)
            if face_locations:
                face_coords.append(face_locations[0])  # Assuming one face per image
            else:
                raise Exception(f"No face detected in image: {image_file}. Cannot proceed.")
        else:
            raise Exception(f"Error loading image: {image_path}")

    # Calculate common crop coordinates
    tops, rights, bottoms, lefts = zip(*face_coords)
    top = min(tops)
    right = max(rights)
    bottom = max(bottoms)
    left = min(lefts)

    # Apply padding and extra headroom
    padding = 20
    face_height = max(bottoms) - min(tops)
    head_extra = int(face_height * 0.5)

    # Adjust the coordinates
    top = max(0, top - padding - head_extra)
    bottom = bottom + padding
    left = max(0, left - padding)
    right = right + padding

    # Second pass: Crop all images using the common coordinates
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        image = cv2.imread(image_path)
        cropped_image = image[top:bottom, left:right]
        cv2.imwrite(os.path.join(output_dir, f"cropped_{image_file}"), cropped_image)

def run_pydic(cropped_dir):
    """Run PyDIC analysis on the cropped images."""
    correl_wind_size = (30, 30)
    correl_grid_size = (6, 6)
    dic_file = os.path.join(cropped_dir, 'result.dic')

    pydic.init(os.path.join(cropped_dir, '*.png'), correl_wind_size, correl_grid_size, dic_file, area_of_intersest='all')
    pydic.FixedscaleValues = [15, 3, 0]
    pydic.read_dic_file(dic_file, FixedScale=False, interpolation='raw', save_image=False, scale_disp=2, scale_grid=1)

def auto_create_mask_and_process_csv(cropped_dir):
    """Automatically create a mask using SAM and apply it to the CSV files."""
    mask_file_path = os.path.join(cropped_dir, 'mask_x.npy')
    csv_dir_path = os.path.join(cropped_dir, 'pydic', 'result')
    modified_csv_dir_path = os.path.join(cropped_dir, 'csv')
    os.makedirs(modified_csv_dir_path, exist_ok=True)

    auto_create_mask(cropped_dir, mask_file_path)
    process_csv_files(csv_dir_path, cropped_dir, mask_file_path, modified_csv_dir_path)
    print("Automatic masking and CSV processing complete.")

def auto_create_mask(image_dir_path, mask_file_path):
    """Create a face mask using SAM with center and cheek points as prompts."""

    # Load the image
    image_files = sorted([f for f in os.listdir(image_dir_path) if f.endswith('.png') and f.startswith('cropped_')])
    if not image_files:
        raise Exception(f"No PNG images found in directory: {image_dir_path}")

    img_path = os.path.join(image_dir_path, image_files[0])
    img = cv2.imread(img_path)
    if img is None:
        raise Exception("Error loading image for mask creation.")

    # Convert image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get image dimensions
    height, width = img_rgb.shape[:2]

    # Define points
    center_point = [width // 2, height // 2]
    cheek_offset = width // 5  # Adjust this value as needed
    left_cheek_point = [center_point[0] - cheek_offset, center_point[1]]
    right_cheek_point = [center_point[0] + cheek_offset, center_point[1]]

    # Prepare input points and labels
    input_points = np.array([center_point, left_cheek_point, right_cheek_point])
    input_labels = np.array([1, 1, 1])  # Labels: 1 for foreground

    # Load the SAM model
    model_type = "vit_h"  # Choose from 'vit_h', 'vit_l', 'vit_b'
    checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'sam_vit_h_4b8939.pth')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # Run the model
    predictor.set_image(img_rgb)
    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=False,
    )

    # Process the mask
    mask = masks[0]
    mask_bool_inverted = ~mask.astype(bool)
    np.save(mask_file_path, mask_bool_inverted)

def process_csv_files(csv_dir_path, image_dir_path, mask_file_path, modified_csv_dir_path):
    """Apply the mask to each CSV file."""
    mask = np.load(mask_file_path)
    mask = ~mask  

    csv_files = sorted([f for f in os.listdir(csv_dir_path) if f.endswith('.csv')])
    if not csv_files:
        raise Exception(f"No CSV files found in directory: {csv_dir_path}")

    for csv_file in tqdm(csv_files, desc="Processing CSV files"):
        csv_file_path = os.path.join(csv_dir_path, csv_file)
        df = pd.read_csv(csv_file_path)

        x_unique = np.sort(df['pos_x'].unique())
        y_unique = np.sort(df['pos_y'].unique())

        X, Y = np.meshgrid(x_unique, y_unique)

        resized_mask = zoom(mask.astype(float), (
            len(y_unique) / mask.shape[0],
            len(x_unique) / mask.shape[1]
        ), order=0).astype(bool)

        Z = df.pivot(index='pos_y', columns='pos_x', values='disp_xy_indi')

        Z_masked = Z.where(resized_mask)

        df_masked = Z_masked.stack().reset_index()
        df_masked.columns = ['pos_y', 'pos_x', 'disp_xy_indi']
        df_final = df.drop(columns=['disp_xy_indi']).merge(df_masked, on=['pos_x', 'pos_y'], how='inner')

        df_final.to_csv(os.path.join(modified_csv_dir_path, f'modified_{csv_file}'), index=False)

def generate_heatmaps_and_vectors(cropped_dir, csv_dir, overlay_dir, settings):
    """Generate heatmaps and vector fields from the CSV files and overlay them on images."""

    # Unpack settings
    OverLay = settings['OverLay']
    Contour = settings['Contour']
    Vector = settings['Vector']
    FixedScaleValue = settings['FixedScaleValue']

    png_dir = cropped_dir
    csv_dir = os.path.join(cropped_dir, 'csv')
    output_dir_path = overlay_dir
    os.makedirs(output_dir_path, exist_ok=True)

    # Define and create subdirectories for different types of outputs
    if Contour:
        heatmap_dir = os.path.join(output_dir_path, 'Heatmap')
        os.makedirs(heatmap_dir, exist_ok=True)
    if Vector:
        vector_dir = os.path.join(output_dir_path, 'Vector_folder')
        os.makedirs(vector_dir, exist_ok=True)

    # List all CSV and PNG files
    csv_files = sorted([f for f in os.listdir(csv_dir) if f.endswith('.csv')])
    png_files = sorted([f for f in os.listdir(png_dir) if f.endswith('.png') and f.startswith('cropped_')])

    # Since each CSV file corresponds to the displacement between two images,
    # we need to pair the CSV file with the second image in the sequence.
    if len(csv_files) != len(png_files) - 1:
        raise Exception("The number of CSV files should be one less than the number of PNG files.")

    for idx, csv_file in enumerate(tqdm(csv_files, total=len(csv_files), desc="Processing files")):
        csv_file_path = os.path.join(csv_dir, csv_file)
        # Use the second image in the pair for visualization
        image_file = png_files[idx + 1]
        image_path = os.path.join(png_dir, image_file)
        df = pd.read_csv(csv_file_path)
        image = Image.open(image_path)
        empty_image = np.zeros((*image.size[::-1], 4), dtype=np.uint8)

        if Contour:
            scale = [i / 100 for i in range(int(100 * FixedScaleValue[0]), int(100 * FixedScaleValue[1]) + 1,
                                             int(100 * (FixedScaleValue[1] - FixedScaleValue[0]) / 25))]
            norm = mcolors.Normalize(vmin=FixedScaleValue[0], vmax=FixedScaleValue[1])

            dpi = image.info['dpi'] if 'dpi' in image.info else (300, 300)
            dpi = (int(dpi[0]), int(dpi[1]))

            # Reshape the data into a 2D grid
            x_unique = np.sort(df['pos_x'].unique())
            y_unique = np.sort(df['pos_y'].unique())
            Z = df.pivot_table(index='pos_y', columns='pos_x', values='disp_xy_indi', fill_value=np.nan).reindex(
                index=y_unique, columns=x_unique, fill_value=np.nan).values

            fig, ax = plt.subplots(figsize=(image.width / dpi[0], image.height / dpi[1]), dpi=dpi[0])

            if OverLay:
                ax.imshow(image, extent=[x_unique.min(), x_unique.max(), y_unique.max(), y_unique.min()])
            else:
                ax.imshow(empty_image, extent=[x_unique.min(), x_unique.max(), y_unique.max(), y_unique.min()])

            # Mask the array where Z values are NaN
            Z_masked = np.ma.masked_where(np.isnan(Z), Z)

            contour = ax.contourf(x_unique, y_unique, Z_masked, levels=scale, cmap=plt.cm.jet, alpha=0.5, norm=norm,
                                  extent=[x_unique.min(), x_unique.max(), y_unique.max(), y_unique.min()])
            ax.axis('off')

            plt.savefig(os.path.join(heatmap_dir, os.path.splitext(image_file)[0] + '_heatmap.png'), dpi=dpi[0],
                        bbox_inches='tight', pad_inches=0)
            plt.close()

        if Vector:
            plt.figure(figsize=(image.width / 100, image.height / 100))
            if OverLay:
                plt.imshow(image, cmap='gray', extent=[0, image.width, image.height, 0])
            else:
                plt.imshow(empty_image, extent=[0, image.width, image.height, 0])

            # Subsample the data to plot only every 10th vector
            df_subsampled = df.iloc[::10, :]

            plt.quiver(
                df_subsampled['pos_x'],
                df_subsampled['pos_y'],
                df_subsampled['disp_x'],
                df_subsampled['disp_y'],
                angles='xy',
                scale_units='xy',
                scale=1,
                color='red',
                width=0.0005
            )
            plt.axis('equal')
            plt.axis('off')
            vector_output_path = os.path.join(vector_dir, os.path.splitext(image_file)[0] + '_vector.png')
            plt.savefig(vector_output_path, transparent=False, format='png', dpi=1000)
            plt.close()

def calculate_symmetry_scores(cropped_dir, csv_dir):
    """Calculate facial symmetry scores based on displacement data within the face mask,
    and display the midline and scores on the face images."""
    csv_dir = os.path.join(cropped_dir, 'csv')
    mask_file_path = os.path.join(cropped_dir, 'mask_x.npy')
    png_dir = cropped_dir
    output_dir_path = os.path.join(cropped_dir, 'OVERLAY', 'Symmetry_Scores')
    os.makedirs(output_dir_path, exist_ok=True)

    csv_files = sorted([f for f in os.listdir(csv_dir) if f.endswith('.csv')])
    png_files = sorted([f for f in os.listdir(png_dir) if f.endswith('.png') and f.startswith('cropped_')])

    if not csv_files:
        raise Exception(f"No CSV files found in directory: {csv_dir}")

    # Load the mask
    mask = np.load(mask_file_path)
    mask = ~mask  # Invert the mask if needed

    symmetry_scores = []

    for idx, csv_file in enumerate(csv_files):
        csv_file_path = os.path.join(csv_dir, csv_file)
        image_file = png_files[idx + 1]  # Use the second image in the pair
        image_path = os.path.join(png_dir, image_file)
        image = Image.open(image_path)
        df = pd.read_csv(csv_file_path)

        # Compute the displacement magnitude
        df['disp_magnitude'] = np.sqrt(df['disp_x']**2 + df['disp_y']**2)

        # Get unique x and y positions
        x_unique = np.sort(df['pos_x'].unique())
        y_unique = np.sort(df['pos_y'].unique())

        # Pivot the displacement data into a 2D grid
        Z_disp = df.pivot_table(index='pos_y', columns='pos_x', values='disp_magnitude', fill_value=np.nan).reindex(
            index=y_unique, columns=x_unique, fill_value=np.nan).values

        # Resize the mask to match the Z_disp grid
        resized_mask = zoom(mask.astype(float), (
            Z_disp.shape[0] / mask.shape[0],
            Z_disp.shape[1] / mask.shape[1]
        ), order=0).astype(bool)

        # Apply the mask to the displacement data
        Z_disp_masked = np.where(resized_mask, Z_disp, np.nan)

        # Flatten the masked displacement data
        disp_masked_flat = Z_disp_masked.flatten()

        # Flatten the x and y grids
        X, Y = np.meshgrid(x_unique, y_unique)
        X_flat = X.flatten()
        Y_flat = Y.flatten()

        # Create a DataFrame with the masked data
        df_masked = pd.DataFrame({
            'pos_x': X_flat,
            'pos_y': Y_flat,
            'disp_magnitude': disp_masked_flat
        })

        # Drop NaN values (points outside the mask)
        df_masked = df_masked.dropna(subset=['disp_magnitude'])

        # Determine the midline of the face based on the masked data
        x_min = df_masked['pos_x'].min()
        x_max = df_masked['pos_x'].max()
        x_mid = (x_min + x_max) / 2

        # Separate the data into left and right halves
        left_df = df_masked[df_masked['pos_x'] < x_mid]
        right_df = df_masked[df_masked['pos_x'] >= x_mid]

        # Handle potential NaN values
        left_disp_mean = left_df['disp_magnitude'].mean(skipna=True)
        right_disp_mean = right_df['disp_magnitude'].mean(skipna=True)

        # Calculate the symmetry score
        score = right_disp_mean - left_disp_mean
        symmetry_scores.append({
            'csv_file': csv_file,
            'left_mean_disp': left_disp_mean,
            'right_mean_disp': right_disp_mean,
            'symmetry_score': score
        })

        # Visualization
        plt.figure(figsize=(image.width / 100, image.height / 100))
        plt.imshow(image, extent=[x_unique.min(), x_unique.max(), y_unique.max(), y_unique.min()])
        plt.axis('off')

        # Plot the midline
        plt.axvline(x=x_mid, color='yellow', linestyle='--', linewidth=2)

        # Annotate the left and right mean displacements
        plt.text(x_min + (x_mid - x_min) / 2, y_unique.min() + 20, f'Left Mean: {left_disp_mean:.2f}', 
                 color='white', fontsize=12, ha='center', va='bottom', bbox=dict(facecolor='black', alpha=0.5))
        plt.text(x_mid + (x_max - x_mid) / 2, y_unique.min() + 20, f'Right Mean: {right_disp_mean:.2f}', 
                 color='white', fontsize=12, ha='center', va='bottom', bbox=dict(facecolor='black', alpha=0.5))

        # Save the visualization
        output_image_path = os.path.join(output_dir_path, f'{os.path.splitext(image_file)[0]}_symmetry.png')
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()

    # Optionally, save the symmetry scores to a CSV file
    scores_df = pd.DataFrame(symmetry_scores)
    output_scores_path = os.path.join(output_dir_path, 'symmetry_scores.csv')
    scores_df.to_csv(output_scores_path, index=False)

    # For simplicity, return the symmetry score of the last processed image
    # You can adjust this to return the score you need
    if symmetry_scores:
        return symmetry_scores[-1]['symmetry_score']
    else:
        return None


#27th april
import numpy as np
import SimpleITK as sitk
import os
from glob import glob
import tensorflow as tf
from model import TwoPathwayCNN
import matplotlib.pyplot as plt
import cv2
from extract_patches import Pipeline
import argparse

def preprocess_scan(patient_dir):
    """Process a single patient scan for prediction"""
    pipeline = Pipeline([patient_dir], Normalize=True)
    return pipeline.train_im[0]

def extract_patches_for_prediction(scan_data, patch_size=128, stride=64):
    """Extract overlapping patches from the scan for prediction"""
    patches = []
    patch_locations = []
    
    # Get dimensions
    n_modalities, depth, height, width = scan_data.shape
    modality_data = scan_data[0:4]  # Extract the 4 modalities (exclude ground truth)
    
    # Iterate through the volume using a sliding window approach
    for z in range(0, depth):
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                # Extract patch
                patch = modality_data[:, z, y:y+patch_size, x:x+patch_size]
                
                # Skip patches that are mostly background
                if np.mean(patch != -9) > 0.5:  # If more than 50% of patch is not background
                    # Reshape to match model input shape: (height, width, channels)
                    patch = np.transpose(patch, (1, 2, 0))
                    patches.append(patch)
                    patch_locations.append((z, y, x))
    
    return np.array(patches), patch_locations

def stitch_predictions(predictions, patch_locations, output_shape, patch_size=128, stride=64):
    """Stitch together the patch predictions into a full volume"""
    # Initialize empty volume for the segmentation
    depth, height, width = output_shape
    segmentation = np.zeros((depth, height, width, 4))
    count_map = np.zeros((depth, height, width, 4))
    
    # Place each patch prediction back into the volume
    for i, (z, y, x) in enumerate(patch_locations):
        pred = predictions[i]
        segmentation[z, y:y+patch_size, x:x+patch_size] += pred
        count_map[z, y:y+patch_size, x:x+patch_size] += 1
    
    # Average overlapping regions
    segmentation = np.divide(segmentation, count_map, where=count_map>0)
    
    # Convert to categorical predictions
    segmentation_argmax = np.argmax(segmentation, axis=-1)
    
    return segmentation_argmax

def visualize_slice(original_slice, segmentation_slice, slice_idx, output_dir):
    """Visualize a slice with its segmentation"""
    plt.figure(figsize=(12, 5))
    
    # Plot original image (FLAIR)
    plt.subplot(1, 2, 1)
    plt.imshow(original_slice, cmap='gray')
    plt.title(f'FLAIR - Slice {slice_idx}')
    plt.axis('off')
    
    # Plot segmentation
    plt.subplot(1, 2, 2)
    cmap = plt.cm.get_cmap('viridis', 4)
    plt.imshow(segmentation_slice, cmap=cmap, vmin=0, vmax=3)
    plt.title('Segmentation')
    plt.axis('off')
    plt.colorbar(ticks=[0, 1, 2, 3], label='Class')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'segmentation_slice_{slice_idx}.png'))
    plt.close()

def predict_volume(model, patient_dir, output_dir=None):
    """Predict segmentation for a patient volume"""
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Preprocess the scan
    scan_data = preprocess_scan(patient_dir)
    
    # Extract patches
    patches, patch_locations = extract_patches_for_prediction(scan_data)
    
    if len(patches) == 0:
        print(f"No valid patches found for {patient_dir}")
        return None
    
    print(f"Processing {len(patches)} patches for prediction")
    
    # Predict in batches to avoid memory issues
    batch_size = 8  # Adjust based on your CPU memory
    predictions = []
    
    for i in range(0, len(patches), batch_size):
        batch = patches[i:i+batch_size]
        pred = model.model.predict(batch)
        predictions.extend(pred)
    
    predictions = np.array(predictions)
    
    # Get output shape (excluding ground truth for the last channel)
    output_shape = scan_data.shape[1:4]
    
    # Stitch predictions back together
    segmentation = stitch_predictions(predictions, patch_locations, output_shape)
    
    # Save segmentation as NIFTI
    if output_dir:
        # Get patient ID from directory
        patient_id = os.path.basename(patient_dir)
        
        # Create SimpleITK image and save as .nii.gz
        sitk_seg = sitk.GetImageFromArray(segmentation.astype(np.uint8))
        sitk.WriteImage(sitk_seg, os.path.join(output_dir, f'{patient_id}_segmentation.nii.gz'))
        
        # Visualize a few slices
        for slice_idx in range(0, segmentation.shape[0], 20):  # Every 20 slices
            if slice_idx < segmentation.shape[0]:
                visualize_slice(
                    scan_data[0, slice_idx],  # FLAIR modality
                    segmentation[slice_idx], 
                    slice_idx,
                    output_dir
                )
    
    return segmentation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict brain tumor segmentation')
    parser.add_argument('--patient_dir', type=str, required=True,
                      help='Path to patient directory')
    parser.add_argument('--output_dir', type=str, default='predictions',
                      help='Directory to save predictions')
    parser.add_argument('--model_weights', type=str, default='twopath_final.h5',
                      help='Path to trained model weights')
    
    args = parser.parse_args()
    
    # Create model and load weights
    model = TwoPathwayCNN(img_shape=(128, 128, 4), load_model_weights=args.model_weights)
    
    # Predict
    predict_volume(model, args.patient_dir, args.output_dir)
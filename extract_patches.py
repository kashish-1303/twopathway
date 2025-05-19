import numpy as np
import os
from glob import glob
import SimpleITK as sitk
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import cv2
from scipy.ndimage import gaussian_filter, map_coordinates
import time
import multiprocessing
from tqdm import tqdm

class OptimizedPipeline:
    
    def __init__(self, list_train, max_patients=None, Normalize=True):
        """
        Initialize the pipeline with the list of training directories
        
        Parameters:
        - list_train: List of patient directories
        - max_patients: Maximum number of patients to process (for testing/debugging)
        - Normalize: Whether to normalize the image data
        """
        if max_patients:
            self.scans_train = list_train[:max_patients]
        else:
            self.scans_train = list_train
            
        print(f"Processing {len(self.scans_train)} patients")
        self.Normalize = Normalize
        self.train_im = None
        
    def read_scans(self):
        """Read and preprocess all scans more efficiently"""
        train_im = []
        skipped_patients = 0
        
        # Use tqdm for progress tracking
        for i, patient_dir in enumerate(tqdm(self.scans_train, desc="Loading patients")):
            try:
                # Get all required files
                flair = glob(os.path.join(patient_dir, '*Flair*.mha'))
                t1 = glob(os.path.join(patient_dir, '*T1*.mha'))
                t1c = glob(os.path.join(patient_dir, '*T1c*.mha'))
                t2 = glob(os.path.join(patient_dir, '*T2*.mha'))
                gt = glob(os.path.join(patient_dir, '*more*.mha'))
                
                # Filter T1 files to exclude T1c
                t1s = [scan for scan in t1 if scan not in t1c]
                
                # Verify we have all necessary files
                if not all([flair, t1s, t1c, t2, gt]):
                    skipped_patients += 1
                    continue
                
                if len(flair) != 1 or len(t2) != 1 or len(gt) != 1 or len(t1s) != 1 or len(t1c) != 1:
                    skipped_patients += 1
                    continue
                
                # Load all scans for this patient
                scans = [flair[0], t1s[0], t1c[0], t2[0], gt[0]]
                
                # Define cropping coordinates (brain region)
                z0, y0, x0 = 1, 29, 42
                z1, y1, x1 = 147, 221, 194
                
                # Load and crop images in a memory-efficient manner
                tmp = []
                for k in range(len(scans)):
                    img = sitk.ReadImage(scans[k])
                    array = sitk.GetArrayFromImage(img)
                    array = array[z0:z1, y0:y1, x0:x1]  # Crop directly after loading
                    tmp.append(array)
                
                tmp = np.array(tmp)
                
                # Normalize if requested
                if self.Normalize:
                    for slice_ix in range(4):  # Only normalize the 4 modalities, not the ground truth
                        for mode_ix in range(tmp.shape[1]):  # For each slice
                            tmp[slice_ix, mode_ix] = self._normalize(tmp[slice_ix, mode_ix])
                
                train_im.append(tmp)
                
            except Exception as e:
                print(f"Error processing patient {patient_dir}: {e}")
                skipped_patients += 1
                continue
                
        print(f"Successfully processed {len(train_im)} patients, skipped {skipped_patients} patients")
        self.train_im = np.array(train_im)
        return self.train_im
    
    def _normalize(self, slice_data):
        """Normalize a single slice"""
        # Find non-zero pixels to avoid skewing normalization with background
        mask = slice_data > 0
        if np.sum(mask) == 0:  # If no foreground pixels, return as is
            return slice_data
            
        # Get percentiles to clip outliers
        b = np.percentile(slice_data[mask], 99)
        t = np.percentile(slice_data[mask], 1)
        
        # Clip values
        slice_data = np.clip(slice_data, t, b)
        
        # Get statistics from non-zero pixels
        image_nonzero = slice_data[mask]
        
        if np.std(image_nonzero) == 0:
            return slice_data
        else:
            # Normalize to zero mean, unit variance on non-zero pixels
            mean_val = np.mean(image_nonzero)
            std_val = np.std(image_nonzero)
            
            # Create output with background marked as -9
            output = np.ones_like(slice_data) * -9
            output[mask] = (slice_data[mask] - mean_val) / std_val
            
            return output
    
    def extract_balanced_patches(self, num_patches, patch_size=(4, 64, 64), tumor_ratio=0.3, random_offset=True):
        """
        Extract patches with a controlled balance between tumor and non-tumor regions
        
        Parameters:
        - num_patches: Total number of patches to extract
        - patch_size: Size of patches (d, h, w)
        - tumor_ratio: Ratio of tumor patches to total patches (0.3 = 30% tumor, 70% non-tumor)
        - random_offset: Whether to add random offset to patch centers for more variability
        """
        start_time = time.time()
        
        if self.train_im is None:
            print("Loading scans first...")
            self.read_scans()
        
        # Unpack patch dimensions
        d, h, w = patch_size
        assert d == 4, "Expected 4 modalities for the patch depth"
        
        # Calculate how many patches to extract from each class
        num_tumor_patches = int(num_patches * tumor_ratio)
        num_non_tumor_patches = num_patches - num_tumor_patches
        
        print(f"Planning to extract {num_tumor_patches} tumor patches and {num_non_tumor_patches} non-tumor patches")
        
        # Access the ground truth and modality data
        gt_im = np.swapaxes(self.train_im, 0, 1)[4]   
        flair = np.swapaxes(self.train_im, 0, 1)[0]  # Use FLAIR for initial tissue mask
        
        # Get shape information
        tmp_shp = gt_im.shape
        
        # Create masks for patch extraction
        print("Creating masks for patch extraction...")
        
        # Flatten arrays for easier processing
        gt_flat = gt_im.reshape(-1).astype(np.uint8)
        flair_flat = flair.reshape(-1).astype(np.float32)
        
        # Get valid tissue indices (not background)
        valid_indices = np.squeeze(np.argwhere((flair_flat != -9.0) & (flair_flat > -1.0)))
        
        # Separate tumor and non-tumor indices
        tumor_indices = np.squeeze(np.argwhere(gt_flat > 0))
        non_tumor_indices = np.intersect1d(valid_indices, np.squeeze(np.argwhere(gt_flat == 0)))
        
        print(f"Found {len(tumor_indices)} tumor voxels and {len(non_tumor_indices)} valid non-tumor voxels")
        
        # Shuffle indices
        np.random.shuffle(tumor_indices)
        np.random.shuffle(non_tumor_indices)
        
        # Reshape back
        gt_im = gt_im.reshape(tmp_shp)
        
        # Lists to store patches and labels
        patches = []
        labels = []
        
        # Extract tumor patches with progress tracking
        print("Extracting tumor patches...")
        tumor_count = 0
        with tqdm(total=num_tumor_patches) as pbar:
            i = 0
            while (tumor_count < num_tumor_patches) and (i < len(tumor_indices)):
                try:
                    # Get the voxel index and convert to patient, slice, y, x coordinates
                    ind = tumor_indices[i]
                    i += 1
                    ind = np.unravel_index(ind, tmp_shp)
                    patient_id, slice_idx = ind[0], ind[1]
                    p = ind[2:]
                    
                    # Add random offset if specified
                    if random_offset:
                        p = (p[0] + np.random.randint(-5, 6), p[1] + np.random.randint(-5, 6))
                    
                    # Calculate patch boundaries
                    p_y = (p[0] - h//2, p[0] + h//2)
                    p_x = (p[1] - w//2, p[1] + w//2)
                    p_x = list(map(int, p_x))
                    p_y = list(map(int, p_y))
                    
                    # Make sure the patch is within bounds
                    if (p_y[0] < 0 or p_y[1] > tmp_shp[2] or p_x[0] < 0 or p_x[1] > tmp_shp[3]):
                        continue
                    
                    # Extract patch and corresponding label
                    patch_data = self.train_im[patient_id][0:4, slice_idx, p_y[0]:p_y[1], p_x[0]:p_x[1]]
                    label_data = gt_im[patient_id, slice_idx, p_y[0]:p_y[1], p_x[0]:p_x[1]]
                    
                    # Check if the patch has the right shape
                    if patch_data.shape != (d, h, w):
                        continue
                    
                    # Check if we have at least some tumor pixels in the extracted patch
                    if np.sum(label_data > 0) < 10:  # Require at least 10 tumor pixels
                        continue
                        
                    patches.append(patch_data)
                    labels.append(label_data)
                    tumor_count += 1
                    pbar.update(1)
                    
                except IndexError:
                    # Skip patches that would go out of bounds
                    continue
        
        # Extract non-tumor patches with progress tracking
        print("Extracting non-tumor patches...")
        non_tumor_count = 0
        with tqdm(total=num_non_tumor_patches) as pbar:
            i = 0
            while (non_tumor_count < num_non_tumor_patches) and (i < len(non_tumor_indices)):
                try:
                    # Get the voxel index and convert to patient, slice, y, x coordinates
                    ind = non_tumor_indices[i]
                    i += 1
                    ind = np.unravel_index(ind, tmp_shp)
                    patient_id, slice_idx = ind[0], ind[1]
                    p = ind[2:]
                    
                    # Add random offset if specified
                    if random_offset:
                        p = (p[0] + np.random.randint(-5, 6), p[1] + np.random.randint(-5, 6))
                    
                    # Calculate patch boundaries
                    p_y = (p[0] - h//2, p[0] + h//2)
                    p_x = (p[1] - w//2, p[1] + w//2)
                    p_x = list(map(int, p_x))
                    p_y = list(map(int, p_y))
                    
                    # Make sure the patch is within bounds
                    if (p_y[0] < 0 or p_y[1] > tmp_shp[2] or p_x[0] < 0 or p_x[1] > tmp_shp[3]):
                        continue
                    
                    # Extract patch and corresponding label
                    patch_data = self.train_im[patient_id][0:4, slice_idx, p_y[0]:p_y[1], p_x[0]:p_x[1]]
                    label_data = gt_im[patient_id, slice_idx, p_y[0]:p_y[1], p_x[0]:p_x[1]]
                    
                    # Check if the patch has the right shape
                    if patch_data.shape != (d, h, w):
                        continue
                    
                    # Check that this is really a non-tumor patch (0 tumor pixels)
                    if np.sum(label_data > 0) > 0:
                        continue
                        
                    patches.append(patch_data)
                    labels.append(label_data)
                    non_tumor_count += 1
                    pbar.update(1)
                    
                except IndexError:
                    # Skip patches that would go out of bounds
                    continue
        
        print(f"Successfully extracted {tumor_count} tumor patches and {non_tumor_count} non-tumor patches")
        print(f"Patch extraction took {time.time() - start_time:.2f} seconds")
        
        # Convert to numpy arrays
        patches = np.array(patches)
        labels = np.array(labels)
        
        # Shuffle the patches
        combined = list(zip(patches, labels))
        np.random.shuffle(combined)
        patches, labels = zip(*combined)
        patches = np.array(patches)
        labels = np.array(labels)
        
        return patches, labels
        
    def augment_patches(self, patches, labels, augmentation_factor=2):
        """
        Apply data augmentation to patches
        
        Parameters:
        - patches: Input patches
        - labels: Corresponding labels
        - augmentation_factor: How many times to multiply the dataset (2 = double the data)
        """
        print(f"Augmenting patches: input shape {patches.shape}")
        start_time = time.time()
        
        augmented_patches = []
        augmented_labels = []
        
        # Add original patches
        augmented_patches.extend(patches)
        augmented_labels.extend(labels)
        
        # Calculate how many more augmented patches we need
        num_additional = (augmentation_factor - 1) * len(patches)
        
        # Create a list of indices to augment
        indices = np.random.choice(len(patches), num_additional, replace=(num_additional > len(patches)))
        
        # List of augmentation functions
        augmentations = [
            # Flip horizontally
            lambda p, l: (np.array([np.fliplr(p[i]) for i in range(p.shape[0])]), np.fliplr(l)),
            
            # Rotate 90 degrees
            lambda p, l: (np.array([np.rot90(p[i]) for i in range(p.shape[0])]), np.rot90(l)),
            
            # Rotate 180 degrees  
            lambda p, l: (np.array([np.rot90(p[i], 2) for i in range(p.shape[0])]), np.rot90(l, 2)),
            
            # Add noise
            lambda p, l: (np.clip(p + np.random.normal(0, 0.1, p.shape), -1, 1), l),
            
            # Adjust contrast
            lambda p, l: (np.clip(p * np.random.uniform(0.9, 1.1), -1, 1), l)
        ]
        
        # Apply random augmentations to selected patches
        print("Applying augmentations...")
        with tqdm(total=num_additional) as pbar:
            for idx in indices:
                # Select a random augmentation
                aug_func = np.random.choice(augmentations)
                
                # Apply the augmentation
                aug_patch, aug_label = aug_func(patches[idx], labels[idx])
                
                augmented_patches.append(aug_patch)
                augmented_labels.append(aug_label)
                pbar.update(1)
        
        # Convert to numpy arrays
        augmented_patches = np.array(augmented_patches)
        augmented_labels = np.array(augmented_labels)
        
        print(f"Augmentation took {time.time() - start_time:.2f} seconds")
        print(f"Final dataset size: {len(augmented_patches)} patches")
        
        return augmented_patches, augmented_labels
    
    def prepare_patches_for_training(self, patches, labels, one_hot=True):
        """
        Prepare patches for training (reshape, one-hot encode labels)
        
        Parameters:
        - patches: Input patches
        - labels: Corresponding labels
        - one_hot: Whether to convert labels to one-hot encoding
        """
        # Reshape patches to (batch, height, width, channels)
        patches = np.transpose(patches, (0, 2, 3, 1)).astype(np.float32)
        
        # Convert grade 4 to grade 3 as per the paper
        labels[labels == 4] = 3
        
        if one_hot:
            # Convert labels to one-hot encoding
            shp = labels.shape
            labels = labels.reshape(-1)
            labels = to_categorical(labels, num_classes=4).astype(np.uint8)
            labels = labels.reshape(shp[0], shp[1], shp[2], 4)
        
        return patches, labels

def process_dataset(data_path, num_patches=8000, output_path='./', max_patients=None, patch_size=(4, 64, 64), tumor_ratio=0.3):
    """
    Process the full dataset pipeline
    
    Parameters:
    - data_path: Path to the BRATS dataset
    - num_patches: Number of patches to extract
    - output_path: Where to save the processed data
    - max_patients: Maximum number of patients to process (for testing)
    - patch_size: Size of patches to extract
    - tumor_ratio: Ratio of tumor patches to total patches
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Get paths to all patient directories
    path_HGG = glob(os.path.join(data_path, 'BRATS2015', 'training', 'HGG', '*'))
    path_LGG = glob(os.path.join(data_path, 'BRATS2015', 'training', 'LGG', '*'))
    path_all = path_HGG + path_LGG
    
    # Shuffle patients
    np.random.shuffle(path_all)
    
    print(f"Found {len(path_all)} patient directories")
    
    # Create pipeline
    pipe = OptimizedPipeline(path_all, max_patients=max_patients, Normalize=True)
    
    # Load scans
    pipe.read_scans()
    
    # Extract balanced patches
    patches, labels = pipe.extract_balanced_patches(
        num_patches=num_patches,
        patch_size=patch_size,
        tumor_ratio=tumor_ratio
    )
    
    # Augment patches (doubles the dataset size)
    patches, labels = pipe.augment_patches(patches, labels, augmentation_factor=2)
    
    # Prepare patches for training
    X, y = pipe.prepare_patches_for_training(patches, labels)
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Save the data
    np.save(os.path.join(output_path, 'X_train.npy'), X_train)
    np.save(os.path.join(output_path, 'X_val.npy'), X_val)
    np.save(os.path.join(output_path, 'y_train.npy'), y_train)
    np.save(os.path.join(output_path, 'y_val.npy'), y_val)
    
    print("Shapes of saved arrays:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}")
    print(f"y_val: {y_val.shape}")
    
    # Verify class distribution
    total_voxels = np.prod(y_train.shape[:-1])
    class_counts = []
    
    for i in range(4):
        class_count = np.sum(y_train[..., i])
        percentage = class_count / total_voxels * 100
        class_counts.append((i, class_count, percentage))
    
    print("\nClass distribution in training set:")
    for class_idx, count, percentage in class_counts:
        class_name = ["Background", "Edema", "Non-enhancing core", "Enhancing core"][class_idx]
        print(f"Class {class_idx} ({class_name}): {count} voxels ({percentage:.2f}%)")
    
    return X_train, y_train, X_val, y_val

if __name__ == "__main__":
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Process a small dataset for testing
    X_train, y_train, X_val, y_val = process_dataset(
        data_path=current_dir, 
        num_patches=1000,  # Small number for testing
        max_patients=5,    # Only 5 patients for testing
        tumor_ratio=0.3    # 30% tumor patches
    )
    
    print("Dataset processing completed successfully!")

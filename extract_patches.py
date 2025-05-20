

import random
import numpy as np
from glob import glob
import SimpleITK as sitk
from tensorflow.keras.utils import to_categorical
import os
from sklearn.model_selection import train_test_split
import cv2
# Add imports for elastic transform
from scipy.ndimage import gaussian_filter, map_coordinates

class Pipeline(object):
    
    def __init__(self, list_train, Normalize=True):
        self.scans_train = list_train
        self.train_im = self.read_scans(Normalize)
        print("Shape of self.train_im:", self.train_im.shape)
        
    def read_scans(self, Normalize):
        train_im = []

        print("Number of scans to process:", len(self.scans_train))

        for i in range(len(self.scans_train)):
            if i % 10 == 0:
                print('iteration [{}]'.format(i))

            patient_dir = self.scans_train[i]
            
            flair = glob(os.path.join(patient_dir, '*Flair*.mha'))
            t1 = glob(os.path.join(patient_dir, '*T1*.mha'))
            t1c = glob(os.path.join(patient_dir, '*T1c*.mha'))
            t2 = glob(os.path.join(patient_dir, '*T2*.mha'))
            gt = glob(os.path.join(patient_dir, '*more*.mha'))

            t1s = [scan for scan in t1 if scan not in t1c]

            if not all([flair, t1s, t2, gt]):
                print(f"Missing files for patient: {patient_dir}")
                continue

            if len(flair) != 1 or len(t2) != 1 or len(gt) != 1 or len(t1s) != 1 or len(t1c) != 1:
                print(f"Unexpected number of files for patient: {patient_dir}")
                continue

            scans = [flair[0], t1s[0], t1c[0], t2[0], gt[0]]
            
            try:
                tmp = [sitk.GetArrayFromImage(sitk.ReadImage(scans[k])) for k in range(len(scans))]

                z0, y0, x0 = 1, 29, 42
                z1, y1, x1 = 147, 221, 194
                tmp = np.array(tmp)
                tmp = tmp[:, z0:z1, y0:y1, x0:x1]

                if Normalize:
                    tmp = self.norm_slices(tmp)

                train_im.append(tmp)
                del tmp
            except Exception as e:
                print(f"Error processing patient {patient_dir}: {e}")
                continue

        print("Number of successfully processed scans:", len(train_im))
        return np.array(train_im)
    
    def sample_patches_randomly(self, num_patches, d, h, w):
        patches, labels = [], []
        count = 0
        print(self.train_im.shape)

        gt_im = np.swapaxes(self.train_im, 0, 1)[4]   
        msk = np.swapaxes(self.train_im, 0, 1)[0]
        tmp_shp = gt_im.shape

        gt_im = gt_im.reshape(-1).astype(np.uint8)
        msk = msk.reshape(-1).astype(np.float32)

        indices = np.squeeze(np.argwhere((msk!=-9.0) & (msk!=0.0)))
        del msk

        np.random.shuffle(indices)

        gt_im = gt_im.reshape(tmp_shp)

        i = 0
        pix = len(indices)
        while (count < num_patches) and (pix > i):
            ind = indices[i]
            i += 1
            ind = np.unravel_index(ind, tmp_shp)
            patient_id, slice_idx = ind[0], ind[1]
            p = ind[2:]
            p_y = (p[0] - h//2, p[0] + h//2)
            p_x = (p[1] - w//2, p[1] + w//2)
            p_x = list(map(int, p_x))
            p_y = list(map(int, p_y))
            
            try:
                # Make sure the patch is within bounds
                if (p_y[0] < 0 or p_y[1] > tmp_shp[2] or p_x[0] < 0 or p_x[1] > tmp_shp[3]):
                    continue
                
                tmp = self.train_im[patient_id][0:4, slice_idx, p_y[0]:p_y[1], p_x[0]:p_x[1]]
                lbl = gt_im[patient_id, slice_idx, p_y[0]:p_y[1], p_x[0]:p_x[1]]

                if tmp.shape != (d, h, w):
                    continue
                patches.append(tmp)
                labels.append(lbl)
                count += 1
            except IndexError:
                # Skip patches that would go out of bounds
                continue
            
        return np.array(patches), np.array(labels)
    
    def sample_patches_with_balanced_classes(self, num_patches, d, h, w):
        """Sample patches with a better balance between tumor and non-tumor regions"""
        patches, labels = [], []
        count = 0
        
        # Access the ground truth and modality data
        gt_im = np.swapaxes(self.train_im, 0, 1)[4]   
        flair = np.swapaxes(self.train_im, 0, 1)[0]  # Use FLAIR for initial tissue mask
        
        # Get shape information
        tmp_shp = gt_im.shape
        
        # Create separate indices for tumor and non-tumor regions
        gt_flat = gt_im.reshape(-1).astype(np.uint8)
        flair_flat = flair.reshape(-1).astype(np.float32)
        
        # Valid tissue indices (not background)
        valid_indices = np.squeeze(np.argwhere((flair_flat != -9.0) & (flair_flat != 0.0)))
        
        # Separate tumor and non-tumor indices
        tumor_indices = np.squeeze(np.argwhere(gt_flat > 0))
        non_tumor_indices = np.intersect1d(valid_indices, np.squeeze(np.argwhere(gt_flat == 0)))
        
        # Calculate how many patches to extract from each class
        # We'll oversample tumor regions to address class imbalance
        # num_tumor_patches = min(int(num_patches * 0.7), len(tumor_indices))
        num_tumor_patches = min(int(num_patches * 0.25), len(tumor_indices))
        num_non_tumor_patches = num_patches - num_tumor_patches
        
        print(f"Sampling {num_tumor_patches} tumor patches and {num_non_tumor_patches} non-tumor patches")
        
        # Shuffle indices
        np.random.shuffle(tumor_indices)
        np.random.shuffle(non_tumor_indices)
        
        # Reshape back
        gt_im = gt_im.reshape(tmp_shp)
        
        # Extract tumor patches
        i = 0
        tumor_count = 0
        while (tumor_count < num_tumor_patches) and (i < len(tumor_indices)):
            try:
                ind = tumor_indices[i]
                i += 1
                ind = np.unravel_index(ind, tmp_shp)
                patient_id, slice_idx = ind[0], ind[1]
                p = ind[2:]
                p_y = (p[0] - h//2, p[0] + h//2)
                p_x = (p[1] - w//2, p[1] + w//2)
                p_x = list(map(int, p_x))
                p_y = list(map(int, p_y))
                
                # Make sure the patch is within bounds
                if (p_y[0] < 0 or p_y[1] > tmp_shp[2] or p_x[0] < 0 or p_x[1] > tmp_shp[3]):
                    continue
                
                tmp = self.train_im[patient_id][0:4, slice_idx, p_y[0]:p_y[1], p_x[0]:p_x[1]]
                lbl = gt_im[patient_id, slice_idx, p_y[0]:p_y[1], p_x[0]:p_x[1]]
                
                if tmp.shape != (d, h, w):
                    continue
                    
                patches.append(tmp)
                labels.append(lbl)
                tumor_count += 1
            except IndexError:
                # Skip patches that would go out of bounds
                continue
        
        # Extract non-tumor patches
        i = 0
        non_tumor_count = 0
        while (non_tumor_count < num_non_tumor_patches) and (i < len(non_tumor_indices)):
            try:
                ind = non_tumor_indices[i]
                i += 1
                ind = np.unravel_index(ind, tmp_shp)
                patient_id, slice_idx = ind[0], ind[1]
                p = ind[2:]
                p_y = (p[0] - h//2, p[0] + h//2)
                p_x = (p[1] - w//2, p[1] + w//2)
                p_x = list(map(int, p_x))
                p_y = list(map(int, p_y))
                
                # Make sure the patch is within bounds
                if (p_y[0] < 0 or p_y[1] > tmp_shp[2] or p_x[0] < 0 or p_x[1] > tmp_shp[3]):
                    continue
                
                tmp = self.train_im[patient_id][0:4, slice_idx, p_y[0]:p_y[1], p_x[0]:p_x[1]]
                lbl = gt_im[patient_id, slice_idx, p_y[0]:p_y[1], p_x[0]:p_x[1]]
                
                if tmp.shape != (d, h, w):
                    continue
                    
                patches.append(tmp)
                labels.append(lbl)
                non_tumor_count += 1
            except IndexError:
                # Skip patches that would go out of bounds
                continue
        
        print(f"Successfully extracted {tumor_count} tumor patches and {non_tumor_count} non-tumor patches")
        
        # Combine and shuffle
        combined = list(zip(patches, labels))
        np.random.shuffle(combined)
        patches, labels = zip(*combined)
        
        return np.array(patches), np.array(labels)
    
    # def sample_multiscale_patches(self, num_patches, scales=[(4, 64, 64), (4, 128, 128)]):
    #     """Extract patches at multiple scales with balanced class representation"""
    #     all_patches = []
    #     all_labels = []
        
    #     # Distribute patches across scales
    #     patches_per_scale = num_patches // len(scales)
        
    #     for scale in scales:
    #         d, h, w = scale
    #         patches, labels = self.sample_patches_with_balanced_classes(patches_per_scale, d, h, w)
            
    #         if len(patches) == 0:
    #             print(f"Warning: No patches extracted for scale {scale}")
    #             continue
                
    #         # If the scales differ, resize patches to a standard size
    #         if h != 128 or w != 128:
    #             resized_patches = []
    #             for patch in patches:
    #                 # Resize each modality
    #                 resized_patch = np.zeros((d, 128, 128))
    #                 for i in range(d):
    #                     resized_patch[i] = cv2.resize(patch[i], (128, 128))
    #                 resized_patches.append(resized_patch)
    #             patches = np.array(resized_patches)
                
    #             # Resize labels
    #             resized_labels = []
    #             for label in labels:
    #                 resized_labels.append(cv2.resize(label, (128, 128), interpolation=cv2.INTER_NEAREST))
    #             labels = np.array(resized_labels)
            
    #         all_patches.append(patches)
    #         all_labels.append(labels)
        
    #     # Combine patches from different scales
    #     if all_patches:
    #         combined_patches = np.concatenate(all_patches)
    #         combined_labels = np.concatenate(all_labels)
    #         return combined_patches, combined_labels
    #     else:
    #         print("Error: No patches were extracted at any scale")
    #         return np.array([]), np.array([])

    def sample_multiscale_patches(self, num_patches, scales=[(4, 64, 64), (4, 128, 128)]):
        """Extract patches at multiple scales with balanced class representation"""
        all_patches = []
        all_labels = []
        
        # Distribute patches across scales
        patches_per_scale = num_patches // len(scales)
        
        for scale in scales:
            d, h, w = scale
            print(f"Extracting patches at scale {scale}")
            patches, labels = self.sample_patches_with_balanced_classes(patches_per_scale, d, h, w)
            
            if patches is None or len(patches) == 0:
                print(f"Warning: No patches extracted for scale {scale}")
                continue
                
            print(f"Extracted {len(patches)} patches at scale {scale}")
                
            # If the scales differ, resize patches to a standard size
            if h != 128 or w != 128:
                print(f"Resizing patches from {h}x{w} to 128x128")
                resized_patches = []
                for patch in patches:
                    # Resize each modality
                    resized_patch = np.zeros((d, 128, 128))
                    for i in range(d):
                        resized_patch[i] = cv2.resize(patch[i], (128, 128))
                    resized_patches.append(resized_patch)
                patches = np.array(resized_patches)
                
                # Resize labels
                resized_labels = []
                for label in labels:
                    resized_labels.append(cv2.resize(label, (128, 128), interpolation=cv2.INTER_NEAREST))
                labels = np.array(resized_labels)
            
            all_patches.append(patches)
            all_labels.append(labels)
        
        # Check if we have any patches before concatenating
        if not all_patches:
            print("Error: No patches were extracted at any scale")
            return None, None
            
        try:
            # Combine patches from different scales
            combined_patches = np.concatenate(all_patches)
            combined_labels = np.concatenate(all_labels)
            print(f"Combined {len(combined_patches)} patches from all scales")
            return combined_patches, combined_labels
        except Exception as e:
            print(f"Error concatenating patches: {e}")
            if len(all_patches) == 1:
                return all_patches[0], all_labels[0]
            return None, None
            
    def norm_slices(self, slice_not):
        normed_slices = np.zeros((5, 146, 192, 152)).astype(np.float32)
        for slice_ix in range(4):
            normed_slices[slice_ix] = slice_not[slice_ix]
            for mode_ix in range(146):
                normed_slices[slice_ix][mode_ix] = self._normalize(slice_not[slice_ix][mode_ix])
        normed_slices[-1] = slice_not[-1]
        return normed_slices    
   
    def _normalize(self, slice):
        b = np.percentile(slice, 99)
        t = np.percentile(slice, 1)
        slice = np.clip(slice, t, b)
        image_nonzero = slice[np.nonzero(slice)]
        if np.std(slice) == 0 or np.std(image_nonzero) == 0:
            return slice
        else:
            tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
            tmp[tmp == tmp.min()] = -9
            return tmp
        
    # def augment_patches(self, patches, labels):
    #     """Apply data augmentation to patches as per paper"""
    #     augmented_patches = []
    #     augmented_labels = []
        
    #     for i in range(len(patches)):
    #         patch = patches[i]
    #         label = labels[i]
            
    #         # Original patch
    #         augmented_patches.append(patch)
    #         augmented_labels.append(label)
            
    #         # Flipped patch (horizontal)
    #         flipped_patch = np.copy(patch)
    #         flipped_label = np.copy(label)
    #         for j in range(patch.shape[0]):
    #             flipped_patch[j] = np.fliplr(patch[j])
    #         flipped_label = np.fliplr(label)
    #         augmented_patches.append(flipped_patch)
    #         augmented_labels.append(flipped_label)
            
    #         # Rotated patches (90, 180, 270 degrees)
    #         for k in range(1, 4):  # 90, 180, 270 degrees
    #             rotated_patch = np.copy(patch)
    #             rotated_label = np.copy(label)
    #             for j in range(patch.shape[0]):
    #                 rotated_patch[j] = np.rot90(patch[j], k=k)
    #             rotated_label = np.rot90(label, k=k)
    #             augmented_patches.append(rotated_patch)
    #             augmented_labels.append(rotated_label)
            
    #         # Gaussian noise (for robustness)
    #         noisy_patch = np.copy(patch)
    #         for j in range(patch.shape[0]):
    #             noise = np.random.normal(0, 0.1, patch[j].shape)
    #             noisy_patch[j] = patch[j] + noise
    #         augmented_patches.append(noisy_patch)
    #         augmented_labels.append(label)  # Label remains the same
            
    #         # Small random intensity shifts
    #         intensity_patch = np.copy(patch)
    #         for j in range(patch.shape[0]):
    #             shift = np.random.uniform(-0.1, 0.1)
    #             intensity_patch[j] = np.clip(patch[j] + shift, -1, 1)
    #         augmented_patches.append(intensity_patch)
    #         augmented_labels.append(label)
            
    #         # Add elastic deformation
    #         try:
    #             elastic_patch = np.copy(patch)
    #             for j in range(patch.shape[0]):
    #                 elastic_patch[j] = self.elastic_transform(patch[j])
    #             # Don't apply elastic transform to labels as it could change the segmentation boundaries
    #             augmented_patches.append(elastic_patch)
    #             augmented_labels.append(label)
    #         except Exception as e:
    #             print(f"Skipping elastic transform due to error: {e}")
            
    #     return np.array(augmented_patches), np.array(augmented_labels)

    def augment_patches(self, patches, labels):
        augmented_patches = []
        augmented_labels = []
        
        for i in range(len(patches)):
            patch = patches[i]
            label = labels[i]
            
            # Check if this is a tumor patch
            is_tumor_patch = np.any(label > 0)
            
            # Original patch
            augmented_patches.append(patch)
            augmented_labels.append(label)
            
            # Apply fewer augmentations to tumor patches
            if is_tumor_patch:
                # For tumor patches, only add horizontal flip
                flipped_patch = np.copy(patch)
                flipped_label = np.copy(label)
                for j in range(patch.shape[0]):
                    flipped_patch[j] = np.fliplr(patch[j])
                flipped_label = np.fliplr(label)
                augmented_patches.append(flipped_patch)
                augmented_labels.append(flipped_label)
            else:
                # For non-tumor patches, apply all augmentations as before
                # Flipped patch (horizontal)
                flipped_patch = np.copy(patch)
                flipped_label = np.copy(label)
                for j in range(patch.shape[0]):
                    flipped_patch[j] = np.fliplr(patch[j])
                flipped_label = np.fliplr(label)
                augmented_patches.append(flipped_patch)
                augmented_labels.append(flipped_label)
                
                # Rotated patches (90, 180, 270 degrees)
                for k in range(1, 4):
                    rotated_patch = np.copy(patch)
                    rotated_label = np.copy(label)
                    for j in range(patch.shape[0]):
                        rotated_patch[j] = np.rot90(patch[j], k=k)
                    rotated_label = np.rot90(label, k=k)
                    augmented_patches.append(rotated_patch)
                    augmented_labels.append(rotated_label)
                
                # Gaussian noise (for robustness)
                noisy_patch = np.copy(patch)
                for j in range(patch.shape[0]):
                    noise = np.random.normal(0, 0.1, patch[j].shape)
                    noisy_patch[j] = patch[j] + noise
                augmented_patches.append(noisy_patch)
                augmented_labels.append(label)
        return np.array(augmented_patches), np.array(augmented_labels)
        
    def elastic_transform(self, image, alpha=15, sigma=5):
        """Apply elastic deformation to image"""
        shape = image.shape
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
        
        distorted_image = map_coordinates(image, indices, order=1, mode='reflect')
        return distorted_image.reshape(shape)

# if __name__ == '__main__':
#     current_dir = os.path.dirname(os.path.abspath(__file__))

#     path_HGG = glob(os.path.join(current_dir, 'BRATS2015', 'training', 'HGG', '*'))
#     path_LGG = glob(os.path.join(current_dir, 'BRATS2015', 'training', 'LGG', '*'))
#     path_all = path_HGG + path_LGG

#     np.random.seed(2022)
#     np.random.shuffle(path_all)

#     np.random.seed(1555)
#     start, end = 0, 10
#     num_patches = 120 * (end - start) * 2
#     h, w, d = 128, 128, 4 

#     pipe = Pipeline(list_train=path_all[start:end], Normalize=True)
    
#     # Use the balanced sampling method instead of random sampling
#     # Use multi-scale patch extraction with multiple scales
#     try:
#         Patches, Y_labels = pipe.sample_multiscale_patches(num_patches)

#         # Apply data augmentation
#         Patches, Y_labels = pipe.augment_patches(Patches, Y_labels)
        
#         # Check if patches were successfully extracted
#         if len(Patches) == 0:
#             print("Error: No patches were extracted. Falling back to random sampling.")
#             Patches, Y_labels = pipe.sample_patches_randomly(num_patches, d, h, w)
#             if len(Patches) > 0:
#                 Patches, Y_labels = pipe.augment_patches(Patches, Y_labels)
#             else:
#                 print("Error: Random sampling also failed to extract patches.")
#                 exit(1)

#         # Process the patches for training
#         Patches = np.transpose(Patches, (0, 2, 3, 1)).astype(np.float32)

#         Y_labels[Y_labels == 4] = 3

#         shp = Y_labels.shape[0]
#         Y_labels = Y_labels.reshape(-1)
#         Y_labels = to_categorical(Y_labels).astype(np.uint8)
#         Y_labels = Y_labels.reshape(shp, h, w, 4)

#         shuffle = list(zip(Patches, Y_labels))
#         np.random.seed(180)
#         np.random.shuffle(shuffle)
#         Patches = np.array([shuffle[i][0] for i in range(len(shuffle))])
#         Y_labels = np.array([shuffle[i][1] for i in range(len(shuffle))])
#         del shuffle
        
#         print("Size of the patches : ", Patches.shape)
#         print("Size of their corresponding targets : ", Y_labels.shape)

#         np.save("x_training_test", Patches.astype(np.float32))
#         np.save("y_training_test", Y_labels.astype(np.uint8))
        
#     except Exception as e:
#         print(f"Error during patch extraction: {e}")
#         import traceback
#         traceback.print_exc()
if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))

    path_HGG = glob(os.path.join(current_dir, 'BRATS2015', 'training', 'HGG', '*'))
    path_LGG = glob(os.path.join(current_dir, 'BRATS2015', 'training', 'LGG', '*'))
    path_all = path_HGG + path_LGG

    np.random.seed(2022)
    np.random.shuffle(path_all)

    np.random.seed(1555)
    start, end = 0, 10
    num_patches = 120 * (end - start) * 2
    h, w, d = 128, 128, 4 

    pipe = Pipeline(list_train=path_all[start:end], Normalize=True)
    
    try:
        # Use multi-scale patch extraction
        print("Extracting patches using multi-scale approach...")
        multi_scale_patches, multi_scale_labels = pipe.sample_multiscale_patches(num_patches)
        
        # Check if multi-scale extraction was successful
        if multi_scale_patches is None or len(multi_scale_patches) == 0:
            print("Multi-scale patch extraction failed. Falling back to balanced sampling.")
            balanced_patches, balanced_labels = pipe.sample_patches_with_balanced_classes(num_patches, d, h, w)
            
            if balanced_patches is None or len(balanced_patches) == 0:
                print("Balanced sampling failed. Falling back to random sampling.")
                random_patches, random_labels = pipe.sample_patches_randomly(num_patches, d, h, w)
                
                if random_patches is None or len(random_patches) == 0:
                    print("All patch extraction methods failed. Exiting.")
                    exit(1)
                else:
                    Patches, Y_labels = random_patches, random_labels
            else:
                Patches, Y_labels = balanced_patches, balanced_labels
        else:
            Patches, Y_labels = multi_scale_patches, multi_scale_labels
        
        print(f"Successfully extracted {len(Patches)} patches")
        
        # Apply data augmentation
        print("Applying data augmentation...")
        try:
            Augmented_Patches, Augmented_Y_labels = pipe.augment_patches(Patches, Y_labels)
            
            # Check if augmentation was successful
            if Augmented_Patches is None or len(Augmented_Patches) == 0:
                print("Warning: Augmentation failed. Using original patches.")
                Augmented_Patches, Augmented_Y_labels = Patches, Y_labels
                
            Patches, Y_labels = Augmented_Patches, Augmented_Y_labels
            print(f"After augmentation: {len(Patches)} patches")
            
        except Exception as e:
            print(f"Error during augmentation: {e}")
            print("Continuing with original patches without augmentation.")

        # Process the patches for training
        print("Processing patches for training...")
        Patches = np.transpose(Patches, (0, 2, 3, 1)).astype(np.float32)

        # Convert label 4 to 3 (standard BRATS format)
        Y_labels[Y_labels == 4] = 3

        # Convert to categorical
        shp = Y_labels.shape[0]
        Y_labels = Y_labels.reshape(-1)
        Y_labels = to_categorical(Y_labels).astype(np.uint8)
        Y_labels = Y_labels.reshape(shp, h, w, 4)

        # Shuffle data
        print("Shuffling data...")
        shuffle = list(zip(Patches, Y_labels))
        np.random.seed(180)
        np.random.shuffle(shuffle)
        Patches = np.array([shuffle[i][0] for i in range(len(shuffle))])
        Y_labels = np.array([shuffle[i][1] for i in range(len(shuffle))])
        del shuffle
        
        print("Size of the patches : ", Patches.shape)
        print("Size of their corresponding targets : ", Y_labels.shape)

        # Class balance check
        total_pixels = Y_labels.size // 4  # Divide by 4 since we have 4 classes
        class_counts = [np.sum(Y_labels[:,:,:,i]) for i in range(4)]
        class_percentages = [count/total_pixels*100 for count in class_counts]
        print("Class distribution (%):")
        print(f"  Background: {class_percentages[0]:.2f}%")
        print(f"  Non-enhancing tumor: {class_percentages[1]:.2f}%")
        print(f"  Edema: {class_percentages[2]:.2f}%")
        print(f"  Enhancing tumor: {class_percentages[3]:.2f}%")

        # Save processed data
        print("Saving processed data...")
        np.save("x_training_test", Patches.astype(np.float32))
        np.save("y_training_test", Y_labels.astype(np.uint8))
        print("Data saved successfully")
        
    except Exception as e:
        print(f"Error during patch extraction: {e}")
        import traceback
        traceback.print_exc()


# import numpy as np 
# import SimpleITK as sitk 
# import os 
# from model import TwoPathwayGroupCNN 
# from losses import gen_dice_loss, dice_whole_metric, dice_core_metric, dice_en_metric 
# from tensorflow.keras.models import load_model 
# from skimage.transform import resize
# class Prediction: 
#     def __init__(self, model_path, batch_size_test=2): 
#         self.model = self.load_model(model_path) 
#         self.batch_size_test = batch_size_test 
   
   
   
#     def load_model(self, model_path): 
#         return load_model(model_path, custom_objects={ 
#             'gen_dice_loss': gen_dice_loss, 
#             'dice_whole_metric': dice_whole_metric, 
#             'dice_core_metric': dice_core_metric, 
#             'dice_en_metric': dice_en_metric }) 
   
   
   
#     def predict_volume(self, filepath_image, show=False): 
#         test_image = self.load_and_preprocess(filepath_image) 
#         prediction = self.model.predict(test_image, batch_size=self.batch_size_test, verbose=1 if show else 0) 
#         prediction = np.argmax(prediction, axis=-1) 
#         prediction[prediction == 3] = 4 
#         return prediction 
    
    


#     def load_and_preprocess(self, filepath_image):
#         image = sitk.ReadImage(filepath_image)
#         image_array = sitk.GetArrayFromImage(image)
        
#         # Ensure the input has the correct shape (1, 128, 128, 4)
#         image_array = np.expand_dims(image_array, axis=0)
#         if image_array.shape[-1] != 4:
#             image_array = np.repeat(image_array, 4, axis=-1)
#         image_array = image_array[:, :128, :128, :]
        
#         # Normalize the image
#         image_array = (image_array - np.mean(image_array)) / np.std(image_array)
        
#         return image_array
   
   
#     def save_prediction(self, prediction, output_path): 
#         prediction_sitk = sitk.GetImageFromArray(prediction.squeeze().astype(np.uint8))
#         sitk.WriteImage(prediction_sitk, output_path) 

# if __name__ == "__main__":
#     model_path = "/Users/a1/Desktop/Brain-Tumour-Segmentation/brain_segmentation/TwoPathwayGroupCNN.05_0.462.keras"
#     predictor = Prediction(model_path)
    
#     # Specify the input image file path
#     input_path = "imgs/Flair.png"
#     output_path = "predictions_twopath/prediction_Flair.png"
    
#     print(f"Processing {input_path}...")
#     prediction = predictor.predict_volume(input_path, show=True)
#     predictor.save_prediction(prediction, output_path)
#     print(f"Prediction saved to {output_path}")

import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import SimpleITK as sitk
from tensorflow.keras.models import load_model
import nibabel as nib
from skimage.transform import resize
from glob import glob
import warnings
from losses import gen_dice_loss, dice_whole_metric, dice_core_metric, dice_en_metric

class BRATSSegmentationPredictor:
    def __init__(self, model_path):
        """
        Initialize the predictor with a trained model
        
        Parameters:
        - model_path: Path to the trained model (.keras or .h5 file)
        """
        self.model_path = model_path
        self.model = self._load_model()
        self.target_size = (128, 128)  # Expected input size for the model
        
    def _load_model(self):
        """
        Load the trained model with custom loss functions
        
        Returns:
        - Loaded Keras model
        """
        print(f"Loading model from {self.model_path}...")
        try:
            model = load_model(
                self.model_path,
                custom_objects={
                    'gen_dice_loss': gen_dice_loss,
                    'dice_whole_metric': dice_whole_metric,
                    'dice_core_metric': dice_core_metric,
                    'dice_en_metric': dice_en_metric
                }
            )
            print("Model loaded successfully!")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def load_mha_file(self, file_path):
        """
        Load a .mha file using SimpleITK
        
        Parameters:
        - file_path: Path to the .mha file
        
        Returns:
        - Numpy array of the image data and metadata
        """
        try:
            image = sitk.ReadImage(file_path)
            array = sitk.GetArrayFromImage(image)
            spacing = image.GetSpacing()
            origin = image.GetOrigin()
            direction = image.GetDirection()
            
            return array, {"spacing": spacing, "origin": origin, "direction": direction}
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            raise
    
    def load_brats_case(self, patient_folder):
        """
        Load all modalities for a BRATS patient and combine them
        
        Parameters:
        - patient_folder: Path to the patient folder containing multiple .mha files
        
        Returns:
        - Combined 4-channel volume
        """
        print(f"Loading BRATS case from {patient_folder}")
        
        # Find all .mha files in the folder
        mha_files = glob(os.path.join(patient_folder, "*.mha"))
        
        if len(mha_files) < 4:
            print(f"Warning: Expected at least 4 modalities, found {len(mha_files)}")
        
        # Dictionary to hold each modality
        modalities = {}
        metadata = None
        
        # Try to identify and load each modality
        for file_path in mha_files:
            filename = os.path.basename(file_path).lower()
            
            # Identify modality from filename
            if "flair" in filename:
                key = "flair"
            elif "t1c" in filename or "t1ce" in filename or "t1gd" in filename:
                key = "t1ce"
            elif "t1" in filename:
                key = "t1"
            elif "t2" in filename:
                key = "t2"
            else:
                print(f"Unknown modality for file: {filename}")
                continue
                
            # Load the modality
            data, file_metadata = self.load_mha_file(file_path)
            modalities[key] = data
            
            # Store metadata from first file
            if metadata is None:
                metadata = file_metadata
        
        # Check if we have all required modalities
        required_modalities = ["flair", "t1", "t1ce", "t2"]
        for mod in required_modalities:
            if mod not in modalities:
                print(f"Warning: Missing {mod} modality")
        
        # Create the 4-channel volume with consistent ordering
        volume_channels = []
        channel_order = []
        
        for mod in required_modalities:
            if mod in modalities:
                volume_channels.append(modalities[mod])
                channel_order.append(mod)
            else:
                # Use zeros if a modality is missing
                if len(volume_channels) > 0:  # Make sure we have at least one channel to get the shape
                    shape = volume_channels[0].shape
                    volume_channels.append(np.zeros(shape))
                    channel_order.append(f"zero_{mod}")
        
        # Convert list of 3D arrays to a single 4D array (slices, height, width, channels)
        if volume_channels:
            # Transpose from (channels, depth, height, width) to (depth, height, width, channels)
            volume_data = np.stack(volume_channels, axis=-1)
            print(f"Combined volume shape: {volume_data.shape}, Channel order: {channel_order}")
            return volume_data, metadata, channel_order
        else:
            raise ValueError("No valid modalities found in the patient folder")
    
    def preprocess_volume(self, volume_data):
        """
        Preprocess the 4D volume for prediction
        
        Parameters:
        - volume_data: 4D array (depth, height, width, channels)
        
        Returns:
        - Preprocessed volume ready for prediction
        """
        print("Preprocessing volume...")
        
        # Normalize each channel separately
        for i in range(volume_data.shape[-1]):
            channel = volume_data[..., i]
            
            # Skip normalization if the channel is all zeros (missing modality)
            if np.max(channel) == 0:
                continue
                
            # Apply intensity normalization
            p1, p99 = np.percentile(channel[channel > 0], (1, 99))
            channel = np.clip(channel, p1, p99)
            
            # Normalize to [0, 1]
            channel = (channel - p1) / (p99 - p1)
            volume_data[..., i] = channel
        
        return volume_data
    
    def predict_volume(self, patient_folder, output_path=None):
        """
        Load a BRATS case, preprocess it, predict segmentation and save the result
        
        Parameters:
        - patient_folder: Path to folder containing .mha files for all modalities
        - output_path: Path to save the prediction result (optional)
        
        Returns:
        - Segmentation mask and saves it to output_path if specified
        """
        # Load and combine modalities
        volume_data, metadata, _ = self.load_brats_case(patient_folder)
        
        # Preprocess the volume
        preprocessed_volume = self.preprocess_volume(volume_data)
        
        # Store original shape for later reconstruction
        original_shape = preprocessed_volume.shape[:-1]  # Without channels
        
        # Process each slice
        print("Running prediction slice by slice...")
        segmentation_slices = []
        
        for i in range(preprocessed_volume.shape[0]):
            # Get the slice
            slice_data = preprocessed_volume[i]
            
            # Resize to target size if needed
            if slice_data.shape[0:2] != self.target_size:
                slice_data = resize(
                    slice_data, 
                    (self.target_size[0], self.target_size[1], slice_data.shape[-1]),
                    anti_aliasing=True, 
                    preserve_range=True
                )
            
            # Add batch dimension
            slice_data = np.expand_dims(slice_data, axis=0)
            
            # Predict
            pred = self.model.predict(slice_data, verbose=0)
            
            # Convert to label map (argmax across classes)
            pred_label = np.argmax(np.squeeze(pred, axis=0), axis=-1)
            
            # Resize back to original slice size if needed
            if pred_label.shape != original_shape[1:3]:
                pred_label = resize(
                    pred_label, 
                    original_shape[1:3], 
                    order=0,  # nearest neighbor for labels
                    preserve_range=True,
                    anti_aliasing=False
                ).astype(np.int32)
            
            segmentation_slices.append(pred_label)
        
        # Combine slices into a volume
        segmentation_volume = np.stack(segmentation_slices, axis=0)
        
        # Save result if output path is provided
        if output_path:
            self.save_prediction(segmentation_volume, output_path, metadata)
            print(f"Segmentation saved to {output_path}")
        
        return segmentation_volume
    
    def save_prediction(self, segmentation, output_path, metadata=None):
        """
        Save the prediction as a .nii.gz or .mha file
        
        Parameters:
        - segmentation: 3D segmentation volume
        - output_path: Path to save the result
        - metadata: Original scan metadata (optional)
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Determine output format based on extension
        _, ext = os.path.splitext(output_path)
        if ext == '.gz':  # .nii.gz
            # Save as NIfTI
            affine = np.eye(4)  # Default affine if no metadata
            if metadata:
                # Convert SimpleITK metadata to NIfTI affine if possible
                pass  # This is complex and omitted for brevity
                
            nib_img = nib.Nifti1Image(segmentation, affine)
            nib.save(nib_img, output_path)
        else:  # .mha or other
            # Save as MHA using SimpleITK
            seg_image = sitk.GetImageFromArray(segmentation.astype(np.uint8))
            
            if metadata:
                seg_image.SetSpacing(metadata["spacing"])
                seg_image.SetOrigin(metadata["origin"])
                seg_image.SetDirection(metadata["direction"])
                
            sitk.WriteImage(seg_image, output_path)
    
    def visualize_slice(self, patient_folder, slice_idx, output_path=None):
        """
        Visualize a slice from each modality and its prediction
        
        Parameters:
        - patient_folder: Path to patient folder
        - slice_idx: Slice index to visualize
        - output_path: Path to save the visualization (optional)
        """
        # Load and preprocess the volume
        volume_data, _, modality_order = self.load_brats_case(patient_folder)
        preprocessed_volume = self.preprocess_volume(volume_data)
        
        # Run prediction on the selected slice
        slice_data = preprocessed_volume[slice_idx]
        if slice_data.shape[0:2] != self.target_size:
            slice_data = resize(
                slice_data, 
                (self.target_size[0], self.target_size[1], slice_data.shape[-1]),
                anti_aliasing=True, 
                preserve_range=True
            )
        
        # Add batch dimension
        slice_data_batch = np.expand_dims(slice_data, axis=0)
        
        # Predict
        pred = self.model.predict(slice_data_batch, verbose=0)
        
        # Convert to label map
        pred_label = np.argmax(np.squeeze(pred, axis=0), axis=-1)
        
        # Create a visualization with all modalities and prediction
        n_rows = 2
        n_cols = 3
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
        
        # Plot each modality
        for i, modality in enumerate(modality_order[:4]):  # Up to 4 modalities
            row = i // n_cols
            col = i % n_cols
            axes[row, col].imshow(slice_data[..., i], cmap='gray')
            axes[row, col].set_title(f'Modality: {modality}')
            axes[row, col].axis('off')
        
        # Plot prediction
        axes[1, 2].imshow(pred_label, cmap='viridis')
        axes[1, 2].set_title('Prediction')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            print(f"Visualization saved to {output_path}")
        else:
            plt.show()


def process_brats_dataset(model_path, dataset_root, output_dir):
    """
    Process the entire BRATS dataset structure
    
    Parameters:
    - model_path: Path to the trained model
    - dataset_root: Root directory of the BRATS dataset
    - output_dir: Directory to save prediction results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize predictor
    predictor = BRATSSegmentationPredictor(model_path)
    
    # Find all patient folders
    for subdir in ["training", "testing"]:
        subdir_path = os.path.join(dataset_root, subdir)
        if not os.path.exists(subdir_path):
            continue
            
        # Look for HGG and LGG folders
        for grade_dir in ["HGG", "LGG"]:
            grade_path = os.path.join(subdir_path, grade_dir)
            if not os.path.exists(grade_path):
                continue
                
            # Create corresponding output directory
            grade_output_path = os.path.join(output_dir, subdir, grade_dir)
            os.makedirs(grade_output_path, exist_ok=True)
            
            # Find all patient folders
            patient_folders = [f for f in os.listdir(grade_path) 
                              if os.path.isdir(os.path.join(grade_path, f))]
            
            print(f"Found {len(patient_folders)} patients in {grade_path}")
            
            # Process each patient
            for patient_folder in patient_folders:
                patient_path = os.path.join(grade_path, patient_folder)
                output_path = os.path.join(grade_output_path, f"{patient_folder}_seg.mha")
                
                print(f"Processing {patient_path}")
                try:
                    predictor.predict_volume(patient_path, output_path)
                    
                    # Optionally create a visualization for the middle slice
                    middle_slice = predictor.load_brats_case(patient_path)[0].shape[0] // 2
                    vis_output = os.path.join(grade_output_path, f"{patient_folder}_visualization.png")
                    predictor.visualize_slice(patient_path, middle_slice, vis_output)
                    
                except Exception as e:
                    print(f"Error processing {patient_path}: {e}")
                    continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BRATS Brain MRI Segmentation')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--dataset', type=str, help='Path to BRATS dataset root directory')
    parser.add_argument('--output', type=str, help='Path to output directory')
    parser.add_argument('--patient', type=str, help='Process a single patient folder')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    
    args = parser.parse_args()
    
    if args.patient:
        # Process a single patient
        predictor = BRATSSegmentationPredictor(args.model)
        output_path = os.path.join(args.output, f"{os.path.basename(args.patient)}_seg.mha") if args.output else None
        predictor.predict_volume(args.patient, output_path)
        
        if args.visualize:
            # Visualize middle slice
            middle_slice = predictor.load_brats_case(args.patient)[0].shape[0] // 2
            vis_path = os.path.join(args.output, f"{os.path.basename(args.patient)}_vis.png") if args.output else None
            predictor.visualize_slice(args.patient, middle_slice, vis_path)
    else:
        # Process the entire dataset
        if not args.dataset or not args.output:
            parser.error("--dataset and --output are required for batch processing")
        
        process_brats_dataset(args.model, args.dataset, args.output)
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import cv2

class BrainTumorPredictor:
    def __init__(self, model_path='models/2pg_cnn_final.keras'):
        """
        Initialize the predictor with trained model
        """
        self.model_path = model_path
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Load the trained model"""
        try:
            print(f"Loading model from {self.model_path}...")
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            print("Model loaded successfully!")
            
            # Recompile with custom metrics if needed
            from losses import (combined_loss, dice_whole_metric, dice_core_metric, 
                               dice_enhancing_metric, sensitivity_metric, specificity_metric)
            
            self.model.compile(
                optimizer='adam',
                loss=combined_loss,
                metrics=[dice_whole_metric, dice_core_metric, dice_enhancing_metric, 
                        sensitivity_metric, specificity_metric]
            )
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Make sure the model file exists and is valid")
            
    def preprocess_image(self, image):
        """
        Preprocess single image for prediction
        Expected input: (128, 128, 4) - 4 channel brain MRI
        """
        if image.shape != (128, 128, 4):
            print(f"Warning: Image shape {image.shape} doesn't match expected (128, 128, 4)")
            
        # Normalize image
        image = image.astype(np.float32)
        
        # Normalize each channel separately
        for i in range(4):
            channel = image[:, :, i]
            if channel.max() > channel.min():
                image[:, :, i] = (channel - channel.min()) / (channel.max() - channel.min())
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict_single_image(self, image, threshold=0.5):
        """
        Predict tumor segmentation for a single image
        
        Args:
            image: Input image (128, 128, 4)
            threshold: Threshold for binary classification
            
        Returns:
            prediction: Segmentation mask (128, 128, 4)
            has_tumor: Boolean indicating presence of tumor
            tumor_confidence: Confidence scores for each class
        """
        # Preprocess
        processed_image = self.preprocess_image(image)
        
        # Predict
        prediction = self.model.predict(processed_image, verbose=0)
        prediction = prediction[0]  # Remove batch dimension
        
        # Apply threshold
        binary_prediction = (prediction > threshold).astype(np.uint8)
        
        # Check for tumor presence (any non-background class)
        has_tumor = np.any(binary_prediction[:, :, 1:])  # Classes 1, 2, 3 are tumor
        
        # Calculate confidence for each class
        tumor_confidence = {
            'background': np.mean(prediction[:, :, 0]),
            'necrotic': np.mean(prediction[:, :, 1]),
            'edema': np.mean(prediction[:, :, 2]),
            'enhancing': np.mean(prediction[:, :, 3])
        }
        
        return binary_prediction, has_tumor, tumor_confidence
    
    def evaluate_on_test_set(self, X_test, y_test, n_samples=30):
        """
        Evaluate model on test set and create confusion matrix
        
        Args:
            X_test: Test images
            y_test: Test labels  
            n_samples: Number of samples to evaluate
        """
        print(f"Evaluating on {min(n_samples, len(X_test))} test samples...")
        
        # Select random samples
        if len(X_test) > n_samples:
            indices = np.random.choice(len(X_test), n_samples, replace=False)
            X_sample = X_test[indices]
            y_sample = y_test[indices]
        else:
            X_sample = X_test
            y_sample = y_test
            n_samples = len(X_test)
        
        # Store predictions and ground truth
        y_true_binary = []  # True tumor presence (binary)
        y_pred_binary = []  # Predicted tumor presence (binary)
        
        detailed_results = []
        
        print("Processing samples...")
        for i in range(n_samples):
            # Get ground truth
            true_mask = y_sample[i]
            true_has_tumor = np.any(true_mask[:, :, 1:])  # Any tumor class
            
            # Get prediction
            pred_mask, pred_has_tumor, confidence = self.predict_single_image(X_sample[i])
            
            y_true_binary.append(int(true_has_tumor))
            y_pred_binary.append(int(pred_has_tumor))
            
            detailed_results.append({
                'sample_id': i,
                'true_tumor': true_has_tumor,
                'pred_tumor': pred_has_tumor,
                'confidence': confidence,
                'correct': true_has_tumor == pred_has_tumor
            })
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{n_samples} samples")
        
        # Calculate metrics
        accuracy = accuracy_score(y_true_binary, y_pred_binary)
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        
        # Create confusion matrix
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        
        # Print results
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Total samples evaluated: {n_samples}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall (Sensitivity): {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"True Negatives (No Tumor): {cm[0,0]}")
        print(f"False Positives: {cm[0,1]}")
        print(f"False Negatives: {cm[1,0]}")
        print(f"True Positives (Tumor): {cm[1,1]}")
        
        # Plot confusion matrix
        self.plot_confusion_matrix(cm, ['No Tumor', 'Tumor'])
        
        # Show sample predictions
        self.show_sample_predictions(X_sample, y_sample, detailed_results, n_show=6)
        
        return detailed_results, cm
    
    def plot_confusion_matrix(self, cm, class_names):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - Tumor Detection')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()
        
        # Calculate additional metrics from confusion matrix
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            print(f"\nDetailed Metrics:")
            print(f"Sensitivity (True Positive Rate): {sensitivity:.4f}")
            print(f"Specificity (True Negative Rate): {specificity:.4f}")
            print(f"False Positive Rate: {fp/(fp+tn):.4f}" if (fp+tn) > 0 else "False Positive Rate: 0.0000")
            print(f"False Negative Rate: {fn/(fn+tp):.4f}" if (fn+tp) > 0 else "False Negative Rate: 0.0000")
    
    def show_sample_predictions(self, X_sample, y_sample, results, n_show=6):
        """Show sample predictions with visualizations"""
        fig, axes = plt.subplots(n_show, 4, figsize=(16, 4*n_show))
        
        for i in range(min(n_show, len(results))):
            result = results[i]
            idx = result['sample_id']
            
            # Original image (show FLAIR channel)
            axes[i, 0].imshow(X_sample[idx][:, :, 0], cmap='gray')
            axes[i, 0].set_title(f'Sample {idx} - FLAIR')
            axes[i, 0].axis('off')
            
            # Ground truth (show combined tumor mask)
            true_tumor_mask = np.any(y_sample[idx][:, :, 1:], axis=2)
            axes[i, 1].imshow(true_tumor_mask, cmap='Reds')
            axes[i, 1].set_title(f'Ground Truth\nTumor: {result["true_tumor"]}')
            axes[i, 1].axis('off')
            
            # Prediction
            pred_mask, _, _ = self.predict_single_image(X_sample[idx])
            pred_tumor_mask = np.any(pred_mask[:, :, 1:], axis=2)
            axes[i, 2].imshow(pred_tumor_mask, cmap='Blues')
            axes[i, 2].set_title(f'Prediction\nTumor: {result["pred_tumor"]}')
            axes[i, 2].axis('off')
            
            # Overlay
            overlay = X_sample[idx][:, :, 0].copy()
            overlay = cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            
            # Add true tumor in red
            overlay[true_tumor_mask, 0] = 255
            # Add predicted tumor in blue  
            overlay[pred_tumor_mask, 2] = 255
            
            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title(f'Overlay\nCorrect: {result["correct"]}')
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    """Main function to run predictions"""
    
    # Initialize predictor
    predictor = BrainTumorPredictor('models/2pg_cnn_final.keras')
    
    if predictor.model is None:
        print("Failed to load model. Exiting...")
        return
    
    # Load test data
    print("Loading test data...")
    try:
        # Load your validation data as test data
        X_test = np.load('x_validation.npy')
        y_test = np.load('y_validation.npy')
        
        print(f"Test data loaded: {X_test.shape}")
        
    except FileNotFoundError:
        print("Test data files not found. Make sure x_validation.npy and y_validation.npy exist.")
        return
    
    # Run evaluation
    results, confusion_matrix = predictor.evaluate_on_test_set(
        X_test, y_test, n_samples=30
    )
    
    print("\nEvaluation completed!")
    
    # Test single image prediction
    print("\n" + "="*60)
    print("SINGLE IMAGE PREDICTION EXAMPLE")
    print("="*60)
    
    # Pick a random sample
    sample_idx = np.random.randint(0, len(X_test))
    sample_image = X_test[sample_idx]
    sample_label = y_test[sample_idx]
    
    # Predict
    pred_mask, has_tumor, confidence = predictor.predict_single_image(sample_image)
    true_has_tumor = np.any(sample_label[:, :, 1:])
    
    print(f"Sample {sample_idx}:")
    print(f"  Ground Truth: {'Tumor' if true_has_tumor else 'No Tumor'}")
    print(f"  Prediction: {'Tumor' if has_tumor else 'No Tumor'}")
    print(f"  Correct: {true_has_tumor == has_tumor}")
    print(f"  Confidence Scores:")
    for class_name, conf in confidence.items():
        print(f"    {class_name}: {conf:.4f}")

if __name__ == "__main__":
    main()
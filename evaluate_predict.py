
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import os
from model import TwoPathwayCNN  # Import your model class


def load_saved_model():
    """
    Load the saved model directly instead of recreating it
    """
    weights_path = 'twopath_phase1_test.keras'
    
    if os.path.exists(weights_path):
        try:
            print(f"Loading model from {weights_path}...")
            # Load with safe_mode=False to handle lambda functions
            model = tf.keras.models.load_model(weights_path, safe_mode=False)
            print("Model loaded successfully!")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to recreating model...")
            return recreate_model()
    else:
        print(f"Model file {weights_path} not found. Recreating model...")
        return recreate_model()
def recreate_model(img_shape=(128, 128, 4)):
    """
    Recreate the model instead of loading it from saved file
    This avoids the Lambda layer serialization issues
    """
    print("Recreating model architecture...")
    model = TwoPathwayCNN(img_shape=img_shape)
    
    # Load weights if available
    weights_path = 'twopath_phase1_test.keras'
    if os.path.exists(weights_path):
        try:
            print(f"Loading weights from {weights_path}...")
            model.model.load_weights(weights_path)
            print("Weights loaded successfully!")
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Continuing with initialized weights.")
    else:
        print(f"Weights file {weights_path} not found. Using initialized weights.")
    
    return model.model

def dice_coefficient(y_true, y_pred, smooth=1.0):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def predict_and_evaluate(model, X_val, y_val):
    """
    Make predictions and evaluate with confusion matrices and additional metrics
    """
    print("Making predictions...")
    # Make predictions
    predictions = model.predict(X_val)
    
    # Convert predictions to class labels
    pred_labels = np.argmax(predictions, axis=-1)
    true_labels = np.argmax(y_val, axis=-1)
    
    # Compute confusion matrices for each tumor region
    region_names = ['Whole Tumor', 'Tumor Core', 'Enhancing Tumor']
    
    results = {}
    
    # Create binary masks for each region
    # Whole tumor (all labels > 0)
    true_whole = (true_labels > 0).astype(int)
    pred_whole = (pred_labels > 0).astype(int)
    
    # Tumor core (labels 1 and 3)
    true_core = np.logical_or(true_labels == 1, true_labels == 3).astype(int)
    pred_core = np.logical_or(pred_labels == 1, pred_labels == 3).astype(int)
    
    # Enhancing tumor (label 3)
    true_en = (true_labels == 3).astype(int)
    pred_en = (pred_labels == 3).astype(int)
    
    true_regions = [true_whole, true_core, true_en]
    pred_regions = [pred_whole, pred_core, pred_en]
    
    # Calculate and display confusion matrices
    for i, region in enumerate(region_names):
        # Flatten arrays for confusion matrix
        true_flat = true_regions[i].flatten()
        pred_flat = pred_regions[i].flatten()
        
        # Calculate confusion matrix
        cm = confusion_matrix(true_flat, pred_flat)
        
        # Extract TP, TN, FP, FN
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Handle case where certain classes might be missing
            tn = cm[0, 0] if cm.shape[0] > 0 and cm.shape[1] > 0 else 0
            fp = cm[0, 1] if cm.shape[0] > 0 and cm.shape[1] > 1 else 0
            fn = cm[1, 0] if cm.shape[0] > 1 and cm.shape[1] > 0 else 0
            tp = cm[1, 1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
            
        # Calculate metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        
        # Calculate accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        # Calculate prediction rate (percentage of pixels classified as positive)
        pred_positive_rate = np.mean(pred_flat) if len(pred_flat) > 0 else 0
        true_positive_rate = np.mean(true_flat) if len(true_flat) > 0 else 0
        
        results[region] = {
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'Precision': precision,
            'Dice': dice,
            'Accuracy': accuracy,
            'Prediction Rate': pred_positive_rate,
            'True Positive Rate': true_positive_rate,
            'Confusion Matrix': cm
        }
        
        print(f"\n--- {region} Results ---")
        print(f"True Positives: {tp}")
        print(f"True Negatives: {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"Sensitivity: {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Dice Score: {dice:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Prediction Rate: {pred_positive_rate:.4f}")
        print(f"True Positive Rate: {true_positive_rate:.4f}")
    
    # Calculate overall accuracy across all classes
    overall_accuracy = accuracy_score(true_labels.flatten(), pred_labels.flatten())
    print(f"\n--- Overall Results ---")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    
    results['Overall'] = {
        'Accuracy': overall_accuracy
    }
    
    return results, pred_labels

def plot_confusion_matrices(results):
    """
    Plot confusion matrices for each tumor region
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, region in enumerate(results.keys()):
        if region == 'Overall':
            continue
            
        cm = results[region]['Confusion Matrix']
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[i])
        axes[i].set_title(f'{region} Confusion Matrix\nAcc: {results[region]["Accuracy"]:.4f}, Dice: {results[region]["Dice"]:.4f}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
        axes[i].set_xticklabels(['Negative', 'Positive'])
        axes[i].set_yticklabels(['Negative', 'Positive'])
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    plt.close()

def plot_metrics_comparison(results):
    """
    Plot bar chart comparing metrics across tumor regions
    """
    regions = [r for r in results.keys() if r != 'Overall']
    metrics = ['Accuracy', 'Dice', 'Sensitivity', 'Specificity', 'Precision', 'Prediction Rate']
    
    # Prepare data
    metric_data = {metric: [results[region][metric] for region in regions] for metric in metrics}
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bar_width = 0.15
    opacity = 0.8
    index = np.arange(len(regions))
    
    for i, metric in enumerate(metrics):
        plt.bar(index + i*bar_width, metric_data[metric], bar_width,
                alpha=opacity, label=metric)
    
    plt.xlabel('Tumor Region')
    plt.ylabel('Score')
    plt.title('Performance Metrics by Tumor Region')
    plt.xticks(index + bar_width * 2, regions)
    plt.legend()
    plt.tight_layout()
    plt.savefig('metrics_comparison.png')
    plt.close()

def visualize_predictions(X_val, y_val, predictions, sample_indices=None, num_samples=3):
    """
    Visualize sample predictions compared to ground truth
    """
    # Convert predictions from one-hot to label format if needed
    if len(predictions.shape) == 4 and predictions.shape[-1] > 1:  # One-hot encoded
        pred_labels = np.argmax(predictions, axis=-1)
    else:
        pred_labels = predictions
    
    if sample_indices is None:
        # Randomly select samples to visualize
        sample_indices = np.random.choice(len(X_val), min(num_samples, len(X_val)), replace=False)
    
    for i, idx in enumerate(sample_indices):
        plt.figure(figsize=(15, 5))
        
        # Get sample image, true mask, and prediction
        image = X_val[idx, :, :, 0]  # Assuming first channel is the T1 or FLAIR image
        true_mask = np.argmax(y_val[idx], axis=-1)
        pred_mask = pred_labels[idx]
        
        # Plot original image
        plt.subplot(1, 3, 1)
        plt.imshow(image, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        # Plot true segmentation
        plt.subplot(1, 3, 2)
        plt.imshow(true_mask, cmap='jet')
        plt.title('Ground Truth')
        plt.axis('off')
        
        # Plot predicted segmentation
        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask, cmap='jet')
        plt.title('Prediction')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'prediction_sample_{i}.png')
        plt.close()

if __name__ == "__main__":
    # Load validation data
    print("Loading validation data...")
    try:
        # Try to load previously saved validation data
        X_val = np.load('x_validation.npy')
        y_val = np.load('y_validation.npy')
    except FileNotFoundError:
        # If not found, load the original data and split again
        # (should match the same splitting logic used in training)
        X_data = np.load('x_training_test.npy')
        y_data = np.load('y_training_test.npy')
        
        # Take only the subset used for testing (5% of data)
        test_size = int(X_data.shape[0] * 0.05)
        X_data = X_data[:test_size]
        y_data = y_data[:test_size]
        
        # Split into train and validation using the same random seed
        from sklearn.model_selection import train_test_split
        _, X_val, _, y_val = train_test_split(
            X_data, y_data, test_size=0.2, random_state=42
        )
    
    print(f"Validation data shape: {X_val.shape}, {y_val.shape}")
    
    # Instead of loading the model from disk, recreate it and load just the weights
    # This avoids the Lambda layer serialization issues
    model = load_saved_model()
    
    # Make predictions and evaluate
    results, pred_labels = predict_and_evaluate(model, X_val, y_val)
    
    # Plot confusion matrices
    plot_confusion_matrices(results)
    
    # Plot metrics comparison
    plot_metrics_comparison(results)
    
    # Visualize some predictions
    visualize_predictions(X_val, y_val, pred_labels)
    
    print("\nEvaluation completed! Confusion matrices saved to 'confusion_matrices.png'")
    print("Metrics comparison saved to 'metrics_comparison.png'")
    
    # Display overall summary
    print("\n===== SUMMARY =====")
    print(f"Overall Accuracy: {results['Overall']['Accuracy']:.4f}")
    for region in results:
        if region == 'Overall':
            continue
        dice = results[region]['Dice']
        sens = results[region]['Sensitivity']
        spec = results[region]['Specificity']
        acc = results[region]['Accuracy']
        pred_rate = results[region]['Prediction Rate']
        print(f"{region}: Accuracy={acc:.4f}, Dice={dice:.4f}, Sensitivity={sens:.4f}, Specificity={spec:.4f}, Prediction Rate={pred_rate:.4f}")

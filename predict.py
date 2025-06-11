import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from losses import combined_loss, dice_whole_metric, dice_core_metric, dice_enhancing_metric, sensitivity_metric, specificity_metric

def load_test_data(num_samples=30):
    """
    Load test data - using validation data as test since you have train/val split
    """
    print("Loading test data...")
    
    # Load validation data as test data
    X_test = np.load('x_validation.npy')
    y_test = np.load('y_validation.npy')
    
    # Take only the first num_samples for evaluation
    if len(X_test) > num_samples:
        X_test = X_test[:num_samples]
        y_test = y_test[:num_samples]
    
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    return X_test, y_test

def predict_and_evaluate(model_path='models/2pg_cnn_final.keras', num_samples=30):
    """
    Load model, predict on test data and generate confusion matrix
    """
    print("="*60)
    print("BRAIN TUMOR SEGMENTATION PREDICTION")
    print("="*60)
    
    # Load trained model with custom objects
    print("Loading trained model...")
    # model = load_model(model_path, custom_objects={
    #     'combined_loss': combined_loss,
    #     'dice_whole_metric': dice_whole_metric,
    #     'dice_core_metric': dice_core_metric,
    #     'dice_enhancing_metric': dice_enhancing_metric,
    #     'sensitivity_metric': sensitivity_metric,
    #     'specificity_metric': specificity_metric
    # })
    # Recreate model and load weights from the saved model
    from model import TwoPathwayGroupCNN
    model_builder = TwoPathwayGroupCNN(img_shape=(128, 128, 4))
    model = model_builder.model

    # Try to load weights from the keras file
    try:
        temp_model = tf.keras.models.load_model(model_path, compile=False)
        model.set_weights(temp_model.get_weights())
    except:
        print("Loading weights manually...")
    print("Model loaded successfully!")
    # Compile the model with custom metrics
    print("Compiling model with custom metrics...")
    model.compile(
        optimizer='adam',
        loss=combined_loss,
        metrics=[dice_whole_metric, dice_core_metric, dice_enhancing_metric, 
                sensitivity_metric, specificity_metric]
)
    
    # Load test data
    X_test, y_test = load_test_data(num_samples)
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test, batch_size=8, verbose=1)
    
    # Convert predictions and true labels to class indices
    y_pred_classes = np.argmax(y_pred, axis=-1)  # Shape: (num_samples, 128, 128)
    y_true_classes = np.argmax(y_test, axis=-1)  # Shape: (num_samples, 128, 128)
    
    # Flatten for confusion matrix calculation
    y_pred_flat = y_pred_classes.flatten()
    y_true_flat = y_true_classes.flatten()
    
    # Calculate confusion matrix
    print("Calculating confusion matrix...")
    cm = confusion_matrix(y_true_flat, y_pred_flat)
    
    # Class names
    class_names = ['Background', 'Edema', 'Non-Enhancing', 'Enhancing']
    
    # Print classification report
    print("\nClassification Report:")
    print("="*50)
    report = classification_report(y_true_flat, y_pred_flat, 
                                 target_names=class_names, 
                                 digits=4)
    print(report)
    
    # Calculate per-class metrics
    print("\nPer-Class Pixel Accuracy:")
    print("="*30)
    for i, class_name in enumerate(class_names):
        if i < len(cm):
            # True positives for each class
            tp = cm[i, i]
            # Total actual pixels for each class
            total_actual = np.sum(cm[i, :])
            if total_actual > 0:
                accuracy = tp / total_actual
                print(f"{class_name:15s}: {accuracy:.4f} ({tp}/{total_actual})")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Brain Tumor Segmentation')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate overall metrics
    print("\nOverall Metrics:")
    print("="*20)
    
    # Overall pixel accuracy
    overall_accuracy = np.trace(cm) / np.sum(cm)
    print(f"Overall Pixel Accuracy: {overall_accuracy:.4f}")
    
    # Calculate Dice scores using the model's built-in metrics
    print("\nEvaluating with model metrics...")
    model_metrics = model.evaluate(X_test, y_test, batch_size=8, verbose=0)
    
    metric_names = ['Loss', 'Dice Whole', 'Dice Core', 'Dice Enhancing', 'Sensitivity', 'Specificity']
    print("\nModel Evaluation Results:")
    print("-" * 30)
    for name, value in zip(metric_names, model_metrics):
        print(f"{name:15s}: {value:.4f}")
    
    # Visualize some predictions
    visualize_predictions(X_test, y_test, y_pred, num_examples=3)
    
    return cm, y_pred, model_metrics

def visualize_predictions(X_test, y_true, y_pred, num_examples=3):
    """
    Visualize some prediction examples
    """
    print(f"\nVisualizing {num_examples} prediction examples...")
    
    fig, axes = plt.subplots(num_examples, 4, figsize=(16, 4*num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_examples):
        # Original image (first modality - FLAIR)
        axes[i, 0].imshow(X_test[i, :, :, 0], cmap='gray')
        axes[i, 0].set_title(f'Input Image {i+1} (FLAIR)')
        axes[i, 0].axis('off')
        
        # True segmentation
        true_seg = np.argmax(y_true[i], axis=-1)
        axes[i, 1].imshow(true_seg, cmap='viridis', vmin=0, vmax=3)
        axes[i, 1].set_title(f'Ground Truth {i+1}')
        axes[i, 1].axis('off')
        
        # Predicted segmentation
        pred_seg = np.argmax(y_pred[i], axis=-1)
        axes[i, 2].imshow(pred_seg, cmap='viridis', vmin=0, vmax=3)
        axes[i, 2].set_title(f'Prediction {i+1}')
        axes[i, 2].axis('off')
        
        # Difference map
        diff = (true_seg != pred_seg).astype(int)
        axes[i, 3].imshow(diff, cmap='Reds')
        axes[i, 3].set_title(f'Errors {i+1}')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_examples.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main function to run prediction and evaluation
    """
    try:
        # Run prediction and evaluation
        cm, predictions, metrics = predict_and_evaluate(
            model_path='models/2pg_cnn_final.keras', 
            num_samples=30  # Evaluate on 30 samples
        )
        
        print("\n" + "="*60)
        print("PREDICTION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Files saved:")
        print("- confusion_matrix.png")
        print("- prediction_examples.png")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you have:")
        print("1. Trained model: models/2pg_cnn_final.keras")
        print("2. Test data: x_validation.npy, y_validation.npy")
        print("3. losses.py file with custom loss functions")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check your model and data files")

if __name__ == "__main__":
    main()

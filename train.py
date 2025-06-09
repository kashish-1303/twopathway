

import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from model import TwoPathwayGroupCNN
from losses import (combined_loss, dice_whole_metric, dice_core_metric, 
                   dice_enhancing_metric, sensitivity_metric, specificity_metric, get_class_weights)
import time
from sklearn.model_selection import train_test_split
import json

class MomentumScheduler(tf.keras.callbacks.Callback):
    """
    Custom callback to schedule momentum as mentioned in paper:
    "We gradually increase the momentum coefficient from 0.5 to 0.9 during training"
    """
    def __init__(self):
        super(MomentumScheduler, self).__init__()
        
    def on_epoch_begin(self, epoch, logs=None):
        # Gradually increase momentum from 0.5 to 0.9
        momentum = min(0.5 + epoch * 0.01, 0.9)
        
        # Fix: Check if optimizer has momentum attribute and set it properly
        if hasattr(self.model.optimizer, 'momentum'):
            if hasattr(self.model.optimizer.momentum, 'assign'):
                # For TensorFlow 2.x
                self.model.optimizer.momentum.assign(momentum)
            else:
                # Fallback method
                self.model.optimizer.momentum = momentum
        
        print(f"Epoch {epoch + 1}: Momentum = {momentum:.3f}")

def create_balanced_dataset(X, y, samples_per_class=5000):
    """
    Create balanced dataset - simplified since your patches are already balanced
    """
    print("Creating balanced dataset...")
    
    # Your patches are already balanced from extract_patches.py
    # Just shuffle and potentially subsample if needed
    
    total_samples = len(X)
    if total_samples > samples_per_class * 4:  # 4 classes
        # Randomly sample if we have too many patches
        indices = np.random.choice(total_samples, samples_per_class * 4, replace=False)
        balanced_X = X[indices]
        balanced_y = y[indices]
    else:
        balanced_X = X
        balanced_y = y
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(balanced_X))
    balanced_X = balanced_X[shuffle_idx]
    balanced_y = balanced_y[shuffle_idx]
    
    print(f"Balanced dataset created: {balanced_X.shape}")
    
    # Check class distribution
    for class_idx in range(4):
        class_pixels = np.sum(balanced_y[:,:,:,class_idx])
        total_pixels = balanced_y.shape[0] * balanced_y.shape[1] * balanced_y.shape[2]
        percentage = (class_pixels / total_pixels) * 100
        print(f"Class {class_idx}: {percentage:.2f}% of pixels")
    
    return balanced_X, balanced_y

def train_phase1(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """
    Phase 1: Train on balanced patches as mentioned in paper
    "Once the initial training on done on balance dataset"
    """
    print("="*60)
    print("PHASE 1: Training on balanced patches")
    print("="*60)
    
    # Create balanced dataset
    X_train_balanced, y_train_balanced = create_balanced_dataset(X_train, y_train)
    
    # Compile model with SGD optimizer as per paper
    # "We set the learning rate to 0.005 with the decay 0.1"
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=0.005,
        momentum=0.5,  # Will be scheduled
    )
    
    model.compile(
        optimizer=optimizer,
        loss=combined_loss,
        metrics=[dice_whole_metric, dice_core_metric, dice_enhancing_metric, 
                sensitivity_metric, specificity_metric]
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            'models/2pg_cnn_phase1_best.keras',
            monitor='val_dice_whole_metric',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,  # As per paper: "decay 0.1"
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),
        MomentumScheduler(),
        EarlyStopping(
            monitor='val_dice_whole_metric',
            patience=20,
            mode='max',
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Calculate class weights for balanced training
    # class_weights = get_class_weights(y_train_balanced)
    
    print(f"Training Phase 1 with {len(X_train_balanced)} balanced samples...")
    # print(f"Class weights: {class_weights}")
    
    # Start training
    start_time = time.time()
    
    history = model.fit(
        X_train_balanced, y_train_balanced,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"Phase 1 training completed in {training_time:.2f} seconds")
    print(f"Average time per epoch: {training_time/epochs:.2f} seconds")
    
    return history

def train_phase2(model, X_train, y_train, X_val, y_val, epochs=30, batch_size=32):
    """
    Phase 2: Fine-tune on unbalanced data (natural distribution)
    "we moved to un-balanced nature of the data and train the output layer only"
    """
    print("="*60)
    print("PHASE 2: Fine-tuning on unbalanced data (output layer only)")
    print("="*60)
    
    # Freeze all layers except the final output layers
    for layer in model.layers[:-3]:  # Keep last 3 layers trainable
        layer.trainable = False
    
    # Recompile with lower learning rate for fine-tuning
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=0.001,  # Lower learning rate for fine-tuning
        momentum=0.9,
    )
    
    model.compile(
        optimizer=optimizer,
        loss=combined_loss,
        metrics=[dice_whole_metric, dice_core_metric, dice_enhancing_metric, 
                sensitivity_metric, specificity_metric]
    )
    
    # Callbacks for phase 2
    callbacks = [
        ModelCheckpoint(
            'models/2pg_cnn_phase2_best.keras',
            monitor='val_dice_whole_metric',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_dice_whole_metric',
            patience=10,
            mode='max',
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    print(f"Fine-tuning Phase 2 with {len(X_train)} unbalanced samples...")
    print("Training only the output layers...")
    
    # Start fine-tuning
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"Phase 2 fine-tuning completed in {training_time:.2f} seconds")
    
    # Unfreeze all layers for final result
    for layer in model.layers:
        layer.trainable = True
    
    return history

def evaluate_model(model, X_test, y_test, batch_size=32):
    """
    Evaluate model using BraTS metrics as mentioned in paper
    """
    print("="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Predict on test set
    y_pred = model.predict(X_test, batch_size=batch_size, verbose=1)
    
    # Calculate metrics
    results = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
    
    # Print results
    metric_names = ['loss', 'dice_whole', 'dice_core', 'dice_enhancing', 'sensitivity', 'specificity']
    
    print("\nTest Results:")
    print("-" * 40)
    for name, value in zip(metric_names, results):
        print(f"{name:15s}: {value:.4f}")
    
    return results, y_pred

def plot_training_history(history_phase1, history_phase2=None, save_path='training_plots.png'):
    """
    Plot training history for both phases
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Phase 1 plots
    if history_phase1:
        # Loss plot
        axes[0, 0].plot(history_phase1.history['loss'], label='Train Loss (Phase 1)', color='blue')
        axes[0, 0].plot(history_phase1.history['val_loss'], label='Val Loss (Phase 1)', color='red')
        axes[0, 0].set_title('Training Loss - Phase 1')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Dice score plot
        axes[0, 1].plot(history_phase1.history['dice_whole_metric'], label='Train Dice (Phase 1)', color='blue')
        axes[0, 1].plot(history_phase1.history['val_dice_whole_metric'], label='Val Dice (Phase 1)', color='red')
        axes[0, 1].set_title('Dice Score - Phase 1')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Phase 2 plots
    if history_phase2:
        # Loss plot
        axes[1, 0].plot(history_phase2.history['loss'], label='Train Loss (Phase 2)', color='green')
        axes[1, 0].plot(history_phase2.history['val_loss'], label='Val Loss (Phase 2)', color='orange')
        axes[1, 0].set_title('Training Loss - Phase 2')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Dice score plot
        axes[1, 1].plot(history_phase2.history['dice_whole_metric'], label='Train Dice (Phase 2)', color='green')
        axes[1, 1].plot(history_phase2.history['val_dice_whole_metric'], label='Val Dice (Phase 2)', color='orange')
        axes[1, 1].set_title('Dice Score - Phase 2')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Dice Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def save_training_results(history_phase1, history_phase2, results, save_dir='results'):
    """
    Save training results and history
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save training history
    history_data = {
        'phase1': history_phase1.history if history_phase1 else None,
        'phase2': history_phase2.history if history_phase2 else None,
        'test_results': {
            'loss': float(results[0]),
            'dice_whole': float(results[1]),
            'dice_core': float(results[2]),
            'dice_enhancing': float(results[3]),
            'sensitivity': float(results[4]),
            'specificity': float(results[5])
        }
    }
    
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history_data, f, indent=2)
    
    print(f"Training results saved to {save_dir}/")

def main_training_pipeline(X_train, y_train, X_val, y_val, X_test, y_test, 
                          img_shape=(128, 128, 4), phase1_epochs=50, phase2_epochs=30, batch_size=32):
    """
    Complete training pipeline implementing the two-phase approach from the paper
    """
    print("="*80)
    print("TWO-PATHWAY-GROUP CNN TRAINING PIPELINE")
    print("Implementing the approach from: 'Efficient Brain Tumor Segmentation'")
    print("="*80)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Initialize model
    print("Initializing Two-Pathway-Group CNN model...")
    model = TwoPathwayGroupCNN(img_shape=img_shape)
    print("Model initialized successfully!")
    print(f"Model parameters: {model.model.count_params():,}")
    
    # Display model summary
    model.summary()
    
    # Phase 1: Balanced training
    print(f"\nStarting Phase 1 training for {phase1_epochs} epochs...")
    history_phase1 = train_phase1(
        model.model, X_train, y_train, X_val, y_val, 
        epochs=phase1_epochs, batch_size=batch_size
    )
    
    # Phase 2: Unbalanced fine-tuning
    print(f"\nStarting Phase 2 fine-tuning for {phase2_epochs} epochs...")
    history_phase2 = train_phase2(
        model.model, X_train, y_train, X_val, y_val, 
        epochs=phase2_epochs, batch_size=batch_size
    )
    
    # Final evaluation
    print("\nEvaluating final model...")
    results, predictions = evaluate_model(model.model, X_test, y_test, batch_size=batch_size)
    
    # Plot training history
    print("\nGenerating training plots...")
    plot_training_history(history_phase1, history_phase2)
    
    # Save results
    print("\nSaving training results...")
    save_training_results(history_phase1, history_phase2, results)
    
    # Save final model
    model.model.model.save('models/2pg_cnn_final.keras')
    print("Final model saved to models/2pg_cnn_final.keras")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    return model, history_phase1, history_phase2, results

def load_brats_data(data_dir="./", img_shape=(128, 128, 4), test_split=0.2):
    """
    Load preprocessed BraTS patches from your extract_patches.py output
    """
    print("Loading BraTS dataset...")
    
    try:
        # Load your saved files (update these names to match your actual files)
        X_train = np.load('x_training.npy')  # Shape: (N, 128, 128, 4)
        y_train = np.load('y_training.npy')  # Shape: (N, 128, 128, 4) - one-hot encoded
        X_val = np.load('x_validation.npy')
        y_val = np.load('y_validation.npy')
        
        # Create a test set from validation data (since you only have train/val)
        X_val, X_test, y_val, y_test = train_test_split(
            X_val, y_val, test_size=0.5, random_state=42
        )
        
        print(f"Loaded data shapes:")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_val: {X_val.shape}")
        print(f"  X_test: {X_test.shape}")
        print(f"  y_train: {y_train.shape}")
        print(f"  y_val: {y_val.shape}")
        print(f"  y_test: {y_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    except FileNotFoundError as e:
        print(f"Data files not found: {e}")
        print("Make sure you've run extract_patches.py first")
        print("Expected files: x_training.npy, y_training.npy, x_validation.npy, y_validation.npy")
        return None, None, None, None, None, None

if __name__ == "__main__":
    # Configuration
    DATA_DIR = "./"  # Current directory where your .npy files are
    IMG_SHAPE = (128, 128, 4)  # Matches your patch size
    PHASE1_EPOCHS = 5
    PHASE2_EPOCHS = 3
    BATCH_SIZE = 16  # Reduce if you have memory issues
    
    # Load data (this will use your saved .npy files)
    X_train, X_val, X_test, y_train, y_val, y_test = load_brats_data(
        DATA_DIR, img_shape=IMG_SHAPE
    )
    
    if X_train is not None:
        print("Data loaded successfully!")
        print(f"Training on {len(X_train)} patches")
        
        # Run training pipeline
        model, hist1, hist2, results = main_training_pipeline(
            X_train, y_train, X_val, y_val, X_test, y_test,
            img_shape=IMG_SHAPE,
            phase1_epochs=PHASE1_EPOCHS,
            phase2_epochs=PHASE2_EPOCHS,
            batch_size=BATCH_SIZE
        )
        
        print("Training pipeline completed successfully!")
    else:
        print("Please run extract_patches.py first to generate the training data.")

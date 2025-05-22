
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
import matplotlib.pyplot as plt
from model import TwoPathwayCNN
import time

# class GradientAccumulationCallback(tf.keras.callbacks.Callback):
#     def __init__(self, accumulation_steps=4):
#         super().__init__()
#         self.accumulation_steps = accumulation_steps
#         self.gradients = None

#     def on_batch_end(self, batch, logs=None):
#         if batch % self.accumulation_steps == 0:
#             self.model.optimizer.apply_gradients()

def train_phase1(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """
    Phase 1: Train on balanced patches
    """
    print("Starting Phase 1: Training on balanced patches...")
    
    # Create model checkpoint callback - updated to use .keras extension
    checkpoint = ModelCheckpoint(
        'twopath_phase1.keras', 
        monitor='val_dice_whole_metric', 
        verbose=1, 
        save_best_only=True, 
        mode='max'
    )
    
    # Learning rate schedule: Start with 0.005 and decay by 0.1
    # Replace the lr_schedule function with:
    def lr_schedule(epoch):
        initial_lr = 0.001
        if epoch < 5:
            return initial_lr
        elif epoch < 10:
            return initial_lr * 0.5
        else:
            return initial_lr * 0.1
    
    # Momentum schedule: Gradually increase from 0.5 to 0.9
    def momentum_schedule(epoch):
        return min(0.5 + epoch * 0.01, 0.9)
    
    lr_scheduler = LearningRateScheduler(lr_schedule)
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=1,
        restore_best_weights=True
    )
    # grad_accum = GradientAccumulationCallback(accumulation_steps=4),
    # Train the model - Phase 1
    start_time = time.time()
    history = model.model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, lr_scheduler, early_stopping],
        verbose=1
    )
    end_time = time.time()
    
    print(f"Phase 1 training completed in {end_time - start_time:.2f} seconds")
    
    # Save training history
    np.save('training_history_phase1.npy', history.history)
    
    # Plot training history
    plot_training_history(history, 'phase1')
    
    return history

def train_phase2(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
    """
    Phase 2: Fine-tune only the output layer on unbalanced dataset
    """
    print("Starting Phase 2: Fine-tuning on unbalanced dataset...")
    
    # Load best weights from phase 1 - updated path
    model.model.load_weights('twopath_phase1.keras')
    
    trainable_start = len(model.model.layers) // 2
    for i, layer in enumerate(model.model.layers):
        layer.trainable = i >= trainable_start
    
    # Recompile model with lower learning rate
    model.model.compile(
        loss='categorical_crossentropy',  # Switch to standard cross-entropy
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=[dice_whole_metric, dice_core_metric, dice_en_metric]
    )
    
    # Create model checkpoint callback - updated to use .keras extension
    checkpoint = ModelCheckpoint(
        'twopath_final.keras', 
        monitor='val_dice_whole_metric', 
        verbose=1, 
        save_best_only=True, 
        mode='max'
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1,
        restore_best_weights=True
    )
    
    # Train the model - Phase 2
    start_time = time.time()
    history = model.model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )
    end_time = time.time()
    
    print(f"Phase 2 training completed in {end_time - start_time:.2f} seconds")
    
    # Save training history
    np.save('training_history_phase2.npy', history.history)
    
    # Plot training history
    plot_training_history(history, 'phase2')
    
    return history

def plot_training_history(history, phase):
    """
    Plot training metrics
    """
    plt.figure(figsize=(12, 10))
    
    # Plot training & validation loss
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Model loss ({phase})')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Plot dice whole metric
    plt.subplot(2, 2, 2)
    plt.plot(history.history['dice_whole_metric'])
    plt.plot(history.history['val_dice_whole_metric'])
    plt.title(f'Dice Whole Tumor ({phase})')
    plt.ylabel('Dice')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot dice core metric
    plt.subplot(2, 2, 3)
    plt.plot(history.history['dice_core_metric'])
    plt.plot(history.history['val_dice_core_metric'])
    plt.title(f'Dice Tumor Core ({phase})')
    plt.ylabel('Dice')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot dice enhancing metric
    plt.subplot(2, 2, 4)
    plt.plot(history.history['dice_en_metric'])
    plt.plot(history.history['val_dice_en_metric'])
    plt.title(f'Dice Enhancing Tumor ({phase})')
    plt.ylabel('Dice')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.tight_layout()
    plt.savefig(f'training_history_{phase}.png')
    plt.close()

if __name__ == "__main__":
    # Load preprocessed data
    X_train = np.load('x_training.npy')
    y_train = np.load('y_training.npy')
    
    print(f"Loaded training data: {X_train.shape}, {y_train.shape}")
    
    # Split into training and validation sets
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape}, {y_train.shape}")
    print(f"Validation set: {X_val.shape}, {y_val.shape}")
    
    # Define input shape based on loaded data
    img_shape = X_train.shape[1:]
    print(f"Input shape: {img_shape}")
    
    # Create model
    model = TwoPathwayCNN(img_shape=img_shape)
    model.summary()
    
    # First phase of training with balanced data
    history_phase1 = train_phase1(
        model, 
        X_train, y_train, 
        X_val, y_val, 
        epochs=8, 
        batch_size=4  # Smaller batch size for CPU
    )
    
    # Second phase of training to fine-tune output layer
    history_phase2 = train_phase2(
        model, 
        X_train, y_train, 
        X_val, y_val, 
        epochs=4, 
        batch_size=4  # Smaller batch size for CPU
    )
    
    print("Training completed successfully!")





#27th april(complete code)
# import numpy as np
# import os
# import tensorflow as tf
# from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
# import matplotlib.pyplot as plt
# from model import TwoPathwayCNN
# import time

# # Set memory growth for GPU
# physical_devices = tf.config.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     try:
#         for device in physical_devices:
#             tf.config.experimental.set_memory_growth(device, True)
#     except:
#         print("Memory growth setting failed. Continuing with default settings.")

# def train_phase1(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
#     """
#     Phase 1: Train on balanced patches
#     """
#     print("Starting Phase 1: Training on balanced patches...")
    
#     # Create model checkpoint callback - updated to use .keras extension
#     checkpoint = ModelCheckpoint(
#         'twopath_phase1.keras', 
#         monitor='val_dice_whole_metric', 
#         verbose=1, 
#         save_best_only=True, 
#         mode='max'
#     )
    
#     # Learning rate schedule: Start with 0.005 and decay by 0.1
#     def lr_schedule(epoch):
#         initial_lr = 0.005
#         if epoch < 10:
#             return initial_lr
#         elif epoch < 20:
#             return initial_lr * 0.1
#         else:
#             return initial_lr * 0.01
    
#     # Momentum schedule: Gradually increase from 0.5 to 0.9
#     def momentum_schedule(epoch):
#         return min(0.5 + epoch * 0.01, 0.9)
    
#     lr_scheduler = LearningRateScheduler(lr_schedule)
    
#     # Early stopping
#     early_stopping = EarlyStopping(
#         monitor='val_loss',
#         patience=10,
#         verbose=1,
#         restore_best_weights=True
#     )
    
#     # Train the model - Phase 1
#     start_time = time.time()
#     history = model.model.fit(
#         X_train, y_train,
#         validation_data=(X_val, y_val),
#         epochs=epochs,
#         batch_size=batch_size,
#         callbacks=[checkpoint, lr_scheduler, early_stopping],
#         verbose=1
#     )
#     end_time = time.time()
    
#     print(f"Phase 1 training completed in {end_time - start_time:.2f} seconds")
    
#     # Save training history
#     np.save('training_history_phase1.npy', history.history)
    
#     # Plot training history
#     plot_training_history(history, 'phase1')
    
#     return history

# def train_phase2(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
#     """
#     Phase 2: Fine-tune only the output layer on unbalanced dataset
#     """
#     print("Starting Phase 2: Fine-tuning on unbalanced dataset...")
    
#     # Load best weights from phase 1 - updated path
#     model.model.load_weights('twopath_phase1.keras')
    
#     # Freeze all layers except the output layer
#     for layer in model.model.layers[:-1]:
#         layer.trainable = False
    
#     # Recompile model with lower learning rate
#     model.model.compile(
#         loss='categorical_crossentropy',  # Switch to standard cross-entropy
#         optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#         metrics=[dice_whole_metric, dice_core_metric, dice_en_metric]
#     )
    
#     # Create model checkpoint callback - updated to use .keras extension
#     checkpoint = ModelCheckpoint(
#         'twopath_final.keras', 
#         monitor='val_dice_whole_metric', 
#         verbose=1, 
#         save_best_only=True, 
#         mode='max'
#     )
    
#     # Early stopping
#     early_stopping = EarlyStopping(
#         monitor='val_loss',
#         patience=10,
#         verbose=1,
#         restore_best_weights=True
#     )
    
#     # Train the model - Phase 2
#     start_time = time.time()
#     history = model.model.fit(
#         X_train, y_train,
#         validation_data=(X_val, y_val),
#         epochs=epochs,
#         batch_size=batch_size,
#         callbacks=[checkpoint, early_stopping],
#         verbose=1
#     )
#     end_time = time.time()
    
#     print(f"Phase 2 training completed in {end_time - start_time:.2f} seconds")
    
#     # Save training history
#     np.save('training_history_phase2.npy', history.history)
    
#     # Plot training history
#     plot_training_history(history, 'phase2')
    
#     return history

# def plot_training_history(history, phase):
#     """
#     Plot training metrics
#     """
#     plt.figure(figsize=(12, 10))
    
#     # Plot training & validation loss
#     plt.subplot(2, 2, 1)
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title(f'Model loss ({phase})')
#     plt.ylabel('Loss')
#     plt.xlabel('Epoch')
#     plt.legend(['Train', 'Validation'], loc='upper right')
    
#     # Plot dice whole metric
#     plt.subplot(2, 2, 2)
#     plt.plot(history.history['dice_whole_metric'])
#     plt.plot(history.history['val_dice_whole_metric'])
#     plt.title(f'Dice Whole Tumor ({phase})')
#     plt.ylabel('Dice')
#     plt.xlabel('Epoch')
#     plt.legend(['Train', 'Validation'], loc='lower right')
    
#     # Plot dice core metric
#     plt.subplot(2, 2, 3)
#     plt.plot(history.history['dice_core_metric'])
#     plt.plot(history.history['val_dice_core_metric'])
#     plt.title(f'Dice Tumor Core ({phase})')
#     plt.ylabel('Dice')
#     plt.xlabel('Epoch')
#     plt.legend(['Train', 'Validation'], loc='lower right')
    
#     # Plot dice enhancing metric
#     plt.subplot(2, 2, 4)
#     plt.plot(history.history['dice_en_metric'])
#     plt.plot(history.history['val_dice_en_metric'])
#     plt.title(f'Dice Enhancing Tumor ({phase})')
#     plt.ylabel('Dice')
#     plt.xlabel('Epoch')
#     plt.legend(['Train', 'Validation'], loc='lower right')
    
#     plt.tight_layout()
#     plt.savefig(f'training_history_{phase}.png')
#     plt.close()

# if __name__ == "__main__":
#     # Load preprocessed data
#     X_train = np.load('x_training.npy')
#     y_train = np.load('y_training.npy')
    
#     print(f"Loaded training data: {X_train.shape}, {y_train.shape}")
    
#     # Split into training and validation sets
#     from sklearn.model_selection import train_test_split
#     X_train, X_val, y_train, y_val = train_test_split(
#         X_train, y_train, test_size=0.2, random_state=42
#     )
    
#     print(f"Training set: {X_train.shape}, {y_train.shape}")
#     print(f"Validation set: {X_val.shape}, {y_val.shape}")
    
#     # Define input shape based on loaded data
#     img_shape = X_train.shape[1:]
#     print(f"Input shape: {img_shape}")
    
#     # Create model
#     model = TwoPathwayCNN(img_shape=img_shape)
#     model.summary()
    
#     # First phase of training with balanced data
#     history_phase1 = train_phase1(
#         model, 
#         X_train, y_train, 
#         X_val, y_val, 
#         epochs=30, 
#         batch_size=8  # Smaller batch size for CPU
#     )
    
#     # Second phase of training to fine-tune output layer
#     history_phase2 = train_phase2(
#         model, 
#         X_train, y_train, 
#         X_val, y_val, 
#         epochs=10, 
#         batch_size=8  # Smaller batch size for CPU
#     )
    
#     print("Training completed successfully!")


#test code:
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
import matplotlib.pyplot as plt
from model import TwoPathwayCNN
import time

# Set memory growth for GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except:
        print("Memory growth setting failed. Continuing with default settings.")

def train_phase1(model, X_train, y_train, X_val, y_val, epochs=2, batch_size=8):
    """
    Phase 1: Train on balanced patches - Reduced epochs for testing
    """
    print("Starting Phase 1: Testing training setup...")
    
    # Create model checkpoint callback
    checkpoint = ModelCheckpoint(
        'twopath_phase1_test.keras', 
        monitor='val_dice_whole_metric', 
        verbose=1, 
        save_best_only=True, 
        mode='max'
    )
    
    # Simplified learning rate schedule for testing
    lr_scheduler = LearningRateScheduler(lambda epoch: 0.001)
    
    # Train the model - Phase 1 (reduced epochs)
    start_time = time.time()
    history = model.model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, lr_scheduler],
        verbose=1
    )
    end_time = time.time()
    
    print(f"Phase 1 test training completed in {end_time - start_time:.2f} seconds")
    
    # Save training history
    np.save('test_history_phase1.npy', history.history)
    
    # Plot training history
    plot_training_history(history, 'test_phase1')
    
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
    
    print(f"Original data shape: {X_train.shape}, {y_train.shape}")
    
    # Take only a small subset for testing (5% of the data)
    test_size = int(X_train.shape[0] * 0.05)  # 5% of data
    X_train = X_train[:test_size]
    y_train = y_train[:test_size]
    
    print(f"Reduced test data shape: {X_train.shape}, {y_train.shape}")
    
    # Split into training and validation sets
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print(f"Test training set: {X_train.shape}, {y_train.shape}")
    print(f"Test validation set: {X_val.shape}, {y_val.shape}")
    
    # Define input shape based on loaded data
    img_shape = X_train.shape[1:]
    print(f"Input shape: {img_shape}")
    
    # Create model
    model = TwoPathwayCNN(img_shape=img_shape)
    model.summary()
    
    # Test training with reduced epochs
    history_phase1 = train_phase1(
        model, 
        X_train, y_train, 
        X_val, y_val, 
        epochs=2,  # Just 2 epochs for testing
        batch_size=8
    )
    
    print("Test training completed successfully!")
    print("If this ran without errors, your full training setup should work fine.")
    print("You can now run the full training with the original train.py")

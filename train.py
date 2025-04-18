import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
import math


if not os.path.exists('brain_segmentation1'):
    os.makedirs('brain_segmentation1')

class Training(object):
    def __init__(self, batch_size, nb_epoch, load_model_resume_training=None):
        """
        Initialize the Training class
        
        Parameters:
        - batch_size: Size of batches for training
        - nb_epoch: Number of epochs to train
        - load_model_resume_training: Path to pre-trained model to resume training
        """
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.load_model_resume_training = load_model_resume_training
        
        # Import the model here to avoid circular imports
        from model import TwoPathwayCNN
        
        if load_model_resume_training is not None:
            # Load pre-trained model with custom objects if needed
            from tensorflow.keras.models import load_model
            from losses import gen_dice_loss, dice_whole_metric, dice_core_metric, dice_en_metric
            
            self.model = load_model(
                load_model_resume_training,
                custom_objects={
                    'gen_dice_loss': gen_dice_loss,
                    'dice_whole_metric': dice_whole_metric,
                    'dice_core_metric': dice_core_metric,
                    'dice_en_metric': dice_en_metric
                }
            )
            print("Pre-trained model loaded from:", load_model_resume_training)
        else:
            # Create a new TwoPathwayCNN model
            model_instance = TwoPathwayCNN(img_shape=(128, 128, 4))
            self.model = model_instance.model
            print("TwoPathwayCNN model initialized")
            
        print("Number of trainable parameters:", self.model.count_params())

    def fit_2pg(self, X_train, Y_train, X_valid, Y_valid):
        """
        Train the model using the approach described in the paper
        
        Parameters:
        - X_train: Training input data
        - Y_train: Training target data
        - X_valid: Validation input data
        - Y_valid: Validation target data
        
        Returns:
        - history: Training history object
        """
        print("Preparing data generator...")
        train_generator = self.img_msk_gen(X_train, Y_train, seed=42)
        
        print("Setting up callbacks...")
        checkpointer = ModelCheckpoint(
            filepath='brain_segmentation1/TwoPathwayCNN.{epoch:02d}_{val_loss:.3f}.keras', 
            verbose=1,
            save_best_only=True
        )
        
        # Momentum scheduler to gradually increase momentum from 0.5 to 0.9
        momentum_scheduler = MomentumScheduler(
            initial_momentum=0.5,
            final_momentum=0.9,
            epochs=self.nb_epoch
        )
        
        # Learning rate scheduler using the specified approach
        lr_scheduler = LearningRateScheduler(self.lr_schedule)
        
        # Early stopping if no improvement
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        # Combine all callbacks
        callbacks = [
            checkpointer,
            early_stopping,
            lr_scheduler,
            momentum_scheduler
        ]
        
        # Configure the model with SGD optimizer as described
        initial_learning_rate = 0.005
        initial_momentum = 0.5
        optimizer = SGD(
            learning_rate=initial_learning_rate,
            momentum=initial_momentum,
            decay=0.1
        )
        
        # Recompile model with the new optimizer configuration
        from losses import gen_dice_loss, dice_whole_metric, dice_core_metric, dice_en_metric
        
        self.model.compile(
            optimizer=optimizer,
            loss=gen_dice_loss,
            metrics=[
                dice_whole_metric,
                dice_core_metric,
                dice_en_metric
            ]
        )
        
        print("Starting model training with TwoPathwayCNN using paper's approach...")
        start_time = time.time()
        
        history = self.model.fit(
            train_generator,
            steps_per_epoch=max(1, len(X_train) // self.batch_size),
            epochs=self.nb_epoch,
            validation_data=(X_valid, Y_valid),
            verbose=1,
            callbacks=callbacks
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        minutes = int(training_time // 60)
        seconds = int(training_time % 60)
        print(f"Model training completed in {minutes} minutes and {seconds} seconds.")
        
        return history

    def img_msk_gen(self, X_train, Y_train, seed):
        """
        Create a generator for data augmentation
        
        Parameters:
        - X_train: Training input data
        - Y_train: Training target data
        - seed: Random seed for reproducibility
        
        Returns:
        - Generator yielding batches of augmented data
        """
        # Data augmentation for both images and masks
        datagen = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True,
            rotation_range=20,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1
        )
        
        datagen_msk = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True,
            rotation_range=20,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1
        )
        
        image_generator = datagen.flow(X_train, batch_size=self.batch_size, seed=seed)
        mask_generator = datagen_msk.flow(Y_train, batch_size=self.batch_size, seed=seed)
        
        while True:
            X_batch = next(image_generator)
            Y_batch = next(mask_generator)
            yield (X_batch, Y_batch)

    def lr_schedule(self, epoch):
        """
        Learning rate scheduler based on the paper's description
        
        Initial rate: 0.005
        Decay: 0.1 at each epoch
        
        Parameters:
        - epoch: Current epoch number
        
        Returns:
        - learning rate for the current epoch
        """
        initial_lr = 0.005
        decay = 0.1
        
        # Calculate learning rate with decay factor for each epoch
        lr = initial_lr * (decay ** epoch)
        
        # Add a minimum threshold to prevent learning rate becoming too small
        lr = max(lr, 1e-6)
        
        print(f'Learning rate for epoch {epoch}: {lr}')
        return lr
    
    def save_model(self, model_name, val_loss=None):
        """
        Save the model to disk
        
        Parameters:
        - model_name: Path where to save the model, without extension
        - val_loss: Optional validation loss to include in filename
        """
        if val_loss is not None:
            save_path = f'{model_name}_{val_loss:.3f}.keras'
            self.model.save(save_path)
            print(f'Model saved to {save_path}')
        else:
            save_path = f'{model_name}.keras'
            self.model.save(save_path)
            print(f'Model saved to {save_path}')


# Custom callback to gradually increase momentum during training
class MomentumScheduler(tf.keras.callbacks.Callback):
    def __init__(self, initial_momentum=0.5, final_momentum=0.9, epochs=50):
        super(MomentumScheduler, self).__init__()
        self.initial_momentum = initial_momentum
        self.final_momentum = final_momentum
        self.epochs = epochs
    
    def on_epoch_begin(self, epoch, logs=None):
        # Calculate momentum value based on current epoch
        # Gradually increase from initial to final value
        progress = epoch / float(self.epochs)
        momentum = self.initial_momentum + progress * (self.final_momentum - self.initial_momentum)
        
        # Ensure momentum is within bounds
        momentum = min(max(momentum, self.initial_momentum), self.final_momentum)
        
        # Update optimizer momentum
        self.model.optimizer.momentum = momentum
        print(f'Momentum for epoch {epoch}: {momentum}')


if __name__ == "__main__":
    import time
    
    print("Loading training data...")
    try:
        X_patches = np.load("x_training.npy").astype(np.float32)
        Y_labels = np.load("y_training.npy").astype(np.float32)
        
        print("Data loaded successfully")
        print("X_patches shape:", X_patches.shape)
        print("Y_labels shape:", Y_labels.shape)
        
        # Split the data into training and validation sets
        X_train, X_valid, Y_train, Y_valid = train_test_split(
            X_patches, Y_labels, test_size=0.2, random_state=42
        )
        
        print("After split:")
        print("Training data shape:", X_train.shape)
        print("Training labels shape:", Y_train.shape)
        print("Validation data shape:", X_valid.shape)
        print("Validation labels shape:", Y_valid.shape)

        # Prepare multiscale data
        # def prepare_multiscale_data(X_patches):
        #     """Prepare multiscale features for the model"""
        #     import cv2
            
        #     # Original scale features
        #     original_features = X_patches
            
        #     # Downsampled features
        #     downsampled_features = np.zeros_like(X_patches)
            
        #     for i in range(len(X_patches)):
        #         for j in range(X_patches.shape[3]):  # For each modality channel
        #             # Downsample by factor of 2
        #             downsampled = cv2.resize(X_patches[i, :, :, j], 
        #                                     (X_patches.shape[2]//2, X_patches.shape[1]//2), 
        #                                     interpolation=cv2.INTER_AREA)
        #             # Upsample back to original size
        #             upsampled = cv2.resize(downsampled, 
        #                                 (X_patches.shape[2], X_patches.shape[1]), 
        #                                 interpolation=cv2.INTER_LINEAR)
        #             downsampled_features[i, :, :, j] = upsampled
            
        #     return np.concatenate([original_features, downsampled_features], axis=3)
        def prepare_multiscale_data(X_patches):
            """Create proper multiscale input according to the paper"""
            import cv2
            
            # Create a new array with double the channels
            multiscale = np.zeros((X_patches.shape[0], X_patches.shape[1], X_patches.shape[2], X_patches.shape[3]*2), 
                                dtype=X_patches.dtype)
            
            # Copy original channels to first 4 channels
            multiscale[:,:,:,:4] = X_patches
            
            # Generate downsampled features for each channel
            for i in range(len(X_patches)):
                for j in range(X_patches.shape[3]):
                    # Downsample by factor of 2
                    downsampled = cv2.resize(X_patches[i, :, :, j], 
                                        (X_patches.shape[2]//2, X_patches.shape[1]//2), 
                                        interpolation=cv2.INTER_AREA)
                    # Upsample back to original size
                    upsampled = cv2.resize(downsampled, 
                                        (X_patches.shape[2], X_patches.shape[1]), 
                                        interpolation=cv2.INTER_LINEAR)
                    # Store in second set of channels
                    multiscale[i, :, :, j+4] = upsampled
            
            return multiscale

        # Apply multiscale processing to your data
        X_train_multiscale = prepare_multiscale_data(X_train)
        X_valid_multiscale = prepare_multiscale_data(X_valid)

        # Update model input shape to accept 8 channels (4 original + 4 downsampled)
        from model import TwoPathwayCNN
        
        # Ensure the TwoPathwayCNN model includes dropout layers as mentioned in the paper
        # You may need to modify the model.py file to include dropout with probability mentioned in the paper
        model_instance = TwoPathwayCNN(img_shape=(128, 128, 8))
        
        # Initialize training with TwoPathwayCNN model
        brain_seg = Training(batch_size=32, nb_epoch=50, load_model_resume_training=None)
        brain_seg.model = model_instance.model
        
        # Train the model
        history = brain_seg.fit_2pg(X_train_multiscale, Y_train, X_valid_multiscale, Y_valid)
        
        # Save the final model with validation loss in filename
        try:
            final_val_loss = history.history['val_loss'][-1]
            brain_seg.save_model('brain_segmentation1/TwoPathwayCNN_final', val_loss=final_val_loss)
        except Exception as e:
            print(f"Warning: Couldn't save model with validation loss: {e}")
            brain_seg.save_model('brain_segmentation1/TwoPathwayCNN_final')
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the training data files exist in the current directory.")
    except Exception as e:
        print(f"An error occurred: {e}")

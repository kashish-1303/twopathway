import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os

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
        Train the model
        
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
        
        lr_scheduler = LearningRateScheduler(self.lr_schedule)
        
        print("Starting model training with TwoPathwayCNN...")
        history = self.model.fit(
            train_generator,
            steps_per_epoch=max(1, len(X_train) // self.batch_size),
            epochs=self.nb_epoch,
            validation_data=(X_valid, Y_valid),
            verbose=1,
            callbacks=[checkpointer, lr_scheduler]
        )
        print("Model training completed.")
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

    # def lr_schedule(self, epoch):
    #     """
    #     Learning rate scheduler
        
    #     Parameters:
    #     - epoch: Current epoch number
        
    #     Returns:
    #     - learning rate for the current epoch
    #     """
    #     lr = 1e-3
    #     if epoch > 180:
    #         lr *= 0.5e-3
    #     elif epoch > 150:
    #         lr *= 1e-3
    #     elif epoch > 120:
    #         lr *= 1e-2
    #     elif epoch > 80:
    #         lr *= 1e-1
    #     print('Learning rate:', lr)
    #     return lr

    # Update lr_schedule method in Training class
    def lr_schedule(self, epoch):
        """
        Learning rate scheduler based on the paper's recommendations
        """
        initial_lr = 1e-3
        if epoch > 150:
            lr = initial_lr * 0.01
        elif epoch > 100:
            lr = initial_lr * 0.1
        else:
            lr = initial_lr
        print('Learning rate:', lr)
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

if __name__ == "__main__":
    print("Loading training data...")
    try:
        X_patches = np.load("x_training.npy").astype(np.float32)
        Y_labels = np.load("y_training.npy").astype(np.float32)
        
        # If you want to use a subset for testing
        # N = 16
        # X_patches = X_patches[:N]
        # Y_labels = Y_labels[:N]
        
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
        
        # Initialize training with TwoPathwayCNN model
        brain_seg = Training(batch_size=8, nb_epoch=2, load_model_resume_training=None)
        
        # Train the model
        history = brain_seg.fit_2pg(X_train, Y_train, X_valid, Y_valid)
        
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

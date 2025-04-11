
# import numpy as np
# import random
# import json
# from glob import glob
# from keras.models import model_from_json,load_model
# from keras.preprocessing.image import ImageDataGenerator
# from keras.callbacks import  ModelCheckpoint,Callback,LearningRateScheduler
# import keras.backend as K
# from model import Unet_model
# from losses import *
# #from keras.utils.visualize_util import plot



# class SGDLearningRateTracker(Callback):
#     def on_epoch_begin(self, epoch, logs={}):
#         optimizer = self.model.optimizer
#         lr = K.get_value(optimizer.lr)
#         decay = K.get_value(optimizer.decay)
#         lr=lr/10
#         decay=decay*10
#         K.set_value(optimizer.lr, lr)
#         K.set_value(optimizer.decay, decay)
#         print('LR changed to:',lr)
#         print('Decay changed to:',decay)



# class Training(object):
    

#     def __init__(self, batch_size,nb_epoch,load_model_resume_training=None):

#         self.batch_size = batch_size
#         self.nb_epoch = nb_epoch

#         #loading model from path to resume previous training without recompiling the whole model
#         if load_model_resume_training is not None:
#             self.model =load_model(load_model_resume_training,custom_objects={'gen_dice_loss': gen_dice_loss,'dice_whole_metric':dice_whole_metric,'dice_core_metric':dice_core_metric,'dice_en_metric':dice_en_metric})
#             print("pre-trained model loaded!")
#         else:
#             unet =Unet_model(img_shape=(128,128,4))
#             self.model=unet.model
#             print("U-net CNN compiled!")

                    
#     def fit_unet(self,X33_train,Y_train,X_patches_valid=None,Y_labels_valid=None):

#         train_generator=self.img_msk_gen(X33_train,Y_train,9999)
#         checkpointer = ModelCheckpoint(filepath='brain_segmentation/ResUnet.{epoch:02d}_{val_loss:.3f}.hdf5', verbose=1)
#         self.model.fit_generator(train_generator,steps_per_epoch=len(X33_train)//self.batch_size,epochs=self.nb_epoch, validation_data=(X_patches_valid,Y_labels_valid),verbose=1, callbacks = [checkpointer,SGDLearningRateTracker()])
#         #self.model.fit(X33_train,Y_train, epochs=self.nb_epoch,batch_size=self.batch_size,validation_data=(X_patches_valid,Y_labels_valid),verbose=1, callbacks = [checkpointer,SGDLearningRateTracker()])

#     def img_msk_gen(self,X33_train,Y_train,seed):

#         '''
#         a custom generator that performs data augmentation on both patches and their corresponding targets (masks)
#         '''
#         datagen = ImageDataGenerator(horizontal_flip=True,data_format="channels_last")
#         datagen_msk = ImageDataGenerator(horizontal_flip=True,data_format="channels_last")
#         image_generator = datagen.flow(X33_train,batch_size=4,seed=seed)
#         y_generator = datagen_msk.flow(Y_train,batch_size=4,seed=seed)
#         while True:
#             yield(image_generator.next(), y_generator.next())


#     def save_model(self, model_name):
#         '''
#         INPUT string 'model_name': path where to save model and weights, without extension
#         Saves current model as json and weights as h5df file
#         '''

#         model_tosave = '{}.json'.format(model_name)
#         weights = '{}.hdf5'.format(model_name)
#         json_string = self.model.to_json()
#         self.model.save_weights(weights)
#         with open(model_tosave, 'w') as f:
#             json.dump(json_string, f)
#         print ('Model saved.')

#     def load_model(self, model_name):
#         '''
#         Load a model
#         INPUT  (1) string 'model_name': filepath to model and weights, not including extension
#         OUTPUT: Model with loaded weights. can fit on model using loaded_model=True in fit_model method
#         '''
#         print ('Loading model {}'.format(model_name))
#         model_toload = '{}.json'.format(model_name)
#         weights = '{}.hdf5'.format(model_name)
#         with open(model_toload) as f:
#             m = next(f)
#         model_comp = model_from_json(json.loads(m))
#         model_comp.load_weights(weights)
#         print ('Model loaded.')
#         self.model = model_comp
#         return model_comp



# if __name__ == "__main__":
#     #set arguments

#     #reload already trained model to resume training
#     #model_to_load="./pretrained_weights/ResUnet.epoch_02.hdf5" 
#     #save=None

#     #compile the model
#     brain_seg = Training(batch_size=4,nb_epoch=3,load_model_resume_training=None)

#     print("number of trainabale parameters:",brain_seg.model.count_params())
#     #print(brain_seg.model.summary())
#     #plot(brain_seg.model, to_file='model_architecture.png', show_shapes=True)

#     #load data from disk
#     Y_labels=np.load("y_training.npy").astype(np.uint8)
#     X_patches=np.load("x_training.npy").astype(np.float32)
#     Y_labels_valid=np.load("y_valid.npy").astype(np.uint8)
#     X_patches_valid=np.load("x_valid.npy").astype(np.float32)
#     print("loading patches done\n")

#     #print(X_patches.shape,Y_labels.shape,Y_labels_valid.shape,X_patches_valid.shape)

#     # fit model
#     brain_seg.fit_unet(X_patches,Y_labels,X_patches_valid,Y_labels_valid)#*

#     #brain_seg.save_model('models/' + model_1)




# # #final code
# # import numpy as np
# # import random
# # import json
# # import os
# # from glob import glob
# # from tensorflow.keras.models import load_model
# # from tensorflow.keras.preprocessing.image import ImageDataGenerator
# # from tensorflow.keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
# # import tensorflow.keras.backend as K
# # from tensorflow.keras.optimizers import SGD
# # from tensorflow.keras.optimizers.schedules import ExponentialDecay
# # from model import Unet_model
# # from losses import *
# # from sklearn.model_selection import train_test_split

# # # Create directory for saving models if it doesn't exist
# # if not os.path.exists('brain_segmentation'):
# #     os.makedirs('brain_segmentation')

# # class SGDLearningRateTracker(Callback):
# #     def on_epoch_begin(self, epoch, logs=None):
# #         optimizer = self.model.optimizer
# #         lr = optimizer.learning_rate.numpy()
# #         print('Current LR:', lr)

# # class Training(object):
# #     def __init__(self, batch_size, nb_epoch, load_model_resume_training=None):
# #         self.batch_size = batch_size
# #         self.nb_epoch = nb_epoch

# #         if load_model_resume_training is not None:
# #             self.model = self.load_model(load_model_resume_training)
# #             print("pre-trained model loaded!")
# #         else:
# #             unet = Unet_model(img_shape=(128, 128, 4))
# #             self.model = unet.model
            
# #             # Define learning rate schedule
# #             initial_learning_rate = 0.01
# #             lr_schedule = ExponentialDecay(
# #                 initial_learning_rate,
# #                 decay_steps=100000,
# #                 decay_rate=0.96,
# #                 staircase=True)
            
# #             # Compile the model with the learning rate schedule
# #             self.model.compile(optimizer=SGD(learning_rate=lr_schedule),
# #                                loss=gen_dice_loss,
# #                                metrics=[dice_whole_metric, dice_core_metric, dice_en_metric])
            
# #             print("U-net CNN compiled!")

# #     def fit_unet(self, X_train, Y_train, X_valid, Y_valid):
# #         train_generator = self.img_msk_gen(X_train, Y_train, 9999)
# #         checkpointer = ModelCheckpoint(
# #             filepath='brain_segmentation/ResUnet.{epoch:02d}_{val_loss:.3f}.keras', 
# #             verbose=1,
# #             save_best_only=True
# #         )
# #         self.model.fit(
# #             train_generator,
# #             steps_per_epoch=len(X_train) // self.batch_size,
# #             epochs=self.nb_epoch,
# #             validation_data=(X_valid, Y_valid),
# #             verbose=1,
# #             callbacks=[checkpointer, SGDLearningRateTracker()]
# #         )

# #     def img_msk_gen(self, X_train, Y_train, seed):
# #         datagen = ImageDataGenerator(horizontal_flip=True, data_format="channels_last")
# #         datagen_msk = ImageDataGenerator(horizontal_flip=True, data_format="channels_last")
# #         image_generator = datagen.flow(X_train, batch_size=self.batch_size, seed=seed)
# #         y_generator = datagen_msk.flow(Y_train, batch_size=self.batch_size, seed=seed)
# #         while True:
# #             X_batch = next(image_generator)
# #             Y_batch = next(y_generator)
# #             yield X_batch, Y_batch

# #     def save_model(self, model_name):
# #         self.model.save(f'{model_name}.keras')
# #         print('Model saved.')

# #     def load_model(self, model_name):
# #         print(f'Loading model {model_name}')
# #         model = load_model(model_name, custom_objects={
# #             'gen_dice_loss': gen_dice_loss,
# #             'dice_whole_metric': dice_whole_metric,
# #             'dice_core_metric': dice_core_metric,
# #             'dice_en_metric': dice_en_metric
# #         })
# #         print('Model loaded.')
# #         return model

# # if __name__ == "__main__":
# #     # Set arguments
# #     brain_seg = Training(batch_size=4, nb_epoch=1, load_model_resume_training=None)

# #     print("number of trainable parameters:", brain_seg.model.count_params())

# #     # Load data from disk
# #     X_patches = np.load("x_training.npy").astype(np.float32)
# #     Y_labels = np.load("y_training.npy").astype(np.uint8)
# #     print("loading patches done\n")

# #     # Split data into training and validation sets
# #     X_train, X_valid, Y_train, Y_valid = train_test_split(
# #         X_patches, Y_labels, test_size=0.2, random_state=42
# #     )

# #     print("Training data shape:", X_train.shape)
# #     print("Training labels shape:", Y_train.shape)
# #     print("Validation data shape:", X_valid.shape)
# #     print("Validation labels shape:", Y_valid.shape)

# #     # Fit model
# #     brain_seg.fit_unet(X_train, Y_train, X_valid, Y_valid)

# # import numpy as np
# # from model import TwoPathwayGroupCNN
# # from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
# # from tensorflow.keras.optimizers import Adam
# # from tensorflow.keras.preprocessing.image import ImageDataGenerator

# # class Training(object):
# #     def __init__(self, batch_size, nb_epoch, load_model_resume_training=None):
# #         self.batch_size = batch_size
# #         self.nb_epoch = nb_epoch
# #         self.load_model_resume_training = load_model_resume_training
# #         self.model = self.get_model()

# #     def get_model(self):
# #         model = TwoPathwayGroupCNN(img_shape=(128, 128, 4), 
# #                                    load_model_weights=self.load_model_resume_training)
# #         return model.model

# #     def fit_unet(self, X_train, Y_train, X_valid, Y_valid):
# #         train_generator = self.img_msk_gen(X_train, Y_train, 9999)
# #         checkpointer = ModelCheckpoint(
# #             filepath='brain_segmentation/TwoPathwayGroupCNN.{epoch:02d}_{val_loss:.3f}.keras', 
# #             verbose=1,
# #             save_best_only=True
# #         )
        
# #         lr_scheduler = LearningRateScheduler(self.lr_schedule)
        
# #         self.model.fit(
# #             train_generator,
# #             steps_per_epoch=len(X_train) // self.batch_size,
# #             epochs=self.nb_epoch,
# #             validation_data=(X_valid, Y_valid),
# #             verbose=1,
# #             callbacks=[checkpointer, lr_scheduler]
# #         )

# #     def img_msk_gen(self, X_train, Y_train, seed):
# #         datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
# #         datagen_msk = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
# #         image_generator = datagen.flow(X_train, batch_size=self.batch_size, seed=seed)
# #         mask_generator = datagen_msk.flow(Y_train, batch_size=self.batch_size, seed=seed)
# #         while True:
# #             yield (next(image_generator), next(mask_generator))

# #     def lr_schedule(self, epoch):
# #         lr = 1e-3
# #         if epoch > 180:
# #             lr *= 0.5e-3
# #         elif epoch > 150:
# #             lr *= 1e-3
# #         elif epoch > 120:
# #             lr *= 1e-2
# #         elif epoch > 80:
# #             lr *= 1e-1
# #         print('Learning rate: ', lr)
# #         return lr

# # # Main execution
# # if __name__ == "__main__":
# #     X_patches = np.load("x_training.npy").astype(np.float32)
# #     Y_labels = np.load("y_training.npy").astype(np.uint8)
    
# #     from sklearn.model_selection import train_test_split
# #     X_train, X_valid, Y_train, Y_valid = train_test_split(X_patches, Y_labels, test_size=0.2, random_state=42)
# #     print("X_train shape:", X_train.shape)
# #     print("Y_train shape:", Y_train.shape)
    
# #     brain_seg = Training(batch_size=32, nb_epoch=200)
# #     brain_seg.fit_unet(X_train, Y_train, X_valid, Y_valid)

# import numpy as np
# from model import TwoPathwayGroupCNN
# from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from sklearn.model_selection import train_test_split

# class Training(object):
#     def __init__(self, batch_size, nb_epoch, load_model_resume_training=None):
#         self.batch_size = batch_size
#         self.nb_epoch = nb_epoch
#         self.load_model_resume_training = load_model_resume_training
#         self.model = self.get_model()

#     def get_model(self):
#         model = TwoPathwayGroupCNN(img_shape=(128, 128, 4), 
#         load_model_weights=self.load_model_resume_training)
#         return model.model

#     def fit_unet(self, X_train, Y_train, X_valid, Y_valid):
#         print("Preparing data generator...")
#         train_generator = self.img_msk_gen(X_train, Y_train, 9999)
#         print("Setting up callbacks...")
#         checkpointer = ModelCheckpoint(
#             filepath='brain_segmentation/TwoPathwayGroupCNN.{epoch:02d}_{val_loss:.3f}.keras', 
#             verbose=1,
#             save_best_only=True
#         )
        
#         lr_scheduler = LearningRateScheduler(self.lr_schedule)
#         print("Starting model fit...")
#         print("Starting model training...")
#         self.model.fit(
#             train_generator,
#             steps_per_epoch=len(X_train) // self.batch_size,
#             epochs=self.nb_epoch,
#             validation_data=(X_valid, Y_valid),
#             verbose=1,
#             callbacks=[checkpointer, lr_scheduler]
#         )
#         print("Model training completed.")

#     def img_msk_gen(self, X_train, Y_train, seed):
#         datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
#         datagen_msk = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
#         image_generator = datagen.flow(X_train, batch_size=self.batch_size, seed=seed)
#         mask_generator = datagen_msk.flow(Y_train, batch_size=self.batch_size, seed=seed)
#         while True:
#             yield (next(image_generator), next(mask_generator))

#     def lr_schedule(self, epoch):
#         lr = 1e-3
#         if epoch > 180:
#             lr *= 0.5e-3
#         elif epoch > 150:
#             lr *= 1e-3
#         elif epoch > 120:
#             lr *= 1e-2
#         elif epoch > 80:
#             lr *= 1e-1
#         print('Learning rate: ', lr)
#         return lr

# if __name__ == "__main__":
    
#     X_patches = np.load("x_training.npy").astype(np.float32)
#     Y_labels = np.load("y_training.npy").astype(np.float32)
#     N= 16
#     X_patches=X_patches[:N]
#     Y_labels=Y_labels[:N]
    
#     print("X_train shape:", X_patches.shape)
#     print("Y_train shape:", Y_labels.shape)
    
#     X_train, X_valid, Y_train, Y_valid = train_test_split(X_patches, Y_labels, test_size=0.2, random_state=42)

#     print("after split")
#     print("x",X_train.shape)
#     print("x",Y_train.shape)
#     print("x",Y_valid.shape)
#     print("x",X_valid.shape)
    
#     brain_seg = Training(batch_size=2, nb_epoch=1)


import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os

# Create directory for saving models if it doesn't exist
if not os.path.exists('brain_segmentation'):
    os.makedirs('brain_segmentation')

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
        from model import TwoPathwayGroupCNN
        
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
            # Create a new TwoPathwayGroupCNN model
            model_instance = TwoPathwayGroupCNN(img_shape=(128, 128, 4))
            self.model = model_instance.model
            print("TwoPathwayGroupCNN model initialized")
            
        print("Number of trainable parameters:", self.model.count_params())

    def fit_unet(self, X_train, Y_train, X_valid, Y_valid):
        """
        Train the model
        
        Parameters:
        - X_train: Training input data
        - Y_train: Training target data
        - X_valid: Validation input data
        - Y_valid: Validation target data
        """
        print("Preparing data generator...")
        train_generator = self.img_msk_gen(X_train, Y_train, seed=42)
        
        print("Setting up callbacks...")
        checkpointer = ModelCheckpoint(
            filepath='brain_segmentation/TwoPathwayGroupCNN.{epoch:02d}_{val_loss:.3f}.keras', 
            verbose=1,
            save_best_only=True
        )
        
        lr_scheduler = LearningRateScheduler(self.lr_schedule)
        
        print("Starting model training with TwoPathwayGroupCNN...")
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

    def lr_schedule(self, epoch):
        """
        Learning rate scheduler
        
        Parameters:
        - epoch: Current epoch number
        
        Returns:
        - learning rate for the current epoch
        """
        lr = 1e-3
        if epoch > 180:
            lr *= 0.5e-3
        elif epoch > 150:
            lr *= 1e-3
        elif epoch > 120:
            lr *= 1e-2
        elif epoch > 80:
            lr *= 1e-1
        print('Learning rate:', lr)
        return lr
    
    def save_model(self, model_name):
        """
        Save the model to disk
        
        Parameters:
        - model_name: Path where to save the model, without extension
        """
        self.model.save(f'{model_name}.keras')
        print(f'Model saved to {model_name}.keras')


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
        
        # Initialize training with TwoPathwayGroupCNN model
        brain_seg = Training(batch_size=8, nb_epoch=50, load_model_resume_training=None)
        
        # Train the model
        brain_seg.fit_unet(X_train, Y_train, X_valid, Y_valid)
        
        # Save the final model
        brain_seg.save_model('brain_segmentation/TwoPathwayGroupCNN_final')
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the training data files exist in the current directory.")
    except Exception as e:
        print(f"An error occurred: {e}")
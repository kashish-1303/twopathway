

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, 
    Activation, Conv2DTranspose, UpSampling2D, concatenate, add, PReLU, GaussianNoise
)
from tensorflow.keras.optimizers import SGD, Adam
from losses import gen_dice_loss, dice_whole_metric, dice_core_metric, dice_en_metric

K.set_image_data_format("channels_last")



class TwoPathwayGroupCNN(object):
    def __init__(self, img_shape=(128, 128, 4), load_model_weights=None):
        self.img_shape = img_shape
        self.load_model_weights = load_model_weights
        self.model = self.build_model()

    def build_model(self):
        input_layer = Input(shape=self.img_shape)
        
        # Local pathway
        local_path = self.local_pathway(input_layer)
        
        # Global pathway
        global_path = self.global_pathway(input_layer)
        
        # Concatenate local and global pathways
        concatenated = concatenate([local_path, global_path])
        
        # Final convolutions
        output = self.final_convolutions(concatenated)
        
        model = Model(inputs=input_layer, outputs=output)
        
        model.compile(loss=gen_dice_loss, optimizer=Adam(learning_rate=0.001),
                    metrics=[dice_whole_metric, dice_core_metric, dice_en_metric])
        
        if self.load_model_weights:
            model.load_weights(self.load_model_weights)
        
        return model

    def local_pathway(self, input_layer):
        x = self.group_conv_block(input_layer, 64, 5)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = self.group_conv_block(x, 64, 5)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = self.group_conv_block(x, 64, 5)
        return x

    def global_pathway(self, input_layer):
        x = self.group_conv_block(input_layer, 160, 13)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = self.group_conv_block(x, 160, 13)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = self.group_conv_block(x, 160, 13)
        return x

    def final_convolutions(self, input_layer):
        x = self.group_conv_block(input_layer, 64, 21)
        x = UpSampling2D(size=(2, 2))(x)
        x = self.group_conv_block(x, 64, 21)
        x = UpSampling2D(size=(2, 2))(x)
        x = self.group_conv_block(x, 64, 21)
        x = Conv2D(4, 1, activation='softmax', padding='same')(x)
        return x

    def group_conv_block(self, input_layer, filters, kernel_size):
        x = Conv2D(filters, kernel_size, padding='same')(input_layer)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, BatchNormalization, 
    Activation, concatenate, UpSampling2D, Dropout,
    AveragePooling2D, Lambda, Reshape, Add
)
from tensorflow.keras.optimizers import Adam
import numpy as np

# Import losses
from losses import gen_dice_loss, dice_whole_metric, dice_core_metric, dice_en_metric

class TwoPathwayCNN:
    def __init__(self, img_shape=(128, 128, 4), load_model_weights=None):
        self.img_shape = img_shape
        self.load_model_weights = load_model_weights
        self.model = self.build_model()
        
    def p4m_group_conv(self, x, filters, kernel_size, padding='same'):
        """
        Implementation of p4m group convolution (translations, rotations, and reflections)
        """
        # Define all transformations in the p4m group
        transformations = [
            # (m=0, r=0): Identity
            lambda x: x,
            
            # (m=0, r=1,2,3): Rotations by 90°, 180°, 270°
            lambda x: tf.image.rot90(x, k=1),
            lambda x: tf.image.rot90(x, k=2),
            lambda x: tf.image.rot90(x, k=3),
            
            # (m=1, r=0): Reflection (horizontal flip)
            lambda x: tf.image.flip_left_right(x),
            
            # (m=1, r=1): Reflection + 90° rotation
            lambda x: tf.image.flip_left_right(tf.image.rot90(x, k=1)),
            
            # (m=1, r=2): Reflection + 180° rotation (vertical flip)
            lambda x: tf.image.flip_up_down(x),
            
            # (m=1, r=3): Reflection + 270° rotation
            lambda x: tf.image.flip_left_right(tf.image.rot90(x, k=3))
        ]
        
        # Define inverse transformations
        inverse_transformations = [
            # Identity inverse
            lambda x: x,
            
            # Rotation inverses
            lambda x: tf.image.rot90(x, k=3),
            lambda x: tf.image.rot90(x, k=2),
            lambda x: tf.image.rot90(x, k=1),
            
            # Reflection inverse
            lambda x: tf.image.flip_left_right(x),
            
            # Reflection + rotation inverses
            lambda x: tf.image.rot90(tf.image.flip_left_right(x), k=3),
            lambda x: tf.image.flip_up_down(x),
            lambda x: tf.image.rot90(tf.image.flip_left_right(x), k=1)
        ]
        
        # Apply each transformation, convolve, and apply inverse
        outputs = []
        
        # Process each transformation
        for i, (transform, inv_transform) in enumerate(zip(transformations, inverse_transformations)):
            # Apply transformation
            transformed = Lambda(
                lambda x, transform=transform: transform(x),
                output_shape=K.int_shape(x)[1:]
            )(x)
            
            # Apply convolution
            conv = Conv2D(
                filters, 
                kernel_size, 
                padding=padding,
                name=f'group_conv_{i}'
            )(transformed)
            
            # Apply inverse transformation
            inv_transformed = Lambda(
                lambda x, inv_transform=inv_transform: inv_transform(x),
                output_shape=(K.int_shape(x)[1], K.int_shape(x)[2], filters)
            )(conv)
            
            outputs.append(inv_transformed)
        
        # Combine all outputs
        return Add()(outputs)
        
    def group_pooling(self, x):
        """
        Group pooling: takes max across orientations
        """
        return Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(x)
    
    def local_pathway(self, x):
        """
        Local pathway with 7x7 receptive field
        """
        # Use p4m group convolution
        x = self.p4m_group_conv(x, 64, 7)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)
        
        # Second convolution block
        x = self.p4m_group_conv(x, 128, 5)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)
        
        # Third convolution block
        x = self.p4m_group_conv(x, 256, 3)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.3)(x)
        
        return x
    
    def global_pathway(self, x):
        """
        Global pathway with 13x13 receptive field
        """
        # Global pathway with larger receptive field
        x = self.p4m_group_conv(x, 64, 13)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        # Second global convolution
        x = self.p4m_group_conv(x, 128, 9)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.3)(x)
        
        return x
    
    def build_model(self):
        input_layer = Input(shape=self.img_shape)
        
        # Build local pathway
        local_features = self.local_pathway(input_layer)
        
        # Build global pathway
        global_features = self.global_pathway(input_layer)
        
        # Concatenate local and global pathways
        merged = concatenate([local_features, global_features])
        
        # Final classification layers
        x = self.p4m_group_conv(merged, 128, 5)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        # Additional convolutional layer
        x = self.p4m_group_conv(x, 64, 3)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.3)(x)
        
        # Final 1x1 convolution for classification
        x = Conv2D(4, 1, activation='softmax', padding='same')(x)
        
        model = Model(inputs=input_layer, outputs=x)
        
        # Use Adam optimizer with proper learning rate
        model.compile(
            loss=gen_dice_loss, 
            optimizer=Adam(learning_rate=0.005),  # As per paper
            metrics=[dice_whole_metric, dice_core_metric, dice_en_metric]
        )
        
        if self.load_model_weights:
            model.load_weights(self.load_model_weights)
        
        return model
    
    def get_config(self):
        return {
            "img_shape": self.img_shape,
            "load_model_weights": self.load_model_weights
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
            
    def summary(self):
        return self.model.summary()

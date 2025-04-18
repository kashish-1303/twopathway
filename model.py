import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, BatchNormalization, 
    Activation, concatenate, UpSampling2D, Dropout,
    AveragePooling2D, Lambda
)
from tensorflow.keras.optimizers import Adam
import numpy as np

# Assume these are defined elsewhere in your code
from losses import gen_dice_loss, dice_whole_metric, dice_core_metric, dice_en_metric

class TwoPathwayCNN:
    def __init__(self, img_shape=(128, 128, 8), load_model_weights=None):
        self.img_shape = img_shape
        self.load_model_weights = load_model_weights
        self.model = self.build_model()
        

    def group_conv(self, x, filters, kernel_size, groups=32, padding='same'):
        """
        More efficient implementation of group convolution
        """
        # Check if groups parameter makes sense for the input
        input_channels = K.int_shape(x)[-1]
        if input_channels < groups:
            groups = max(1, input_channels)
        
        if groups == 1:
            return Conv2D(filters, kernel_size, padding=padding)(x)
        
        # Ensure filters is divisible by groups
        filters_per_group = filters // groups
        
        # Process each group
        group_outputs = []
        for i in range(groups):
            # Extract the group's channels using slice
            start_channel = i * (input_channels // groups)
            end_channel = (i + 1) * (input_channels // groups) if i < groups - 1 else input_channels
            
            # Extract input slice
            group_input = Lambda(lambda x: x[:, :, :, start_channel:end_channel])(x)
            
            # Apply convolution to this group
            group_output = Conv2D(filters_per_group, kernel_size, padding=padding)(group_input)
            group_outputs.append(group_output)
        
        # Concatenate all group outputs
        return concatenate(group_outputs) if len(group_outputs) > 1 else group_outputs[0]
    
    def maxout(self, inputs, num_units=2):
        """
        Implementation of maxout activation as described in the paper.
        """
        shape = inputs.get_shape().as_list()
        # Reshape for maxout: (batch_size, h, w, num_features // num_units, num_units)
        out_shape = (-1, shape[1], shape[2], shape[3] // num_units, num_units)
        reshaped = tf.reshape(inputs, out_shape)
        # Perform maxout operation
        outputs = tf.reduce_max(reshaped, axis=4)
        return outputs
    
    def local_pathway(self, x):
        """
        Local pathway implementation with decreasing kernel sizes
        """
        # First local convolution block: 7x7 kernel, maxout, 4x4 pooling
        x = self.group_conv(x, 64*2, 7)  # *2 for maxout
        x = Lambda(lambda x: self.maxout(x))(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(4, 4))(x)
        
        # Second local convolution block: 5x5 kernel, maxout, 2x2 pooling
        x = self.group_conv(x, 64*2, 5)
        x = Lambda(lambda x: self.maxout(x))(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        # Third local convolution block: 3x3 kernel, maxout, no pooling
        x = self.group_conv(x, 64*2, 3)
        x = Lambda(lambda x: self.maxout(x))(x)
        x = BatchNormalization()(x)
        
        return x
    
    def global_pathway(self, x):
        """
        Global pathway implementation with consistent large kernel
        """
        # Global convolution block: 13x13 kernel, maxout, no pooling
        x = self.group_conv(x, 160*2, 13)
        x = Lambda(lambda x: self.maxout(x))(x)
        x = BatchNormalization()(x)
        
        return x
    
    # Add to Pipeline class in paste-3.txt
    def extract_multiscale_features(self, patches):
        """
        Extract features at multiple scales as described in the paper
        """
        # Original scale features
        original_features = patches
        
        # Downsampled features (for capturing larger context)
        downsampled_features = []
        for patch in patches:
            modalities = []
            for i in range(patch.shape[0]):
                # Downsample by factor of 2
                downsampled = cv2.resize(patch[i], 
                                        (patch.shape[2]//2, patch.shape[1]//2), 
                                        interpolation=cv2.INTER_AREA)
                # Upsample back to original size to maintain dimensions
                upsampled = cv2.resize(downsampled, 
                                    (patch.shape[2], patch.shape[1]), 
                                    interpolation=cv2.INTER_LINEAR)
                modalities.append(upsampled)
            downsampled_features.append(np.stack(modalities))
        
        # Combine multi-scale features
        combined_features = np.concatenate([
            original_features, 
            np.array(downsampled_features)
        ], axis=1)  # Concatenate along modality dimension
        
        return combined_features
    
    def build_model(self):
        input_layer = Input(shape=self.img_shape)
        
        # Build local pathway
        local_features = self.local_pathway(input_layer)
        
        # Build global pathway
        global_features = self.global_pathway(input_layer)
        
        # Resize global features to match local features dimensions
        local_shape = K.int_shape(local_features)
        h, w = local_shape[1], local_shape[2]
        
        # Resize global_path to match local_path dimensions
        global_features_resized = tf.keras.layers.Resizing(h, w)(global_features)
        
        # Concatenate local and global pathways
        merged = concatenate([local_features, global_features_resized])
        
        # Final processing with 21x21 convolutions
        x = self.group_conv(merged, 160*2, 21)
        x = Lambda(lambda x: self.maxout(x))(x)
        x = BatchNormalization()(x)
        
        x = self.group_conv(x, 160*2, 21)
        x = Lambda(lambda x: self.maxout(x))(x)
        x = BatchNormalization()(x)
        
        x = self.group_conv(x, 160*2, 21)
        x = Lambda(lambda x: self.maxout(x))(x)
        x = BatchNormalization()(x)
        
        # Upsample back to original image dimensions
        x = tf.keras.layers.Resizing(self.img_shape[0], self.img_shape[1])(x)
        
        # Final classification layer
        output = Conv2D(4, 1, activation='softmax', padding='same')(x)
        
        model = Model(inputs=input_layer, outputs=output)
        
        # Use a lower learning rate to prevent NaN issues
        model.compile(
            loss=gen_dice_loss, 
            optimizer=Adam(learning_rate=0.0001),  # Reduced learning rate
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

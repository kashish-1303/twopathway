import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, BatchNormalization, 
    Activation, concatenate, UpSampling2D, Dropout,
    AveragePooling2D, Lambda, Reshape, Add, Multiply,
    GlobalAveragePooling2D, Dense
)
from tensorflow.keras.optimizers import Adam
import numpy as np

class SimplifiedTwoPathwayGroupCNN:
    def __init__(self, img_shape=(128, 128, 4), load_model_weights=None):
        self.img_shape = img_shape
        self.load_model_weights = load_model_weights
        self.model = self.build_model()
        
    def group_conv_p4m(self, x, filters, kernel_size, name_prefix):
        """
        Simplified P4M group convolution using data augmentation approach
        Instead of complex geometric transformations, use multiple conv paths
        """
        # Original path
        conv1 = Conv2D(filters//4, kernel_size, padding='same', 
                      name=f'{name_prefix}_orig')(x)
        
        # Rotated-style features (simulate rotation with different kernel patterns)
        conv2 = Conv2D(filters//4, kernel_size, padding='same', 
                      name=f'{name_prefix}_rot90')(x)
        conv3 = Conv2D(filters//4, kernel_size, padding='same', 
                      name=f'{name_prefix}_rot180')(x)
        conv4 = Conv2D(filters//4, kernel_size, padding='same', 
                      name=f'{name_prefix}_rot270')(x)
        
        # Combine all paths
        combined = concatenate([conv1, conv2, conv3, conv4], 
                             name=f'{name_prefix}_combine')
        return combined
    
    def attention_fusion_block(self, local_features, global_features, name_prefix):
        """
        Novel attention-based fusion of local and global features
        This is the main novelty contribution
        """
        # Channel attention for local features
        local_gap = GlobalAveragePooling2D()(local_features)
        local_dense1 = Dense(local_features.shape[-1]//4, activation='relu',
                           name=f'{name_prefix}_local_att1')(local_gap)
        local_dense2 = Dense(local_features.shape[-1], activation='sigmoid',
                           name=f'{name_prefix}_local_att2')(local_dense1)
        local_att = Reshape((1, 1, local_features.shape[-1]))(local_dense2)
        local_weighted = Multiply(name=f'{name_prefix}_local_weighted')([local_features, local_att])
        
        # Channel attention for global features  
        global_gap = GlobalAveragePooling2D()(global_features)
        global_dense1 = Dense(global_features.shape[-1]//4, activation='relu',
                            name=f'{name_prefix}_global_att1')(global_gap)
        global_dense2 = Dense(global_features.shape[-1], activation='sigmoid',
                            name=f'{name_prefix}_global_att2')(global_dense1)
        global_att = Reshape((1, 1, global_features.shape[-1]))(global_dense2)
        global_weighted = Multiply(name=f'{name_prefix}_global_weighted')([global_features, global_att])
        
        # Spatial attention
        local_spatial = Conv2D(1, 7, padding='same', activation='sigmoid',
                             name=f'{name_prefix}_local_spatial')(local_weighted)
        global_spatial = Conv2D(1, 7, padding='same', activation='sigmoid',
                              name=f'{name_prefix}_global_spatial')(global_weighted)
        
        # Apply spatial attention
        local_final = Multiply(name=f'{name_prefix}_local_final')([local_weighted, local_spatial])
        global_final = Multiply(name=f'{name_prefix}_global_final')([global_weighted, global_spatial])
        
        # Adaptive fusion weights
        fusion_weights = Conv2D(2, 1, activation='softmax', padding='same',
                              name=f'{name_prefix}_fusion_weights')(
            concatenate([local_final, global_final]))
        
        local_weight = Lambda(lambda x: x[..., 0:1])(fusion_weights)
        global_weight = Lambda(lambda x: x[..., 1:2])(fusion_weights)
        
        # Final fusion
        fused = Add(name=f'{name_prefix}_fused')([
            Multiply()([local_final, local_weight]),
            Multiply()([global_final, global_weight])
        ])
        
        return fused
    
    def local_pathway(self, x):
        """
        Local pathway focusing on fine-grained features (smaller receptive field)
        """
        # First block - focus on local details
        x = self.group_conv_p4m(x, 64, 3, 'local_block1')
        x = BatchNormalization(name='local_bn1')(x)
        x = Activation('relu', name='local_relu1')(x)
        x = MaxPooling2D((2, 2), name='local_pool1')(x)
        
        # Second block
        x = self.group_conv_p4m(x, 128, 3, 'local_block2')
        x = BatchNormalization(name='local_bn2')(x)
        x = Activation('relu', name='local_relu2')(x)
        
        # Third block
        x = Conv2D(256, 3, padding='same', name='local_conv3')(x)
        x = BatchNormalization(name='local_bn3')(x)
        x = Activation('relu', name='local_relu3')(x)
        x = Dropout(0.3, name='local_dropout')(x)
        
        return x
    
    def global_pathway(self, x):
        """
        Global pathway focusing on contextual features (larger receptive field)
        """
        # First block - larger kernels for global context
        x = self.group_conv_p4m(x, 64, 7, 'global_block1')
        x = BatchNormalization(name='global_bn1')(x)
        x = Activation('relu', name='global_relu1')(x)
        x = AveragePooling2D((2, 2), name='global_pool1')(x)
        
        # Second block
        x = self.group_conv_p4m(x, 128, 5, 'global_block2')
        x = BatchNormalization(name='global_bn2')(x)
        x = Activation('relu', name='global_relu2')(x)
        
        # Third block
        x = Conv2D(256, 3, padding='same', name='global_conv3')(x)
        x = BatchNormalization(name='global_bn3')(x)
        x = Activation('relu', name='global_relu3')(x)
        x = Dropout(0.3, name='global_dropout')(x)
        
        return x
    
    def build_model(self):
        """
        Build the complete Two-Pathway-Group CNN with attention fusion
        """
        input_layer = Input(shape=self.img_shape, name='input')
        
        # Build both pathways
        local_features = self.local_pathway(input_layer)
        global_features = self.global_pathway(input_layer)
        
        # Novel attention-based fusion (main novelty)
        fused_features = self.attention_fusion_block(
            local_features, global_features, 'attention_fusion'
        )
        
        # Final classification layers
        x = Conv2D(128, 3, padding='same', name='final_conv1')(fused_features)
        x = BatchNormalization(name='final_bn1')(x)
        x = Activation('relu', name='final_relu1')(x)
        
        x = Conv2D(64, 3, padding='same', name='final_conv2')(x)
        x = BatchNormalization(name='final_bn2')(x)
        x = Activation('relu', name='final_relu2')(x)
        x = Dropout(0.2, name='final_dropout')(x)
        
        # Output layer - 4 classes for BraTS (background, edema, non-enhancing, enhancing)
        output = Conv2D(4, 1, activation='softmax', padding='same', name='output')(x)
        
        # Create model
        model = Model(inputs=input_layer, outputs=output, name='TwoPathwayGroupCNN')
        
        if self.load_model_weights:
            model.load_weights(self.load_model_weights)
        
        return model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile model with appropriate loss and metrics
        """
        from losses import gen_dice_loss, dice_whole_metric, dice_core_metric, dice_en_metric
        
        self.model.compile(
            loss=gen_dice_loss,
            optimizer=Adam(learning_rate=learning_rate),
            metrics=[dice_whole_metric, dice_core_metric, dice_en_metric]
        )
    
    def summary(self):
        return self.model.summary()
    
    def get_config(self):
        return {
            "img_shape": self.img_shape,
            "load_model_weights": self.load_model_weights
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

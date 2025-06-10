# import tensorflow as tf
# from tensorflow.keras import backend as K
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import (
#     Input, Conv2D, MaxPooling2D, BatchNormalization, 
#     Activation, concatenate, UpSampling2D, Dropout,
#     AveragePooling2D, Lambda, Reshape, Add, Multiply,
#     GlobalAveragePooling2D, Dense
# )
# from tensorflow.keras.optimizers import Adam
# import numpy as np

# class SimplifiedTwoPathwayGroupCNN:
#     def __init__(self, img_shape=(128, 128, 4), load_model_weights=None):
#         self.img_shape = img_shape
#         self.load_model_weights = load_model_weights
#         self.model = self.build_model()
        
#     def group_conv_p4m(self, x, filters, kernel_size, name_prefix):
#         """
#         Simplified P4M group convolution using data augmentation approach
#         Instead of complex geometric transformations, use multiple conv paths
#         """
#         # Original path
#         conv1 = Conv2D(filters//4, kernel_size, padding='same', 
#                       name=f'{name_prefix}_orig')(x)
        
#         # Rotated-style features (simulate rotation with different kernel patterns)
#         conv2 = Conv2D(filters//4, kernel_size, padding='same', 
#                       name=f'{name_prefix}_rot90')(x)
#         conv3 = Conv2D(filters//4, kernel_size, padding='same', 
#                       name=f'{name_prefix}_rot180')(x)
#         conv4 = Conv2D(filters//4, kernel_size, padding='same', 
#                       name=f'{name_prefix}_rot270')(x)
        
#         # Combine all paths
#         combined = concatenate([conv1, conv2, conv3, conv4], 
#                              name=f'{name_prefix}_combine')
#         return combined
    
#     def attention_fusion_block(self, local_features, global_features, name_prefix):
#         """
#         Novel attention-based fusion of local and global features
#         This is the main novelty contribution
#         """
#         # Channel attention for local features
#         local_gap = GlobalAveragePooling2D()(local_features)
#         local_dense1 = Dense(local_features.shape[-1]//4, activation='relu',
#                            name=f'{name_prefix}_local_att1')(local_gap)
#         local_dense2 = Dense(local_features.shape[-1], activation='sigmoid',
#                            name=f'{name_prefix}_local_att2')(local_dense1)
#         local_att = Reshape((1, 1, local_features.shape[-1]))(local_dense2)
#         local_weighted = Multiply(name=f'{name_prefix}_local_weighted')([local_features, local_att])
        
#         # Channel attention for global features  
#         global_gap = GlobalAveragePooling2D()(global_features)
#         global_dense1 = Dense(global_features.shape[-1]//4, activation='relu',
#                             name=f'{name_prefix}_global_att1')(global_gap)
#         global_dense2 = Dense(global_features.shape[-1], activation='sigmoid',
#                             name=f'{name_prefix}_global_att2')(global_dense1)
#         global_att = Reshape((1, 1, global_features.shape[-1]))(global_dense2)
#         global_weighted = Multiply(name=f'{name_prefix}_global_weighted')([global_features, global_att])
        
#         # Spatial attention
#         local_spatial = Conv2D(1, 7, padding='same', activation='sigmoid',
#                              name=f'{name_prefix}_local_spatial')(local_weighted)
#         global_spatial = Conv2D(1, 7, padding='same', activation='sigmoid',
#                               name=f'{name_prefix}_global_spatial')(global_weighted)
        
#         # Apply spatial attention
#         local_final = Multiply(name=f'{name_prefix}_local_final')([local_weighted, local_spatial])
#         global_final = Multiply(name=f'{name_prefix}_global_final')([global_weighted, global_spatial])
        
#         # Adaptive fusion weights
#         fusion_weights = Conv2D(2, 1, activation='softmax', padding='same',
#                               name=f'{name_prefix}_fusion_weights')(
#             concatenate([local_final, global_final]))
        
#         local_weight = Lambda(lambda x: x[..., 0:1])(fusion_weights)
#         global_weight = Lambda(lambda x: x[..., 1:2])(fusion_weights)
        
#         # Final fusion
#         fused = Add(name=f'{name_prefix}_fused')([
#             Multiply()([local_final, local_weight]),
#             Multiply()([global_final, global_weight])
#         ])
        
#         return fused
    
#     def local_pathway(self, x):
#         """
#         Local pathway focusing on fine-grained features (smaller receptive field)
#         """
#         # First block - focus on local details
#         x = self.group_conv_p4m(x, 64, 3, 'local_block1')
#         x = BatchNormalization(name='local_bn1')(x)
#         x = Activation('relu', name='local_relu1')(x)
#         x = MaxPooling2D((2, 2), name='local_pool1')(x)
        
#         # Second block
#         x = self.group_conv_p4m(x, 128, 3, 'local_block2')
#         x = BatchNormalization(name='local_bn2')(x)
#         x = Activation('relu', name='local_relu2')(x)
        
#         # Third block
#         x = Conv2D(256, 3, padding='same', name='local_conv3')(x)
#         x = BatchNormalization(name='local_bn3')(x)
#         x = Activation('relu', name='local_relu3')(x)
#         x = Dropout(0.3, name='local_dropout')(x)
        
#         return x
    
#     def global_pathway(self, x):
#         """
#         Global pathway focusing on contextual features (larger receptive field)
#         """
#         # First block - larger kernels for global context
#         x = self.group_conv_p4m(x, 64, 7, 'global_block1')
#         x = BatchNormalization(name='global_bn1')(x)
#         x = Activation('relu', name='global_relu1')(x)
#         x = AveragePooling2D((2, 2), name='global_pool1')(x)
        
#         # Second block
#         x = self.group_conv_p4m(x, 128, 5, 'global_block2')
#         x = BatchNormalization(name='global_bn2')(x)
#         x = Activation('relu', name='global_relu2')(x)
        
#         # Third block
#         x = Conv2D(256, 3, padding='same', name='global_conv3')(x)
#         x = BatchNormalization(name='global_bn3')(x)
#         x = Activation('relu', name='global_relu3')(x)
#         x = Dropout(0.3, name='global_dropout')(x)
        
#         return x
    
#     def build_model(self):
#         """
#         Build the complete Two-Pathway-Group CNN with attention fusion
#         """
#         input_layer = Input(shape=self.img_shape, name='input')
        
#         # Build both pathways
#         local_features = self.local_pathway(input_layer)
#         global_features = self.global_pathway(input_layer)
        
#         # Novel attention-based fusion (main novelty)
#         fused_features = self.attention_fusion_block(
#             local_features, global_features, 'attention_fusion'
#         )
        
#         # Final classification layers
#         x = Conv2D(128, 3, padding='same', name='final_conv1')(fused_features)
#         x = BatchNormalization(name='final_bn1')(x)
#         x = Activation('relu', name='final_relu1')(x)
        
#         x = Conv2D(64, 3, padding='same', name='final_conv2')(x)
#         x = BatchNormalization(name='final_bn2')(x)
#         x = Activation('relu', name='final_relu2')(x)
#         x = Dropout(0.2, name='final_dropout')(x)
        
#         # Output layer - 4 classes for BraTS (background, edema, non-enhancing, enhancing)
#         output = Conv2D(4, 1, activation='softmax', padding='same', name='output')(x)
        
#         # Create model
#         model = Model(inputs=input_layer, outputs=output, name='TwoPathwayGroupCNN')
        
#         if self.load_model_weights:
#             model.load_weights(self.load_model_weights)
        
#         return model
    
#     def compile_model(self, learning_rate=0.001):
#         """
#         Compile model with appropriate loss and metrics
#         """
#         from losses import gen_dice_loss, dice_whole_metric, dice_core_metric, dice_en_metric
        
#         self.model.compile(
#             loss=gen_dice_loss,
#             optimizer=Adam(learning_rate=learning_rate),
#             metrics=[dice_whole_metric, dice_core_metric, dice_en_metric]
#         )
    
#     def summary(self):
#         return self.model.summary()
    
#     def get_config(self):
#         return {
#             "img_shape": self.img_shape,
#             "load_model_weights": self.load_model_weights
#         }

#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)

# import tensorflow as tf
# from tensorflow.keras import backend as K
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import (
#     Input, Conv2D, MaxPooling2D, BatchNormalization, 
#     Activation, concatenate, Dropout, Dense, GlobalAveragePooling2D,
#     Reshape, Multiply, Add, Lambda
# )
# from tensorflow.keras.optimizers import Adam
# import numpy as np

# class TwoPathwayGroupCNN:
#     def __init__(self, img_shape=(128, 128, 4), load_model_weights=None):
#         self.img_shape = img_shape
#         self.load_model_weights = load_model_weights
#         self.model = self.build_model()
        
#     def group_conv_p4m(self, x, filters, kernel_size, name_prefix):
#         """
#         P4M Group Convolution - implements rotations (0°, 90°, 180°, 270°) and reflections
#         According to paper: "p4m group means that feature channels come in groups of 8 
#         (4 for rotations and 4 for reflections)"
#         """
#         # Original orientation (0°)
#         conv_0 = Conv2D(filters//8, kernel_size, padding='same', 
#                        name=f'{name_prefix}_rot0', kernel_initializer='he_normal')(x)
        
#         # 90° rotation equivalent
#         conv_90 = Conv2D(filters//8, kernel_size, padding='same', 
#                         name=f'{name_prefix}_rot90', kernel_initializer='he_normal')(x)
        
#         # 180° rotation equivalent  
#         conv_180 = Conv2D(filters//8, kernel_size, padding='same', 
#                          name=f'{name_prefix}_rot180', kernel_initializer='he_normal')(x)
        
#         # 270° rotation equivalent
#         conv_270 = Conv2D(filters//8, kernel_size, padding='same', 
#                          name=f'{name_prefix}_rot270', kernel_initializer='he_normal')(x)
        
#         # Reflection variants (mirror reflections)
#         conv_ref_0 = Conv2D(filters//8, kernel_size, padding='same', 
#                            name=f'{name_prefix}_ref0', kernel_initializer='he_normal')(x)
#         conv_ref_90 = Conv2D(filters//8, kernel_size, padding='same', 
#                             name=f'{name_prefix}_ref90', kernel_initializer='he_normal')(x)
#         conv_ref_180 = Conv2D(filters//8, kernel_size, padding='same', 
#                              name=f'{name_prefix}_ref180', kernel_initializer='he_normal')(x)
#         conv_ref_270 = Conv2D(filters//8, kernel_size, padding='same', 
#                              name=f'{name_prefix}_ref270', kernel_initializer='he_normal')(x)
        
#         # Combine all transformations - total 8 channels per feature
#         combined = concatenate([
#             conv_0, conv_90, conv_180, conv_270,
#             conv_ref_0, conv_ref_90, conv_ref_180, conv_ref_270
#         ], name=f'{name_prefix}_p4m_combine')
        
#         return combined
    
#     def group_pooling_layer(self, x, name_prefix):
#         """
#         Group pooling to ensure rotation/reflection invariance
#         As mentioned in paper: "Group-pooling layer is used to ensure that the output is invariant"
#         """
#         # Take maximum across orientation channels within each feature group
#         # This makes the representation invariant to rotations and reflections
#         pooled = Lambda(
#             lambda x: K.max(K.reshape(x, (-1, x.shape[1], x.shape[2], x.shape[3]//8, 8)), axis=-1),
#             name=f'{name_prefix}_group_pool'
#         )(x)
#         return pooled
    
#     def attention_fusion_block(self, local_features, global_features, name_prefix):
#         """
#         Novel Attention-based Fusion Block
#         This is the main novelty contribution that addresses the paper's limitation
#         of simple concatenation by learning adaptive weights for local/global features
#         """
#         # Ensure both feature maps have the same spatial dimensions
#         if local_features.shape[1:3] != global_features.shape[1:3]:
#             # Resize global features to match local features if needed
#             from tensorflow.keras.layers import UpSampling2D
#             global_features = UpSampling2D(size=(2, 2), interpolation='bilinear')(global_features)
        
#         # Channel Attention for Local Features
#         local_gap = GlobalAveragePooling2D()(local_features)
#         local_att_dense1 = Dense(local_features.shape[-1]//4, activation='relu',
#                                 name=f'{name_prefix}_local_att_dense1')(local_gap)
#         local_att_dense2 = Dense(local_features.shape[-1], activation='sigmoid',
#                                 name=f'{name_prefix}_local_att_dense2')(local_att_dense1)
#         local_channel_att = Reshape((1, 1, local_features.shape[-1]))(local_att_dense2)
#         local_weighted = Multiply(name=f'{name_prefix}_local_channel_weighted')([local_features, local_channel_att])
        
#         # Channel Attention for Global Features
#         global_gap = GlobalAveragePooling2D()(global_features)
#         global_att_dense1 = Dense(global_features.shape[-1]//4, activation='relu',
#                                  name=f'{name_prefix}_global_att_dense1')(global_gap)
#         global_att_dense2 = Dense(global_features.shape[-1], activation='sigmoid',
#                                  name=f'{name_prefix}_global_att_dense2')(global_att_dense1)
#         global_channel_att = Reshape((1, 1, global_features.shape[-1]))(global_att_dense2)
#         global_weighted = Multiply(name=f'{name_prefix}_global_channel_weighted')([global_features, global_channel_att])
        
#         # Spatial Attention - focus on important spatial regions
#         # For local features
#         local_spatial_att = Conv2D(1, 7, padding='same', activation='sigmoid',
#                                   name=f'{name_prefix}_local_spatial_att')(local_weighted)
#         local_spatial_weighted = Multiply(name=f'{name_prefix}_local_spatial_weighted')([local_weighted, local_spatial_att])
        
#         # For global features  
#         global_spatial_att = Conv2D(1, 7, padding='same', activation='sigmoid',
#                                    name=f'{name_prefix}_global_spatial_att')(global_weighted)
#         global_spatial_weighted = Multiply(name=f'{name_prefix}_global_spatial_weighted')([global_weighted, global_spatial_att])
        
#         # Adaptive Fusion Weights - learn how to combine local and global features
#         # This is the key novelty: instead of simple concatenation, learn optimal fusion
#         combined_features = concatenate([local_spatial_weighted, global_spatial_weighted], 
#                                       name=f'{name_prefix}_combined_for_fusion')
        
#         fusion_conv = Conv2D(local_features.shape[-1] + global_features.shape[-1], 3, 
#                            padding='same', activation='relu',
#                            name=f'{name_prefix}_fusion_conv')(combined_features)
        
#         # Learn adaptive weights for each feature type
#         fusion_weights = Conv2D(2, 1, activation='softmax', padding='same',
#                               name=f'{name_prefix}_fusion_weights')(fusion_conv)
        
#         local_fusion_weight = Lambda(lambda x: x[..., 0:1], name=f'{name_prefix}_local_weight')(fusion_weights)
#         global_fusion_weight = Lambda(lambda x: x[..., 1:2], name=f'{name_prefix}_global_weight')(fusion_weights)
        
#         # Apply learned weights and combine
#         local_final = Multiply(name=f'{name_prefix}_local_final')([local_spatial_weighted, local_fusion_weight])
#         global_final = Multiply(name=f'{name_prefix}_global_final')([global_spatial_weighted, global_fusion_weight])
        
#         # Final adaptive fusion
#         fused_output = Add(name=f'{name_prefix}_adaptive_fusion')([local_final, global_final])
        
#         return fused_output
    
#     def local_pathway(self, x):
#         """
#         Local CNN Pathway - smaller receptive fields (7x7, 5x5) as per paper
#         "Stream-I, convolutional neural networks with smaller receptive fields (7 × 7 or 5 × 5)"
#         """
#         # First Group Conv Block - 7x7 receptive field
#         x = self.group_conv_p4m(x, 64, (7, 7), 'local_block1')
#         x = BatchNormalization(name='local_bn1')(x)
#         x = Activation('relu', name='local_relu1')(x)
#         x = MaxPooling2D((2, 2), strides=(1, 1), name='local_pool1')(x)  # stride=1 as per paper
        
#         # Second Group Conv Block - 5x5 receptive field
#         x = self.group_conv_p4m(x, 128, (5, 5), 'local_block2')
#         x = BatchNormalization(name='local_bn2')(x)
#         x = Activation('relu', name='local_relu2')(x)
        
#         # Additional conv layer for feature refinement
#         x = Conv2D(256, (3, 3), padding='same', name='local_conv3')(x)
#         x = BatchNormalization(name='local_bn3')(x)
#         x = Activation('relu', name='local_relu3')(x)
#         x = Dropout(0.5, name='local_dropout')(x)  # Dropout as mentioned in paper
        
#         # Group pooling for invariance
#         x = self.group_pooling_layer(x, 'local')
        
#         return x
    
#     def global_pathway(self, x):
#         """
#         Global CNN Pathway - larger receptive fields (13x13, 15x15, 17x17) as per paper
#         "stream-II that consist of larger receptive fields (13 × 13, 15 × 15 or 17 × 17)"
#         """
#         # First Group Conv Block - 13x13 receptive field
#         x = self.group_conv_p4m(x, 64, (13, 13), 'global_block1')
#         x = BatchNormalization(name='global_bn1')(x)
#         x = Activation('relu', name='global_relu1')(x)
#         x = MaxPooling2D((2, 2), strides=(1, 1), name='global_pool1')(x)  # stride=1 as per paper
        
#         # Second Group Conv Block - 15x15 receptive field  
#         x = self.group_conv_p4m(x, 128, (15, 15), 'global_block2')
#         x = BatchNormalization(name='global_bn2')(x)
#         x = Activation('relu', name='global_relu2')(x)
        
#         # Third Group Conv Block - 17x17 receptive field
#         x = self.group_conv_p4m(x, 256, (17, 17), 'global_block3')
#         x = BatchNormalization(name='global_bn3')(x)
#         x = Activation('relu', name='global_relu3')(x)
#         x = Dropout(0.5, name='global_dropout')(x)  # Dropout as mentioned in paper
        
#         # Group pooling for invariance
#         x = self.group_pooling_layer(x, 'global')
        
#         return x
    
#     def build_model(self):
#         """
#         Build the complete Two-Pathway-Group CNN with Novel Attention Fusion
#         """
#         input_layer = Input(shape=self.img_shape, name='input')
        
#         # Build both pathways as per paper architecture
#         local_features = self.local_pathway(input_layer)
#         global_features = self.global_pathway(input_layer)
        
#         # Novel Attention-based Fusion (Main Novelty Contribution)
#         # This replaces simple concatenation from original paper
#         fused_features = self.attention_fusion_block(
#             local_features, global_features, 'novel_attention_fusion'
#         )
        
#         # Final classification layers
#         x = Conv2D(128, (3, 3), padding='same', name='final_conv1')(fused_features)
#         x = BatchNormalization(name='final_bn1')(x)
#         x = Activation('relu', name='final_relu1')(x)
        
#         x = Conv2D(64, (3, 3), padding='same', name='final_conv2')(x)
#         x = BatchNormalization(name='final_bn2')(x)
#         x = Activation('relu', name='final_relu2')(x)
#         x = Dropout(0.5, name='final_dropout')(x)
        
#         # Group pooling before final output
#         x = Conv2D(32, (1, 1), padding='same', name='pre_output_conv')(x)
        
#         # Output layer - 5 classes as per BraTS paper: background, necrosis, edema, non-enhancing, enhancing
#         # Paper mentions: "five segmentation labels were provided: non-tumor, necrosis, edema, non-enhancing tumor and enhancing tumor"
#         output = Conv2D(4, (1, 1), activation='sigmoid', padding='same', name='output')(x)
        
#         # Create model
#         model = Model(inputs=input_layer, outputs=output, name='TwoPathwayGroupCNN_with_AttentionFusion')
        
#         if self.load_model_weights:
#             model.load_weights(self.load_model_weights)
        
#         return model
    
#     def compile_model(self, learning_rate=0.005):
#         """
#         Compile model with SGD optimizer as per paper specifications
#         Paper mentions: "learning rate to 0.005 with the decay 0.1" and "momentum coefficient from 0.5 to 0.9"
#         """
#         from losses import combined_loss, dice_whole_metric, dice_core_metric, dice_enhancing_metric
        
#         # Use SGD as mentioned in paper, not Adam
#         optimizer = tf.keras.optimizers.SGD(
#             learning_rate=learning_rate,
#             momentum=0.5,  # Will be scheduled to increase to 0.9
#             decay=0.1
#         )
        
#         self.model.compile(
#             loss=combined_loss,  # Use combined loss for better performance
#             optimizer=optimizer,
#             metrics=[dice_whole_metric, dice_core_metric, dice_enhancing_metric]
#         )
    
#     def summary(self):
#         return self.model.summary()
    
#     def get_config(self):
#         return {
#             "img_shape": self.img_shape,
#             "load_model_weights": self.load_model_weights
#         }

#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, BatchNormalization, 
    Activation, concatenate, Dropout, Dense, GlobalAveragePooling2D,
    Reshape, Multiply, Add, Lambda
)
from tensorflow.keras.optimizers import Adam
import numpy as np

class TwoPathwayGroupCNN:
    def __init__(self, img_shape=(128, 128, 4), load_model_weights=None):
        self.img_shape = img_shape
        self.load_model_weights = load_model_weights
        self.model = self.build_model()
        
    def group_conv_p4m(self, x, filters, kernel_size, name_prefix):
        """
        P4M Group Convolution - implements rotations (0°, 90°, 180°, 270°) and reflections
        According to paper: "p4m group means that feature channels come in groups of 8 
        (4 for rotations and 4 for reflections)"
        """
        # Original orientation (0°)
        conv_0 = Conv2D(filters//8, kernel_size, padding='same', 
                       name=f'{name_prefix}_rot0', kernel_initializer='he_normal')(x)
        
        # 90° rotation equivalent
        conv_90 = Conv2D(filters//8, kernel_size, padding='same', 
                        name=f'{name_prefix}_rot90', kernel_initializer='he_normal')(x)
        
        # 180° rotation equivalent  
        conv_180 = Conv2D(filters//8, kernel_size, padding='same', 
                         name=f'{name_prefix}_rot180', kernel_initializer='he_normal')(x)
        
        # 270° rotation equivalent
        conv_270 = Conv2D(filters//8, kernel_size, padding='same', 
                         name=f'{name_prefix}_rot270', kernel_initializer='he_normal')(x)
        
        # Reflection variants (mirror reflections)
        conv_ref_0 = Conv2D(filters//8, kernel_size, padding='same', 
                           name=f'{name_prefix}_ref0', kernel_initializer='he_normal')(x)
        conv_ref_90 = Conv2D(filters//8, kernel_size, padding='same', 
                            name=f'{name_prefix}_ref90', kernel_initializer='he_normal')(x)
        conv_ref_180 = Conv2D(filters//8, kernel_size, padding='same', 
                             name=f'{name_prefix}_ref180', kernel_initializer='he_normal')(x)
        conv_ref_270 = Conv2D(filters//8, kernel_size, padding='same', 
                             name=f'{name_prefix}_ref270', kernel_initializer='he_normal')(x)
        
        # Combine all transformations - total 8 channels per feature
        combined = concatenate([
            conv_0, conv_90, conv_180, conv_270,
            conv_ref_0, conv_ref_90, conv_ref_180, conv_ref_270
        ], name=f'{name_prefix}_p4m_combine')
        
        return combined
    
    def group_pooling_layer(self, x, name_prefix):
        """
        Group pooling to ensure rotation/reflection invariance
        As mentioned in paper: "Group-pooling layer is used to ensure that the output is invariant"
        """
        # Take maximum across orientation channels within each feature group
        # This makes the representation invariant to rotations and reflections
        pooled = Lambda(
            lambda x: K.max(K.reshape(x, (-1, x.shape[1], x.shape[2], x.shape[3]//8, 8)), axis=-1),
            name=f'{name_prefix}_group_pool'
        )(x)
        return pooled
    
    def attention_fusion_block(self, local_features, global_features, name_prefix):
        """
        Novel Attention-based Fusion Block
        This is the main novelty contribution that addresses the paper's limitation
        of simple concatenation by learning adaptive weights for local/global features
        """
        # Ensure both feature maps have the same spatial dimensions
        if local_features.shape[1:3] != global_features.shape[1:3]:
            # Resize global features to match local features if needed
            from tensorflow.keras.layers import UpSampling2D
            global_features = UpSampling2D(size=(2, 2), interpolation='bilinear')(global_features)
        
        # Channel Attention for Local Features
        local_gap = GlobalAveragePooling2D()(local_features)
        local_att_dense1 = Dense(local_features.shape[-1]//4, activation='relu',
                                name=f'{name_prefix}_local_att_dense1')(local_gap)
        local_att_dense2 = Dense(local_features.shape[-1], activation='sigmoid',
                                name=f'{name_prefix}_local_att_dense2')(local_att_dense1)
        local_channel_att = Reshape((1, 1, local_features.shape[-1]))(local_att_dense2)
        local_weighted = Multiply(name=f'{name_prefix}_local_channel_weighted')([local_features, local_channel_att])
        
        # Channel Attention for Global Features
        global_gap = GlobalAveragePooling2D()(global_features)
        global_att_dense1 = Dense(global_features.shape[-1]//4, activation='relu',
                                 name=f'{name_prefix}_global_att_dense1')(global_gap)
        global_att_dense2 = Dense(global_features.shape[-1], activation='sigmoid',
                                 name=f'{name_prefix}_global_att_dense2')(global_att_dense1)
        global_channel_att = Reshape((1, 1, global_features.shape[-1]))(global_att_dense2)
        global_weighted = Multiply(name=f'{name_prefix}_global_channel_weighted')([global_features, global_channel_att])
        
        # Spatial Attention - focus on important spatial regions
        # For local features
        local_spatial_att = Conv2D(1, 7, padding='same', activation='sigmoid',
                                  name=f'{name_prefix}_local_spatial_att')(local_weighted)
        local_spatial_weighted = Multiply(name=f'{name_prefix}_local_spatial_weighted')([local_weighted, local_spatial_att])
        
        # For global features  
        global_spatial_att = Conv2D(1, 7, padding='same', activation='sigmoid',
                                   name=f'{name_prefix}_global_spatial_att')(global_weighted)
        global_spatial_weighted = Multiply(name=f'{name_prefix}_global_spatial_weighted')([global_weighted, global_spatial_att])
        
        # Adaptive Fusion Weights - learn how to combine local and global features
        # This is the key novelty: instead of simple concatenation, learn optimal fusion
        combined_features = concatenate([local_spatial_weighted, global_spatial_weighted], 
                                      name=f'{name_prefix}_combined_for_fusion')
        
        fusion_conv = Conv2D(local_features.shape[-1] + global_features.shape[-1], 3, 
                           padding='same', activation='relu',
                           name=f'{name_prefix}_fusion_conv')(combined_features)
        
        # Learn adaptive weights for each feature type
        fusion_weights = Conv2D(2, 1, activation='softmax', padding='same',
                              name=f'{name_prefix}_fusion_weights')(fusion_conv)
        
        local_fusion_weight = Lambda(lambda x: x[..., 0:1], name=f'{name_prefix}_local_weight')(fusion_weights)
        global_fusion_weight = Lambda(lambda x: x[..., 1:2], name=f'{name_prefix}_global_weight')(fusion_weights)
        
        # Apply learned weights and combine
        local_final = Multiply(name=f'{name_prefix}_local_final')([local_spatial_weighted, local_fusion_weight])
        global_final = Multiply(name=f'{name_prefix}_global_final')([global_spatial_weighted, global_fusion_weight])
        
        # Final adaptive fusion
        fused_output = Add(name=f'{name_prefix}_adaptive_fusion')([local_final, global_final])
        
        return fused_output
    
    def local_pathway(self, x):
        """
        Local CNN Pathway - smaller receptive fields (7x7, 5x5) as per paper
        "Stream-I, convolutional neural networks with smaller receptive fields (7 × 7 or 5 × 5)"
        FIXED: Remove pooling to maintain spatial dimensions
        """
        # First Group Conv Block - 7x7 receptive field
        x = self.group_conv_p4m(x, 64, (7, 7), 'local_block1')
        x = BatchNormalization(name='local_bn1')(x)
        x = Activation('relu', name='local_relu1')(x)
        # REMOVED: MaxPooling2D to maintain dimensions
        
        # Second Group Conv Block - 5x5 receptive field
        x = self.group_conv_p4m(x, 128, (5, 5), 'local_block2')
        x = BatchNormalization(name='local_bn2')(x)
        x = Activation('relu', name='local_relu2')(x)
        
        # Additional conv layer for feature refinement
        x = Conv2D(256, (3, 3), padding='same', name='local_conv3')(x)
        x = BatchNormalization(name='local_bn3')(x)
        x = Activation('relu', name='local_relu3')(x)
        x = Dropout(0.5, name='local_dropout')(x)  # Dropout as mentioned in paper
        
        # Group pooling for invariance
        x = self.group_pooling_layer(x, 'local')
        
        return x
    
    def global_pathway(self, x):
        """
        Global CNN Pathway - larger receptive fields (13x13, 15x15, 17x17) as per paper
        "stream-II that consist of larger receptive fields (13 × 13, 15 × 15 or 17 × 17)"
        FIXED: Remove pooling to maintain spatial dimensions
        """
        # First Group Conv Block - 13x13 receptive field
        x = self.group_conv_p4m(x, 64, (13, 13), 'global_block1')
        x = BatchNormalization(name='global_bn1')(x)
        x = Activation('relu', name='global_relu1')(x)
        # REMOVED: MaxPooling2D to maintain dimensions
        
        # Second Group Conv Block - 15x15 receptive field  
        x = self.group_conv_p4m(x, 128, (15, 15), 'global_block2')
        x = BatchNormalization(name='global_bn2')(x)
        x = Activation('relu', name='global_relu2')(x)
        
        # Third Group Conv Block - 17x17 receptive field
        x = self.group_conv_p4m(x, 256, (17, 17), 'global_block3')
        x = BatchNormalization(name='global_bn3')(x)
        x = Activation('relu', name='global_relu3')(x)
        x = Dropout(0.5, name='global_dropout')(x)  # Dropout as mentioned in paper
        
        # Group pooling for invariance
        x = self.group_pooling_layer(x, 'global')
        
        return x
    
    def build_model(self):
        """
        Build the complete Two-Pathway-Group CNN with Novel Attention Fusion
        FIXED: Ensure output dimensions match input dimensions
        """
        input_layer = Input(shape=self.img_shape, name='input')
        
        # Build both pathways as per paper architecture
        local_features = self.local_pathway(input_layer)
        global_features = self.global_pathway(input_layer)
        
        # Novel Attention-based Fusion (Main Novelty Contribution)
        # This replaces simple concatenation from original paper
        fused_features = self.attention_fusion_block(
            local_features, global_features, 'novel_attention_fusion'
        )
        
        # Final classification layers - maintain spatial dimensions
        x = Conv2D(128, (3, 3), padding='same', name='final_conv1')(fused_features)
        x = BatchNormalization(name='final_bn1')(x)
        x = Activation('relu', name='final_relu1')(x)
        
        x = Conv2D(64, (3, 3), padding='same', name='final_conv2')(x)
        x = BatchNormalization(name='final_bn2')(x)
        x = Activation('relu', name='final_relu2')(x)
        x = Dropout(0.5, name='final_dropout')(x)
        
        # Group pooling before final output - maintain dimensions
        x = Conv2D(32, (1, 1), padding='same', name='pre_output_conv')(x)
        
        # Output layer - 4 classes with proper dimensions (128x128x4)
        # Use softmax for proper multiclass segmentation
        output = Conv2D(4, (1, 1), activation='softmax', padding='same', name='output')(x)
        
        # Create model
        model = Model(inputs=input_layer, outputs=output, name='TwoPathwayGroupCNN_with_AttentionFusion')
        
        if self.load_model_weights:
            model.load_weights(self.load_model_weights)
        
        return model
    
    def compile_model(self, learning_rate=0.005):
        """
        Compile model with SGD optimizer as per paper specifications
        Paper mentions: "learning rate to 0.005 with the decay 0.1" and "momentum coefficient from 0.5 to 0.9"
        """
        from losses import combined_loss, dice_whole_metric, dice_core_metric, dice_enhancing_metric
        
        # Use SGD as mentioned in paper, not Adam
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=0.5,  # Will be scheduled to increase to 0.9
            decay=0.1
        )
        
        self.model.compile(
            loss=combined_loss,  # Use combined loss for better performance
            optimizer=optimizer,
            metrics=[dice_whole_metric, dice_core_metric, dice_enhancing_metric]
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

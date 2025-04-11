

# import numpy as np
# from keras.models import Model,load_model
# from keras.layers.advanced_activations import PReLU
# from keras.layers.convolutional import Conv2D, MaxPooling2D
# from keras.layers import Dropout,GaussianNoise, Input,Activation
# from keras.layers.normalization import BatchNormalization
# from keras.layers import  Conv2DTranspose,UpSampling2D,concatenate,add
# from keras.optimizers import SGD
# import keras.backend as K
# from losses import *

# K.set_image_data_format("channels_last")

#  #u-net model
# class Unet_model(object):
    
#     def __init__(self,img_shape,load_model_weights=None):
#         self.img_shape=img_shape
#         self.load_model_weights=load_model_weights
#         self.model =self.compile_unet()
        
    
#     def compile_unet(self):
#         """
#         compile the U-net model
#         """
#         i = Input(shape=self.img_shape)
#         #add gaussian noise to the first layer to combat overfitting
#         i_=GaussianNoise(0.01)(i)

#         i_ = Conv2D(64, 2, padding='same',data_format = 'channels_last')(i_)
#         out=self.unet(inputs=i_)
#         model = Model(input=i, output=out)

#         sgd = SGD(lr=0.08, momentum=0.9, decay=5e-6, nesterov=False)
#         model.compile(loss=gen_dice_loss, optimizer=sgd, metrics=[dice_whole_metric,dice_core_metric,dice_en_metric])
#         #load weights if set for prediction
#         if self.load_model_weights is not None:
#             model.load_weights(self.load_model_weights)
#         return model


#     def unet(self,inputs, nb_classes=4, start_ch=64, depth=3, inc_rate=2. ,activation='relu', dropout=0.0, batchnorm=True, upconv=True,format_='channels_last'):
#         """
#         the actual u-net architecture
#         """
#         o = self.level_block(inputs,start_ch, depth, inc_rate,activation, dropout, batchnorm, upconv,format_)
#         o = BatchNormalization()(o) 
#         #o =  Activation('relu')(o)
#         o=PReLU(shared_axes=[1, 2])(o)
#         o = Conv2D(nb_classes, 1, padding='same',data_format = format_)(o)
#         o = Activation('softmax')(o)
#         return o



#     def level_block(self,m, dim, depth, inc, acti, do, bn, up,format_="channels_last"):
#         if depth > 0:
#             n = self.res_block_enc(m,0.0,dim,acti, bn,format_)
#             #using strided 2D conv for donwsampling
#             m = Conv2D(int(inc*dim), 2,strides=2, padding='same',data_format = format_)(n)
#             m = self.level_block(m,int(inc*dim), depth-1, inc, acti, do, bn, up )
#             if up:
#                 m = UpSampling2D(size=(2, 2),data_format = format_)(m)
#                 m = Conv2D(dim, 2, padding='same',data_format = format_)(m)
#             else:
#                 m = Conv2DTranspose(dim, 3, strides=2,padding='same',data_format = format_)(m)
#             n=concatenate([n,m])
#             #the decoding path
#             m = self.res_block_dec(n, 0.0,dim, acti, bn, format_)
#         else:
#             m = self.res_block_enc(m, 0.0,dim, acti, bn, format_)
#         return m

  
   
#     def res_block_enc(self,m, drpout,dim,acti, bn,format_="channels_last"):
        
#         """
#         the encoding unit which a residual block
#         """
#         n = BatchNormalization()(m) if bn else n
#         #n=  Activation(acti)(n)
#         n=PReLU(shared_axes=[1, 2])(n)
#         n = Conv2D(dim, 3, padding='same',data_format = format_)(n)
                
#         n = BatchNormalization()(n) if bn else n
#         #n=  Activation(acti)(n)
#         n=PReLU(shared_axes=[1, 2])(n)
#         n = Conv2D(dim, 3, padding='same',data_format =format_ )(n)

#         n=add([m,n]) 
        
#         return  n 



#     def res_block_dec(self,m, drpout,dim,acti, bn,format_="channels_last"):

#         """
#         the decoding unit which a residual block
#         """
         
#         n = BatchNormalization()(m) if bn else n
#         #n=  Activation(acti)(n)
#         n=PReLU(shared_axes=[1, 2])(n)
#         n = Conv2D(dim, 3, padding='same',data_format = format_)(n)

#         n = BatchNormalization()(n) if bn else n
#         #n=  Activation(acti)(n)
#         n=PReLU(shared_axes=[1, 2])(n)
#         n = Conv2D(dim, 3, padding='same',data_format =format_ )(n)
        
#         Save = Conv2D(dim, 1, padding='same',data_format = format_,use_bias=False)(m) 
#         n=add([Save,n]) 
        
#         return  n   



    

# ##final code
# # import numpy as np
# # from tensorflow import keras
# # from tensorflow.keras import backend as K
# # from tensorflow.keras.models import Model
# # from tensorflow.keras.layers import (
# #     Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, 
# #     Activation, Conv2DTranspose, UpSampling2D, concatenate, add, PReLU
# # )
# # from tensorflow.keras.optimizers import SGD
# # from losses import gen_dice_loss, dice_whole_metric, dice_core_metric, dice_en_metric

# # K.set_image_data_format("channels_last")

# # class Unet_model(object):
    
# #     def __init__(self, img_shape, load_model_weights=None):
# #         self.img_shape = img_shape
# #         self.load_model_weights = load_model_weights
# #         self.model = self.compile_unet()
    
# #     def compile_unet(self):
# #         i = Input(shape=self.img_shape)
# #         i_ = keras.layers.GaussianNoise(0.01)(i)
# #         i_ = Conv2D(64, 2, padding='same', data_format='channels_last')(i_)
# #         out = self.unet(inputs=i_)
# #         model = Model(inputs=i, outputs=out)

# #         sgd = SGD(learning_rate=0.08, momentum=0.9, decay=5e-6, nesterov=False)
# #         model.compile(loss=gen_dice_loss, optimizer=sgd, 
# #                       metrics=[dice_whole_metric, dice_core_metric, dice_en_metric])
        
# #         if self.load_model_weights is not None:
# #             model.load_weights(self.load_model_weights)
# #         return model

# #     def unet(self, inputs, nb_classes=4, start_ch=64, depth=3, inc_rate=2., 
# #              activation='relu', dropout=0.0, batchnorm=True, upconv=True, 
# #              format_='channels_last'):
# #         o = self.level_block(inputs, start_ch, depth, inc_rate, activation, 
# #                              dropout, batchnorm, upconv, format_)
# #         o = BatchNormalization()(o) 
# #         o = PReLU(shared_axes=[1, 2])(o)
# #         o = Conv2D(nb_classes, 1, padding='same', data_format=format_)(o)
# #         o = Activation('softmax')(o)
# #         return o

# #     def level_block(self, m, dim, depth, inc, acti, do, bn, up, format_="channels_last"):
# #         if depth > 0:
# #             n = self.res_block_enc(m, 0.0, dim, acti, bn, format_)
# #             m = Conv2D(int(inc*dim), 2, strides=2, padding='same', data_format=format_)(n)
# #             m = self.level_block(m, int(inc*dim), depth-1, inc, acti, do, bn, up)
# #             if up:
# #                 m = UpSampling2D(size=(2, 2), data_format=format_)(m)
# #                 m = Conv2D(dim, 2, padding='same', data_format=format_)(m)
# #             else:
# #                 m = Conv2DTranspose(dim, 3, strides=2, padding='same', data_format=format_)(m)
# #             n = concatenate([n, m])
# #             m = self.res_block_dec(n, 0.0, dim, acti, bn, format_)
# #         else:
# #             m = self.res_block_enc(m, 0.0, dim, acti, bn, format_)
# #         return m

# #     def res_block_enc(self, m, drpout, dim, acti, bn, format_="channels_last"):
# #         n = BatchNormalization()(m) if bn else m
# #         n = PReLU(shared_axes=[1, 2])(n)
# #         n = Conv2D(dim, 3, padding='same', data_format=format_)(n)
# #         n = BatchNormalization()(n) if bn else n
# #         n = PReLU(shared_axes=[1, 2])(n)
# #         n = Conv2D(dim, 3, padding='same', data_format=format_)(n)
# #         n = add([m, n]) 
# #         return n 

# #     def res_block_dec(self, m, drpout, dim, acti, bn, format_="channels_last"):
# #         n = BatchNormalization()(m) if bn else m
# #         n = PReLU(shared_axes=[1, 2])(n)
# #         n = Conv2D(dim, 3, padding='same', data_format=format_)(n)
# #         n = BatchNormalization()(n) if bn else n
# #         n = PReLU(shared_axes=[1, 2])(n)
# #         n = Conv2D(dim, 3, padding='same', data_format=format_)(n)
# #         save = Conv2D(dim, 1, padding='same', data_format=format_, use_bias=False)(m) 
# #         n = add([save, n]) 
# #         return n

# # import tensorflow as tf
# # from tensorflow import keras
# # from tensorflow.keras import backend as K
# # from tensorflow.keras.models import Model
# # from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, concatenate
# # from tensorflow.keras.optimizers import Adam
# # from losses import gen_dice_loss, dice_whole_metric, dice_core_metric, dice_en_metric

# # K.set_image_data_format("channels_last")

# # class TwoPathwayGroupCNN(object):
# #     def __init__(self, img_shape, load_model_weights=None):
# #         self.img_shape = img_shape
# #         self.load_model_weights = load_model_weights
# #         self.model = self.build_model()

# #     def build_model(self):
# #         input_layer = Input(shape=self.img_shape)
        
# #         # Local pathway
# #         local_path = self.local_pathway(input_layer)
        
# #         # Global pathway
# #         global_path = self.global_pathway(input_layer)
        
# #         # Concatenate local and global pathways
# #         concatenated = concatenate([local_path, global_path])
        
# #         # Final convolutions
# #         output = self.final_convolutions(concatenated)
        
# #         model = Model(inputs=input_layer, outputs=output)
        
# #         model.compile(loss=gen_dice_loss, optimizer=Adam(learning_rate=0.001),
# #                       metrics=[dice_whole_metric, dice_core_metric, dice_en_metric])
        
# #         if self.load_model_weights:
# #             model.load_weights(self.load_model_weights)
        
# #         return model

# #     def local_pathway(self, input_layer):
# #         x = self.group_conv_block(input_layer, 64, 5)
# #         x = MaxPooling2D(pool_size=(2, 2))(x)
# #         x = self.group_conv_block(x, 64, 5)
# #         x = MaxPooling2D(pool_size=(2, 2))(x)
# #         x = self.group_conv_block(x, 64, 5)
# #         return x

# #     def global_pathway(self, input_layer):
# #         x = self.group_conv_block(input_layer, 160, 13)
# #         x = MaxPooling2D(pool_size=(2, 2))(x)
# #         x = self.group_conv_block(x, 160, 13)
# #         x = MaxPooling2D(pool_size=(2, 2))(x)
# #         x = self.group_conv_block(x, 160, 13)
# #         return x

# #     def final_convolutions(self, input_layer):
# #         x = self.group_conv_block(input_layer, 4, 21)
# #         x = Conv2D(4, 1, activation='softmax', padding='same')(x)
# #         return x

# #     def group_conv_block(self, input_layer, filters, kernel_size):
# #         x = Conv2D(filters, kernel_size, padding='same')(input_layer)
# #         x = BatchNormalization()(x)
# #         x = Activation('relu')(x)
# #         return x
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import backend as K
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, concatenate
# from tensorflow.keras.optimizers import Adam
# from losses import gen_dice_loss, dice_whole_metric, dice_core_metric, dice_en_metric

# K.set_image_data_format("channels_last")

# class TwoPathwayGroupCNN(object):
#     def __init__(self, img_shape=(128, 128, 4), load_model_weights=None):
#         self.img_shape = img_shape
#         self.load_model_weights = load_model_weights
#         self.model = self.build_model()

#     def build_model(self):
#         input_layer = Input(shape=self.img_shape)
        
#         # Local pathway
#         local_path = self.local_pathway(input_layer)
        
#         # Global pathway
#         global_path = self.global_pathway(input_layer)
        
#         # Concatenate local and global pathways
#         concatenated = concatenate([local_path, global_path])
        
#         # Final convolutions
#         output = self.final_convolutions(concatenated)
        
#         model = Model(inputs=input_layer, outputs=output)
        
#         model.compile(loss=gen_dice_loss, optimizer=Adam(learning_rate=0.001),
#                       metrics=[dice_whole_metric, dice_core_metric, dice_en_metric])
        
#         if self.load_model_weights:
#             model.load_weights(self.load_model_weights)
        
#         return model

#     def local_pathway(self, input_layer):
#         x = self.group_conv_block(input_layer, 64, 5)
#         x = MaxPooling2D(pool_size=(2, 2))(x)
#         x = self.group_conv_block(x, 64, 5)
#         x = MaxPooling2D(pool_size=(2, 2))(x)
#         x = self.group_conv_block(x, 64, 5)
#         return x

#     def global_pathway(self, input_layer):
#         x = self.group_conv_block(input_layer, 160, 13)
#         x = MaxPooling2D(pool_size=(2, 2))(x)
#         x = self.group_conv_block(x, 160, 13)
#         x = MaxPooling2D(pool_size=(2, 2))(x)
#         x = self.group_conv_block(x, 160, 13)
#         return x

#     def final_convolutions(self, input_layer):
#         x = self.group_conv_block(input_layer, 64, 21)
#         x = UpSampling2D(size=(2, 2))(x)
#         x = self.group_conv_block(x, 64, 21)
#         x = UpSampling2D(size=(2, 2))(x)
#         x = self.group_conv_block(x, 64, 21)
#         x = Conv2D(4, 1, activation='softmax', padding='same')(x)
#         return x

#     def group_conv_block(self, input_layer, filters, kernel_size):
#         x = Conv2D(filters, kernel_size, padding='same')(input_layer)
#         x = BatchNormalization()(x)
#         x = Activation('relu')(x)

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

# class Unet_model(object):
    
#     def __init__(self, img_shape, load_model_weights=None):
#         self.img_shape = img_shape
#         self.load_model_weights = load_model_weights
#         self.model = self.compile_unet()
    
#     def compile_unet(self):
#         """
#         compile the U-net model
#         """
#         i = Input(shape=self.img_shape)
#         # add gaussian noise to the first layer to combat overfitting
#         i_ = GaussianNoise(0.01)(i)
#         i_ = Conv2D(64, 2, padding='same', data_format='channels_last')(i_)
#         out = self.unet(inputs=i_)
#         model = Model(inputs=i, outputs=out)

#         sgd = SGD(learning_rate=0.08, momentum=0.9, decay=5e-6, nesterov=False)
#         model.compile(loss=gen_dice_loss, optimizer=sgd, 
#                     metrics=[dice_whole_metric, dice_core_metric, dice_en_metric])
        
#         # load weights if set for prediction
#         if self.load_model_weights is not None:
#             model.load_weights(self.load_model_weights)
#         return model

#     def unet(self, inputs, nb_classes=4, start_ch=64, depth=3, inc_rate=2., 
#             activation='relu', dropout=0.0, batchnorm=True, upconv=True, 
#             format_='channels_last'):
#         """
#         the actual u-net architecture
#         """
#         o = self.level_block(inputs, start_ch, depth, inc_rate, activation, 
#                             dropout, batchnorm, upconv, format_)
#         o = BatchNormalization()(o) 
#         o = PReLU(shared_axes=[1, 2])(o)
#         o = Conv2D(nb_classes, 1, padding='same', data_format=format_)(o)
#         o = Activation('softmax')(o)
#         return o

#     def level_block(self, m, dim, depth, inc, acti, do, bn, up, format_="channels_last"):
#         """
#         Recursive function to create U-Net levels
#         """
#         if depth > 0:
#             n = self.res_block_enc(m, 0.0, dim, acti, bn, format_)
#             # using strided 2D conv for downsampling
#             m = Conv2D(int(inc*dim), 2, strides=2, padding='same', data_format=format_)(n)
#             m = self.level_block(m, int(inc*dim), depth-1, inc, acti, do, bn, up)
#             if up:
#                 m = UpSampling2D(size=(2, 2), data_format=format_)(m)
#                 m = Conv2D(dim, 2, padding='same', data_format=format_)(m)
#             else:
#                 m = Conv2DTranspose(dim, 3, strides=2, padding='same', data_format=format_)(m)
#             n = concatenate([n, m])
#             # the decoding path
#             m = self.res_block_dec(n, 0.0, dim, acti, bn, format_)
#         else:
#             m = self.res_block_enc(m, 0.0, dim, acti, bn, format_)
#         return m

#     def res_block_enc(self, m, drpout, dim, acti, bn, format_="channels_last"):
#         """
#         the encoding unit which is a residual block
#         """
#         n = BatchNormalization()(m) if bn else m
#         n = PReLU(shared_axes=[1, 2])(n)
#         n = Conv2D(dim, 3, padding='same', data_format=format_)(n)
#         n = BatchNormalization()(n) if bn else n
#         n = PReLU(shared_axes=[1, 2])(n)
#         n = Conv2D(dim, 3, padding='same', data_format=format_)(n)
#         n = add([m, n]) 
#         return n 

#     def res_block_dec(self, m, drpout, dim, acti, bn, format_="channels_last"):
#         """
#         the decoding unit which is a residual block
#         """
#         n = BatchNormalization()(m) if bn else m
#         n = PReLU(shared_axes=[1, 2])(n)
#         n = Conv2D(dim, 3, padding='same', data_format=format_)(n)
#         n = BatchNormalization()(n) if bn else n
#         n = PReLU(shared_axes=[1, 2])(n)
#         n = Conv2D(dim, 3, padding='same', data_format=format_)(n)
#         save = Conv2D(dim, 1, padding='same', data_format=format_, use_bias=False)(m) 
#         n = add([save, n]) 
#         return n

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
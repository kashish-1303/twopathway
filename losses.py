

import tensorflow as tf
import tensorflow.keras.backend as K

def weighted_log_loss(y_true, y_pred):
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    weights = tf.constant([1, 5, 2, 4], dtype=tf.float32)
    loss = y_true * K.log(y_pred) * weights
    return -K.mean(K.sum(loss, axis=-1))

def dice(y_true, y_pred):
    print("in dice")
    print(f"yt_true shape:{K.int_shape(y_true)},dtype:{y_true.dtype}")
    print(f"yt_pred shape:{K.int_shape(y_pred)},dtype:{y_pred.dtype}")
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    print(f"yt_true_f shape:{K.int_shape(y_true_f)},dtype:{y_true_f.dtype}")
    print(f"yt_pred_f shape:{K.int_shape(y_pred_f)},dtype:{y_pred_f.dtype}")
    intersection = K.sum(y_true_f * y_pred_f)
    smooth=1.0
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def gen_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    
    num_classes = K.int_shape(y_pred)[-1]
    dice_sum = 0
    for index in range(num_classes):
        dice_sum += dice(y_true[..., index], y_pred[..., index])
    
    dice_mean = dice_sum / num_classes
    return 1 - dice_mean + weighted_log_loss(y_true, y_pred)

def dice_whole_metric(y_true, y_pred):
    print(f"yt_true shape:{K.int_shape(y_true)},dtype:{y_true.dtype}")
    print(f"yt_pred shape:{K.int_shape(y_pred)},dtype:{y_pred.dtype}")
    y_true_f = K.flatten(K.sum(y_true, axis=-1))
    y_pred_f = K.flatten(K.sum(y_pred, axis=-1))
    print(f"yt_true_f shape:{K.int_shape(y_true_f)},dtype:{y_true_f.dtype}")
    print(f"yt_pred_f shape:{K.int_shape(y_pred_f)},dtype:{y_pred_f.dtype}")
    return dice(y_true_f, y_pred_f)

def dice_core_metric(y_true, y_pred):
    indices = tf.constant([1, 3])
    y_true_core = tf.gather(y_true, indices, axis=-1)
    y_pred_core = tf.gather(y_pred, indices, axis=-1)
    y_true_f = K.flatten(K.sum(y_true_core, axis=-1))
    y_pred_f = K.flatten(K.sum(y_pred_core, axis=-1))
    return dice(y_true_f, y_pred_f)

def dice_en_metric(y_true, y_pred):
    return dice(y_true[..., 3], y_pred[..., 3])


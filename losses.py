

#27th april
import tensorflow as tf
import tensorflow.keras.backend as K

def weighted_log_loss(y_true, y_pred):
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # Weights for different classes [background, edema, non-enhancing, enhancing]
    weights = tf.constant([1, 5, 2, 4], dtype=tf.float32)
    loss = y_true * K.log(y_pred) * weights
    return -K.mean(K.sum(loss, axis=-1))

def dice(y_true, y_pred):
    # Cast inputs to float32 to ensure type consistency
    y_true_f = K.flatten(K.cast(y_true, 'float32'))
    y_pred_f = K.flatten(K.cast(y_pred, 'float32'))
    intersection = K.sum(y_true_f * y_pred_f)
    smooth = 1.0
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
    # Ensure consistent types
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    
    # Convert to binary: tumor vs. non-tumor (classes 1,2,3)
    y_true_bin = K.cast(K.sum(y_true[..., 1:], axis=-1) > 0, 'float32')
    y_pred_bin = K.cast(K.sum(y_pred[..., 1:], axis=-1) > 0, 'float32')
    return dice(y_true_bin, y_pred_bin)

def dice_core_metric(y_true, y_pred):
    # Ensure consistent types
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    
    # Core consists of labels 1 and 3 (original 4)
    y_true_core = tf.gather(y_true, [1, 3], axis=-1)
    y_pred_core = tf.gather(y_pred, [1, 3], axis=-1)
    y_true_bin = K.cast(K.sum(y_true_core, axis=-1) > 0, 'float32')
    y_pred_bin = K.cast(K.sum(y_pred_core, axis=-1) > 0, 'float32') 
    return dice(y_true_bin, y_pred_bin)

def dice_en_metric(y_true, y_pred):
    # Ensure consistent types
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    
    return dice(y_true[..., 3], y_pred[..., 3])
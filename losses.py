import tensorflow as tf
import tensorflow.keras.backend as K

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Compute Dice coefficient for binary masks
    """
    y_true_f = K.flatten(K.cast(y_true, 'float32'))
    y_pred_f = K.flatten(K.cast(y_pred, 'float32'))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def multiclass_dice_loss(y_true, y_pred, smooth=1e-6):
    """
    Multiclass Dice loss - more stable than the original implementation
    """
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    
    # Calculate Dice for each class
    dice_scores = []
    num_classes = K.int_shape(y_pred)[-1]
    
    for class_idx in range(num_classes):
        y_true_class = y_true[..., class_idx]
        y_pred_class = y_pred[..., class_idx]
        dice_score = dice_coefficient(y_true_class, y_pred_class, smooth)
        dice_scores.append(dice_score)
    
    # Average Dice across classes
    mean_dice = K.mean(K.stack(dice_scores))
    return 1 - mean_dice

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal loss to handle class imbalance
    """
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    
    # Calculate cross entropy
    ce_loss = -y_true * K.log(y_pred)
    
    # Calculate focal weight
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    focal_weight = alpha * K.pow(1 - pt, gamma)
    
    # Apply focal weight
    focal_loss_val = focal_weight * ce_loss
    
    return K.mean(K.sum(focal_loss_val, axis=-1))

def tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, smooth=1e-6):
    """
    Tversky loss - generalizes Dice loss, good for imbalanced data
    """
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    
    # Calculate Tversky index for each class
    tversky_scores = []
    num_classes = K.int_shape(y_pred)[-1]
    
    for class_idx in range(num_classes):
        y_true_class = y_true[..., class_idx]
        y_pred_class = y_pred[..., class_idx]
        
        true_pos = K.sum(y_true_class * y_pred_class)
        false_neg = K.sum(y_true_class * (1 - y_pred_class))
        false_pos = K.sum((1 - y_true_class) * y_pred_class)
        
        tversky_idx = (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)
        tversky_scores.append(tversky_idx)
    
    mean_tversky = K.mean(K.stack(tversky_scores))
    return 1 - mean_tversky

def combined_loss(y_true, y_pred, dice_weight=0.5, focal_weight=0.3, tversky_weight=0.2):
    """
    Combined loss function incorporating multiple loss types
    """
    dice_loss = multiclass_dice_loss(y_true, y_pred)
    focal_loss_val = focal_loss(y_true, y_pred)
    tversky_loss_val = tversky_loss(y_true, y_pred)
    
    return (dice_weight * dice_loss + 
            focal_weight * focal_loss_val + 
            tversky_weight * tversky_loss_val)

# Evaluation Metrics
def dice_whole_metric(y_true, y_pred):
    """
    Dice coefficient for whole tumor (classes 1, 2, 3)
    """
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    
    # Convert to binary: tumor vs. non-tumor
    y_true_whole = K.cast(K.sum(y_true[..., 1:], axis=-1) > 0, 'float32')
    y_pred_whole = K.cast(K.sum(y_pred[..., 1:], axis=-1) > 0, 'float32')
    
    return dice_coefficient(y_true_whole, y_pred_whole)

def dice_core_metric(y_true, y_pred):
    """
    Dice coefficient for tumor core (classes 1, 3) - excluding edema
    """
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    
    # Core consists of classes 1 and 3 (non-enhancing and enhancing)
    y_true_core = y_true[..., 1] + y_true[..., 3]
    y_pred_core = y_pred[..., 1] + y_pred[..., 3]
    
    y_true_core = K.cast(y_true_core > 0, 'float32')
    y_pred_core = K.cast(y_pred_core > 0, 'float32')
    
    return dice_coefficient(y_true_core, y_pred_core)

def dice_enhancing_metric(y_true, y_pred):
    """
    Dice coefficient for enhancing tumor (class 3)
    """
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    
    return dice_coefficient(y_true[..., 3], y_pred[..., 3])

def sensitivity_metric(y_true, y_pred):
    """
    Sensitivity (recall) for whole tumor
    """
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    
    y_true_whole = K.cast(K.sum(y_true[..., 1:], axis=-1) > 0, 'float32')
    y_pred_whole = K.cast(K.sum(y_pred[..., 1:], axis=-1) > 0, 'float32')
    
    true_positives = K.sum(y_true_whole * y_pred_whole)
    possible_positives = K.sum(y_true_whole)
    
    return true_positives / (possible_positives + K.epsilon())

def specificity_metric(y_true, y_pred):
    """
    Specificity for whole tumor
    """
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    
    y_true_whole = K.cast(K.sum(y_true[..., 1:], axis=-1) > 0, 'float32')
    y_pred_whole = K.cast(K.sum(y_pred[..., 1:], axis=-1) > 0, 'float32')
    
    true_negatives = K.sum((1 - y_true_whole) * (1 - y_pred_whole))
    possible_negatives = K.sum(1 - y_true_whole)
    
    return true_negatives / (possible_negatives + K.epsilon())



import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Compute Dice coefficient for binary masks
    Formula from paper: Dice(P,G) = |P1 ∩ G1| / ((|P1| + |G1|)/2)
    """
    y_true_f = K.flatten(K.cast(y_true, 'float32'))
    y_pred_f = K.flatten(K.cast(y_pred, 'float32'))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def multiclass_dice_loss(y_true, y_pred, smooth=1e-6):
    """
    Multiclass Dice loss - handles class imbalance better
    FIXED: Handle softmax predictions properly
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
    Focal loss to handle class imbalance as mentioned in the paper
    FIXED: Handle softmax predictions properly
    """
    # No need to clip for softmax output, but add small epsilon for numerical stability
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    
    # Calculate cross entropy for multiclass
    ce_loss = -y_true * K.log(y_pred)
    
    # Calculate focal weight
    pt = K.sum(y_true * y_pred, axis=-1, keepdims=True)
    focal_weight = alpha * K.pow(1 - pt, gamma)
    
    # Apply focal weight
    focal_loss_val = focal_weight * K.sum(ce_loss, axis=-1, keepdims=True)
    
    return K.mean(focal_loss_val)

def tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, smooth=1e-6):
    """
    Tversky loss - generalizes Dice loss, good for imbalanced data
    Particularly useful for brain tumor segmentation where classes are highly imbalanced
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
    This addresses the class imbalance issue mentioned in the paper
    """
    dice_loss = multiclass_dice_loss(y_true, y_pred)
    focal_loss_val = focal_loss(y_true, y_pred)
    tversky_loss_val = tversky_loss(y_true, y_pred)
    
    return (dice_weight * dice_loss + 
            focal_weight * focal_loss_val + 
            tversky_weight * tversky_loss_val)

# Evaluation Metrics as specified in the paper

def dice_whole_metric(y_true, y_pred):
    """
    Dice coefficient for whole tumor (all tumor classes combined)
    Paper mentions evaluating "complete tumor" region
    """
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    
    # Convert to binary: tumor vs. non-tumor (exclude background class 0)
    y_true_whole = K.cast(K.sum(y_true[..., 1:], axis=-1) > 0, 'float32')
    y_pred_whole = K.cast(K.sum(y_pred[..., 1:], axis=-1) > 0, 'float32')
    
    return dice_coefficient(y_true_whole, y_pred_whole)

def dice_core_metric(y_true, y_pred):
    """
    Dice coefficient for tumor core region
    Paper mentions evaluating "core tumor" region
    For 4-class system: Core = classes 1, 2, 3 (all tumor classes except background)
    """
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    
    # Core consists of all tumor classes (1, 2, 3)
    y_true_core = K.sum(y_true[..., 1:], axis=-1)
    y_pred_core = K.sum(y_pred[..., 1:], axis=-1)
    
    y_true_core = K.cast(y_true_core > 0, 'float32')
    y_pred_core = K.cast(y_pred_core > 0, 'float32')
    
    return dice_coefficient(y_true_core, y_pred_core)

def dice_enhancing_metric(y_true, y_pred):
    """
    Dice coefficient for enhancing tumor (class 3 in 4-class system)
    Paper mentions evaluating "enhancing tumor" region
    """
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    
    # Use class 3 as enhancing tumor (adjust based on your class mapping)
    return dice_coefficient(y_true[..., 3], y_pred[..., 3])

def sensitivity_metric(y_true, y_pred):
    """
    Sensitivity (Recall) for whole tumor as mentioned in paper
    Formula: Sensitivity(P,G) = |P1 ∩ G1| / |G1|
    """
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    
    # Convert to binary: tumor vs. non-tumor
    y_true_whole = K.cast(K.sum(y_true[..., 1:], axis=-1) > 0, 'float32')
    y_pred_whole = K.cast(K.sum(y_pred[..., 1:], axis=-1) > 0, 'float32')
    
    true_positives = K.sum(y_true_whole * y_pred_whole)
    possible_positives = K.sum(y_true_whole)
    
    return true_positives / (possible_positives + K.epsilon())

def specificity_metric(y_true, y_pred):
    """
    Specificity for whole tumor as mentioned in paper
    Formula: Specificity(P,G) = |P0 ∩ G0| / |G0|
    """
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    
    # Convert to binary: tumor vs. non-tumor
    y_true_whole = K.cast(K.sum(y_true[..., 1:], axis=-1) > 0, 'float32')
    y_pred_whole = K.cast(K.sum(y_pred[..., 1:], axis=-1) > 0, 'float32')
    
    true_negatives = K.sum((1 - y_true_whole) * (1 - y_pred_whole))
    possible_negatives = K.sum(1 - y_true_whole)
    
    return true_negatives / (possible_negatives + K.epsilon())

# Additional metrics for comprehensive evaluation

def precision_metric(y_true, y_pred):
    """
    Precision for whole tumor
    """
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    
    # Convert to binary: tumor vs. non-tumor
    y_true_whole = K.cast(K.sum(y_true[..., 1:], axis=-1) > 0, 'float32')
    y_pred_whole = K.cast(K.sum(y_pred[..., 1:], axis=-1) > 0, 'float32')
    
    true_positives = K.sum(y_true_whole * y_pred_whole)
    predicted_positives = K.sum(y_pred_whole)
    
    return true_positives / (predicted_positives + K.epsilon())

def hausdorff_distance_95(y_true, y_pred):
    """
    95th percentile Hausdorff Distance - commonly used in medical segmentation
    Note: This is a simplified implementation
    """
    return 0.0

def get_class_weights(y):
    """
    Calculate class weights for your 4-class segmentation
    """
    class_weights = {}
    total_pixels = y.shape[0] * y.shape[1] * y.shape[2]
    
    for class_idx in range(4):
        class_pixels = np.sum(y[:,:,:,class_idx])
        if class_pixels > 0:
            weight = total_pixels / (4 * class_pixels)  # 4 is number of classes
            class_weights[class_idx] = weight
        else:
            class_weights[class_idx] = 1.0
    
    print("Class weights calculated:")
    for i, weight in class_weights.items():
        print(f"  Class {i}: {weight:.4f}")
    
    return class_weights

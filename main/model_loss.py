from keras import backend as K
import parameter as para
import helper


def total_loss(y_true, y_pred):
    """
    squeezeDet loss function for object detection and classification
    :param y_true: ground truth with shape [batchsize, #anchors, inputmasks+boxes+deltas+labels]
    :param y_pred: predicted result with shape [batchsize, #anchors, classes+4+1+padded]
    :return: a tensor of the total loss
    """
    cls_loss = class_loss(y_true, y_pred)
    box_loss = bbox_loss(y_true, y_pred)
    cf_loss = conf_loss(y_true, y_pred)
    
    # add above losses
    total_loss = cls_loss + cf_loss + box_loss
    return total_loss

# Sublosses
def bbox_loss(y_true, y_pred):
    """
    squeezeDet loss function for object detection and classification
    :param y_true: ground truth with shape [batchsize, #anchors, inputmasks+boxes+deltas+labels]
    :param y_pred: predicted result with shape [batchsize, #anchors, classes+4+1+padded]
    :return: bbox loss
    """            
    # Split y_true
    input_masks = y_true[:, :, 0]
    input_masks = K.expand_dims(input_masks, axis=-1)
    box_deltas = y_true[:, :, 5:9]
    # Used to normalize bbox and class loss
    num_objects = K.sum(input_masks)
    
    # Before computing the losses we need to get the unpadded and splited network outputs
    pred_class_probs, pred_conf, pred_box_delta = helper.split_predictions(y_pred)

    bbox_loss = (K.sum(5.0 * K.square(input_masks * (pred_box_delta - box_deltas))) / num_objects)
    return bbox_loss

def conf_loss(y_true, y_pred):
    """
    squeezeDet loss function for object detection and classification
    :param y_true: ground truth with shape [batchsize, #anchors, inputmasks+boxes+deltas+labels]
    :param y_pred: predicted result with shape [batchsize, #anchors, classes+4+1+padded]
    :return: conf loss
    """
    # Split y_true
    input_masks = y_true[:, :, 0]
    input_masks = K.expand_dims(input_masks, axis=-1)
    boxes = y_true[:, :, 1:5]
    # Used to normalize bbox and class loss
    num_objects = K.sum(input_masks)
    # Before computing the losses we need to get the unpadded and splited network outputs
    pred_class_probs, pred_conf, pred_box_delta = helper.split_predictions(y_pred)

    # Get the bounding boxes
    det_boxes = helper.convert_deltas_to_bboxes(pred_box_delta)

    unstacked_boxes_pred = []
    unstacked_boxes_input = []
    for i in range(4):
        unstacked_boxes_pred.append(det_boxes[:, :, i])
        unstacked_boxes_input.append(boxes[:, :, i])
    # Get the ious
    ious = helper.tensor_iou(helper.convert_bbox_diagonal(unstacked_boxes_pred), helper.convert_bbox_diagonal(unstacked_boxes_input),input_masks)

    input_masks = K.reshape(input_masks, [para.BATCH_SIZE, para.ANCHORS])
    conf_loss = K.mean(K.sum(K.square((ious - pred_conf)) * (input_masks * 75.0 / num_objects + (1 - input_masks) * 100.0 / (para.ANCHORS - num_objects)),axis=[1]))
    return conf_loss

def class_loss(y_true, y_pred):
    """
    squeezeDet loss function for object detection and classification
    :param y_true: ground truth with shape [batchsize, #anchors, inputmasks+boxes+deltas+labels]
    :param y_pred: predicted result with shape [batchsize, #anchors, classes+4+1+padded]
    :return: class loss
    """
    # Split y_true
    input_masks = y_true[:, :, 0]
    input_masks = K.expand_dims(input_masks, axis=-1)
    labels = y_true[:, :, 9:]
    # Used to normalize bbox and class loss
    num_objects = K.sum(input_masks)
    # Before computing the losses we need to get the unpadded and splited network outputs
    pred_class_probs, pred_conf, pred_box_delta = helper.split_predictions(y_pred)
    # Cross-entropy: q * -log(p) + (1-q) * -log(1-p)
    # Add a small value into log to prevent NAN loss
    class_loss = K.sum((labels * (-K.log(pred_class_probs + 1e-7)) + (1 - labels) * (-K.log(1 - pred_class_probs + 1e-7)))* input_masks) / num_objects
    return class_loss

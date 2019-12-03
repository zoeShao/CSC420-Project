
import numpy as np
from tensorflow import keras
from keras import backend as K
import parameter as para


def batch_iou(boxes, box):
    """
    Compute the Intersection-Over-Union of a batch of boxes with another box.
    :param boxes: 2D array of [cx, cy, width, height].
    :param box: a single array of [cx, cy, width, height]
    :returns: array of ious
    """
    int_xlr = np.maximum(
        np.minimum(boxes[:,0]+0.5*boxes[:,2], box[0]+0.5*box[2]) - \
        np.maximum(boxes[:,0]-0.5*boxes[:,2], box[0]-0.5*box[2]), 0)
    int_ytb = np.maximum(
        np.minimum(boxes[:,1]+0.5*boxes[:,3], box[1]+0.5*box[3]) - \
        np.maximum(boxes[:,1]-0.5*boxes[:,3], box[1]-0.5*box[3]), 0)
    intersection = int_xlr * int_ytb
    area1, area2 = boxes[:,2]*boxes[:,3], box[2]*box[3]
    union =  area1 + area2 - intersection
    return intersection/union

def tensor_iou(box1, box2, input_mask):
    """Computes pairwise IOU of two lists of boxes
    :param box1: array of [xmin, ymin, xmax, ymax] format
    :param box2: array of [xmin, ymin, xmax, ymax] format
    :param input_mask: indicating which boxes to compute
    :returns: iou of the two boxes
    """
    xl = K.maximum(box1[0], box2[0])
    xr = K.minimum(box1[2], box2[2])
    yb = K.maximum(box1[1], box2[1])
    yt = K.minimum(box1[3], box2[3])
    intersection = K.maximum(0.0, xr - xl) * K.maximum(0.0, yt - yb)
    w1 = box1[2] - box1[0]
    h1 = box1[3] - box1[1]
    w2 = box2[2] - box2[0]
    h2 = box2[3] - box2[1]
    area1, area2 = w1 * h1, w2 * h2
    union = area1 + area2 - intersection
    return intersection / (union + 1e-7) * K.reshape(input_mask, [para.BATCH_SIZE, para.ANCHORS]) # To prevent zero division

def non_maximum_suppression(boxes, probs, thresh):
    """
    Non-Maximum supression.
    :param boxes: array of [cx, cy, w, h] format
    :param probs: array of probabilities
    :param thresh: threshold to decide overlapping
    :returns: array of True or False.
    """
    order_desc = probs.argsort()[::-1]
    res = [True]*len(order_desc)
    for i in range(len(order_desc)-1):
      overlaps = batch_iou(boxes[order_desc[i+1:]], boxes[order_desc[i]])
      for j, overlap in enumerate(overlaps):
        if overlap > thresh:
          res[order_desc[j+i+1]] = False
    return res

def sparse_to_dense(sp_indices, output_shape, values, default_value=0):
    """
    Build a dense matrix from sparse representations.
    """
    assert len(sp_indices) == len(values), \
        'Length of sp_indices is not equal to length of values'
    array = np.ones(output_shape) * default_value
    for idx, value in zip(sp_indices, values):
      array[tuple(idx)] = value
    return array

def convert_bbox_diagonal(bbox):
    """
    Convert a bounding box of form [cx, cy, w, h] to the form of 
    [xmin, ymin, xmax, ymax]. 
    """
    cx, cy, w, h = bbox
    res_box = [[]]*4
    res_box[0] = cx - w/2
    res_box[1] = cy - h/2
    res_box[2] = cx + w/2
    res_box[3] = cy + h/2
    return res_box

def convert_bbox_center(bbox):
    """
    Convert a bbox of form [xmin, ymin, xmax, ymax] to the form of 
    [cx, cy, w, h]. 
    """
    xmin, ymin, xmax, ymax = bbox
    res_box = [[]]*4
    width = xmax - xmin + 1.0
    height = ymax - ymin + 1.0
    res_box[0]  = xmin + 0.5*width
    res_box[1]  = ymin + 0.5*height
    res_box[2]  = width
    res_box[3]  = height
    return res_box

def convert_deltas_to_bboxes(pred_box_delta):
    """
    Converts prediction deltas to bounding boxes
    :param pred_box_delta: tensor of deltas
    :returns: tensor of bounding boxes
    """

    def exp_helper_tensor(w, thresh=1.0):
        """ Safe exponential function. """
        slope = np.exp(thresh)
        lin_bool = w > thresh
        lin_region = K.cast(lin_bool, dtype='float32')
        lin_out = slope*(w - thresh + 1.)
        exp_out = K.exp(K.switch(lin_bool, K.zeros_like(w), w))
        out = lin_region*lin_out + (1.-lin_region)*exp_out
        return out

    # Get the coordinates and sizes of the anchor boxes and deltas 
    anchor_x = para.ANCHOR_BOX[:, 0]
    anchor_y = para.ANCHOR_BOX[:, 1]
    anchor_w = para.ANCHOR_BOX[:, 2]
    anchor_h = para.ANCHOR_BOX[:, 3]
    delta_x = pred_box_delta[:, :, 0]
    delta_y = pred_box_delta[:, :, 1]
    delta_w = pred_box_delta[:, :, 2]
    delta_h = pred_box_delta[:, :, 3]
    # Get the anchor box values
    box_cx = K.identity(anchor_x + delta_x * anchor_w)
    box_cy = K.identity(anchor_y + delta_y * anchor_h)
    box_w = K.identity(anchor_w * exp_helper_tensor(delta_w))
    box_h = K.identity(anchor_h * exp_helper_tensor(delta_h))
    # Tranform into a [xmin, ymin, xmax, ymax] format
    xmins, ymins, xmaxs, ymaxs = convert_bbox_diagonal([box_cx, box_cy, box_w, box_h])
    xmins = K.minimum(K.maximum(0.0, xmins), para.IMAGE_WIDTH - 1.0)
    ymins = K.minimum(K.maximum(0.0, ymins), para.IMAGE_HEIGHT - 1.0)
    xmaxs = K.maximum(K.minimum(para.IMAGE_WIDTH - 1.0, xmaxs), 0.0)
    ymaxs = K.maximum(K.minimum(para.IMAGE_HEIGHT - 1.0, ymaxs), 0.0)
    det_boxes = K.permute_dimensions(K.stack(convert_bbox_center([xmins, ymins, xmaxs, ymaxs])), (1, 2, 0))
    return (det_boxes)

def split_predictions(y_pred):
    """
    :param y_pred: network prediction output
    :return: unpadded and sliced predictions
    """
    # Compute non padded prediction part
    n_outputs = para.NUM_CLASSES + 1 + 4
    y_pred = y_pred[:, :, 0:n_outputs]
    y_pred = K.reshape(y_pred, (para.BATCH_SIZE, para.NUM_VERTICAL_ANCHORS, para.NUM_HORIZ_ANCHORS, -1))
    # Get the index, need to compute the number of total class probabilities(n classes for each anchor)
    idx_class_probs = para.ANCHOR_PER_GRID * para.NUM_CLASSES
    # Get class pred scores
    class_probs = K.reshape(y_pred[:, :, :, :idx_class_probs], [-1, para.NUM_CLASSES])
    class_probs = K.softmax(class_probs)
    class_probs = K.reshape(class_probs, [para.BATCH_SIZE, para.ANCHORS, para.NUM_CLASSES])
    # Get the index, neet to compute the number of confidence scores(1 score for each anchor)
    idx_confidence_scores = para.ANCHOR_PER_GRID * 1 + idx_class_probs
    # Get the confidence scores (sigmoid for probabilities)
    conf_score = K.reshape(y_pred[:, :, :, idx_class_probs:idx_confidence_scores], [para.BATCH_SIZE, para.ANCHORS])
    conf_score = K.sigmoid(conf_score)
    # Get the bounding box deltas
    box_delta = K.reshape(y_pred[:, :, :, idx_confidence_scores:], [para.BATCH_SIZE, para.ANCHORS, 4])
    return [class_probs, conf_score, box_delta]

def convert_deltas_to_bboxes_np(pred_box_delta):
    """
    Converts prediction deltas to bounding boxes
    :param pred_box_delta: tensor of deltas
    :returns: tensor of bounding boxes
    """

    def exp_helper_np(w, thresh=1.0):
        """ Safe exponential function for numpy. """
        slope = np.exp(thresh)
        lin_bool = w > thresh
        lin_region = lin_bool.astype(float)
        lin_out = slope*(w - thresh + 1.)
        exp_out = np.exp(np.where(lin_bool, np.zeros_like(w), w))
        out = lin_region*lin_out + (1.-lin_region)*exp_out
        return out

    # Get the coordinates and sizes of the anchor boxes and deltas
    anchor_x = para.ANCHOR_BOX[:, 0]
    anchor_y = para.ANCHOR_BOX[:, 1]
    anchor_w = para.ANCHOR_BOX[:, 2]
    anchor_h = para.ANCHOR_BOX[:, 3]
    delta_x = pred_box_delta[:, :, 0]
    delta_y = pred_box_delta[:, :, 1]
    delta_w = pred_box_delta[:, :, 2]
    delta_h = pred_box_delta[:, :, 3]
    # Get the anchor box values
    box_cx = anchor_x + delta_x * anchor_w
    box_cy = anchor_y + delta_y * anchor_h
    box_w = anchor_w * exp_helper_np(delta_w)
    box_h = anchor_h * exp_helper_np(delta_h)
    # Tranform into a [xmin, ymin, xmax, ymax] format
    xmins, ymins, xmaxs, ymaxs = convert_bbox_diagonal([box_cx, box_cy, box_w, box_h])
    xmins = np.minimum(np.maximum(0.0, xmins), para.IMAGE_WIDTH - 1.0)
    ymins = np.minimum(np.maximum(0.0, ymins), para.IMAGE_HEIGHT - 1.0)
    xmaxs = np.maximum(np.minimum(para.IMAGE_WIDTH - 1.0, xmaxs), 0.0)
    ymaxs = np.maximum(np.minimum(para.IMAGE_HEIGHT - 1.0, ymaxs), 0.0)
    det_boxes = np.transpose(np.stack(convert_bbox_center([xmins, ymins, xmaxs, ymaxs])), (1, 2, 0))
    return (det_boxes)

def split_predictions_np(y_pred):
    """
    :param y_pred: network prediction output
    :return: unpadded and sliced predictions
    """
    # Compute non padded prediction part
    n_outputs = para.NUM_CLASSES + 1 + 4
    y_pred = y_pred[:, :, 0:n_outputs]
    y_pred = np.reshape(y_pred, (para.BATCH_SIZE, para.NUM_VERTICAL_ANCHORS, para.NUM_HORIZ_ANCHORS, -1))
    # Get the index, need to compute the number of total class probabilities(n classes for each anchor)
    idx_class_probs = para.ANCHOR_PER_GRID * para.NUM_CLASSES
    # Get class pred scores
    class_probs = np.reshape(y_pred[:, :, :, :idx_class_probs], [-1, para.NUM_CLASSES])
    class_probs = softmax(class_probs)
    class_probs = np.reshape(class_probs, [para.BATCH_SIZE, para.ANCHORS, para.NUM_CLASSES])
    # Get the index, neet to compute the number of confidence scores(1 score for each anchor)
    idx_confidence_scores = para.ANCHOR_PER_GRID * 1 + idx_class_probs
    # Get the confidence scores (sigmoid for probabilities)
    conf_score = np.reshape(y_pred[:, :, :, idx_class_probs:idx_confidence_scores], [para.BATCH_SIZE, para.ANCHORS])
    conf_score = sigmoid(conf_score)
    # Get the bounding box deltas
    box_delta = np.reshape(y_pred[:, :, :, idx_confidence_scores:], [para.BATCH_SIZE, para.ANCHORS, 4])
    return [class_probs, conf_score, box_delta]

def softmax(x, axis=-1):
    """ Compute softmax values. """
    e_x = np.exp(x - np.max(x))
    return e_x / np.expand_dims(np.sum(e_x,axis=axis), axis=axis)

def sigmoid(x):
    """ Sigmoid function """
    return 1 / (1 + np.exp(-x))

def get_annotation(ytrue_file):
    """ Get annotations from file. """
    with open(ytrue_file, 'r') as f:
        lines = f.readlines()
    f.close()
    annotations = []
    # Get annotation bounding box for each line
    for line in lines:
        item = line.strip().split(' ')
        # Only get the class we care
        try:
            cls = para.CLASS_TO_IDX[item[0].lower().strip()]
            # Get bounding box
            xmin = float(item[4])
            ymin = float(item[5])
            xmax = float(item[6])
            ymax = float(item[7])
            # Transform to [cx, cy, w, h] format
            cx, cy, w, h = convert_bbox_center([xmin, ymin, xmax, ymax])
            annotations.append([cx, cy, w, h, cls])
        except:
            continue
    return annotations
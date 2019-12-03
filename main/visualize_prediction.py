import cv2
import numpy as np
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
import parameter as para
import helper
from SqueezeDet import SqueezeDet
import data_generators as dg

def filter_result(y_pred):
    """
    Return list of all filtered boxes, list of all filtered classes,
    and list of all filtered scores.
    """
    filtered_boxes = []
    filtered_classes = []
    filtered_scores = []
    detection_num = 0
    # Get the none padded part of prediction
    pred_class_probs, pred_conf, pred_box_delta = helper.split_predictions_np(y_pred)
    # Transform deltas to detected bounding boxes
    det_boxes = helper.convert_deltas_to_bboxes_np(pred_box_delta)
    # Get the probabilities for the class
    probs = pred_class_probs * np.reshape(pred_conf, [para.BATCH_SIZE, para.ANCHORS, 1])
    det_probs = np.max(probs, 2)
    det_class = np.argmax(probs, 2)
    for i in range(para.BATCH_SIZE):
        # Filter with non_maximum_suppression
        filtered_bbox, filtered_score, filtered_class = filter_each(det_boxes[i], det_probs[i], det_class[i])
        keep_idices = [idx for idx in range(len(filtered_score)) if filtered_score[idx] > float(para.SHOW_RESULT_THRESH)]
        final_boxes = [filtered_bbox[idx] for idx in keep_idices]
        filtered_boxes.append(final_boxes)
        final_class = [filtered_class[idx] for idx in keep_idices]
        filtered_classes.append(final_class)
        final_probs = [filtered_score[idx] for idx in keep_idices]
        filtered_scores.append(final_probs)
        detection_num += len(filtered_bbox)
    return filtered_boxes, filtered_classes, filtered_scores

def filter_each(boxes, probs, cls_idx, top_n=64, prob_thres=0.005):
    """Filter predicted bounding box with probability threshold and
    non-maximum supression.
    :param boxes: array of [cx, cy, w, h] format
    :param probs: array of probabilities
    :param cls_idx: array of class indices
    :top_n: integer indicating top n detection
    :probe_thres: float of probability threshold
    :returns: array of filtered bounding boxes, array of filtered probabilities,
    and array of filtered class indices
    """
    filtered_bbox = []
    filtered_score = []
    filtered_cls = []

    if top_n < len(probs) and top_n > 0:
      order_desc = probs.argsort()[:-top_n-1:-1]
      boxes = boxes[order_desc]
      probs = probs[order_desc]
      cls_idx = cls_idx[order_desc]
    else:
      filtered_idx = np.nonzero(probs>prob_thres)[0]
      boxes = boxes[filtered_idx]
      probs = probs[filtered_idx] 
      cls_idx = cls_idx[filtered_idx]

    for cls in range(para.NUM_CLASSES):
      idx_per_cls = [idx for idx in range(len(probs)) if cls_idx[idx] == cls]
      # Perform Non-maximum suppresion
      keep = helper.non_maximum_suppression(boxes[idx_per_cls], probs[idx_per_cls], thresh=0.4)
      for i in range(len(keep)):
        if keep[i]:
          filtered_bbox.append(boxes[idx_per_cls[i]])
          filtered_score.append(probs[idx_per_cls[i]])
          filtered_cls.append(cls)
    return filtered_bbox, filtered_score, filtered_cls


def visualize_pr_and_ytrue(model, generator):
    """ 
    Generates images with ground truth bboxes and predicted bboxes.
    :param model: model of SqueezeDet
    :param generator: data generator that yields images and ground truth
    :returns: numpy array of images with ground truth and prediction boxes
    """
    print("  Getting Visualizations, please wait...")
    res_imgs = []
    for i in range(len(generator)-1):
        images, y_true, orig_imgs = generator.__getitem__(i)
        y_pred = model.predict(images)
        # Create visualizations
        imgs_with_boxes = add_pr_and_ytrue_boxes(orig_imgs, y_true, y_pred)
        try:
            res_imgs.append(np.stack(imgs_with_boxes))
        except:
            pass
    try:
        return np.stack(res_imgs).reshape((-1, para.IMAGE_HEIGHT, para.IMAGE_WIDTH, 3))
    except:
        return np.zeros((para.BATCH_SIZE*(len(generator)-1), para.IMAGE_HEIGHT, para.IMAGE_WIDTH, 3))


def add_pr_and_ytrue_boxes(images, y_true, y_pred):
    """
    Takes a batch of images and creates bounding box visualization.
    """
    img_with_boxes = []
    # Get ground truth boxes and labels
    boxes = y_true[:, :, 1:5]
    labels = y_true[:, :, 9:]
    filtered_boxes, filtered_classes, filtered_scores = filter_result(y_pred)
    for i, img in enumerate(images):
        # Get the ytrue boxes
        ytrue_boxes = boxes[i][boxes[i] > 0].reshape((-1,4))
        # Get the ytrue labels
        ytrue_labels = []
        for j, coords in enumerate(boxes[i,:]):
            if np.sum(coords) > 0:
                for k, label in enumerate(labels[i, j]):
                    if label == 1:
                        ytrue_labels.append(k)
        # For predicted part:
        for j, det_box in enumerate(filtered_boxes[i]):
            # Transform into xmin, ymin, xmax, ymax format
            det_box = helper.convert_bbox_diagonal(det_box)
            class_string = para.CLASS_NAMES[filtered_classes[i][j]]
            prob_string = "%.2f" % filtered_scores[i][j]
            # Draw bounding box and add text
            cv2.rectangle(img, (int(det_box[0]), int(det_box[3])), (int(det_box[2]), int(det_box[1])), para.CLASS_TO_COLOR[class_string], 2)
            if filtered_scores[i][j] is not None:
                # Write probability of detection on top line of the bbox:
                cv2.rectangle(img, (int(det_box[0]), int(det_box[1])), (int(det_box[2]), int(det_box[1]-12)),
                            para.CLASS_TO_COLOR[class_string], -1)
                cv2.putText(img, class_string[:2] + ":" + prob_string, (int(det_box[0])+2, int(det_box[1])-2), 2, 0.4,
                            (255,255,255), 0)
        # For ground truth part:
        for j, ytrue_box in enumerate(ytrue_boxes):
            # Transform into xmin, ymin, xmax, ymax format
            ytrue_box = helper.convert_bbox_diagonal(ytrue_box)
            # Draw bounding box and add text
            cv2.rectangle(img, (int(ytrue_box[0]), int(ytrue_box[3])), (int(ytrue_box[2]), int(ytrue_box[1])), (0, 255, 0), 2)
            cv2.putText(img, para.CLASS_NAMES[int(ytrue_labels[j])], (int(ytrue_box[0]), int(ytrue_box[1])), para.FONT, 0.5,
                        (0, 255, 0), 1, cv2.LINE_AA)
        img_with_boxes.append(img[:,:, [2,1,0]])
    return img_with_boxes

if __name__ == "__main__":
    img_file = "../kitti_data/img_train.txt"
    ytrue_file = "../kitti_data/ytrue_train.txt"
    val_file = "../kitti_data/img_val.txt"
    val_ytrue_file = "../kitti_data/ytrue_val.txt"
    checkpoint_dir = "../kitti_data/checkpoints"
    
    # open files with images and ground truths files with full path names
    with open(img_file) as imgs:
        img_names = imgs.read().splitlines()
    imgs.close()
    with open(ytrue_file) as ytrues:
        ytrue_names = ytrues.read().splitlines()
    ytrues.close()
    with open(val_file) as vals:
        val_names = vals.read().splitlines()
    vals.close()
    with open(val_ytrue_file) as val_ytrues:
        val_ytrue_names = val_ytrues.read().splitlines()
    val_ytrues.close()
    test_names = val_names[32:65]
    test_ytrue_names = val_ytrue_names[32:65]
    val_names = val_names[0:32]
    val_ytrue_names = val_ytrue_names[0:32]
    
    # Initiate a model
    squeeze = SqueezeDet()

    squeeze.load_weights(checkpoint_dir + "/squeeze_ep140.h5")
    vis_generator = dg.DataGenOrigin(test_names, test_ytrue_names, batch_size=para.BATCH_SIZE, img_height=384, img_width=1248)
    imgs = visualize_pr_and_ytrue(model=squeeze, generator=vis_generator)
    
    
    # show result
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 16
    fig_size[1] = 14
    plt.rcParams["figure.figsize"] = fig_size
    idx = 9
    img = imgs[idx] / np.max(imgs[idx])
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    plt.imshow(img)
    plt.show()
    # imsave('predict ' + str(idx) + '.png', img)
    
    idx = 8
    img = imgs[idx] / np.max(imgs[idx])
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    plt.imshow(img)
    plt.show()
    # imsave('predict ' + str(idx) + '.png', img)
    
    # failed example
    idx = 22
    img = imgs[idx] / np.max(imgs[idx])
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    plt.imshow(img)
    plt.show()
    # imsave('predict ' + str(idx) + '.png', img)
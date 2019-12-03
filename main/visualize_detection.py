import cv2
import numpy as np
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm

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

# for Patch matching
pm_bboxes = []

def visualize_pr(model, generator):
    """
    Generates images with predicted bboxes.
    :param model: model of SqueezeDet
    :param generator: data generator that yields images and ground truth
    :returns: numpy array of images with prediction boxes
    """
    print("  Getting Visualizations, please wait...")
    res_imgs = []
    for i in range(len(generator)-1):
        images, orig_imgs = generator.__getitem__(i)
        y_pred = model.predict(images)
        # Create visualizations
        imgs_with_boxes = add_pr_boxes(orig_imgs, y_pred)
        try:
            res_imgs.append(np.stack(imgs_with_boxes))
        except:
            pass
    try:
        np.savez('../kitti_data/pm_bboxes.npz', pm_bboxes)
        return np.stack(res_imgs).reshape((-1, para.IMAGE_HEIGHT, para.IMAGE_WIDTH, 3))
    except:
        np.savez('../kitti_data/pm_bboxes.npz', pm_bboxes)
        return np.zeros((para.BATCH_SIZE*(len(generator)-1), para.IMAGE_HEIGHT, para.IMAGE_WIDTH, 3))


def add_pr_boxes(images, y_pred):
    """
    Takes a batch of images and creates bounding box visualization.
    """
    img_with_boxes = []
    filtered_boxes, filtered_classes, filtered_scores = filter_result(y_pred)

    for i, img in enumerate(images):
        pm_bboxes_per_image = []
        for j, det_box in enumerate(filtered_boxes[i]):
            # Transform into xmin, ymin, xmax, ymax format
            det_box = helper.convert_bbox_diagonal(det_box)

            # Added the detected boudning box for patch matching 
            pm_bboxes_per_image.append(det_box)
            
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
        pm_bboxes.append(pm_bboxes_per_image)
        img_with_boxes.append(img[:,:, [2,1,0]])
    return img_with_boxes


if __name__ == "__main__":
    checkpoint_dir = "../kitti_data/checkpoints"
    img_video_file = '../kitti_data/images_video.txt'
    with open(img_video_file) as video_imgs:
        img_video_names = video_imgs.read().splitlines()
    video_imgs.close()
    
    # Initiate a model
    squeeze = SqueezeDet()

    squeeze.load_weights(checkpoint_dir + "/squeeze_ep140.h5")
    vis_generator = dg.DataGenVideo(img_video_names, batch_size=para.BATCH_SIZE, img_height=384, img_width=1248)
    detect_imgs = visualize_pr(model=squeeze, generator=vis_generator)
    
    # show result
    idx = 37
    img = detect_imgs[idx] / np.max(detect_imgs[idx])
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    plt.imshow(img)
    plt.show()
    
    idx = 374
    img = detect_imgs[idx] / np.max(detect_imgs[idx])
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    plt.imshow(img)
    plt.show()
    
    # Failed example
    # Detect a trash can as a car
    idx = 162
    img = detect_imgs[idx] / np.max(detect_imgs[idx])
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    plt.imshow(img)
    plt.show()
    
    # ------- generate video demo------------
    results_dir = '../kitti_data/video_img/'
    # results_dir = '../kitti_data/video_epoch110/'
    # results_dir = '../kitti_data/video_epoch140/'
    for i in tqdm(range(len(detect_imgs))):
        img = detect_imgs[i] / np.max(detect_imgs[i])
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        pred_path = results_dir + '000000' + str(i) + '.png'
        imsave(pred_path, img)
        
    height, width, layers = detect_imgs[0].shape
    size = (width, height)
    out = cv2.VideoWriter('../kitti_data/project_video140.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)
    for i in tqdm(range(len(detect_imgs))):
        image_path = results_dir + '000000' + str(i) + '.png'
        image = cv2.imread(image_path, -1)
        out.write(image)
    out.release()
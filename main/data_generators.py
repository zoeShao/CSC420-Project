import cv2
import numpy as np
import parameter as para
import helper
import keras

class DataGen(keras.utils.Sequence):
    def __init__(self, img_names, ytrue_names, batch_size=4, img_height=384, img_width=1248):
        self.img_names = img_names
        self.ytrue_names = ytrue_names
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.on_epoch_end()
        
    def __load__(self, img_name, ytrue_name):
        aidx_per_image, delta_per_image = [], []
        aidx_set = set()
        # Read image
        img = cv2.imread(img_name).astype(np.float32, copy=False)
        # Store original width and height
        orig_h, orig_w, _ = [float(v) for v in img.shape]
        # Rescale image
        img = cv2.resize(img, (para.IMAGE_WIDTH, para.IMAGE_HEIGHT))
        img = (img - np.mean(img))/ np.std(img)
        # Load annotations(format:[[cx, cy, w, h, cls],...,[cx, cy, w, h, cls]])
        annotations = helper.get_annotation(ytrue_name)
        # Get all the classes for that image 
        labels_per_image = [a[4] for a in annotations]
        # Get all the corresponding bounding boxes  
        bboxes_per_image = np.array([a[0:4]for a in annotations])
        # scale annotation
        x_scale = para.IMAGE_WIDTH / orig_w
        bboxes_per_image[:, 0::2] = bboxes_per_image[:, 0::2] * x_scale # scale parameter(x, w)
        y_scale = para.IMAGE_HEIGHT / orig_h
        bboxes_per_image[:, 1::2] = bboxes_per_image[:, 1::2] * y_scale # scale parameter(y, h)
        # Go through all the bounding boxes for that image
        for i in range(len(bboxes_per_image)):
            overlaps = helper.batch_iou(para.ANCHOR_BOX, bboxes_per_image[i])
            # Set anchor box index to the number of all the anchor box
            aidx = para.ANCHORS
            # Choose the one with the largest overlaps
            for overlap_idx in np.argsort(overlaps)[::-1]:
                if overlaps[overlap_idx] <= 0:
                    break
                if overlap_idx not in aidx_set:
                    aidx_set.add(overlap_idx)
                    aidx = overlap_idx
                    break
            # Case when the largest available overlap is 0
            if aidx == para.ANCHORS:
                # Choose the one that has the smallest distance
                dist = np.sum(np.square(bboxes_per_image[i] - para.ANCHOR_BOX), axis=1)
                for dist_idx in np.argsort(dist):
                    if dist_idx not in aidx_set:
                        aidx_set.add(dist_idx)
                        aidx = dist_idx
                        break
            # Get deltas for loss regression
            box_cx, box_cy, box_w, box_h = bboxes_per_image[i]
            delta = [0] * 4
            delta[0] = (box_cx - para.ANCHOR_BOX[aidx][0]) / para.ANCHOR_BOX[aidx][2]
            delta[1] = (box_cy - para.ANCHOR_BOX[aidx][1]) / para.ANCHOR_BOX[aidx][3]
            delta[2] = np.log(box_w / para.ANCHOR_BOX[aidx][2])
            delta[3] = np.log(box_h / para.ANCHOR_BOX[aidx][3])
            aidx_per_image.append(aidx)
            delta_per_image.append(delta)

        return img, labels_per_image, bboxes_per_image, delta_per_image, aidx_per_image
    
    def __getitem__(self, index):
        if (index + 1) * self.batch_size > len(self.img_names):
            self.batch_size = len(self.img_names) - index * self.batch_size 
        
        img_names_batch = self.img_names[index * self.batch_size : (index + 1) * self.batch_size]
        ytrue_names_batch = self.ytrue_names[index * self.batch_size : (index + 1) * self.batch_size]

        imgs = []
        labels = []
        bboxes = []
        deltas = []
        aidxs = []
        
        for img_name, ytrue_name in zip(img_names_batch, ytrue_names_batch):
            _img, _labels, _bboxes, _deltas, _aidxs = self.__load__(img_name, ytrue_name)
            imgs.append(_img)
            labels.append(_labels)
            bboxes.append(_bboxes)
            deltas.append(_deltas)
            aidxs.append(_aidxs)

        # we need to transform this batch annotations into a form we can feed into the model
        label_indices, bbox_indices, delta_values, input_mask_indices, box_values = [], [], [], [], []
        aidx_set = set()

        for i in range(len(labels)):
            for j in range(len(labels[i])):
                if (i, aidxs[i][j]) not in aidx_set:
                    aidx_set.add((i, aidxs[i][j]))
                    input_mask_indices.append([i, aidxs[i][j]])
                    bbox_indices.extend([[i, aidxs[i][j], k] for k in range(4)])
                    box_values.extend(bboxes[i][j])
                    delta_values.extend(deltas[i][j])
                    label_indices.append([i, aidxs[i][j], labels[i][j]])

        # transform them into matrices
        input_masks = helper.sparse_to_dense(input_mask_indices, [para.BATCH_SIZE, para.ANCHORS],[1.0] * len(input_mask_indices))
        input_masks = np.reshape(input_masks, [para.BATCH_SIZE, para.ANCHORS, 1])

        boxes =  helper.sparse_to_dense(bbox_indices, [para.BATCH_SIZE, para.ANCHORS, 4], box_values)

        box_deltas = helper.sparse_to_dense(bbox_indices, [para.BATCH_SIZE, para.ANCHORS, 4], delta_values)
        
        labels = helper.sparse_to_dense(label_indices, [para.BATCH_SIZE, para.ANCHORS, para.NUM_CLASSES], [1.0] * len(label_indices))

        Y = np.concatenate((input_masks, boxes, box_deltas, labels), axis=-1).astype(np.float32)
        imgs = np.array(imgs)  
        return imgs, Y
    
    def on_epoch_end(self):
        pass
    
    def __len__(self):
        return int(np.ceil(len(self.img_names)/float(self.batch_size)))


class DataGenOrigin(keras.utils.Sequence):
    """ Data generator (with original image). """
    def __init__(self, img_names, ytrue_names, batch_size, img_height=384, img_width=1248):
        self.img_names = img_names
        self.ytrue_names = ytrue_names
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.on_epoch_end()
        
    def __load__(self, img_name, ytrue_name):
        aidx_per_image, delta_per_image = [], []
        aidx_set = set()
        # Read image
        img = cv2.imread(img_name).astype(np.float32, copy=False)
        # Store original width and height
        orig_h, orig_w, _ = [float(v) for v in img.shape]
        # Rescale image
        img = cv2.resize(img, (para.IMAGE_WIDTH, para.IMAGE_HEIGHT))
        orig_img = img
        img = (img - np.mean(img))/ np.std(img)
        # Load annotations(format:[[cx, cy, w, h, cls],...,[cx, cy, w, h, cls]])
        annotations = helper.get_annotation(ytrue_name)
        # Get all the classes for that image 
        labels_per_image = [a[4] for a in annotations]
        # Get all the corresponding bounding boxes  
        bboxes_per_image = np.array([a[0:4]for a in annotations])
        # scale annotation
        x_scale = para.IMAGE_WIDTH / orig_w
        bboxes_per_image[:, 0::2] = bboxes_per_image[:, 0::2] * x_scale # scale parameter(x, w)
        y_scale = para.IMAGE_HEIGHT / orig_h
        bboxes_per_image[:, 1::2] = bboxes_per_image[:, 1::2] * y_scale # scale parameter(y, h)
        # Go through all the bounding boxes for that image
        for i in range(len(bboxes_per_image)):
            overlaps = helper.batch_iou(para.ANCHOR_BOX, bboxes_per_image[i])
            # Set anchor box index to the number of all the anchor box
            aidx = para.ANCHORS
            # Choose the one with the largest overlaps
            for overlap_idx in np.argsort(overlaps)[::-1]:
                if overlaps[overlap_idx] <= 0:
                    break
                if overlap_idx not in aidx_set:
                    aidx_set.add(overlap_idx)
                    aidx = overlap_idx
                    break
            # Case when the largest available overlap is 0
            if aidx == para.ANCHORS:
                # Choose the one that has the smallest distance
                dist = np.sum(np.square(bboxes_per_image[i] - para.ANCHOR_BOX), axis=1)
                for dist_idx in np.argsort(dist):
                    if dist_idx not in aidx_set:
                        aidx_set.add(dist_idx)
                        aidx = dist_idx
                        break
            #compute deltas for regression
            box_cx, box_cy, box_w, box_h = bboxes_per_image[i]
            delta = [0] * 4
            delta[0] = (box_cx - para.ANCHOR_BOX[aidx][0]) / para.ANCHOR_BOX[aidx][2]
            delta[1] = (box_cy - para.ANCHOR_BOX[aidx][1]) / para.ANCHOR_BOX[aidx][3]
            delta[2] = np.log(box_w / para.ANCHOR_BOX[aidx][2])
            delta[3] = np.log(box_h / para.ANCHOR_BOX[aidx][3])
            aidx_per_image.append(aidx)
            delta_per_image.append(delta)

        return img, orig_img, labels_per_image, bboxes_per_image, delta_per_image, aidx_per_image
    
    def __getitem__(self, index):
        if (index + 1) * self.batch_size > len(self.img_names):
            self.batch_size = len(self.img_names) - index * self.batch_size 
        
        img_names_batch = self.img_names[index * self.batch_size : (index + 1) * self.batch_size]
        ytrue_names_batch = self.ytrue_names[index * self.batch_size : (index + 1) * self.batch_size]

        imgs = []
        orig_imgs = []
        labels = []
        bboxes = []
        deltas = []
        aidxs = []
        
        for img_name, ytrue_name in zip(img_names_batch, ytrue_names_batch):
            _img, _orig_img, _labels, _bboxes, _deltas, _aidxs = self.__load__(img_name, ytrue_name)
            imgs.append(_img)
            orig_imgs.append(_orig_img)
            labels.append(_labels)
            bboxes.append(_bboxes)
            deltas.append(_deltas)
            aidxs.append(_aidxs)

        #we need to transform this batch annotations into a form we can feed into the model
        label_indices, bbox_indices, delta_values, input_mask_indices, box_values = [], [], [], [], []
        aidx_set = set()
        #iterate batch
        for i in range(len(labels)):
            #and annotations
            for j in range(len(labels[i])):
                if (i, aidxs[i][j]) not in aidx_set:
                    aidx_set.add((i, aidxs[i][j])) 
                    input_mask_indices.append([i, aidxs[i][j]])
                    bbox_indices.extend([[i, aidxs[i][j], k] for k in range(4)])
                    box_values.extend(bboxes[i][j])
                    delta_values.extend(deltas[i][j])
                    label_indices.append([i, aidxs[i][j], labels[i][j]])

        #transform them into matrices
        input_masks = helper.sparse_to_dense(input_mask_indices, [para.BATCH_SIZE, para.ANCHORS], [1.0] * len(input_mask_indices))
        input_masks = np.reshape(input_masks, [para.BATCH_SIZE, para.ANCHORS, 1])
        
        boxes = helper.sparse_to_dense(bbox_indices, [para.BATCH_SIZE, para.ANCHORS, 4], box_values)

        box_deltas = helper.sparse_to_dense(bbox_indices, [para.BATCH_SIZE, para.ANCHORS, 4], delta_values)
        
        labels = helper.sparse_to_dense(label_indices, [para.BATCH_SIZE, para.ANCHORS, para.NUM_CLASSES], [1.0] * len(label_indices))

        Y = np.concatenate((input_masks, boxes,  box_deltas, labels), axis=-1).astype(np.float32)
        imgs = np.array(imgs)  
        orig_imgs = np.array(orig_imgs) 
        return imgs, Y, orig_imgs
    
    def on_epoch_end(self):
        pass
    
    def __len__(self):
        return int(np.ceil(len(self.img_names)/float(self.batch_size)))
    
class DataGenVideo(keras.utils.Sequence):
    def __init__(self, img_names, batch_size, img_height=384, img_width=1248):
        self.img_names = img_names
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.on_epoch_end()
        
    def __load__(self, img_name):
        # Read image
        img = cv2.imread(img_name).astype(np.float32, copy=False)
        # Rescale image
        img = cv2.resize(img, (para.IMAGE_WIDTH, para.IMAGE_HEIGHT))
        orig_img = img
        img = (img - np.mean(img))/ np.std(img)
        return img, orig_img
    
    def __getitem__(self, index):
        if (index + 1) * self.batch_size > len(self.img_names):
            self.batch_size = len(self.img_names) - index * self.batch_size 
        
        img_names_batch = self.img_names[index * self.batch_size : (index + 1) * self.batch_size]

        imgs = []
        orig_imgs = []
        
        for img_name in img_names_batch:
            _img, _orig_img = self.__load__(img_name)
            imgs.append(_img)
            orig_imgs.append(_orig_img)

        imgs = np.array(imgs)  
        orig_imgs = np.array(orig_imgs) 
        return imgs, orig_imgs
    
    def on_epoch_end(self):
        pass
    
    def __len__(self):
        return int(np.ceil(len(self.img_names)/float(self.batch_size)))
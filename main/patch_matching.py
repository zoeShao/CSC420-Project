import cv2
import numpy as np
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm

import parameter as para
import data_generators as dg

    
# define helper functions
def add_semi_transparent_rect(img, box, color=(255,0,0), alpha=0.3):
    """
    params:
    img: input image
    box: coordinates of boundingbox in [xmin, ymin, xmax, ymax]
    color: tuple of 3 integers
    alpha: alpha value between 0 and 1
    returns image with semi transparent color added at box location
    """
    x1, y1, x2, y2 = np.round(box).astype('int')
    mask = img.copy()
    cv2.rectangle(mask, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(mask, alpha, img, 1 - alpha, 0, img)
    # plt.imshow(img)
    return img

def get_patches(img, box):
    """
    params:
    img: input image
    box: coordinates of boundingbox in [xmin, ymin, xmax, ymax]
    returns all patches specified by boxes
    """
    res = []
    for i in range(len(box)):
        x1, y1, x2, y2 = np.round(box[i]).astype('int')
        res.append(img[y1:y2, x1:x2])
    return res

def sift_match(img1, img2, thres=0.6):
    """
    perform sift maching between two images. 
    returns keypoints from both images and matched keypoints
    """    
    sift = cv2.xfeatures2d.SIFT_create() #cv2.SIFT()
    bf = cv2.BFMatcher()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if len(kp1) <= 1 or len(kp2) <= 1:
        return kp1, kp2, []

    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < thres * n.distance:
            good.append([m])
    return kp1, kp2, good


if __name__ == "__main__":
    # load images names 
    img_video_file = '../kitti_data/images_video.txt'
    with open(img_video_file) as video_imgs:
        img_video_names = video_imgs.read().splitlines()
    video_imgs.close()
    vis_generator = dg.DataGenVideo(img_video_names, batch_size=para.BATCH_SIZE, img_height=384, img_width=1248)
    
    # load bounding boxes
    loaded = np.load('../kitti_data/pm_bboxes.npz', allow_pickle=True)
    pm_bboxes = loaded['arr_0']
    
    # Getting images for Patch matching
    pm_images = []
    for i in range((len(vis_generator)-1)*para.BATCH_SIZE):
        img = cv2.imread(img_video_names[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (para.IMAGE_WIDTH, para.IMAGE_HEIGHT))
        pm_images.append(img)
    
    # constants
    pm_results_dir = '../kitti_data/patch_matching_img/'
    score_thres = 0.01
    num_colors = 20
    colors = np.random.randint(0, 255, (num_colors, 3))
    num_imgs = len(pm_bboxes)
    
    all_colors = set(np.arange(num_colors))
    all_pairs = []
    pm_imgs_with_bboxes = []
    
    img2 = pm_images[0]
    box2 = pm_bboxes[0]

    # here, left image and right image basically means an adjacent pair of frames 
    # r_patch: patches of right image. l_patches is for left images, similarly
    # r_colors: colors assigned to patches in right image. l_colors is for left images, similarly
    r_patches = get_patches(img2, box2)
    r_colors = [-1] * len(r_patches)

    # iterate through all frames
    for idx in tqdm(range(1, num_imgs)):
        # copy right image data to left image and load next right image (frame)
        img1 = img2.copy()
        img2 = pm_images[idx].copy()
        box1, box2 = pm_bboxes[idx-1:idx+1]

        # get all patches from both frames
        l_patches = r_patches.copy()
        r_patches = get_patches(img2, box2)
    
        # calculate scores of all possible matches. score = |matches| / (|kp1| + |kp2|)
        dice_scores = []
        l1 = len(l_patches)
        l2 = len(r_patches)
        # this is the data structure of dice_score. each array elements stores:
        # i1: the index of i1-th patch from left image
        # i2: the index of i2-th patch from left image
        # score: how good is the match between i1-th patch from first frame and i1-th patch from second frame
        for i1 in range(l1):
            l_p = l_patches[i1]
            for i2 in range(l2):
                r_p = r_patches[i2]
                kp1, kp2, good = sift_match(l_p, r_p)
                score = len(good) / (len(kp1) + len(kp2))
                dice_scores.append( [(i1, i2), score] )
    
        # sort scores
        dice_scores = sorted(dice_scores, key=lambda x: x[1], reverse=True)
    
        pairs = []
        used_color = set(r_colors)
        l_colors = r_colors.copy()
        r_colors = [-1] * l2
    
        num_scores = len(dice_scores)
        cnt = 0

        # recursively pick the best scores
        while num_scores > 0:
            cur_best = dice_scores[0]
            if cur_best[1] < score_thres:
                break
            
            # extract pair from stored score information
            good_pair = cur_best[0]
            l_idx, r_idx = good_pair
            pairs.append(good_pair)

            # align the colors assigned to the pair of matches
            if l_colors[l_idx] >= 0:
                r_colors[r_idx] = l_colors[l_idx]
            else:
                unused = list(all_colors - used_color)
                l_colors[l_idx] = unused[0]
                r_colors[r_idx] = unused[0]
                used_color.add(unused[0])
            
            # delete the pair from our score list, as they are already matched
            del dice_scores[0]
            num_scores -= 1
            i = 0

            # remove all scores associated with either matched patch
            # for example if we matched (3,5), then pairs like (3,4) or (1,5) should be dropped
            while i < num_scores:
                item = dice_scores[i]
                pair = item[0]
                if pair[0] == l_idx or pair[1] == r_idx:
                    del dice_scores[i]
                    num_scores -= 1
                    i -= 1
                i += 1
        
        # add calculated color to all patches
        # if a patch does not have a color assigned, then pick one from unused colors
        unused = list(all_colors - used_color)
        for i in range(l1):
            color = l_colors[i]
            if color < 0:
                color = unused[0]
                l_colors[i] = color
                del unused[0]
                used_color.add(color)
            add_semi_transparent_rect(img1, box1[i], tuple(colors[color].tolist()))
    
        for i in range(l2):
            color = r_colors[i]
            if color < 0:
                color = unused[0]
                r_colors[i] = color
                del unused[0]
                used_color.add(color)
            add_semi_transparent_rect(img2, box2[i], tuple(colors[color].tolist()))
    
        # print(pairs)
        all_pairs.append(pairs)
        imsave(pm_results_dir + str(idx-1) + '.jpg', img1)
        pm_imgs_with_bboxes.append(img1)
    
    imsave(pm_results_dir + str(idx) + '.jpg', img2)
    pm_imgs_with_bboxes.append(img2)
    
    # Show result
    for idx in range(5):
        img = pm_imgs_with_bboxes[308 + idx*3]
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        plt.imshow(img)
        plt.show()
    
    # failed example:
    # One car failed to be matched
    for idx in range(3):
        img = pm_imgs_with_bboxes[28 + idx]
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        plt.imshow(img)
        plt.show()
        
    # ------generate video demo for patch matching--------------
    height, width, layers = pm_imgs_with_bboxes[0].shape
    size = (width, height)
    out = cv2.VideoWriter('../kitti_data/project_pm_video.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)
    for i in tqdm(range(len(pm_imgs_with_bboxes))):
        image_path = pm_results_dir + str(i) + '.jpg'
        image = cv2.imread(image_path, -1)
        out.write(image)
    out.release()
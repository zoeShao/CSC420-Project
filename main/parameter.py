import numpy as np
import cv2

# -------------------Image Parameter Part-------------------------
IMAGE_WIDTH = 1248
IMAGE_HEIGHT = 384
# -------------------Classification Parameter Part-------------------------
CLASS_TO_COLOR = {"car": (0, 191, 255),
                  "cyclist": (255, 0, 191),
                  "pedestrian": (255, 191, 0)}
CLASS_NAMES = ['cyclist', 'pedestrian', 'car']
# Number of categories
NUM_CLASSES = len(CLASS_NAMES)
# Convert to a class index dictionary
CLASS_TO_IDX = dict(zip(CLASS_NAMES, range(NUM_CLASSES)))
# Colour of class in Visualization
FONT = cv2.FONT_HERSHEY_SIMPLEX
# -------------------Train Part-------------------------
BATCH_SIZE = 4
# -------------------Prediction Threshold Part-------------------------
SHOW_RESULT_THRESH = 0.59
# -------------------Anchor Box Parameter Part-------------------------
def set_grid_anchors():
    Height, Width, Num = NUM_VERTICAL_ANCHORS, NUM_HORIZ_ANCHORS, ANCHOR_PER_GRID
    anchor_shapes = np.reshape([ANCHOR_RATIO] * Height * Width, (Height, Width, Num, 2))
    center_x = np.array([np.arange(1, Width+1)*float(IMAGE_WIDTH)/(Width+1)]*Height*Num)
    center_x = np.reshape(np.transpose(np.reshape(center_x, (Num, Height, Width)),(1, 2, 0)),(Height, Width, Num, 1))
    center_y = np.array([np.arange(1, Height+1)*float(IMAGE_HEIGHT)/(Height+1)]*Width*Num)
    center_y = np.reshape(np.transpose(np.reshape(center_y, (Num, Width, Height)),(2, 1, 0)),(Height, Width, Num, 1))
    anchors = np.reshape(np.concatenate((center_x, center_y, anchor_shapes), axis=3), (-1, 4))
    return anchors
# Default anchor box parameter
ANCHOR_RATIO = np.array([[36., 37.], [366., 174.], [115., 59.], 
                        [162., 87.], [38., 90.], [258., 173.], 
                        [224., 108.], [78., 170.], [72., 43.]])
ANCHOR_PER_GRID = len(ANCHOR_RATIO)
NUM_VERTICAL_ANCHORS = 24
NUM_HORIZ_ANCHORS = 78
# Get the anchors for the grid
ANCHOR_BOX = set_grid_anchors()
ANCHORS = len(ANCHOR_BOX)
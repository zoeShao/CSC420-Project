# CSC420 Project: Detect and Track Objects for Autonomous driving
#### Group members: Yuting Shao, Jijun Xiao  
## General Ideas:
Comprehensive perception of the surrounding environment is one of the most crucial tasks in autonomous driving scenario. Our goal is to detect various objects, including cars and pedestrians, as well as changes in their positions.

### Getting Started
You can either choose to run on colab or run locally, please follow the instruction below:

__1. Mounted on google drive:__
  * Download and upload [preprocessor.ipynb](https://github.com/zoeShao/CSC420-Project/blob/master/preprocessor.ipynb) and [project.ipynb](https://github.com/zoeShao/CSC420-Project/blob/master/project.ipynb) on your google drive.
  * Open [preprocessor.ipynb](https://github.com/zoeShao/CSC420-Project/blob/master/preprocessor.ipynb) and follow the instructions to download and preprocess the KITTI dataset.
  * Wait for the file to upload on drive and then open [project.ipynb](https://github.com/zoeShao/CSC420-Project/blob/master/project.ipynb).
  
__2. Run locally:__
 * Clone the repo:
 
   `git clone https://github.com/zoeShao/CSC420-Project.git`
 * Inside the repo folder, download Kitti dataset [images](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip) and [labels](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip)
 * Also, for demo purpose, download Kitti raw data from [here](https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0009/2011_09_26_drive_0009_extract.zip)
 * Run the script:
 
    `sh set_up.sh`
 * __*Note*__: Patch matching depends on the result of detecting objects, so if you want to run `patch_matching.py` you need to run `visualize_detection.py` first.

## 1. Detect objects
We implemented SqueezeDet to detect objects and their bounding boxes described in this paper: *SqueezeDet: Unified, Small, Low Power Fully Convolutional Neural Networks for Real-Time Object Detection for Autonomous Driving* https://arxiv.org/abs/1612.01051. 

### Demo
<a href="https://youtu.be/r-Qd8Y_hBc8" target="_blank"><img src="https://github.com/zoeShao/CSC420-Project/blob/master/illustration/object%20detection.png" 
alt="Detect objects Demo picture" border="10" /></a>

## 2. Patch matching

### Demo
![Patch matching Demo gif](https://github.com/zoeShao/CSC420-Project/blob/master/illustration/patch%20matching.gif)


### Acknowledgements
We Thank Bichen Wu, Alvin Wan, Forrest Iandola, Peter H. Jin, Kurt Keutzer for providing the open-source code of their great work SqueezeDet. The original implementation is at [here](https://github.com/BichenWuUCB/squeezeDet).

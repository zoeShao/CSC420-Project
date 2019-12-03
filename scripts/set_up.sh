cd ../
unzip data_object_image_2.zip > /dev/null
unzip data_object_label_2.zip > /dev/null
unzip 2011_09_26_drive_0009_extract.zip > /dev/null

mkdir kitti_data
mkdir kitti_data/checkpoints
mkdir kitti_data/video_img
mkdir kitti_data/patch_matching_img

cd kitti_data
find '../training/image_2/' -name "*png" | sort > images.txt
find '../training/label_2/' -name "*txt" | sort > labels.txt
find '../2011_09_26/2011_09_26_drive_0009_extract/image_02/' -name "*png" | sort > images_video.txt

python ../scripts/split_dataset.py
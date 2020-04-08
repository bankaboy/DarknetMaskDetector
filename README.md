## Introduction :
This file contains the steps taken in this project to implement the given problem<br>
This is a library called darknet which is available on GitHub - https://github.com/AlexeyAB/darknet.<br>
I have used this to train a custom model to detect to classes - 
* People with masks
* People without masks
The model can work on images, videos and webcams and present live results.<br>

## Steps to setup - 
 * ### Downloading necessary materials - 
 Materials - https://drive.google.com/open?id=1cniJAl1KFK3F_YV0CAEjoI4CUSJlQO5F
 1. Training_cfg_obj_files - 
    * Put yolo-obj.cfg in cfg folder of darknet_mask_detection folder.
    * Put obj.names, obj.data and train.txt in data folder of darknet_mask_detection folder.
 2. Weights files - Contains weights files generated over training.
    * Download yolo-obj_4000.weights, rename to yolo-obj_masks.weights and keep in darknet_mask_detection folder.
    * yolov3.weights - is the original weights file.
    * darknet53.conv.74 - weights file to start training from.
 3. obj.zip is the dataset along with annotations files. Not needed to run model.
 4. Evaluation.zip contains the detection images.

Currently the library is in its raw form without any object files or dependencies setup.<br>
The library is supported by NVIDIA CUDA and it is benificial to have OpenCV as well.<br>
These are steps to setup the library in two different types of environments.<br>
 * ### Setup on a non gpu instance -
 1. Go to Makefile.
 2. In the top keep GPU, CUDNN, CUDNN_HALF = 0.
 3. Set LIBSO=1.
 4. If you want to enable OPENCV and OPENMP set them equal to 1.
 5. Save the Makefile and exit.
 6. Open a terminal in the main darknet_mask_detection directory, write make and enter.

 * ### Setup on a gpu instance -
 1. Go to Makefile.
 2. In the top keep GPU, CUDNN = 1 and CUDNN_HALF = 0.
 3. Set CUDNN_HALF=1 if GPU is NVIDIA Volta, Xavier or Higher.
 4. Set LIBSO=1.
 5. If you want to enable OPENCV and OPENMP set them equal to 1.
 6. For specific GPUs (Line 26-44), comment line 18-22. Below specific GPU name, uncoment "ARCH=---".
 7. Save the Makefile and exit.
 8. Open a terminal in the main darknet_mask_detection directory, write make and enter.

## Steps for running created model -
The model can be launched to work on images, videos and live feed - 
### Steps to run on webcam -
1. Open a terminal in the main darknet_mask_detection folder.
2. Run the command -
./darknet detector demo ./data/obj.data ./cfg/yolo-obj.cfg ./yolo-obj_maksks.weights -c 0

### Steps to run on particular image - 
1. Put the image to be tested in the darknet_mask_detection folder.
2. Open a terminal in the main darknet_mask_detection folder.
3. Run the command -
./darknet detector test ./data/obj.data ./cfg/yolo-obj.cfg ./yolo-obj_masks.weights <img_path>

### Steps to run on particular video -
1. Put the video to be tested in the darknet_mask_detection folder.
2. Open a terminal in the main darknet_mask_detection folder.
3. Run the command -
./darknet detector demo ./data/obj.data ./cfg/yolo-obj.cfg ./yolo-obj_masks.weights -ext_output <video_path>

## Evaluation - 
The model was evaluated on the following cases - 
    1. Single person without mask.
    2. Single person wearing mask.
    3. Crowd of people without masks.
    4. Crowd of people wearing masks.

## Process description - 
* The initial models were trained for 90 different classes.
* To train a new model for people with and without mask, the following steps were taken -
    1. Created a dataset by obtaining images from the net for both cases.
    2. Auto-annotated images of people not wearing masks using existing model.
    3. Self annotated images of people wearing masks using Yolo_Mark annotation tool.
    4. Combined images and annotations files of both classes into single dataset.
    5. Augmented data using data annotation.
    6. Configured cfg, names, data and training files to set training parameters.
    7. Complete training at test model.

## Appendix -
These are the resources used to implement the project - 
1. Yolo Object Detection Framework - https://github.com/AlexeyAB/darknet
2. Yolo Mark Annotation Tool - https://github.com/AlexeyAB/Yolo_mark
3. Data Augmentation - https://github.com/Paperspace/DataAugmentationForObjectDetection + An Additional self Script to augment annotations as well.
4. Google colab notebook usd to train - https://colab.research.google.com/drive/1FlLl-ZOYx7_a8zjudzKvdrwMNRFgWylG

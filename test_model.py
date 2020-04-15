import argparse
import os
import sys
import cv2
import time

def create_arg_parser():
    parser = argparse.ArgumentParser(description='This script takes input image or video and gives the output to the user.')
    parser.add_argument('--input_image',
                    help='Run the script inside the main darknet folder. Give path to the image file. Best if image is in darknet folder and path is relative to main darknet folder.')
    parser.add_argument('--input_video',
                    help='Run the script inside the main darknet folder. Give path to the video file. Best if video is in darknet folder and path is relative to main darknet folder.')
    # parser.add_argument('--output',
    #                 help='Path to the output file')
    return parser

allowed_images = ['png','jpg','jpeg']
allowed_videos = ['mp4','mkv','mpeg','avi','mov']

arg_parser = create_arg_parser()
parsed_args = arg_parser.parse_args(sys.argv[1:])

image = parsed_args.input_image
video = parsed_args.input_video

if image:
    attributes = image.split(".")
    ext = attributes[-1]
    if ext in allowed_images:
        os.system("./darknet detector test ./data/obj.data ./cfg/yolo-obj.cfg ./yolo-obj_masks.weights "+image)
        pred = cv2.imread('predictions.jpg')
        cv2.imshow('Predictions',pred)
if video:
    attributes = video.split(".")
    ext = attributes[-1]
    if ext in allowed_videos:
        print("The output will now be shown in the terminal and on the video.\n")
        time.sleep(2)
        os.system("./darknet detector demo ./data/obj.data ./cfg/yolo-obj.cfg ./yolo-obj_masks.weights -ext_output "+video)


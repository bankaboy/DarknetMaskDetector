from ctypes import *
import math
import random
import os
import cv2
import argparse
import numpy as np
import time
import sys
import darknet
import glob
import gc

folder = "./people_without_masks/"


def dump_garbage():
    """
    show us what's the garbage about
    """
        
    # force collection
    print ("\nGARBAGE:")
    gc.collect()

    print ("\nGARBAGE OBJECTS:")
    for x in gc.garbage:
        s = str(x)
        if len(s) > 80: s = s[:80]
        print (type(x),"\n  ", s)

# def create_arg_parser():

#     parser = argparse.ArgumentParser(description=':)')
#     parser.add_argument('--input',
#                     help='Path to the image file')
#     parser.add_argument('--output',
#                     help='Path to the output file')
#     return parser

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax
    
def find_label_number(label):
    with open("./cfg/coco.data") as metaFH:
        metaContents = metaFH.read()
        import re
        match = re.search("names *= *(.*)$", metaContents,
                            re.IGNORECASE | re.MULTILINE)
        if match:
            result = match.group(1)
        else:
            result = None
        try:
            if os.path.exists(result):
                with open(result) as namesFH:
                    namesList = namesFH.read().strip().split("\n")
                    altNames = [x.strip() for x in namesList]
                    return altNames.index(label)
        except TypeError:
            pass

def cvDrawBoxes(detections, img, filename):
    newfile = folder+f'{filename}.txt'
    height, width, _ = img.shape
    print(filename)
    with open(newfile, "w+") as txt:
        print(newfile)
        for detection in detections:
            x, y, w, h = detection[2][0],\
                detection[2][1],\
                detection[2][2],\
                detection[2][3]
            label = find_label_number(detection[0].decode())
            if(label == 0):
                txt.write(f"{label} {x/width} {y/height} {w/width} {h/height} \n")


netMain = None
metaMain = None
altNames = None

def YOLO():

    global metaMain, netMain, altNames
    configPath = "./cfg/yolov3.cfg"
    weightPath = "./yolov3.weights"
    metaPath = "./cfg/coco.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    #cap = cv2.VideoCapture(0)
    # folder_name = parsed_args.input
    _,_,filenames = next(os.walk(folder))
    for image_name in filenames:
        print("file name inside folder: ", image_name)
        # print("file being used: ",image_name) # keeping track of current file being used
        # image_name = parsed_args.input
        ext = image_name.split('.')
        print("variable ext: ",ext)
        # Create an image we reuse for each detect
        darknet_image = darknet.make_image(darknet.network_width(netMain),
                                        darknet.network_height(netMain),3)
        frame_read = cv2.imread(folder+image_name)
        # frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_read,
                                    (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                    interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
        cvDrawBoxes(detections, frame_resized, ext[0])
        darknet_image=None
        frame_read=None
        frame_rgb=None
        frame_resized=None
        detections=None

if __name__ == "__main__":
    # arg_parser = create_arg_parser()
    # parsed_args = arg_parser.parse_args(sys.argv[1:])
    # make a leak
    
    allowed_formats = ['jpg','jpeg','png']
    YOLO()

    # show the dirt ;-
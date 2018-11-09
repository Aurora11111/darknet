import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import pandas as pd
from PIL import Image
import os
import glob
import pandas as pd
import json
import cv2
import numpy as np
from PIL import Image
from lib.text_connector.text_connect_cfg import Config as TextLineCfg
import os


sets=[('2007', 'train'), ('2007', 'val')]


def convert(size, box):
    #print(size[0],size[1])
    dw = 1.0/(size[0])
    dh = 1.0/(size[1])

    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    print(x,y,w,h)
    return (x,y,w,h)


def resize_im(im, scale, max_scale=None):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f


def draw_boxes(img, boxes, scale):
    box_id = 0
    img = img.copy()
    text_recs = np.zeros((len(boxes), 8), np.int)
    for box in boxes:
        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
            continue

        color = (255, 0, 0)  # red
        #color = (0, 255, 0)  # green

        cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)
        cv2.line(img, (int(box[2]), int(box[3])), (int(box[4]), int(box[5])), color, 2)
        cv2.line(img, (int(box[0]), int(box[1])), (int(box[6]), int(box[7])), color, 2)

        for i in range(8):
            text_recs[box_id, i] = box[i]

        box_id += 1

    img = cv2.resize(img, None, None, fx=1.0/scale, fy=1.0/scale, interpolation=cv2.INTER_LINEAR)
    return text_recs, img


def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0 -1
    y = (box[2] + box[3])/2.0 -1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(year, image_id,anchors_file):
    # df = pd.read_csv(filename, dtype={'data': object})
    # selected = ['text', 'title', 'style', 'structural', 'tag']
    in_file = open('/home/rice/VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    out_file = open('/home/rice/VOCdevkit/VOC%s/labels/%s.txt'%(year, image_id), 'w')

    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    # w = int(size.find('width').text)
    # h = int(size.find('height').text)
    # print("w1,h1",w,h)
    image = '/home/rice/VOCdevkit/VOC2007/JPEGImages/'+image_id +'.jpg'
    w, h = (Image.open(image).convert('RGB')).size
    print("w,h",w,h)
    boxes = []
    for obj in root.iter('object'):
        xmlbox = obj.find('bndbox')
        xmin = int(xmlbox.find('xmin').text)
        ymin = int(xmlbox.find('ymin').text)
        xmax = int(xmlbox.find('xmax').text)
        ymax = int(xmlbox.find('ymax').text)
        box = (xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin)
        boxes.append(box)
        cls_id = '1'
        k = (xmin, ymin, xmax, ymax)
        b = (xmin, xmax, ymin, ymax)
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + "".join([str(a) for a in bb]) + '\n')
        anchors_file.write(" " + ",".join([str(a) for a in k])+ "," + cls_id )

    anchors_file.write('\n')
    # im_name = image
    # img = Image.open(im_name).convert("RGB")
    # img = np.array(img)
    # img2, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
    # text_recs, img_drawed = draw_boxes(img, boxes, scale)
    # name = '/home/rice/PycharmProjects/keras-yolo3-text/test_resultx/' + image_id+'.jpg'
    # Image.fromarray(img_drawed).save(name, 'JPEG')

wd = getcwd()
df = pd.read_csv('labels.csv', dtype={'data': object})

for year, image_set in sets:
    if not os.path.exists('/home/rice/VOCdevkit/VOC%s/labels/'%(year)):
        os.makedirs('/home/rice/VOCdevkit/VOC%s/labels/'%(year))
    image_ids = open('/home/rice/VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    list_file = open('%s_%s.txt'%(year, image_set), 'w')
    anchors_file = open('newanchors1.txt', 'w')
    for image_id in image_ids:
        list_file.write('/home/rice/VOCdevkit/VOC2007/JPEGImages/%s.jpg\n'%(image_id))
        anchors_file.write('/home/rice/VOCdevkit/VOC2007/JPEGImages/%s.jpg' % (image_id))
        convert_annotation(year, image_id, anchors_file)
    list_file.close()



import os
import glob
import pandas as pd
import json
import cv2
import numpy as np
from PIL import Image
from lib.text_connector.text_connect_cfg import Config as TextLineCfg
import os


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


def dict_to_csv(path):
    boxes_list = []

    path = r"./results.txt"
    data = open(path)
    file = open('train.txt', 'w')
    anchors_file = open('newanchors.txt', 'w')
    for line in data:
        #print(line)
        boxes = []
        image = line.split(',')[0]
        mydata = line[len(image)+1:]
        #print(image)
        #print(image[36:-4])
        out_file = open('/home/rice/PycharmProjects/keras-yolo3-text/image/%s.txt' % (image[36:-4]), 'w')
        image = '/home/rice/PycharmProjects/keras-yolo3-text/image/' + image[36:]
        print(image)
        w,h = (Image.open(image).convert('RGB')).size
        file.write('%s\n' % (image))
        anchors_file.write(image)
        da = json.loads(mydata)["words_result"]
        #print(da)
        for location in da:
            width = location["location"]["width"]
            top = location["location"]["top"]
            height = location["location"]["height"]
            left = location["location"]["left"]
            x = width % 16
            xside = 16 - x
            if x != 0:
                width += xside
            left -= xside//2

            ymin = top
            ymax = top + height

            xmin = left
            xmax = left + 16

            for i in range(int(width/16)):
                #print(xmin, xmax)
                value = (image,width,height,'text',xmin,ymin,xmax,ymax)
                boxes_list.append(value)
                box = (xmin,ymin,xmin,ymax,xmax,ymax,xmax,ymin)
                boxes.append(box)
                cls_id = '1'
                k = (int(xmin), int(ymin), int(xmax), int(ymax))
                b = (int(xmin), int(xmax),int(ymin), int(ymax))
                anchors_file.write(" " + ",".join([str(a) for a in k]) + "," + cls_id)
                bb = convert((w, h), b)
                out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
                xmin += 16
                xmax += 16

        # step = 16.0
        # x_left = []
        # x_right = []
        # x_left.append(xmin)
        # x_left_start = int(math.ceil(xmin / 16.0) * 16.0)
        # if x_left_start == xmin:
        #     x_left_start = xmin + 16
        # for i in np.arange(x_left_start, xmax, 16):
        #     x_left.append(i)
        # x_left = np.array(x_left)
        #
        # x_right.append(x_left_start - 1)
        # for i in range(1, len(x_left) - 1):
        #     x_right.append(x_left[i] + 15)
        # x_right.append(xmax)
        # x_right = np.array(x_right)
        #
        # idx = np.where(x_left == x_right)
        # x_left = np.delete(x_left, idx, axis=0)
        # x_right = np.delete(x_right, idx, axis=0)

        anchors_file.write('\n')
        """
        draw rectangle on image
        """
        # im_name = image = '/home/rice/PycharmProjects/keras-yolo3-text/image/'+image[50:]
        # img = Image.open(im_name).convert("RGB")
        # img = np.array(img)
        # img2, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
        # text_recs, img_drawed = draw_boxes(img, boxes, scale)
        # name = '/home/rice/PycharmProjects/keras-yolo3-text/test_result/' + image[50:]
        # Image.fromarray(img_drawed).save(name, 'JPEG')

    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    boxes_df = pd.DataFrame(boxes_list, columns=column_name)
    return boxes_df


def main():
    xml_df = dict_to_csv('/home/rice/PycharmProjects/keras-yolo3-text/results.txt')
    xml_df.to_csv(('/home/rice/PycharmProjects/keras-yolo3-text/labels.csv'), index=None)
    print('Successfully converted xml to csv.')


main()

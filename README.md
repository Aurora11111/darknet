![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

# Darknet #
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).

some  using detaile can be found in below link:

https://github.com/AlexeyAB/darknet


# text-detection-yolov3 #
this project is used to detect text 
mainly reference:

https://github.com/Aurora11111/keras_ocr

https://github.com/tianzhi0549/CTPN

https://github.com/pjreddie/darknet.git


# datasets prepare #
first, you can down load icdar2017:链接:http://rrc.cvc.uab.es/?ch=7&com=downloads

then :python voc_annotation.py

if your dataset format is json:

python dict_datasets.py

attentions 1:the accordibnate sequences must be:(xmin, xmax,ymin, ymax)

# generate anchors #
python kmeans.py

attentions 1:the accordibnate sequences must be:(xmin, ymin, xmax, ymax)

attentions 2:file structural must be similarity to voc

# training #
./darknet detector train cfg/voc.data cfg/text.cfg darknet53.conv.74

from __future__ import print_function
import tensorflow as tf
from VOC2012 import VOC2012
from new_layer2 import fpn
from tools import outputtensor_to_lableimage
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import pylab


voc2012 = VOC2012('/home/cheng/cheng_prac/mask/Deeplab-v2--ResNet-101--Tensorflow-master/data/VOCdevkit/VOC2012',
                  resize_method='resize', image_size=(320, 320))
# voc2012.read_all_data_and_save()
voc2012.load_all_data()
batch_train_images, batch_train_labels = voc2012.get_batch_train(batch_size=8)
batch_val_images, batch_val_labels = voc2012.get_batch_val(batch_size=8)
print('ok')
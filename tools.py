import tensorflow as tf
import numpy as np
                            # N,W,H,21
def outputtensor_to_lableimage(tensor,rate = 0.5):

    label_colours = tf.constant([(0, 0, 64)
                     # 0=background
        , (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128)
                     # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
        , (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0)
                     # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
        , (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128)
                     # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
        , (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)],dtype=tf.float32)

    prediction_tmp = tf.nn.softmax(tensor)

    zero = tf.zeros_like(prediction_tmp)

    prediction_tmp = tf.where(tf.less_equal(prediction_tmp, rate), x=zero, y=prediction_tmp)

    prediction_tmp = tf.reshape(prediction_tmp, [1, 320, 320, 21])

    position_mat = tf.argmax(prediction_tmp, axis=3)

    prediction2 = tf.one_hot(position_mat, depth=21)

    prediction2 = tf.reshape(prediction2, [320, 320, 21])

    prediction2 = tf.reshape(prediction2, [320 * 320, 21])

    image_tensor = tf.cast(tf.matmul(prediction2,label_colours),dtype=tf.uint8)

    image_tensor = tf.reshape(image_tensor, [320, 320, 3])

    return image_tensor


def lable_to_image(lable):
    label_colours = tf.constant([(0, 0, 64)
                     # 0=background
        , (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128)
                     # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
        , (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0)
                     # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
        , (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128)
                     # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
        , (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)],dtype=tf.float32)
    lable = tf.reshape(lable,[1,320,320])

    prediction2 = tf.one_hot(lable, depth=21)

    prediction2 = tf.reshape(prediction2, [320, 320, 21])

    prediction2 = tf.reshape(prediction2, [320 * 320, 21])

    image_tensor = tf.cast(tf.matmul(prediction2,label_colours),dtype=tf.uint8)

    image_tensor = tf.reshape(image_tensor, [320, 320, 3])

    return image_tensor
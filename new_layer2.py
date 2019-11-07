import tensorflow as tf
import numpy as np

def load_pretrain_vgg(model_dir):
    saver = tf.train.import_meta_graph(model_dir)  # 加载图结构
    gragh = tf.get_default_graph()  # 获取当前图，为了后续训练时恢复变量

    inputimage_placehold = gragh.get_tensor_by_name('Placeholder:0')  #
    part1_output = gragh.get_tensor_by_name('scale1/Relu:0')       # 获取输出变量
    part2_output = gragh.get_tensor_by_name('scale2/block3/Relu:0')  # 获取输出变量
    part3_output = gragh.get_tensor_by_name('scale3/block4/Relu:0')  # 获取输出变量
    part4_output = gragh.get_tensor_by_name('scale4/block6/Relu:0')  # 获取输出变量
    part5_output = gragh.get_tensor_by_name('scale5/block3/Relu:0')   # 获取输出变量

def res_block(input_v,name='block1_layer1'):
    # Store layers weight & bias
    weights = {
        #
        #                                    input               outputs
        name + '_w1': tf.Variable(tf.random_normal([1, 1, int(input_v.shape[3]), int(input_v.shape[3])])),
        name + '_w2': tf.Variable(tf.random_normal([3, 3, int(input_v.shape[3]), int(input_v.shape[3])])),
        name + '_w3': tf.Variable(tf.random_normal([1, 1, int(input_v.shape[3]), int(input_v.shape[3])])),
        name + '_shortw': tf.Variable(tf.random_normal([1, 1, int(input_v.shape[3]),  int(input_v.shape[3])])),
    }

    strides = 1
    output1 = tf.nn.conv2d(input_v, weights[name+'_w1'], strides=[1, strides, strides, 1], padding='SAME')
    bn1 = tf.layers.batch_normalization(output1,training=True)
    bn1 = tf.nn.leaky_relu(bn1)
    #bn1 = output1
    strides = 1
    output2 = tf.nn.conv2d(bn1, weights[name+'_w2'], strides=[1, strides, strides, 1], padding='SAME')
    bn2 = tf.layers.batch_normalization(output2, training=True)
    bn2 = tf.nn.leaky_relu(bn2)
    #bn2 = output2
    strides = 1
    output3 = tf.nn.conv2d(bn2, weights[name + '_w3'], strides=[1, strides, strides, 1], padding='SAME')
    bn3 = tf.layers.batch_normalization(output3, training=True)

    #bn3 = output3
    strides = 1
    short_output = tf.layers.batch_normalization(input_v,training=True)

    all_output = tf.layers.batch_normalization(tf.nn.leaky_relu(tf.add(bn3, short_output, name=None)))
    return all_output


def up_block(input_v,add_v,out_put_shape,name='block1_layer1'):
    weights = {
        #                                                   output               inputs
        name + '_w1': tf.Variable(tf.random_normal([2, 2, int(add_v.shape[3]), int(input_v.shape[3])])),
    }

    strides = 2
    input_up = tf.nn.conv2d_transpose(input_v, weights[name + '_w1'],
                                     output_shape=out_put_shape,
                                     strides=[1, strides, strides, 1], padding='SAME')

    all_output = tf.nn.leaky_relu( tf.layers.batch_normalization(tf.add(input_up, add_v, name=None), training=True))
    return all_output


# out1.shape
# (5, 160, 160, 64)
# out2.shape
# (5, 80, 80, 256)
# out3.shape
# (5, 40, 40, 512)
# out4.shape
# (5, 20, 20, 1024)
# out5.shape
# (5, 10, 10, 2048)

def fpn(batch_size,inputimage_placehold,part1_output,part2_output,part3_output,part4_output,part5_output):

    C2 = part1_output
    #160*160*64
    C3 = part2_output
    #80*80*256
    C4 = part3_output
    #40*40*512
    C5 = part4_output
    #20*20*1024
    C6 = part5_output
    #10*10*2048
    P6 = C6
    #10*10*2048
    P5 = up_block(P6,C5,[batch_size,20,20,1024],name='P5_layer')
    P5 = res_block(P5, name='P5_res_block1')
    P5 = res_block(P5, name='P5_res_block2')

    #20*20*1024
    P4 = up_block(P5,C4,[batch_size,40,40,512],name='P4_layer')
    P4 = res_block(P4, name='P4_res_block1')
    P4 = res_block(P4, name='P4_res_block2')

    #40*40*512
    P3 = up_block(P4,C3,[batch_size,80,80,256],name='P3_layer')
    P3 = res_block(P3, name='P3_res_block1')
    P3 = res_block(P3, name='P3_res_block2')

    #80*80*256
    P2 = up_block(P3,C2,[batch_size,160,160,64],name='P2_layer')
    P2 = res_block(P2, name='P2_res_block1')
    P2 = res_block(P2, name='P2_res_block2')


    #160*160*64

    up_conv1_w1 = tf.Variable(tf.random_normal([2, 2, 21, int(P2.shape[3])]))
    output = tf.nn.leaky_relu(tf.layers.batch_normalization(tf.nn.conv2d_transpose(P2, up_conv1_w1,
                                            output_shape=[batch_size,320 ,320, 21],
                                            strides=[1, 2, 2, 1], padding='SAME'), training=True))
    output = res_block(output, name='out_res_block1')
    output = res_block(output, name='out_res_block2')
    output = res_block(output, name='out_res_block3')

    # output_softmax = tf.nn.softmax(output)

    return output
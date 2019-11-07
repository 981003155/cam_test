from __future__ import print_function
import tensorflow as tf
from VOC2012 import VOC2012
from new_layer2 import fpn
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import pylab
def configure():
    flags = tf.app.flags

    # training

    flags.DEFINE_integer('batch_size', 5, 'batch_size')

    flags.DEFINE_integer('epoch_num', 10, 'num of epoch')

    flags.DEFINE_integer('save_img_epoch', 2, 'save_img_epoch')# save_img_epoch

    flags.DEFINE_integer('display_step', 50, 'display_step_epoch_num')

    flags.DEFINE_integer('num_train_data', 1464, 'the num of train data')

    flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')

    flags.DEFINE_float('weight_decay', 0.0005, 'weight_decay')

    flags.DEFINE_string('board_logs_path', './logs', ' board_logs_path')

    flags.DEFINE_string('model_save_path', "./my_model/fcn_seg_save.ckpt", ' model_save_path')

    flags.DEFINE_string('img_save_path', './img_save/', ' img_save_path')

    flags.DEFINE_string('pretrain_model_path', './resnet_model/ResNet-L50.meta', ' pretrain_model_path')

    flags.DEFINE_boolean('visual', True, 'whether to save predictions for visualization')

    flags.DEFINE_boolean('RELOAD_FLAG', True , 'RELOAD_FLAG')

    flags.FLAGS.__dict__['__parsed'] = False
    return flags.FLAGS



if __name__ == '__main__':

    IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

    label_colours = [(0, 0, 64)
                     # 0=background
        , (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128)
                     # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
        , (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0)
                     # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
        , (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128)
                     # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
        , (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]
    label_colours = np.array(label_colours)
    # label_colours = [(0, 0, 64)
    #                  # 0=background
    #     , (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)
    #                  # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
    #     , (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)
    #                  # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
    #     , (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (255, 0, 0)
    #                  # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
    #     , (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)]
    # label_colours = np.array(label_colours)


    ####load_data
    voc2012 = VOC2012('/home/cheng/cheng_prac/mask/Deeplab-v2--ResNet-101--Tensorflow-master/data/VOCdevkit/VOC2012',resize_method='resize',image_size=(320, 320))
    # voc2012.read_all_data_and_save()
    voc2012.load_all_data()
    # batch_train_images, batch_train_labels = voc2012.get_batch_train(batch_size=8)
    # batch_val_images, batch_val_labels = voc2012.get_batch_val(batch_size=8)
    print('load data finish')
    ####

    ####load parameter
    para = configure()
    ####

    tf.train.import_meta_graph(para.pretrain_model_path)
    gragh = tf.get_default_graph()  # 获取当前图，为了后续训练时恢复变量



    for d in ['/gpu:1']:
        with tf.device(d):
            # tf Graph Input
            lable_placehold = tf.placeholder(tf.int32, [para.batch_size,320,320])
            image_placehold = gragh.get_tensor_by_name('images:0')  #
            part1_output = gragh.get_tensor_by_name('scale1/Relu:0')  # 获取输出变量
            part2_output = gragh.get_tensor_by_name('scale2/block3/Relu:0')  # 获取输出变量
            part3_output = gragh.get_tensor_by_name('scale3/block4/Relu:0')  # 获取输出变量
            part4_output = gragh.get_tensor_by_name('scale4/block6/Relu:0')  # 获取输出变量
            part5_output = gragh.get_tensor_by_name('scale5/block3/Relu:0')  # 获取输出变量

            ####################predit#####################################
            prediction_tmp = fpn(1,image_placehold,part1_output,part2_output,part3_output,part4_output,part5_output)

            prediction_tmp = tf.nn.softmax(prediction_tmp)

            zero = tf.zeros_like(prediction_tmp)

            prediction_tmp = tf.where(tf.less_equal(prediction_tmp,0.5) , x=zero , y=prediction_tmp)

            position_mat = tf.argmax(prediction_tmp, axis=3)

            prediction2 = tf.one_hot(position_mat, depth=21)
            ###############################################################

            ####################train#####################################
            prediction = fpn(para.batch_size,image_placehold,part1_output,part2_output,part3_output,part4_output,part5_output)
            ###############################################################

            pred_in_form = tf.reshape(prediction,[prediction.shape[0]*prediction.shape[1]*prediction.shape[2],-1])

            lable_in_form = tf.reshape(lable_placehold,[-1,])

            with tf.name_scope('Loss'):

                l2_losses = tf.add_n([para.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables()])

                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_in_form, labels=lable_in_form)

                loss = tf.reduce_mean(loss)#+l2_losses


            optimizer = tf.train.AdamOptimizer(para.learning_rate).minimize(loss)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()



    # Start training

    # Create a summary to monitor cost tensor
    tf.summary.scalar("loss", loss)

    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.8
    with tf.Session(config=config) as sess:

        if para.RELOAD_FLAG:
            saver.restore(sess, para.model_save_path)
            print("Model restored.")
        # Run the initializer
        if not para.RELOAD_FLAG:
            sess.run(init)

        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter(para.board_logs_path, graph=tf.get_default_graph())

        # Training cycle
        for epoch in range(para.epoch_num):
            avg_cost = 0.
            iteration_num = int(para.num_train_data / para.batch_size)
            # Loop over all batches
            for i in range(iteration_num):
                batch_train_images, batch_train_labels = voc2012.get_batch_train(batch_size=para.batch_size)
                batch_train_images_norm =(np.array(batch_train_images,dtype=float)/255 - 0.5) / 0.5


                # Run optimization op (backprop) and cost op (to get loss value)
                _, c, summary = sess.run([optimizer, loss, merged_summary_op], feed_dict={image_placehold: batch_train_images_norm,
                                                              lable_placehold: batch_train_labels})

                # Write logs at every iteration
                summary_writer.add_summary(summary, epoch * iteration_num + i)

                # Compute average loss
                avg_cost += c / iteration_num

            # Display logs per epoch step
            #if (epoch + 1) % para.display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), 'iteration=', '%04d' %i, "cost=", "{:.9f}".format(avg_cost))
            if epoch%para.save_img_epoch==1:
                #if i % 20 == 1:
                    batch_train_images, batch_train_labels = voc2012.get_batch_train(1)
                    batch_train_images_norm = (np.array(batch_train_images, dtype=float) / 255-0.5) / 0.5

                    org_img_array = np.uint8((batch_train_images_norm.reshape([320, 320, 3]) * 0.5+ 0.5)*255)
                    org_img_show = Image.fromarray(org_img_array, mode="RGB")
                    org_img_show.save(para.img_save_path+'org'+' %04d' % epoch+'epoch.png')
                    plt.imshow(org_img_show)

                    predict_image_tensor = sess.run([prediction2], feed_dict={image_placehold: batch_train_images_norm})

                    predict_image_tensor_np = np.reshape(predict_image_tensor,[320,320,21])
                    predict_image = np.dot(predict_image_tensor_np,label_colours)
                    predict_image = np.array(predict_image)
                    pred_img_array = np.uint8(predict_image.reshape([320, 320, 3]))
                    pred_img_show = Image.fromarray(pred_img_array, mode="RGB")
                    pred_img_show.save(para.img_save_path+'pred'+' %04d' % epoch+'epoch.png')
                    plt.imshow(pred_img_show)
                    print(para.img_save_path+'pred'+' %04d' % epoch+'epoch.png'+'SAVED')
            save_path = saver.save(sess, para.model_save_path)
        print("Optimization Finished!")





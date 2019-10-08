from __future__ import division
import sys
import time
import numpy as np
import pickle
import os
import tensorflow as tf
from threading import Thread
from keras import backend as K
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from keras_frcnn import config
from keras_frcnn import losses as losses
import keras_frcnn.roi_helpers as roi_helpers
from keras_frcnn.data_generators import imgdata_generator , process_imgdata
from keras.utils import generic_utils
from keras.callbacks import TensorBoard
from keras_frcnn.simple_parser import get_data,getparentdict
from test_frcnn import testimages
from multiprocessing import Process, Queue

def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

sys.setrecursionlimit(40000)
C = config.Config()

from keras_frcnn import resnet as nn
C.num_rois = 32
C.network = 'resnet50'
train_path = '/home/adam/Desktop/kaggleOID/data/annotsfull.csv'
descriptors_path = '/home/adam/Desktop/kaggleOID/data/descriptors.csv'
hierarchypath = '/home/adam/Desktop/kaggleOID/data/bbox_labels_600_hierarchy.json'
useonlybase = False
downloaders_number=4
boxlimit=-1

all_imgs, classes_count, class_mapping , eng_mapping ,probabilities ,classes_count_total = get_data(train_path,descriptors_path,800,boxlimit)
parentdict = getparentdict(hierarchypath)
C.class_mapping=class_mapping

trainingclasses={k:0 for (k,v) in classes_count.items() if v>0}
eng_count={classes_count[k] : [classes_count_total[k],probabilities[k],eng_mapping[k]] for (k,v) in classes_count_total.items()}

for k,v in sorted(eng_count.items()):
    print(k,v)

with open('config.pickle', 'wb') as config_f:
    pickle.dump(C, config_f)

img_input = Input(shape=(None, None, 3))
roi_input = Input(shape=(None, 4))
shared_layers = nn.nn_base(img_input, trainable=True)
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = nn.rpn(shared_layers, num_anchors)
classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count), trainable=True)

model_rpn = Model(img_input, rpn[:2])
model_classifier = Model([img_input, roi_input], classifier)
model_all = Model([img_input, roi_input], rpn[:2] + classifier)

try:
    if useonlybase:
        model_rpn.load_weights('./resnet50_weights_tf_dim_ordering_tf_kernels.h5', by_name=True)
        print('weights loaded only base')
    else:
        model_all.load_weights('./model_all_frcnn{}.hdf5'.format(C.network), by_name=True)
        model_rpn.load_weights('./model_rpn_frcnn{}.hdf5'.format(C.network), by_name=True)
        model_classifier.load_weights('./model_cls_frcnn{}.hdf5'.format(C.network), by_name=True)
        print('weights loaded whole net')
except:
    print('Could not load pretrained model weights')

optimizer = Adam(lr=3e-6) #default 1e-5
optimizer_classifier = Adam(lr=3e-6) #default 1e-5
model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])
model_classifier.compile(optimizer=optimizer_classifier, loss=[losses.class_loss_cls, losses.class_loss_regr(len(classes_count)-1)], metrics={'dense_class_sigm{}'.format(len(classes_count)): 'accuracy'})
model_all.compile(optimizer='sgd', loss='mae')

log_path = './logs'
if not os.path.isdir(log_path):
    os.mkdir(log_path)

callback = TensorBoard(log_path)
callback.set_model(model_all)

epoch_length = 1000
num_epochs = 80
iter_num = 0
train_step = 0
losses = np.zeros((epoch_length, 5))
rpn_accuracy_for_epoch = []
best_loss = np.Inf
traintime=0
data_to_download=Queue()
downloaded_data=Queue()
imgdata_gen=imgdata_generator(all_imgs,trainingclasses,parentdict)
mAPlist=[]

def imgdata_thread_func():
    while(True):
        try:
            if data_to_download.qsize()<60:
                data_to_download.put(next(imgdata_gen))
            else:
                time.sleep(2)
        except Exception as e:
            print("imgdata_thread_ex_{}".format(e))
            continue

def download_process_func():
    while(True):
        try:
            if data_to_download.qsize()>0 and downloaded_data.qsize()<60:
                downloaded_data.put(process_imgdata(data_to_download.get(),nn.get_img_output_length,C))
            else:
                time.sleep(2)
        except Exception as e:
            print("download_thread_ex_{}".format(e))
            continue

imgdata_thread=Thread(target=imgdata_thread_func,args=[])
imgdata_thread.daemon=True
imgdata_thread.start()

downloaders_list=[]

for i in range(downloaders_number):
    downloaders_list.append(Process(target=download_process_func, args=[]))
    downloaders_list[i].daemon = True
    downloaders_list[i].start()

for epoch_num in range(num_epochs):
    progbar = generic_utils.Progbar(epoch_length,width=5,stateful_metrics=['time','im.todown','im.downed'])
    print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))

    for iter_num in range(epoch_length):
        iter_time = time.time()

        while(downloaded_data.qsize()==0):
            time.sleep(0.005)

        X, Y, img_data = downloaded_data.get()

        loss_rpn = model_rpn.train_on_batch(X, Y)
        write_log(callback, ['rpn_cls_loss', 'rpn_reg_loss'], loss_rpn, train_step)
        P_rpn = model_rpn.predict_on_batch(X)
        R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7,max_boxes=300)
        X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping, parentdict)

        if X2 is None:
            rpn_accuracy_for_epoch.append(0)
            continue

        neg_samples = np.where(Y1[0, :, -1] == 1)
        pos_samples = np.where(Y1[0, :, -1] == 0)

        if len(neg_samples)==0 or len(pos_samples)==0:
            rpn_accuracy_for_epoch.append(0)
            continue

        assert C.num_rois > 1
        rpn_accuracy_for_epoch.append((len(pos_samples[0])))

        pos_samples = pos_samples[0]
        neg_samples = neg_samples[0]

        if len(pos_samples) < C.num_rois // 2:
            selected_pos_samples = pos_samples.tolist()
        else:
            selected_pos_samples = np.random.choice(pos_samples, C.num_rois // 2, replace=False).tolist()

        try:
            selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples),replace=False).tolist()
        except:
            selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples),replace=True).tolist()

        sel_samples = selected_pos_samples + selected_neg_samples
        loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]],[Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

        write_log(callback, ['detection_cls_loss', 'detection_reg_loss', 'detection_acc'], loss_class, train_step)
        train_step += 1

        losses[iter_num, 0] = loss_rpn[1]
        losses[iter_num, 1] = loss_rpn[2]
        losses[iter_num, 2] = loss_class[1]
        losses[iter_num, 3] = loss_class[2]
        losses[iter_num, 4] = loss_class[3]

        iter_num += 1

        print()
        print(sorted(((v, eng_mapping[k]) for k, v in trainingclasses.items()), reverse=False))

        progbar.update(iter_num, [('rpn.cls', loss_rpn[1]), ('rpn.rgr', loss_rpn[2]),('dt.cls', loss_class[1]),
                                  ('dt._rgr', loss_class[2]),('im.todown', data_to_download.qsize()),('im.downed', downloaded_data.qsize()),
                                  ('time',time.time()-iter_time),('time_avg',time.time()-iter_time)])
        print()
        print('mAP list:{}'.format(str(mAPlist)))

    loss_rpn_cls = np.mean(losses[:, 0])
    loss_rpn_regr = np.mean(losses[:, 1])
    loss_class_cls = np.mean(losses[:, 2])
    loss_class_regr = np.mean(losses[:, 3])
    class_acc = np.mean(losses[:, 4])

    mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
    rpn_accuracy_for_epoch = []

    if C.verbose:
        print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
        print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
        print('Loss RPN classifier: {}'.format(loss_rpn_cls))
        print('Loss RPN regression: {}'.format(loss_rpn_regr))
        print('Loss Detector classifier: {}'.format(loss_class_cls))
        print('Loss Detector regression: {}'.format(loss_class_regr))

    curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr

    write_log(callback,
              ['mean_overlapping_bboxes', 'mean_rpn_cls_loss', 'mean_rpn_reg_loss',
               'mean_detection_cls_loss', 'mean_detection_reg_loss', 'mean_detection_acc', 'total_loss'],
              [ mean_overlapping_bboxes, loss_rpn_cls, loss_rpn_regr,
               loss_class_cls, loss_class_regr, class_acc, curr_loss],
              epoch_num)

    if C.verbose:
        print('Total loss : {}, saving weights'.format(curr_loss))
    model_rpn.save_weights('./model_rpn_frcnn{}.hdf5'.format(C.network))
    model_classifier.save_weights('./model_cls_frcnn{}.hdf5'.format(C.network))
    model_all.save_weights('./model_all_frcnn{}.hdf5'.format(C.network))

    if epoch_num%4==0:
        tmpmAP=testimages()
        mAPlist.append(tmpmAP)
        write_log(callback,['mAP'],[tmpmAP],epoch_num)

print('Training complete, exiting.')

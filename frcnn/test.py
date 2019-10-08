from __future__ import division
import os
import cv2
import numpy as np
import sys
import pickle
import time
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
import keras_frcnn.resnet as nn

sys.setrecursionlimit(40000)
num_rois=32

def testimages(img_path = '/home/adam/Desktop/kaggleOID/eval/eval_imgs',
               descriptors_path = '/home/adam/Desktop/kaggleOID/data/descriptors.csv',
               output_path = '/home/adam/Desktop/kaggleOID/EVALcsvs/evalFRCNNtmp.csv',
               bbox_threshold = 0.04,
               non_max_sup_thresh = 0.36,
               rpn_to_roi_overlap_thresh = 0.82,
               rest_classes=[]):

    with open('/home/adam/Desktop/kaggleOID/frcnn/config.pickle', 'rb') as f_in:
        C = pickle.load(f_in)

    eng_mapping = {}

    with open(descriptors_path, 'r') as f:
        for i, line in enumerate(f):
            line_split = line.strip().split(',')
            (coded, engcoded) = line_split
            eng_mapping[coded]=engcoded

    eng_mapping['bg'] = 'bg'

    def format_img_size(img, C):
        """ formats the image size based on config """
        img_min_side = float(C.im_size)
        (height, width ,_) = img.shape

        if width <= height:
            ratio = img_min_side/width
            new_height = int(ratio * height)
            new_width = int(img_min_side)
        else:
            ratio = img_min_side/height
            new_width = int(ratio * width)
            new_height = int(img_min_side)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        return img, ratio

    def format_img_channels(img, C):
        """ formats the image channels based on config """
        img = img[:, :, (2, 1, 0)]
        img = img.astype(np.float32)
        img[:, :, 0] -= C.img_channel_mean[0]
        img[:, :, 1] -= C.img_channel_mean[1]
        img[:, :, 2] -= C.img_channel_mean[2]
        img /= C.img_scaling_factor
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def format_img(img, C):
        """ formats an image for model prediction based on config """
        img, ratio = format_img_size(img, C)
        img = format_img_channels(img, C)
        return img, ratio

    # Method to transform the coordinates of the bounding box to its original size
    def get_real_coordinates(ratio, x1, y1, x2, y2):

        real_x1 = int(round(x1 // ratio))
        real_y1 = int(round(y1 // ratio))
        real_x2 = int(round(x2 // ratio))
        real_y2 = int(round(y2 // ratio))

        return (real_x1, real_y1, real_x2 ,real_y2)

    class_mapping = C.class_mapping

    if 'bg' not in class_mapping:
        class_mapping['bg'] = len(class_mapping)

    class_mapping = {v: k for k, v in class_mapping.items()}
    print(class_mapping)
    class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
    C.num_rois = 32

    if C.network == 'resnet50':
        num_features = 1024
    elif C.network == 'xception':
        num_features = 1024
    elif C.network == 'inception_resnet_v2':
        num_features = 1024
    elif C.network == 'vgg':
        num_features = 512

    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, num_features)
    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(C.num_rois, 4))
    feature_map_input = Input(shape=input_shape_features)

    shared_layers = nn.nn_base(img_input, trainable=True)
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
    rpn_layers = nn.rpn(shared_layers, num_anchors)
    classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

    model_rpn = Model(img_input, rpn_layers)
    model_classifier = Model([feature_map_input, roi_input], classifier)

    model_rpn.load_weights('/home/adam/Desktop/kaggleOID/frcnn/model_rpn_frcnn{}.hdf5'.format(C.network), by_name=True)
    model_classifier.load_weights('/home/adam/Desktop/kaggleOID/frcnn/model_cls_frcnn{}.hdf5'.format(C.network), by_name=True)

    model_rpn.compile(optimizer='sgd', loss='mse')
    model_classifier.compile(optimizer='sgd', loss='mse')

    all_imgs = []
    classes = {}

    visualise = True
    with open(output_path,'w') as f:
        f.write('ImageID,LabelName,Score,YMin,XMin,YMax,XMax\n')
        beg_time = time.time()

        for idx, img_name in enumerate(sorted(os.listdir(img_path))):
            if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
                continue
            filepath = os.path.join(img_path,img_name)

            img = cv2.imread(filepath)

            X, ratio = format_img(img, C)
            X = np.transpose(X, (0, 2, 3, 1))

            # get the feature maps and output from the RPN
            [Y1, Y2, F] = model_rpn.predict(X)

            R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=rpn_to_roi_overlap_thresh)

            # convert from (x1,y1,x2,y2) to (x,y,w,h)
            R[:, 2] -= R[:, 0]
            R[:, 3] -= R[:, 1]

            # apply the spatial pyramid pooling to the proposed regions
            bboxes = {}
            probs = {}

            for jk in range(R.shape[0]//C.num_rois + 1):
                ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
                if ROIs.shape[1] == 0:
                    break

                if jk == R.shape[0]//C.num_rois:
                    #pad R
                    curr_shape = ROIs.shape
                    target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
                    ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                    ROIs_padded[:, :curr_shape[1], :] = ROIs
                    ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                    ROIs = ROIs_padded

                [P_cls, P_regr] = model_classifier.predict([F, ROIs])

                for ii in range(P_cls.shape[1]):

                    if np.max(P_cls[0, ii, :]) < bbox_threshold:
                        continue

                    a=np.argwhere(P_cls[0, ii, :]>bbox_threshold)

                    for ixa in np.nditer(a):
                        ix=ixa.item(0)
                        if ix!=(len(class_mapping)-1):
                            cls_name = class_mapping[ix]

                            if len(rest_classes) == 0 or (cls_name in rest_classes):
                                if cls_name not in bboxes:
                                    bboxes[cls_name] = []
                                    probs[cls_name] = []

                                (x, y, w, h) = ROIs[0, ii, :]

                                cls_num = ix
                                try:
                                    (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                                    tx /= C.classifier_regr_std[0]
                                    ty /= C.classifier_regr_std[1]
                                    tw /= C.classifier_regr_std[2]
                                    th /= C.classifier_regr_std[3]
                                    x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                                except:
                                    pass
                                bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
                                probs[cls_name].append(P_cls[0, ii, ix])

            all_dets = []


            printimage=False

            for key in bboxes:
                    new_boxes, new_probs = roi_helpers.non_max_suppression_fast(np.array(bboxes[key]), np.array(probs[key]), overlap_thresh=non_max_sup_thresh)

                    for jk in range(new_boxes.shape[0]):
                        (x1, y1, x2, y2) = new_boxes[jk,:]

                        (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

                        if printimage:
                            cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)

                            textLabel = '{}: {}'.format(eng_mapping[key],int(100*new_probs[jk]))
                            all_dets.append((eng_mapping[key],100*new_probs[jk]))

                            (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
                            textOrg = (real_x1, real_y1-0)

                            cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
                            cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
                            cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

                        (height, width, _) = img.shape
                        f.write('{},{},{:.5f},{:.3f},{:.3f},{:.3f},{:.3f}\n'.format(img_name.split('.')[0],key,new_probs[jk],real_y1/height,real_x1/width,real_y2/height,real_x2/width))

            print(img_name)
            print('Time left: {} , idx: {}'.format((time.time() - beg_time) / 60 / 60 / (idx + 1) * (100000 - idx), idx))

            if printimage:
                cv2.imwrite('/home/adam/Desktop/kaggleOID/frcnn/results_imgs/{}.png'.format(idx),img)

    return True

restricted_classes=['/m/06m11', '/m/01kb5b', '/m/0174n1', '/m/076lb9', '/m/0k0pj', '/m/0kpqd', '/m/0l14j_', '/m/0174k2', '/m/0fszt', '/m/09728', '/m/07j7r', '/m/0f4s2w', '/m/03bbps', '/m/027pcv', '/m/0323sq', '/m/03y6mg', '/m/03__z0', '/m/0cvnqh', '/m/04v6l4', '/m/02jz0l', '/m/0dj6p', '/m/01c648', '/m/01j61q', '/m/025nd', '/m/0h8n5zk', '/m/04hgtk', '/m/09qck', '/m/01h3n', '/m/01mzpv', '/m/0fm3zh', '/m/03dnzn', '/m/0h8mhzd', '/m/01gllr', '/m/0642b4', '/m/04yx4', '/m/015x4r', '/m/01jfsr', '/m/0bt_c3', '/m/0hqkz', '/m/0cyhj_', '/m/03c7gz', '/m/01nq26', '/m/0jyfg', '/m/0d4v4', '/m/0ph39', '/m/0gjbg72', '/m/035r7c', '/m/019jd', '/m/0c06p', '/m/034c16', '/m/03bt1vf', '/m/01lynh', '/m/01z1kdw', '/m/02dgv', '/m/0jg57', '/m/0dkzw', '/m/0b3fp9', '/m/0h8n27j', '/m/0h8n6f9', '/m/017ftj', '/m/044r5d', '/m/0cdl1', '/m/07gql', '/m/0zvk5', '/m/012w5l', '/m/0jwn_', '/m/02wv84t', '/m/06j2d', '/m/06_fw', '/m/02x8cch', '/m/099ssp', '/m/071p9', '/m/01gkx_', '/m/03v5tg', '/m/026t6', '/m/0_k2', '/m/039xj_', '/m/0220r2', '/m/0hg7b', '/m/019dx1', '/m/03kt2w', '/m/03q69', '/m/0dt3t', '/m/0271t', '/m/03q5c7', '/m/0fj52s', '/m/0cjs7', '/m/02f9f_', '/m/09k_b', '/m/02h19r', '/m/07c6l', '/m/05zsy', '/m/025dyy', '/m/07j87', '/m/01vbnl', '/m/05z55', '/m/01x3jk', '/m/0h8l4fh', '/m/02522', '/m/0c_jw', '/m/0crjs', '/m/02xwb', '/m/03rszm', '/m/0388q', '/m/0hf58v5', '/m/0cmf2', '/m/047j0r', '/m/0gm28', '/m/0130jx', '/m/081qc', '/m/0fqfqc', '/m/030610', '/m/01g317', '/m/01bfm9', '/m/0fz0h', '/m/05kyg_', '/m/04brg2', '/m/031n1', '/m/02jvh9', '/m/0k65p', '/m/03grzl', '/m/05r655', '/m/07fbm7', '/m/09tvcd', '/m/06nwz', '/m/05ctyq', '/m/0djtd', '/m/078n6m', '/m/0152hh', '/m/0ll1f78', '/m/02_n6y', '/m/01n5jq', '/m/09rvcxw', '/m/015qff', '/m/024g6', '/m/05gqfk', '/m/0dv77', '/m/057cc', '/m/02pkr5', '/m/03g8mr', '/m/014y4n', '/m/01g3x7', '/m/032b3c', '/m/02rdsp', '/m/02crq1', '/m/0cnyhnx', '/m/05bm6', '/m/0dtln', '/m/05kms', '/m/0fldg', '/m/043nyj', '/m/0grw1', '/m/0dzf4', '/m/0703r8', '/m/07mhn', '/m/050gv4', '/m/014j1m', '/m/04y4h8h', '/m/01s105', '/m/033rq4', '/m/0bt9lr', '/m/079cl', '/m/05vtc', '/m/054xkw', '/m/09g1w']

testimages(img_path = '/home/adam/Desktop/kaggleOID/data/test_imgs/test',
               descriptors_path = '/home/adam/Desktop/kaggleOID/data/descriptors.csv',
                output_path='/home/adam/Desktop/kaggleOID/FULLcsvs/fullFrcnn150cls.csv',
               bbox_threshold = 0.001,
               non_max_sup_thresh = 0.36,
               rpn_to_roi_overlap_thresh = 0.82,
                rest_classes=restricted_classes)
import sys, os
sys.path.append(os.path.join(os.getcwd(),'python/'))

import darknet as dn
import cv2
import numpy as np
import time

dn.set_gpu(0)

descriptors_path = '/home/adam/Desktop/kaggleOID/data/descriptors.csv'
img_path = '/home/adam/Desktop/kaggleOID/eval/eval_imgs'
net = dn.load_net("/home/adam/Desktop/kaggleOID/darknet/cfg/yolov3-openimages.cfg".encode('utf-8'),"/home/adam/Desktop/kaggleOID/darknet/yolov3-openimages.weights".encode('utf-8'), 0)
meta = dn.load_meta("/home/adam/Desktop/kaggleOID/darknet/cfg/openimages.data".encode('utf-8'))

eng_mapping = {}

with open(descriptors_path, 'r') as f:
    for i, line in enumerate(f):
        line_split = line.strip().split(',')
        (coded, engcoded) = line_split
        eng_mapping[engcoded] = coded


with open('/home/adam/Desktop/kaggleOID/EVALcsvs/evalYOLOv3.csv','w') as f:
    f.write('ImageID,LabelName,Score,YMin,XMin,YMax,XMax\n')

    beg_time=time.time()

    for idx, img_name in enumerate(sorted(os.listdir(img_path))):
        filepath = os.path.join(img_path, img_name)
        img = cv2.imread(filepath)
        r = dn.detect(net, meta, filepath.encode('utf-8'), thresh=.0001, hier_thresh=.0001,nms=.35)

        printimage=False

        for bbox in r:
            (x, y, w, h) = bbox[2]
            (x1, y1, x2, y2) = (int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2))

            if printimage:
                cv2.rectangle(img, (x1, y1), (x2, y2), (np.random.randint(0, 255), 0, 0), 2)
                textLabel = '{}: {}'.format(bbox[0], int(100 * bbox[1]))
                (retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                textOrg = (x1, y1 - 0)
                cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),(textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (0, 0, 0), 2)
                cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),(textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (255, 255, 255), -1)
                cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

            (height, width, _) = img.shape

            eng_label=bbox[0].decode("utf-8")

            if eng_label in eng_mapping.keys():
                f.write('{},{},{:.5f},{:.3f},{:.3f},{:.3f},{:.3f}\n'.format(img_name.split('.')[0], eng_mapping[eng_label],bbox[1], y1/height, x1/width, y2/height, x2/width))
            else:
                corresponding_label=None

                for label in eng_mapping.keys():
                    if eng_label.lower() in label.lower():
                        corresponding_label = label
                        break

                if corresponding_label!=None:
                    f.write('{},{},{:.5f},{:.3f},{:.3f},{:.3f},{:.3f}\n'.format(img_name.split('.')[0],eng_mapping[corresponding_label], bbox[1], y1/height, x1/width, y2/height, x2/width))

                    #print(eng_label)
                    #print(corresponding_label)
                else:
                    pass
                    #print('######################'+eng_label)

        print(img_name)
        print('Time left: {} , idx: {}'.format((time.time() - beg_time) / 60 / 60 / (idx + 1) * (7232 - idx), idx))


        if printimage:
            cv2.imwrite('/home/adam/Desktop/kaggleOID/darknet/results_imgs/{}.png'.format(idx), img)
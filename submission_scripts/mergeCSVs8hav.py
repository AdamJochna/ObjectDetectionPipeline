from keras_frcnn.simple_parser import getparentdict
from protos import string_int_label_map_pb2
from google.protobuf import text_format
import pandas as pd
import numpy as np
import time

#frcnn lowered edit of 2


probthresh=0.0022

def Frcnnfunc(x):
    return x/(2-x)

def Yolofunc(x):
    return np.power(x,0.7)

def Incepfunc(x):
    return x

def postFrcnnfunc(x):
    return np.power(x,2)

def postYolofunc(x):
    return x

def postIncepfunc(x):
    return x

def non_max_suppression_fast(boxes, probs, overlap_thresh=0.7):
    if len(boxes) == 0:
        return boxes, probs

    x1 = boxes[:, 1]
    y1 = boxes[:, 0]
    x2 = boxes[:, 3]
    y2 = boxes[:, 2]

    if np.all(np.less(x1, x2)) != True:
        print("err")
        return boxes, probs

    if np.all(np.less(y1, y2)) != True:
        print("err")
        return boxes, probs

    boxes = boxes.astype("float")

    pick = []
    area = (x2 - x1) * (y2 - y1)
    idxs = np.argsort(probs)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1_int = np.maximum(x1[i], x1[idxs[:last]])
        yy1_int = np.maximum(y1[i], y1[idxs[:last]])
        xx2_int = np.minimum(x2[i], x2[idxs[:last]])
        yy2_int = np.minimum(y2[i], y2[idxs[:last]])

        area_int = np.maximum(0, xx2_int - xx1_int) * np.maximum(0, yy2_int - yy1_int)
        overlap = area_int/(area[i] + area[idxs[:last]] - area_int + 1e-6)

        idxs = np.delete(idxs, np.concatenate(([last],np.where(overlap > overlap_thresh)[0])))

    boxes = boxes[pick].astype("float")
    probs = probs[pick]
    return boxes, probs


def _load_labelmap(labelmap_path):
  """Loads labelmap from the labelmap path.

  Args:
    labelmap_path: Path to the labelmap.

  Returns:
    A dictionary mapping class name to class numerical id
    A list with dictionaries, one dictionary per category.
  """

  label_map = string_int_label_map_pb2.StringIntLabelMap()
  with open(labelmap_path, 'r') as fid:
    label_map_string = fid.read()
    text_format.Merge(label_map_string, label_map)
  labelmap_dict = {}
  categories = []
  for item in label_map.item:
    labelmap_dict[item.name] = item.id
    categories.append({'id': item.id, 'name': item.name})
  return labelmap_dict, categories

hierarchypath = '/home/adam/Desktop/kaggleOID/data/bbox_labels_600_hierarchy.json'
parentdict = getparentdict(hierarchypath)

input_class_labelmap='/home/adam/Desktop/kaggleOID/eval/oid_object_detection_challenge_500_label_map.pbtxt'
class_label_map, categories = _load_labelmap(input_class_labelmap)
valid500classes=class_label_map.keys()

ImIdList=[]

with open('/home/adam/Desktop/kaggleOID/EXPcsvs/sample_submission.csv', 'r') as samsub:
    for i, line in enumerate(samsub):
        if i!=0:
            ImIdList.append(line.split(',')[0])

beg=time.time()

with open('/home/adam/Desktop/kaggleOID/inception/predictionsInception.csv', 'r') as fInc, \
    open('/home/adam/Desktop/kaggleOID/FULLcsvs/predictionsFRCNN.csv', 'r') as fFrc, \
    open('/home/adam/Desktop/kaggleOID/FULLcsvs/predictionsYOLOv3_OID.csv', 'r') as fYol, \
    open('/home/adam/Desktop/kaggleOID/EXPcsvs/subfin8.csv', 'w') as fSub:

    line = fInc.readline()
    line = fFrc.readline()
    line = fYol.readline()
    fSub.write('ImageId,PredictionString\n')

    boxlist = []
    NMS_boxes={}
    MARGIN_boxes={}

    for idx, ImID in enumerate(sorted(ImIdList)):
        NMS_boxes[ImID] = []
        MARGIN_boxes[ImID] = []

    for idx, ImID in enumerate(sorted(ImIdList)):

        while (True):
            line = fInc.readline()
            if line == '':
                break
            ImageID, LabelName, Score, YMin, XMin, YMax, XMax = line.split(',')
            XMin = max(min(1.0, float(XMin)), 0.0)
            YMin = max(min(1.0, float(YMin)), 0.0)
            XMax = max(min(1.0, float(XMax)), 0.0)
            YMax = max(min(1.0, float(YMax)), 0.0)
            Score = max(min(1.0, float(Score)), 0.0)
            Score=Incepfunc(Score)

            if XMin < XMax and YMin < YMax:
                if Score > probthresh:
                    Score = postIncepfunc(Score)
                    if LabelName in valid500classes:
                        NMS_boxes[ImageID].append([LabelName,Score, YMin, XMin, YMax, XMax])

                    for parent in parentdict[LabelName]:
                        if parent in valid500classes:
                            NMS_boxes[ImageID].append([parent,Score, YMin, XMin, YMax, XMax])

                elif Score > 0.0003:
                    Score = postIncepfunc(Score)

                    if LabelName in valid500classes:
                        MARGIN_boxes[ImageID].append([LabelName,Score, YMin, XMin, YMax, XMax])

            if ImageID != ImID:
                break

        while (True):
            line = fYol.readline()
            if line == '':
                break
            ImageID, LabelName, Score, YMin, XMin, YMax, XMax = line.split(',')
            XMin = max(min(1.0, float(XMin)), 0.0)
            YMin = max(min(1.0, float(YMin)), 0.0)
            XMax = max(min(1.0, float(XMax)), 0.0)
            YMax = max(min(1.0, float(YMax)), 0.0)
            Score = max(min(1.0, float(Score)), 0.0)
            Score = Yolofunc(Score)

            if XMin < XMax and YMin < YMax:
                if Score > probthresh:
                    Score = postYolofunc(Score)
                    if LabelName in valid500classes:
                        NMS_boxes[ImageID].append([LabelName,Score, YMin, XMin, YMax, XMax])

                    for parent in parentdict[LabelName]:
                        if parent in valid500classes:
                            NMS_boxes[ImageID].append([parent,Score, YMin, XMin, YMax, XMax])
                else:
                    Score = postYolofunc(Score)
                    pass

            if ImageID != ImID:
                break

        while (True):
            line = fFrc.readline()
            if line == '':
                break
            ImageID, LabelName, Score, YMin, XMin, YMax, XMax = line.split(',')
            XMin = max(min(1.0, float(XMin)), 0.0)
            YMin = max(min(1.0, float(YMin)), 0.0)
            XMax = max(min(1.0, float(XMax)), 0.0)
            YMax = max(min(1.0, float(YMax)), 0.0)
            Score = max(min(1.0, float(Score)), 0.0)
            Score = Frcnnfunc(Score)

            if XMin < XMax and YMin < YMax:
                if Score > probthresh:
                    Score = postFrcnnfunc(Score)
                    if LabelName in valid500classes:
                        NMS_boxes[ImageID].append([LabelName, Score, YMin, XMin, YMax, XMax])

                    for parent in parentdict[LabelName]:
                        if parent in valid500classes:
                            NMS_boxes[ImageID].append([parent,Score, YMin, XMin, YMax, XMax])
                else:
                    Score = postFrcnnfunc(Score)
                    pass

            if ImageID != ImID:
                break

        tmpdf=pd.DataFrame.from_records(NMS_boxes[ImID],columns=['LabelName', 'Score', 'YMin', 'XMin', 'YMax', 'XMax'])

        for label, group in tmpdf.groupby('LabelName'):
            boxes = group[['YMin', 'XMin', 'YMax', 'XMax']].values
            probs = group['Score'].values
            boxes, probs = non_max_suppression_fast(boxes, probs, overlap_thresh=0.7)

            for j in range(boxes.shape[0]):
                boxlist.append('{} {} {} {} {} {}'.format(label,
                                                          '{:.3f}'.format(probs.item(j)*0.999+0.001).lstrip('0'),
                                                          '{:.3f}'.format(boxes[j,1]).lstrip('0'),
                                                          '{:.3f}'.format(boxes[j,0]).lstrip('0'),
                                                          '{:.3f}'.format(boxes[j,3]).lstrip('0'),
                                                          '{:.3f}'.format(boxes[j,2]).lstrip('0')))

        for box in MARGIN_boxes[ImID]:
            boxlist.append('{} {} {} {} {} {}'.format(box[0],
                                                      '{:.3f}'.format(box[1]*0.999+0.001).lstrip('0'),
                                                      '{:.2f}'.format(box[3]).lstrip('0'),
                                                      '{:.2f}'.format(box[2]).lstrip('0'),
                                                      '{:.2f}'.format(box[5]).lstrip('0'),
                                                      '{:.2f}'.format(box[4]).lstrip('0')))

        fSub.write(ImID + ',' + str(' '.join(boxlist)) + '\n')
        boxlist=[]
        del NMS_boxes[ImID]
        del MARGIN_boxes[ImID]

        print('Time left: {} , idx: {}'.format((time.time() - beg) / 60 / 60 / (idx + 1) * (99999 - idx), idx))
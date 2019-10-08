# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


from a00_utils_and_constants import *
from a01_ensemble_boxes_functions import *


def create_csv_for_retinanet_multiple_predictions(input_dirs, out_file, label_arr, skip_box_thr=0.001, intersection_thr=0.5, limit_boxes=600, type='avg'):
    out = open(out_file, 'w')
    out.write('ImageID,LabelName,Score,YMin,XMin,YMax,XMax\n')

    d1, d2 = get_description_for_labels()
    files = glob.glob(input_dirs[0] + '*.pkl')
    beg = time.time()

    for idx,f in enumerate(sorted(files)):
        print('Time_left: {} , idx: {}'.format((time.time() - beg) / 60 / 60 / (idx + 1) * (100000 - idx), idx))
        id = os.path.basename(f)[:-4]
        print('Go for ID: {}'.format(id))
        boxes_list = []
        scores_list = []
        labels_list = []
        for i in range(len(input_dirs)):
            f1 = input_dirs[i] + id + '.pkl'
            boxes, scores, labels = load_from_file_fast(f1)
            boxes_list.append(boxes)
            scores_list.append(scores)
            labels_list.append(labels)

        filtered_boxes = filter_boxes_v2(boxes_list, scores_list, labels_list, skip_box_thr)
        merged_boxes = merge_all_boxes_for_image(filtered_boxes, intersection_thr, type)
        print(id, len(filtered_boxes), len(merged_boxes))
        if len(merged_boxes) > limit_boxes:
            # sort by score
            merged_boxes = merged_boxes[merged_boxes[:, 1].argsort()[::-1]][:limit_boxes]

        for i in range(len(merged_boxes)):
            label = int(merged_boxes[i][0])
            score = merged_boxes[i][1]
            b = merged_boxes[i][2:]

            google_name = label_arr[label]
            if '/' not in google_name:
                try:
                    google_name = d2[google_name]
                except:
                    continue

            xmin = b[0]
            if xmin < 0:
                xmin = 0
            if xmin > 1:
                xmin = 1

            xmax = b[2]
            if xmax < 0:
                xmax = 0
            if xmax > 1:
                xmax = 1

            ymin = b[1]
            if ymin < 0:
                ymin = 0
            if ymin > 1:
                ymin = 1

            ymax = b[3]
            if ymax < 0:
                ymax = 0
            if ymax > 1:
                ymax = 1

            if (xmax < xmin):
                print('X min value larger than max value {}: {} {}'.format(label_arr[label], xmin, xmax))
                exit()

            if (ymax < ymin):
                print('Y min value larger than max value {}: {} {}'.format(label_arr[label], xmin, xmax))
                exit()

            if abs(xmax - xmin) < 1e-5:
                print('Too small diff for {}: {} and {}'.format(label_arr[label], xmin, xmax))
                continue

            if abs(ymax - ymin) < 1e-5:
                print('Too small diff for {}: {} and {}'.format(label_arr[label], ymin, ymax))
                continue

            out.write('{},{},{:.5f},{:.3f},{:.3f},{:.3f},{:.3f}\n'.format(f.split('/')[-1].split('.')[0],google_name,score,ymin,xmin,ymax,xmax))

    out.close()



skip_box_thr = 0.001
intersection_thr = 0.55
limit_boxes = 600
type = 'avg'

input_dirs = ['/home/adam/Desktop/kaggleOID/Keras-RetinaNet-for-Open-Images-Challenge-2018-master/cache_retinanet_level_1_resnet101/',
              '/home/adam/Desktop/kaggleOID/Keras-RetinaNet-for-Open-Images-Challenge-2018-master/cache_retinanet_level_1_resnet152/']
labels_arr = LEVEL_1_LABELS
create_csv_for_retinanet_multiple_predictions(input_dirs,
                                     '/home/adam/Desktop/kaggleOID/FULLcsvs/fullZFturboMerged.csv',
                                     labels_arr,
                                     skip_box_thr, intersection_thr, limit_boxes, type=type)

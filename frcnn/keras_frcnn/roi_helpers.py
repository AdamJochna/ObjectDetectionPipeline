import numpy as np
import math
from . import data_generators
import copy
import multiprocessing as mp

def IOUvec(bboxes1, bboxes2, rearange1):
    if rearange1:
        x11, x12, y11, y12 = np.split(bboxes1, 4, axis=1)
    else:
        x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)

    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)

    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))

    interArea = np.maximum((xB - xA), 0) * np.maximum((yB - yA), 0)
    boxAArea = (x12 - x11) * (y12 - y11)
    boxBArea = (x22 - x21) * (y22 - y21)

    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea + 1e-6)
    iou = np.maximum(iou,0)
    iou = np.minimum(iou,1)

    return iou

def calc_iou(R, img_data, C, class_mapping , parentdict):
    bboxes = img_data['bboxes']
    (width, height) = (img_data['width'], img_data['height'])
    (resized_width, resized_height) = data_generators.get_new_img_size(width, height, C.im_size)
    gta = np.empty((len(bboxes), 4))

    for bbox_num, bbox in enumerate(bboxes):
        gta[bbox_num, 0] = bbox['x1'] * (resized_width / width)/C.rpn_stride
        gta[bbox_num, 1] = bbox['x2'] * (resized_width / width)/C.rpn_stride
        gta[bbox_num, 2] = bbox['y1'] * (resized_height / height)/C.rpn_stride
        gta[bbox_num, 3] = bbox['y2'] * (resized_height / height)/C.rpn_stride

    IOUtable = IOUvec(gta, R, True)
    bestIOUfor_proposal=np.amax(IOUtable,axis=0)
    bestIdxfor_proposal=np.argmax(IOUtable, axis=0)

    output = mp.Queue()
    outputlist=[]

    def process_idxR(ix):
        (x1, y1, x2, y2) = R[ix, :]
        best_iou = copy.copy(bestIOUfor_proposal.item(ix))
        best_bbox = copy.copy(bestIdxfor_proposal.item(ix))

        if C.classifier_min_overlap <= best_iou:
            w = x2 - x1
            h = y2 - y1

            if best_iou < C.classifier_max_overlap:
                cls_name = 'bg'
            elif C.classifier_max_overlap <= best_iou:
                cls_name = bboxes[best_bbox]['class']
                cxg = (gta[best_bbox, 0] + gta[best_bbox, 1]) / 2
                cyg = (gta[best_bbox, 2] + gta[best_bbox, 3]) / 2
                cx = x1 + w / 2
                cy = y1 + h / 2
                tx = (cxg - cx) / w
                ty = (cyg - cy) / h
                tw = np.log((gta[best_bbox, 1] - gta[best_bbox, 0]) / w)
                th = np.log((gta[best_bbox, 3] - gta[best_bbox, 2]) / h)

            class_label = np.zeros(len(class_mapping))
            np.put(class_label,class_mapping[cls_name],1)

            for parent in parentdict[cls_name]:
                np.put(class_label, class_mapping[parent], 1)

            coords=np.zeros(4 * (len(class_mapping) - 1))
            labels=np.zeros(4 * (len(class_mapping) - 1))

            if cls_name != 'bg':
                sx, sy, sw, sh = C.classifier_regr_std

                label_pos = 4 * class_mapping[cls_name]
                np.put(coords, [label_pos,label_pos+1,label_pos+2,label_pos+3], [float(sx*tx), float(sy*ty), float(sw*tw), float(sh*th)])
                np.put(labels, [label_pos,label_pos+1,label_pos+2,label_pos+3], [1, 1, 1, 1])

                for parent in parentdict[cls_name]:
                    label_pos = 4 * class_mapping[parent]
                    np.put(coords, [label_pos,label_pos+1,label_pos+2,label_pos+3], [sx * tx, sy * ty, sw * tw, sh * th])
                    np.put(labels, [label_pos,label_pos+1,label_pos+2,label_pos+3], [1, 1, 1, 1])

            output.put((np.array([x1, y1, w, h]),best_iou,copy.deepcopy(class_label),copy.deepcopy(coords),copy.deepcopy(labels)))

    def process_batch_idxR(idx_batch):
        for num in idx_batch:
            process_idxR(num)

    batchesnum=4
    idxbatches=[[] for b in range(batchesnum)]

    for idx in range(R.shape[0]):
        idxbatches[idx%batchesnum].append(idx)

    processes = [mp.Process(target=process_batch_idxR, args=[idx_batch]) for idx_batch in idxbatches]

    for p in processes:
        p.daemon = True
        p.start()

    while True:
        running = any(p.is_alive() for p in processes)
        while not output.empty():
            outputlist.append(output.get())
        if not running:
            break

    for p in processes:
        p.join()

    x_roi = np.empty((len(outputlist),4))
    IoUs = np.empty(len(outputlist))
    y_class_num = np.empty((len(outputlist),len(class_mapping)))
    y_class_regr_coords = np.empty((len(outputlist),4*(len(class_mapping)-1)))
    y_class_regr_label = np.empty((len(outputlist),4*(len(class_mapping)-1)))

    for i in range(len(outputlist)):
        out=outputlist[i]

        x_roi[i]=out[0]
        IoUs[i]=out[1]
        y_class_num[i,:]=out[2]
        y_class_regr_coords[i,:]=out[3]
        y_class_regr_label[i,:]=out[4]

    if len(outputlist) == 0:
        return None, None, None, None
    else:
        return np.expand_dims(x_roi, axis=0), np.expand_dims(y_class_num, axis=0), np.expand_dims(np.concatenate([y_class_regr_label,y_class_regr_coords],axis=1), axis=0), IoUs

def apply_regr(x, y, w, h, tx, ty, tw, th):
    try:
        cx = x + w/2.
        cy = y + h/2.
        cx1 = tx * w + cx
        cy1 = ty * h + cy
        w1 = math.exp(tw) * w
        h1 = math.exp(th) * h
        x1 = cx1 - w1/2.
        y1 = cy1 - h1/2.
        x1 = int(round(x1))
        y1 = int(round(y1))
        w1 = int(round(w1))
        h1 = int(round(h1))

        return x1, y1, w1, h1

    except ValueError:
        return x, y, w, h
    except OverflowError:
        return x, y, w, h
    except Exception as e:
        print(e)
        return x, y, w, h


def apply_regr_np(X, T):
    try:
        x = X[0, :, :]
        y = X[1, :, :]
        w = X[2, :, :]
        h = X[3, :, :]

        tx = T[0, :, :]
        ty = T[1, :, :]
        tw = T[2, :, :]
        th = T[3, :, :]

        cx = x + w/2.
        cy = y + h/2.
        cx1 = tx * w + cx
        cy1 = ty * h + cy

        w1 = np.exp(tw.astype(np.float64)) * w
        h1 = np.exp(th.astype(np.float64)) * h
        x1 = cx1 - w1/2.
        y1 = cy1 - h1/2.

        x1 = np.round(x1)
        y1 = np.round(y1)
        w1 = np.round(w1)
        h1 = np.round(h1)
        return np.stack([x1, y1, w1, h1])
    except Exception as e:
        print(e)
        return X


def non_max_suppression_fast(boxes, probs, overlap_thresh=0.9, max_boxes=300):
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    np.testing.assert_array_less(x1, x2)
    np.testing.assert_array_less(y1, y2)

    if boxes.dtype.kind == "i":
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

        if len(pick) >= max_boxes:
            break

    boxes = boxes[pick].astype("int")
    probs = probs[pick]
    return boxes, probs

def rpn_to_roi(rpn_layer, regr_layer, C, dim_ordering, use_regr=True, max_boxes=300,overlap_thresh=0.9):

    regr_layer = regr_layer / C.std_scaling
    anchor_sizes = C.anchor_box_scales
    anchor_ratios = C.anchor_box_ratios

    assert rpn_layer.shape[0] == 1
    (rows, cols) = rpn_layer.shape[1:3]
    curr_layer = 0
    A = np.zeros((4, rpn_layer.shape[1], rpn_layer.shape[2], rpn_layer.shape[3]))

    for anchor_size in anchor_sizes:
        for anchor_ratio in anchor_ratios:

            anchor_x = (anchor_size * anchor_ratio[0])/C.rpn_stride
            anchor_y = (anchor_size * anchor_ratio[1])/C.rpn_stride
            regr = regr_layer[0, :, :, 4 * curr_layer:4 * curr_layer + 4]
            regr = np.transpose(regr, (2, 0, 1))

            X, Y = np.meshgrid(np.arange(cols),np. arange(rows))

            A[0, :, :, curr_layer] = X - anchor_x/2
            A[1, :, :, curr_layer] = Y - anchor_y/2
            A[2, :, :, curr_layer] = anchor_x
            A[3, :, :, curr_layer] = anchor_y

            if use_regr:
                A[:, :, :, curr_layer] = apply_regr_np(A[:, :, :, curr_layer], regr)

            A[2, :, :, curr_layer] = np.maximum(1, A[2, :, :, curr_layer])
            A[3, :, :, curr_layer] = np.maximum(1, A[3, :, :, curr_layer])
            A[2, :, :, curr_layer] += A[0, :, :, curr_layer]
            A[3, :, :, curr_layer] += A[1, :, :, curr_layer]

            A[0, :, :, curr_layer] = np.maximum(0, A[0, :, :, curr_layer])
            A[1, :, :, curr_layer] = np.maximum(0, A[1, :, :, curr_layer])
            A[2, :, :, curr_layer] = np.minimum(cols-1, A[2, :, :, curr_layer])
            A[3, :, :, curr_layer] = np.minimum(rows-1, A[3, :, :, curr_layer])

            curr_layer += 1

    all_boxes = np.reshape(A.transpose((0, 3, 1,2)), (4, -1)).transpose((1, 0))
    all_probs = rpn_layer.transpose((0, 3, 1, 2)).reshape((-1))

    x1 = all_boxes[:, 0]
    y1 = all_boxes[:, 1]
    x2 = all_boxes[:, 2]
    y2 = all_boxes[:, 3]

    idxs = np.where((x1 - x2 >= 0) | (y1 - y2 >= 0))
    all_boxes = np.delete(all_boxes, idxs, 0)
    all_probs = np.delete(all_probs, idxs, 0)
    result = non_max_suppression_fast(all_boxes, all_probs, overlap_thresh=overlap_thresh, max_boxes=max_boxes)[0]

    return result

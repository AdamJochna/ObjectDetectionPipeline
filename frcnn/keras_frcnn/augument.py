import numpy as np
import imgaug.augmenters as iaa
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import copy

def random_augument_image(img,imgdata):
    ia.seed(np.random.randint(1000))
    bboxlist=[]

    for i, bbox in enumerate(imgdata['bboxes']):
        bboxlist.append(BoundingBox(x1=bbox['x1'], x2=bbox['x2'], y1=bbox['y1'], y2=bbox['y2'], label=bbox['class']))

    bbs = BoundingBoxesOnImage(bboxlist, shape=img.shape)
    seq = iaa.Sequential([iaa.Fliplr(0.5),
                          iaa.Add((-10, 10))])
    image_aug, bbs_aug = seq(image=img, bounding_boxes=bbs)

    bbsarr=bbs_aug.to_xyxy_array('int_')

    for i,bbox in enumerate(imgdata['bboxes']):
        bbox['x1'] = bbsarr.item((i,0))
        bbox['y1'] = bbsarr.item((i,1))
        bbox['x2'] = bbsarr.item((i,2))
        bbox['y2'] = bbsarr.item((i,3))

    return image_aug,imgdata
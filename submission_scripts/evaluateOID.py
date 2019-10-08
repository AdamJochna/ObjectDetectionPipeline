from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time

import pandas as pd
from google.protobuf import text_format

from object_detection.metrics import oid_challenge_evaluation_utils as utils
from object_detection.protos import string_int_label_map_pb2
from object_detection.utils import object_detection_evaluation

input_annotations_boxes='/home/adam/Desktop/kaggleOID/eval/challenge-2019-validation-detection-bbox_expanded_v2.csv'
input_annotations_labels='/home/adam/Desktop/kaggleOID/eval/challenge-2019-validation-detection-human-imagelabels_expanded_v2.csv'
input_class_labelmap='/home/adam/Desktop/kaggleOID/eval/oid_object_detection_challenge_500_label_map.pbtxt'

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


def evaluate(input_predictions_path,restricted=None):
  all_location_annotations = pd.read_csv(input_annotations_boxes)
  all_label_annotations = pd.read_csv(input_annotations_labels)
  all_label_annotations.rename(columns={'Confidence': 'ConfidenceImageLabel'}, inplace=True)
  all_annotations = pd.concat([all_location_annotations, all_label_annotations])
  class_label_map, categories = _load_labelmap(input_class_labelmap)
  challenge_evaluator = (object_detection_evaluation.OpenImagesChallengeEvaluator(categories, evaluate_masks=False))

  all_predictions = pd.read_csv(input_predictions_path)
  valid500classes = class_label_map.keys()
  if restricted!=None:
    valid500classes=list(set(valid500classes).intersection(set(restricted)))

  indexNames = all_predictions[~all_predictions.LabelName.isin(valid500classes)].index
  all_predictions.drop(indexNames, inplace=True)
  groupby_set = all_predictions.groupby('ImageID')
  imId_list=groupby_set.groups.keys()

  images_processed = 0

  for _, groundtruth in enumerate(all_annotations.groupby('ImageID')):
    logging.info('Processing image %d', images_processed)
    image_id, image_groundtruth = groundtruth

    if image_id in imId_list:
      groundtruth_dictionary = utils.build_groundtruth_dictionary(image_groundtruth.loc[image_groundtruth['LabelName'].isin(valid500classes)], class_label_map)
      challenge_evaluator.add_single_ground_truth_image_info(image_id, groundtruth_dictionary)
      prediction_dictionary = utils.build_predictions_dictionary(groupby_set.get_group(image_id),class_label_map)
      challenge_evaluator.add_single_detected_image_info(image_id,prediction_dictionary)

    if images_processed%50==0:
      if images_processed==0:
        beg_time=time.time()
      print(images_processed)
      #print('Time left: {} , idx: {}'.format((time.time() - beg_time) / 60 / 60 / (images_processed + 1) * (7232 - images_processed), images_processed))

    images_processed += 1

  metrics = challenge_evaluator.evaluate()
  print(input_predictions_path)
  print(metrics)
  return metrics['OpenImagesDetectionChallenge_Precision/mAP@0.5IOU']
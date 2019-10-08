import os
import glob
from keras_frcnn.simple_parser import get_data
from protos import string_int_label_map_pb2
from google.protobuf import text_format

input_path='/home/adam/Desktop/kaggleOID/small_eval_restricted_v2/challenge-2019-validation-detection-bbox_expanded.csv'
descriptors_path='/home/adam/Desktop/kaggleOID/data/descriptors.csv'


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

input_class_labelmap='/home/adam/Desktop/kaggleOID/small_eval_restricted_v2/oid_object_detection_challenge_500_label_map.pbtxt'
class_label_map, categories = _load_labelmap(input_class_labelmap)
valid500classes=class_label_map.keys()

all_data, _, _ , eng_mapping ,_ ,classes_total=get_data(input_path,descriptors_path,10000000,-1)
initclasses=[k for k,v in classes_total.items() if (v>0 and v<50)]
expected=10
data_to_use=[]
data_count={k:0 for (k,v) in classes_total.items()}

for cls in initclasses:
    for img in all_data:
        addflag = 0

        for bbox in img['bboxes']:
            if bbox['class'] == cls:
                addflag = 1
                break

        if addflag == 1:
            data_to_use.append(img)
            all_data.remove(img)

            for bbox in img['bboxes']:
                data_count[bbox['class']] += 1

            break


while data_count[min(data_count, key=data_count.get)] < expected:
    currclass = min(data_count, key=data_count.get)

    if data_count[currclass]>=classes_total[currclass]:
        data_count[currclass]+=1
        continue

    for img in all_data:
        addflag = 0

        for bbox in img['bboxes']:
            if bbox['class'] == currclass:
                addflag = 1
                break

        if addflag == 1:
            for bbox in img['bboxes']:
                data_count[bbox['class']] += 1

            data_to_use.append(img)
            all_data.remove(img)
            break

uniqueid=list(set([img['ImageID'] for img in data_to_use]))
print(len(uniqueid))

finalcount={k:0 for (k,v) in classes_total.items()}

for img in data_to_use:
    for bbox in img['bboxes']:
            finalcount[bbox['class']]+=1

for f in sorted(list(set([v for (k,v) in finalcount.items()]))):
    for key in [k for (k,v) in finalcount.items() if v==f]:
        if key in valid500classes:
            print(finalcount[key],classes_total[key],key)


print(len(uniqueid))


files = glob.glob('/home/adam/Desktop/kaggleOID/small_eval_restricted_v2/eval_imgs/*')
for f in files:
    namejpg=f.split('/')[7]
    namenojpg=namejpg.split('.')[0]

    if not namenojpg in uniqueid:
        os.remove(f)

with open('/home/adam/Desktop/kaggleOID/small_eval_restricted_v2/challenge-2019-validation-detection-bbox_expanded.csv', 'r') as f1:
    with open('/home/adam/Desktop/kaggleOID/small_eval_restricted_v2/challenge-2019-validation-detection-bbox_expanded_v2.csv', 'w') as f2:
        for i, line in enumerate(f1):
            if i==0:
                f2.write(line)
            else:
                line_split = line.strip().split(',')
                if line_split[0] in uniqueid:
                    f2.write(line)

with open('/home/adam/Desktop/kaggleOID/small_eval_restricted_v2/challenge-2019-validation-detection-human-imagelabels_expanded.csv', 'r') as f1:
    with open('/home/adam/Desktop/kaggleOID/small_eval_restricted_v2/challenge-2019-validation-detection-human-imagelabels_expanded_v2.csv', 'w') as f2:
        for i, line in enumerate(f1):
            if i==0:
                f2.write(line)
            else:
                line_split = line.strip().split(',')
                if line_split[0] in uniqueid:
                    f2.write(line)
import numpy as np
import os
import sys
import tensorflow as tf
import time

from distutils.version import StrictVersion
import cv2

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

from models import label_map_util

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = '/home/adam/Desktop/kaggleOID/inception_resnet_v2/export_folder/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '/home/adam/Desktop/kaggleOID/models/research/object_detection/data/oid_v4_label_map.pbtxt'
img_path = "/home/adam/Desktop/kaggleOID/eval/eval_imgs"

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=False)

print(category_index)

with open('/home/adam/Desktop/kaggleOID/EVALcsvs/evalInception.csv','w') as f:
    f.write('ImageID,LabelName,Score,YMin,XMin,YMax,XMax\n')

    with detection_graph.as_default():
        with tf.Session() as sess:
            for idx, img_name in enumerate(sorted(os.listdir(img_path))):
                ImgID = img_name.split('.')[0]
                filepath = os.path.join(img_path, img_name)

                image_np = cv2.imread(filepath)
                image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in ['detection_class_names','num_detections', 'detection_boxes', 'detection_scores','detection_classes', 'detection_masks']:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict,feed_dict={image_tensor: image_np_expanded})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.int64)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]

                if (idx == 0):
                    beg = time.time()

                for i in range(output_dict['detection_classes'].shape[0]):
                    if output_dict['detection_scores'].item(i)>=0.00002:
                        f.write('{},{},{:.5f},{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(ImgID,
                                                                                    category_index[output_dict['detection_classes'].item(i)]['name'],
                                                                                    output_dict['detection_scores'].item(i),
                                                                                    output_dict['detection_boxes'][i, 0],
                                                                                    output_dict['detection_boxes'][i, 1],
                                                                                    output_dict['detection_boxes'][i, 2],
                                                                                    output_dict['detection_boxes'][i, 3]))

                print(img_name)
                print('Time left: {} , idx: {}'.format((time.time() - beg) / 60 / 60 / (idx + 1) * (7232 - idx), idx))


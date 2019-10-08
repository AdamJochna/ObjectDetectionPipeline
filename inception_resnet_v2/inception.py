import os

os.system("ulimit -a")

import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from six import BytesIO
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps
import time

print("The following GPU devices are available: %s" % tf.test.gpu_device_name())

module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
img_path = "/home/adam/Desktop/kaggleOID/data/test_imgs/test"

with open('/home/adam/Desktop/kaggleOID/inception/resultsinception.csv','w') as f:
    f.write('ImageID,LabelName,Score,YMin,XMin,YMax,XMax\n')


    with tf.device('/device:GPU:0'):
        with tf.Graph().as_default():
            detector = hub.Module(module_handle)
            image_string_placeholder = tf.placeholder(tf.string)
            decoded_image = tf.image.decode_jpeg(image_string_placeholder)
            # Module accepts as input tensors of shape [1, height, width, 3], i.e. batch
            # of size 1 and type tf.float32.
            decoded_image_float = tf.image.convert_image_dtype(image=decoded_image, dtype=tf.float32)
            module_input = tf.expand_dims(decoded_image_float, 0)
            result = detector(module_input, as_dict=True)
            init_ops = [tf.global_variables_initializer(), tf.tables_initializer()]

            session = tf.Session()
            session.run(init_ops)

            for idx, img_name in enumerate(sorted(os.listdir(img_path))):
                ImgID = img_name.split('.')[0]
                filepath = os.path.join(img_path, img_name)

                with tf.gfile.Open(filepath, "rb") as binfile:
                    image_string = binfile.read()

                result_out, image_out = session.run([result, decoded_image],feed_dict={image_string_placeholder: image_string})
                if(idx==0):
                    beg = time.time()

                for i in range(result_out['detection_class_names'].shape[0]):
                    f.write('{},{},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n'.format(ImgID,
                                                                                result_out['detection_class_names'].item(i).decode("utf-8"),
                                                                                result_out['detection_scores'].item(i),
                                                                                result_out['detection_boxes'][i, 0],
                                                                                result_out['detection_boxes'][i, 1],
                                                                                result_out['detection_boxes'][i, 2],
                                                                                result_out['detection_boxes'][i, 3]))

                print('Time left: {} , idx: {}'.format((time.time() - beg) / 60 / 60 / (idx + 1) * (99999-idx),idx))

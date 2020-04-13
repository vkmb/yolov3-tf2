import os
import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('image', './data/girl.png', 'path to input image')
flags.DEFINE_string('image_folder', '', 'path to input image')
flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_string('output', './output.jpg', 'path to output image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


def main(_argv):
    img_list = []
    img_annotion_file = []
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    if FLAGS.tfrecord:
        dataset = load_tfrecord_dataset(
            FLAGS.tfrecord, FLAGS.classes, FLAGS.size)
        dataset = dataset.shuffle(512)
        img_raw, _label = next(iter(dataset.take(1)))
        img_list.append(img_raw)
        img_annotion_file.append(f'{_label}tfrecord.txt')

    

    elif FLAGS.image_folder:
        if not os.path.exists(FLAGS.image_folder):
            exit()
        file_list =  sorted(os.listdir(FLAGS.image_folder))
        for img_file_name in file_list:
            try:
                img_file_name = FLAGS.image_folder+img_file_name
                if '.jpg' in img_file_name:
                    # logging.info(f'{img_file_name}')
                    img_raw = tf.image.decode_image(open(img_file_name, 'rb').read(), channels=3)
                    img_annotion_file.append('{}.txt'.format(img_file_name.split('.jpg')[0]))
                    img_list.append(img_raw)
                    
            except:
                
                continue
            
    elif FLAGS.image:
        img_raw = tf.image.decode_image(open(FLAGS.image, 'rb').read(), channels=3)
        img_list.append(img_raw)
        img_annotion_file.append('.{}.txt'.format(FLAGS.image.split('.')[1]))
    
    logging.info(f'{len(img_list)} files loaded')
    # p346, p584, p93
    for index, img in enumerate(img_list):
        img = tf.expand_dims(img, 0)
        img = transform_images(img, FLAGS.size)
        # logging.info(f'{img_annotion_file[index]}')
        # t1 = time.time()
        boxes, scores, classes, nums = yolo(img)
        # t2 = time.time()
        # logging.info('time: {}'.format(t2 - t1))

        # logging.info('detections:')
        with open(img_annotion_file[index], mode='a+') as file:
            lines_to_write = []
            line_to_write = ""
            for i in range(nums[0]):
                if int(classes[0][i]) == 0 and  np.array(scores[0][i]) > .75:
                    line_to_write = "2"
                    for bnd in np.array(boxes[0][i]):
                         line_to_write += " "+str(bnd)
                    line_to_write += "\r\n"
                    if line_to_write not in lines_to_write:
                        lines_to_write.append(line_to_write)
                # logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                #                                 np.array(scores[0][i]),
                #                                 np.array(boxes[0][i])))
            file.writelines(lines_to_write)
            file.close()

        # img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
        # img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        # cv2.imwrite(FLAGS.output, img)
        # logging.info('output saved to: {} '.format(img_annotion_file[index]))
        if (index+1) % 100 == 0:
            logging.info('{} files have been processed'.format(index+1))
            # break

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

import os
import math
import shutil
from tqdm import tqdm
from absl.flags import FLAGS
from absl import app, flags, logging

flags.DEFINE_string('src', '', 'path to all images file')
flags.DEFINE_string('train_dst', './data/train', 'path to place train files')
flags.DEFINE_string('val_dst', './data/val', 'path to place validation files')
flags.DEFINE_float('train_size', .6, 'path to place train files')
flags.DEFINE_float('val_size', .4, 'path to place validation files')

def copy(file_name_list, src, dst):
    logging.info(f"\tCopying files to {dst} folder")
    for file_name in tqdm(file_name_list):
        name = file_name.split('.jpg')[0]
        txt_file_name = name + '.txt'
        xml_file_name = name + '.xml'
        if os.path.exists(dst+'/JPEGImage'):
            shutil.copy2(os.path.join(src, file_name), os.path.join(dst+'/JPEGImage', file_name))
        if os.path.exists(os.path.join(src, txt_file_name)) and os.path.exists(dst+'/txt'):
            shutil.copy2(os.path.join(src, txt_file_name), os.path.join(dst+'/txt', txt_file_name))
        if os.path.exists(os.path.join(src, xml_file_name)) and os.path.exists(dst+'/Annotation'):
            shutil.copy2(os.path.join(src, xml_file_name), os.path.join(dst+'/Annotation', xml_file_name))

def main(_argv):
    if FLAGS.src == '' or not os.path.exists(FLAGS.src):
        logging.error("Source folder not found")
        exit()
    if not os.path.exists(FLAGS.train_dst):
        os.makedirs(FLAGS.train_dst)
        os.makedirs(FLAGS.train_dst+'/JPEGImage')
        os.makedirs(FLAGS.train_dst+'/txt')
        os.makedirs(FLAGS.train_dst+'/Annotation')
    if not os.path.exists(FLAGS.val_dst):
        os.makedirs(FLAGS.val_dst)
        os.makedirs(FLAGS.val_dst+'/JPEGImage')
        os.makedirs(FLAGS.val_dst+'/txt')
        os.makedirs(FLAGS.val_dst+'/Annotation')
    
    file_list = os.listdir(FLAGS.src)
    
    neg_files = []
    pos_files = []

    logging.info("\tLoading files from source folder")
    for file_name in tqdm(file_list):
        if file_name.endswith('.jpg'):
            if file_name.startswith('neg'):
                neg_files.append(file_name)
            elif file_name.startswith('pos'):
                pos_files.append(file_name)
        

    train_size = int(len(pos_files) * FLAGS.train_size)
    train_files_list = pos_files[:train_size]
    val_files_list = pos_files[train_size:]

    train_size = int(len(neg_files) * FLAGS.train_size)
    train_files_list += neg_files[:train_size]
    val_files_list += neg_files[train_size:]

    copy(train_files_list, FLAGS.src, FLAGS.train_dst)
    copy(val_files_list, FLAGS.src, FLAGS.val_dst)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

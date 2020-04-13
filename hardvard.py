import os
import shutil
import xml.etree.ElementTree as et
dir_name = './data/Hardhat/Test/Annotation'
img_src = './data/Hardhat/Test/JPEGImage'
destination = './data/val/Annotation'
img_destination = './data/val/JPEGImage'


for file_name in os.listdir(dir_name):
    try:
        
        tree = et.parse(os.path.join(dir_name, file_name))
        img_file_name = ""
        root = tree.getroot()
        for elem in root:
            if elem.tag == 'object':
                for subelem in elem:
                    if subelem.tag == 'name':
                        subelem.text = 'hard hat'
            if elem.tag == 'path':
                elem.text = img_destination
            if elem.tag == 'folder':
                elem.text = './data/'
            if elem.tag == 'filename':
                img_file_name = elem.text
        tree.write("{}/{}".format(destination, file_name))
        shutil.copy2(img_src+'/'+img_file_name, img_destination+'/'+img_file_name)
    except:
        print(file_name)
        continue
    
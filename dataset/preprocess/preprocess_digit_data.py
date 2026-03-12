import os
import numpy as np
from PIL import Image
from glob import glob
from preprocess.xml_parser import get_coco_annotation_from_obj

os.chdir('/data/CWT_Weights')
# annotation_files = sorted(glob('annotation/*'))
# label2id = {str(i):i for i in range(10)}
#
# digit_data_dir = 'digit_data'
# for i in label2id:
#     os.makedirs(os.path.join(digit_data_dir, i))
#
# for f in annotation_files:
#     anns = get_coco_annotation_from_obj(f, label2id)
#     base_path = os.path.basename(f).split('.')[0]
#     img = Image.open('image_for_annotation/{}.jpg'.format(base_path))
#
#     for i, ann in enumerate(anns):
#         bbox = ann['bbox']
#         bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
#         category = ann['category_id']
#         cropped_img = img.crop(bbox)
#         cropped_img.save(os.path.join(digit_data_dir, str(category), base_path + '_%s.jpg' %(i)))
#

# gen train, valid dataset
os.chdir('digit_data')
files = glob('*/*')
np.random.shuffle(files)
num_imgs = len(files)
train_data = files[:int(0.8 * num_imgs)]
valid_data = files[int(0.8 * num_imgs):]

with open('train_data.txt', 'w') as f:
    for line in sorted(train_data):
        f.write(line + '\n')

with open('valid_data.txt', 'w') as f:
    for line in sorted(valid_data):
        f.write(line + '\n')
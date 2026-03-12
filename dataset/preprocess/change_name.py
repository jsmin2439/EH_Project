import os
import shutil

os.chdir('/data/CWT_Weights')
image_list = os.listdir('image_weight')
f = open('original_image_name.txt', 'w')

for idx, name in enumerate(image_list):
    saved_name = str(idx+1).zfill(5) + '.jpg'
    f.write(f'{name}\t{saved_name}\n')
    shutil.copy(f'image_weight/{name}', f'image_for_annotation/{saved_name}')

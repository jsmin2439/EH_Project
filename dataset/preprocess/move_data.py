import shutil
import os
from glob import glob
import cv2

dirs = {ord('1'): 'image_with_number',
        ord('2'): 'image_with_weight',
        ord('3'): 'image_without_number'}
base_dir = 'images'

def move_img(num, img_name):
    move_dir = dirs[num]
    shutil.copy(os.path.join(base_dir, img_name), os.path.join(move_dir, img_name))
    print(f'move {img_name} to the {move_dir}')

done_imgs = open('done.txt', 'a+')
os.chdir('/data/CWT_Weights')
done_imgs.seek(0)
done_img_list = [name.strip() for name in done_imgs.readlines()]
print(done_img_list)

for i, im in enumerate(glob('images/*')):
    img_name = os.path.basename(im)
    if img_name in done_img_list:
        continue
    print(i, img_name)
    img = cv2.imread(im, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (img.shape[0]//4, img.shape[1]//4))
    cv2.imshow('Time', img)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()
    move_img(k, img_name)
    done_imgs.write('%s\n' % img_name)

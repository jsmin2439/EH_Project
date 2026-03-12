import os
import numpy as np
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt

os.chdir('/data/CWT_Weights/digit_data')

# number of data
num_data = []
for i in sorted(glob('*')):
    num_data.append(len(os.listdir(f'{i}')))
print(sum(num_data))
plt.bar(range(10), num_data)
plt.xticks(range(10), [str(i) for i in range(10)])
plt.show()

# mean, std
imgs = []
for idx, img in enumerate(glob('*/*')):
    img = Image.open(img).resize((64,64))
    img = np.array(img) / 255
    imgs.append(img)
    if idx % 1000 == 0:
        print(idx)

imgs = np.array(imgs).reshape(-1, 3)
print(imgs.mean(0))
print(imgs.std(0))

import os
from glob import glob
import numpy as np
from scipy.io import loadmat, savemat
import h5py
import kagglehub

# Download latest version
path = kagglehub.dataset_download("ctrnngtrung/miniimagenet")
# saved to ~/.cache/kagglehub/datasets/
print("Path to dataset files:", path)

"""use data from ~/.cache/kagglehub/datasets/ctrnngtrung/miniimagenet/versions/1
│ ├── 1
│ │ ├── map_clsloc.txt  
│ │ ├── preprocess.py 
│ │ ├── test
│ │ ├── train
"""

# data_dir = "./datasets/SIDD/SIDD_Medium_Raw/Data"
#data_dir = "data/datasets/tests"
#path_all_noisy = glob(os.path.join(data_dir, '**/*NOISY*.MAT'), recursive=True)
path_all_noisy = glob(os.path.join(data_dir, '**/*output*.mat'), recursive=True)
path_all_noisy = sorted(path_all_noisy)
print(path_all_noisy)
print('Number of big images: {:d}'.format(len(path_all_noisy)))


#save_folder = "./datasets/SIDD/SIDD_Medium_Raw_noisy_sub512"
save_folder = "data/datasets/tests_out_raw"
if os.path.exists(save_folder):
    os.system("rm -r {}".format(save_folder))
os.makedirs(save_folder)   

#crop_size = 512
#step = 256
# randomly chosen
crop_size = 128
step = 64

for ii in range(len(path_all_noisy)):
    img_name, extension = os.path.splitext(os.path.basename(path_all_noisy[ii]))
    print(img_name)
    #mat = h5py.File(path_all_noisy[ii])
    mat = loadmat(path_all_noisy[ii])
    print(mat.keys()) # dict_keys(['__header__', '__version__', '__globals__', 'satellite_data', 'metadata'])
    #im = mat['x'].value
    im = mat['satellite_data']
    print(im.shape) # (1, 184, 313)
    #h, w = im.shape
    n, h, w = im.shape
    """if h < crop_size or w < crop_size:
        print(f"Skipping {img_name}, too small ({h}x{w})")
        continue"""
    # prepare to crop
    h_space = np.arange(0, h - crop_size + 1, step)
    print(h_space)
    #if h - (h_space[-1] + crop_size) > 0:
    if h - (h_space[-1] + crop_size) > 0:
        h_space = np.append(h_space, h - crop_size)
    w_space = np.arange(0, w - crop_size + 1, step)
    if w - (w_space[-1] + crop_size) > 0:
        w_space = np.append(w_space, w - crop_size)
    # crop
    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            cropped_img = im[x:x + crop_size, y:y + crop_size]
            cropped_img = np.ascontiguousarray(cropped_img)
            save_path = os.path.join(save_folder, "{}_s{:0>3d}{}".format(img_name, index, extension.lower()))
            savemat(save_path, {"x": cropped_img})

import numpy as np
from imageio import imread
from skimage.transform import resize
from RI_wvl import wvl_660, wvl_517, wvl_450

def get_img(img_name, up_R, up_C, color):

    img = imread(img_name)
    if img.dtype == 'uint8':
        img = img.astype(np.float32) / 255.0
    elif img.dtype == 'uint16':
        img = img.astype(np.float32) / 65535.0
    elif img.dtype == 'float32':
        img = img / 1.0
    else:
        assert('Invalid image type')

    if color == 'w':
        if len(img.shape) == 3:
            img = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2] # Convert to gray scale
        wvlpad_R = 0
        wvlpad_C = 0
        img = resize(img, (up_R, up_C))

    elif color == 'r':
        if len(img.shape) == 3:
            img = img[:,:,0]
        wvl_scale = wvl_450 / wvl_660
        new_R = wvl_scale * up_R
        new_C = wvl_scale * up_C
        new_R = int(round(new_R / 2) * 2)
        new_C = int(round(new_C / 2) * 2)
        wvlpad_R = (up_R - new_R) // 2
        wvlpad_C = (up_C - new_C) // 2
        img = resize(img, (new_R, new_C))
        img = np.pad(img, ((wvlpad_R, wvlpad_R),(wvlpad_C, wvlpad_C)))

    elif color == 'g':
        if len(img.shape) == 3:
            img = img[:,:,1]
        wvl_scale = wvl_450 / wvl_517
        new_R = wvl_scale * up_R
        new_C = wvl_scale * up_C
        new_R = int(round(new_R / 2) * 2)
        new_C = int(round(new_C / 2) * 2)
        wvlpad_R = (up_R - new_R) // 2
        wvlpad_C = (up_C - new_C) // 2
        img = resize(img, (new_R, new_C))
        img = np.pad(img, ((wvlpad_R, wvlpad_R),(wvlpad_C, wvlpad_C)))
 
    elif color == 'b':
        if len(img.shape) == 3:
            img = img[:,:,2]
        wvlpad_R = 0
        wvlpad_C = 0
        img = resize(img, (up_R, up_C))

    assert len(img.shape) == 2

    return img, wvlpad_R, wvlpad_C 

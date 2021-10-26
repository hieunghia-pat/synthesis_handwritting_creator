import numpy as np
import cv2 as cv
import os
from tqdm import tqdm

BASE_DIR = "UIT_HWDB_word_syn"

for folder in tqdm(os.listdir(BASE_DIR)):
    for file in os.listdir(os.path.join(BASE_DIR, folder)):
        image_file = os.path.join(BASE_DIR, folder, file)
        img = cv.imread(image_file)
        rnum = np.random.binomial(1, 0.5)
        if rnum == 0:
            continue
        if img is not None:
            img = np.where(img <= 100, img, np.ones_like(img, dtype=np.uint8)*255)
            cv.imwrite(image_file, img)
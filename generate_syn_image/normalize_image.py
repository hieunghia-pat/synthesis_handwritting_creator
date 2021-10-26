import cv2 as cv
import os 
from tqdm import tqdm

BASE_DIR = "UIT_HWDB_word_syn"
for path in tqdm(os.listdir(BASE_DIR)):
    for subpath in tqdm(os.listdir(os.path.join(BASE_DIR, path))):
        img_file = os.path.join(BASE_DIR, path, subpath)
        img = cv.imread(img_file)
        if img is None: 
            continue
        img = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)
        img = cv.GaussianBlur(img, ksize=(5, 5), sigmaX=3, sigmaY=3)
        cv.imwrite(img_file, img)
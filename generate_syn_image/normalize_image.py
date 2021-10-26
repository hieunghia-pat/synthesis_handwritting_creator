import cv2 as cv
import os 
from tqdm import tqdm

BASE_DIR = "UIT_HWDB_word_syn"
for path in tqdm(os.listdir(BASE_DIR)):
    for subpath in os.listdir(os.path.join(BASE_DIR, path)):
        img_file = os.path.join(BASE_DIR, path, subpath)
        img = cv.imread(img_file)
        if img is None: 
            continue
        img = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)
        img = cv.GaussianBlur(img, ksize=(7, 7), sigmaX=0, sigmaY=0)
        # cv.imwrite(img_file, img)
        cv.imshow("Image", img)
        cv.waitKey()
        cv.destroyWindow("Image")
import cv2 as cv
import os 
from tqdm import tqdm

BASE_DIR = "UIT_HWDB_word_syn"
splits = ["train_data", "test_data"]
for split in splits:
    for folder in tqdm(os.listdir(os.path.join(BASE_DIR, split))):
        for image_file in os.listdir(os.path.join(BASE_DIR, split, folder)):
            image_file = os.path.join(os.path.join(BASE_DIR, split, folder, image_file))
            image = cv.imread(image_file)
            if image is None:
                continue
            image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
            cv.imwrite(image_file, image)
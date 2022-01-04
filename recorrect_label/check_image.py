import cv2 as cv
import os
import json
from tqdm import tqdm

BASE_DIR = "UIT_HWDB_line"

for data_folder in os.listdir(BASE_DIR):
    parent_folder = os.path.join(BASE_DIR, data_folder)
    for folder in os.listdir(parent_folder):
        subfolder = os.path.join(parent_folder, folder)
        json_file = os.path.join(subfolder, "label.json")
        label = json.load(open(json_file))
        for img_id, text in label.items():
            print(text)
    input("Enter to continue")

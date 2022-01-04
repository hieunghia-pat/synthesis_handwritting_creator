import os 
import json
import re 
from tqdm import tqdm

BASE_DIR = "UIT_HWDB_v1"

splits = ["train_data", "test_data"]
for data_folder in tqdm(os.listdir(BASE_DIR)):
    for split in splits:
        parent_folder = os.path.join(BASE_DIR, data_folder, split)
        for folder in tqdm(os.listdir(parent_folder)):
            subfolder = os.path.join(parent_folder, folder)
            txt_file = os.path.join(subfolder, "ground_truth.txt")
            lines = open(txt_file).readlines()
            labels = {}
            for line in lines: 
                line = re.sub('\n', '', line)
                img_file, text = line.split(".png, ")
                labels[f"{img_file}.jpg"] = text

            json.dump(labels, open(os.path.join(subfolder, "label.json"), "w+"), ensure_ascii=False)
            os.remove(txt_file)
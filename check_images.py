import cv2 as cv
import os

BASE_DIR = "UIT_HWDB_line_syn/train_data"
image_folders = os.listdir(BASE_DIR)

max_w = 0
total = 0
for image_folder in image_folders:
    image_files = os.listdir(os.path.join(BASE_DIR, image_folder))
    for image_file in image_files:
        image = cv.imread(os.path.join(BASE_DIR, image_folder, image_file))
        if image is None:
            continue
        h, w, _ = image.shape
        scale = h / 64
        h = 64
        w = int(w / scale)
        if max_w < w:
            max_w = w
        if w > 1024:
            total += 1
        image = cv.resize(image, dsize=(w, h))
        if w > 1024:
            image = cv.resize(image, (1024, 64))
            cv.imshow("Image", image)
            cv.waitKey(500)

print(max_w)
print(total)
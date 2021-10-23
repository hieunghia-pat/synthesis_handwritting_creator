import cv2 as cv
import numpy as np

text_img = cv.imread("9.jpg")
paper_img = cv.imread("paper_image.jpg")

text_img_w, text_img_h, _ = text_img.shape
paper_img_w, paper_img_h, _ = paper_img.shape
paper_img = paper_img[:text_img_w, :text_img_h, :]

cv.imshow("Paper image", paper_img)
cv.imshow("Text image", text_img)

alpha = 0.9
beta = 1 - alpha
# final_img = np.where(text_img <= 50, alpha*text_img + beta*paper_img, paper_img)
final_img = alpha*text_img + beta*paper_img

cv.imshow("Result", final_img.astype(np.uint8))
cv.waitKey()
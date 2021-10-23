import PIL
from PIL import Image, ImageFont, ImageDraw, ImageFilter, ImageOps
from tqdm import tqdm
import os
import sys
import numpy as np
import cv2 as cv

sys.path.append("generate_syn_image")
from functions import Processing, fitcrop
import random

font_files = os.listdir("generate_syn_image/VNfonts")
'''
texts = ["AaĂăÂâBbCcDdĐđEeÊêGgHhIiKkLlMmNnOoÔôƠơPpQqRrSsTtUuƯưVvXxYy",
    "ÁáÀàẢảÃãẠạĂăẮắẰằẲẳẴẵẶặÂâẤấẦầẨẩẪẫẬậĐđÉéÈèẺẻẼẽẸẹ",
	"ÊêẾếỀềỂểỄễỆệÍíÌìỈỉĨĩỊịÓóÒòỎỏÕõỌọÔôỐốỒồỔổỖỗỘộ",
    "ƠơỚớỜờỞởỠỡỢợÚúÙùỦủŨũỤụƯưỨứỪừỬửỮữỰựÝýỲỳỶỷỸỹỴỵ123456789(){}[]?"]
'''
texts=["!@#$%^&*"]

imagefile = "."
for font_file in tqdm(font_files):
	font = ImageFont.truetype(os.path.join("generate_syn_image/VNfonts", font_file),random.randint(80,150))
	images = []
	for text in texts:
		content_size = font.getsize(text)
		image = Image.new("L", (content_size[0]+20, content_size[1]+20),"#ffffff")

		draw = ImageDraw.Draw(image)

		x_coor = 10
		y_coor = 10
		draw.text((x_coor,y_coor), text, font=font, fill="#000000")

		image = Processing(image)
		image = fitcrop(image)
		image = np.array(image)
		image = cv.resize(image, (200, 200))
		images.append(image)
	
	img = np.concatenate(images, axis=0)
	cv.imshow("Image", img)
	key = cv.waitKey()
	cv.destroyAllWindows()
	if key == ord("d"):
		os.remove(os.path.join("generate_syn_image/VNfonts", font_file))
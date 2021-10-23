import PIL
from PIL import Image, ImageFont, ImageDraw, ImageFilter, ImageOps
import random
import cupy as cp
import os
from scipy.ndimage import map_coordinates, gaussian_filter
from tqdm import tqdm
import re

def RandSpace():
	cnt=random.randint(1,2)
	spaces=""
	for i in range(cnt+1):
		spaces+=" "
	return spaces

def getBetaValue(type):
	if type == "stroke":
		path = os.path.join("generate_syn_image/color_channels", random.choice(os.listdir("generate_syn_image/color_channels")))
	elif type == "background":
		path = os.path.join("generate_syn_image/color_channels", random.choice(os.listdir("generate_syn_image/background_color_channels")))
	else: 
		raise Exception("Type must be stroke or background.")

	file = open(path,"r")
	content = file.readlines()[1]
	alpha = float(content[0:content.find(",")])
	beta = float(content[content.find(",")+1:])
	file.close()

	return(alpha, beta)
	
def AddColor(image):
	image = cp.asarray(image)

	# get color for stroke
	alpha, beta = getBetaValue(type="stroke")	
	txt_colored_arr = (cp.random.beta(alpha, beta, image.shape)*100).astype(cp.uint8)

	# get color for background
	alpha, beta = getBetaValue(type="background")
	image = cp.where(image < 120, txt_colored_arr, image)
	bg_colored_arr = (cp.random.beta(alpha, beta, image.shape)*100).astype(cp.uint8)
	image = cp.where(image >= 120, bg_colored_arr, image)

	return Image.fromarray(cp.asnumpy(image))

def fitcrop(image):
	w,h = image.size
	px = image.load()
	found = False
	for i in range(h):
		for j in range(w):
			if (px[j,i]!=(255,255,255)):
				up = i
				found=True
				break
		if (found): break
	found = False
	for i in range(h):
		for j in range(w):
			if (px[j,h-1-i]!=(255,255,255)):
				bottom = h-i
				found=True
				break
		if (found): break
	found = False		
	for j in range(w):
		for i in range(h):
			if (px[j,i]!=(255,255,255)):
				left = j
				found=True
				break
		if (found): break
	found = False
	for j in range(w):
		for i in range(h):
			if (px[w-1-j,i]!=(255,255,255)):
				right = w-j
				found=True
				break
		if (found): break
	return image.crop((left, up, right, bottom))

def pixel_deform(X, alpha=20, sigma=6): #elastic deform on pixelwise basis
	shape = X.shape
	dx = gaussian_filter(cp.random.randn(*shape), sigma, mode="constant", cval=0) * alpha #originally with random_state.rand * 2 - 1
	dy = gaussian_filter(cp.random.randn(*shape), sigma, mode="constant", cval=0) * alpha
	x, y = cp.meshgrid(cp.arange(shape[0]), cp.arange(shape[1]), indexing='ij')
	indices = x+dx, y+dy
	return map_coordinates(X, indices, order=3).reshape(shape)

def Processing01(image): #add blur dilation and smoothed noise
	image = image.filter(ImageFilter.GaussianBlur(radius=1))
	image = image.filter(ImageFilter.MinFilter(3))

	return(image)
'''
def Processing02(image): #add smoothed noise keeping the edge intact
	AddNoise(image)
	image = image.filter(ImageFilter.MedianFilter)
	return(image)
'''
def Processing03(image): #add elastic deformation
	px = cp.asarray(image)
	px = pixel_deform(px, sigma=random.randrange(6,8))
	image = Image.fromarray(px)
	image = image.crop((5,5,image.size[0]-5,image.size[1]-5))
	return(image)

def Processing(image, k=0):
	if (k==0):
		k = random.randint(1,5)
	if (k==1):
		image = ImageOps.colorize(image, black="black", white="white")
		image = AddColor(image)
		image = Processing01(image)
	elif (k==3 or k==2):
		image = Processing03(image)
		image = ImageOps.colorize(image, black="black", white="white")
		image = AddColor(image)
	elif (k==4):
		image = Processing03(image)
		image = ImageOps.colorize(image, black="black", white="white")
		image = AddColor(image)
		image = Processing01(image)
	elif (k==5):
		image = ImageOps.colorize(image, black="black", white="white")
		image = AddColor(image)
	return image

def RenderLineImage(text, imagefile):

	font_path = os.path.join("generate_syn_image/VNfonts", random.choice(os.listdir("generate_syn_image/VNfonts")))
	font = ImageFont.truetype(font_path,random.randint(80,150))
	words=text.split()
	render_text = ""
	content_width = 0
	for i in range(len(words)):	#add random spaces between words
		if (i!=0):
			words[i] = RandSpace() + words[i]
		render_text += words[i]
		content_width += font.getsize(words[i])[0]
	content_height=font.getsize(render_text)[1]
	
	image = Image.new("L",(content_width*2,content_height*3),"#ffffff")
	draw = ImageDraw.Draw(image)
	
	x_coor = content_width/2
	y_coor = content_height
	
	for word in words:
		draw.text((x_coor, y_coor + random.randint(-2,6)), word, font = font, fill="#000000")
		x_coor += font.getsize(word)[0]
	image = Processing(image)
	image = fitcrop(image)
	image.save("{}.png".format(imagefile))

def RenderWordImage(text, imagefile):

	font_path = os.path.join("generate_syn_image/VNfonts", random.choice(os.listdir("generate_syn_image/VNfonts")))
	font = ImageFont.truetype(font_path,random.randint(80,150))
	content_size = font.getsize(text)
	
	image = Image.new("L", (content_size[0]+20, content_size[1]+20),"#ffffff")
	draw = ImageDraw.Draw(image)
	
	x_coor = 10
	y_coor = 10
	draw.text((x_coor,y_coor), text, font=font, fill="#000000")
   
	image = Processing(image)
	image = fitcrop(image)
	image.save("{}.png".format(imagefile))

def CreateLineImgDataset(corpus, startfrom):
	cnt=0
	foldernum=startfrom
	path = "UIT_HWDB_line_syn"
	if not os.path.isdir(os.path.join(path, str(foldernum))):
		os.mkdir(os.path.join(path, str(foldernum)))
	labelfile = open("{}/{}/label.txt".format(path,str(foldernum)),"w+",encoding='utf-8')
	# print("Folder 0\n")
	for text in tqdm(corpus, dynamic_ncols=True, colour="green"):
		text = re.sub(r"[\[\]@#$^&]", "", text)
		cnt +=1
		if (cnt==500):
			foldernum+=1
			# print("Folder {}\n".format(foldernum))
			if not os.path.isdir(os.path.join(path, str(foldernum))):
				os.mkdir(os.path.join(path, str(foldernum)))
			labelfile = open("{}/{}/label.txt".format(path,str(foldernum)),"w+",encoding='utf-8')
			cnt=0
		RenderLineImage(text,"{}{}\line_{}".format(path,foldernum,cnt))
		labelfile.writelines("line_{}.png, {}".format(cnt,text))
		
def CreateWordImgDataset(corpus, startfrom):
	cnt=0
	foldernum=startfrom
	path = "UIT_HWDB_word_syn"
	if not os.path.isdir(os.path.join(path, str(foldernum))):
		os.mkdir(os.path.join(path, str(foldernum)))
	# print("Folder 0\n")
	for text in tqdm(corpus, dynamic_ncols=True, colour="green"):
		text = re.sub(r"[\[\]@#$^&]", "", text)
		cnt +=1
		if (cnt==500):
			foldernum +=1
			# print("Folder {}\n".format(foldernum))
			if not os.path.exists(os.path.join(path, str(foldernum))):
				os.mkdir(os.path.join(path, str(foldernum)))
			cnt=0
		RenderWordImage(text, "{}/{}/word_{}".format(path,foldernum,cnt))

if __name__ == "__main__":
	text = "This a [ ] an example * %!@#$%^&*() line."
	RenderLineImage(text, "saving")
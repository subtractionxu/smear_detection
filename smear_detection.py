import cv2
import os
import sys
import numpy as np

root_dir = 'sample_drive/'
checkpoint_dir = 'checkpoints/'
cam_dir = 'cam_3/'
img_dir = root_dir + cam_dir
diff_dir = checkpoint_dir + 'diff/'
processed_dir = checkpoint_dir + 'processed/'

os.system('rm {}*'.format(diff_dir))
os.system('rm {}*'.format(processed_dir))

def Get_Avg(img_dir):
	img_list = os.listdir(img_dir)
	for n,i in enumerate(img_list):
		pro = ((n + 1) / len(img_list)) * 100.0
		sys.stdout.write('\rAveraging images.. {}/{}'.format(n + 1, len(img_list)))
		img = cv2.imread(img_dir + i)
		if n == 0:
			img_avg = img
		else:
			img_avg = img_avg * n / (n+1) + img / (n + 1)
	return img_avg

def Process(img):
	print ('\nProcessing image..')
	
	# img = cv2.GaussianBlur(img,(5,5),0)
	# img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	# ret,img = cv2.threshold(img,150,255,cv2.THRESH_BINARY)
	img = cv2.equalizeHist(img)
	img = cv2.GaussianBlur(img,(7,7),0)
	# img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,25,4)
	# ret,img = cv2.threshold(img,35,255,cv2.THRESH_BINARY)
	# img = cv2.Canny(img,10,0)

	kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
	img = cv2.dilate(img,kernel)
	return img

def Segment(img,dir,cam):
	print ('Segmenting..')	
	ret,img = cv2.threshold(img,35,255,cv2.THRESH_BINARY)
	img = 255 * np.ones_like(img) - img
	ret, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity = 8)
	threshhold_l = 0.001
	threshhold_u = 0.1
	cc_nums = []
	masks = []
	for i,s in enumerate(stats):
		if s[-1] > threshhold_l * img.size and s[-1] < threshhold_u * img.size:
			cc_nums.append(i)
	for i,cc in enumerate(cc_nums):
		msk = np.array(labels)
		msk = np.where(msk == cc, 255,0)
		cv2.imwrite(checkpoint_dir + cam[:-1] + '_msk_'+ str(i) +'.jpg',msk)
		masks.append(msk)
	return masks

def Append_mask(masks,img,dir,cam):
	for i,msk in enumerate(masks):
		# msk = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)
		# ret,msk = cv2.threshold(msk,35,255,cv2.THRESH_BINARY)
		# canny = cv2.Canny(msk,10,0)
		im2, contours, hierarchy = cv2.findContours(msk,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		cv2.drawContours(msk, contours, -1, (0,255,0), 3)
		cv2.imwrite(checkpoint_dir + cam[:-1] + '_canny_'+ str(i) +'.jpg',msk)






'''
def Get_Avg_array(img_array_list):
	for n,img in enumerate(img_array_list):
		# img = cv2.imread(img_dir + i,0)
		if n == 0:
			img_avg = img
		else:
			img_avg = img_avg * n / (n+1) + img / (n + 1)
	# img_avg = img_sum / len(img_list)
	return img_avg  
'''


img_list = os.listdir(img_dir)
'''
for i in range(1,len(img_list)):
	# Process the previous img
	pre_img_name = img_list[i-1]
	pre_img_path = img_dir + pre_img_name
	pre_img = cv2.imread(pre_img_path,0)
	# pre_img = Process(pre_img_path)
	# cv2.imwrite(processed_dir + pre_img_name,pre_img)

	# Process the current img
	img_name = img_list[i]
	img_path = img_dir + img_name
	img = cv2.imread(img_path,0)
	# img = Process(img_path)
	# cv2.imwrite(processed_dir + img_name,img)
	

	# calculate difference
	img_diff = img - pre_img
	diff_name = img_name[:-4] + '-' + pre_img_name
	cv2.imwrite(diff_dir+diff_name,img_diff)
	print (diff_name)
'''
'''
processed_img_list = []
for i,img_name in enumerate(img_list):
	pro = ((i + 1) / len(img_list)) * 100.0
	sys.stdout.write("\rProcessing images: %.2f%%" % (pro))
	# print (img_name)
	img_path = img_dir + img_name
	img = Process(img_path)
	cv2.imwrite(processed_dir + img_name,img)
	processed_img_list.append(img)
'''


# diff_avg = Get_Avg(diff_dir)
# cv2.imwrite('miniset/diff_avg.jpg',diff_avg)

# Get the average of the input images (RGB)
# img_avg = Get_Avg(img_dir)

'''
diff_avg = Get_Avg(diff_dir)
diff_avg_path = 'miniset/diff_avg.jpg'
cv2.imwrite(diff_avg_path,diff_avg)
'''
img_avg_path = checkpoint_dir + cam_dir[:-1] + '_img_avg.jpg'
# cv2.imwrite(img_avg_path, img_avg)
img_avg = cv2.imread(img_avg_path,0)

# Process the averaged image
img_avg_processed = Process(img_avg)
img_avg_processed_path = checkpoint_dir + cam_dir[:-1] + '_img_avg_processed.jpg'
cv2.imwrite(img_avg_processed_path,img_avg_processed)

# Segment
masks = Segment(img_avg_processed, checkpoint_dir, cam_dir)

# Show mask(s)
demo_img_path = checkpoint_dir + 'cam_3_demo.jpg'
demo_img = cv2.imread(demo_img_path)
Append_mask(masks,demo_img,checkpoint_dir, cam_dir)









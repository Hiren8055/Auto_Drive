import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.segmentation import watershed
from skimage.segmentation import mark_boundaries, find_boundaries
from skimage.filters import sobel
import sys
import cv2
np.set_printoptions(threshold=sys.maxsize)
myimage = io.imread('output15.png')
# plt.imshow(myimage, cmap=plt.cm.gray)
# plt.show()

hist = np.histogram(myimage, bins= np.arange(0,256))
# fig, (ax1, ax2) = plt.subplots(1,2,figsize=(8,3))
# # ax1.imshow(myimage, cmap=plt.cm.gray, interpolation='nearest')
# ax1.axis('off')
# ax2.plot(hist[1][:-1], hist[0], lw=2)
# ax2.set_title('histogram of grey values')
# plt.show()

selected_pixles = []
for i in hist[1][:-1]:
    if hist[0][i] >= 700:
        selected_pixles.append(i)

image = sobel(myimage)
# fig, ax = plt.subplots(figsize=(4, 3))
# ax.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
# ax.axis('off')
# ax.set_title('elevation_map')

markers = np.zeros_like(myimage)
markers[myimage < min(selected_pixles)] = 1
markers[myimage > max(selected_pixles)] = 2
# fig, ax = plt.subplots(figsize=(4, 3))
# ax.imshow(markers, cmap=plt.cm.gray, interpolation='nearest')
# ax.axis('off')
# ax.set_title('markers')
markers = cv2.normalize(markers, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imshow("markers",markers)
seg = watershed(image, markers=markers)
#print(np.shape(seg[1]))
# fig, ax = plt.subplots(figsize=(4, 3))
# ax.imshow(seg, cmap=plt.cm.gray, interpolation='nearest')
# ax.axis('off')
# ax.set_title('segmentation')
# plt.show()



# bound = mark_boundaries(seg,seg, mode='thick')
# #print(np.shape(bound[0]))
# plt.imshow(bound, cmap=plt.cm.gray)
# plt.show()

bounder = find_boundaries(seg, mode='thick').astype(np.uint8)
# fig, ax = plt.subplots(figsize=(4, 3))
# ax.imshow(bounder, cmap=plt.cm.gray, interpolation='nearest')
# white = [255,255,255]

# print("shape: ",bounder.shape)
ret,thresh_binary = cv2.threshold(markers,0,255,cv2.THRESH_BINARY)
    # findcontours
# print(re
#t)
contours, hierarchy = cv2.findContours(image =thresh_binary , mode = cv2.RETR_TREE,method = cv2.CHAIN_APPROX_SIMPLE)

# create an empty mask
mask = np.zeros((512,512),dtype=np.uint8)

# loop through the contours
# print(contours)
j = 0
for i,cnt in enumerate(contours):
        # if the contour has no other contours inside of it
        if hierarchy[0][i][2] == -1 :
                # if the size of the contour is greater than a threshold
                if  cv2.contourArea(cnt) > 10000:
                        cv2.drawContours(mask,[cnt], 0, (255), -1)
                        j = i

cnts = contours[j]         
 
all_pixels = []      
for i in range(0, len(cnts)):
    for j in range(0,len(cnts[i])):
        all_pixels.append(cnts[i][j])  
# print(all_pixels)
image=np.zeros((512,512),np.uint8)
plt.plot()
for z in all_pixels:
    z = z.tolist()
    cv2.circle(image,z,0,(255,255,255),-1)
cv2.imshow("front view",image)
maxHeight = 512
maxWidth = 512
a = np.float32(image)
b = np.float32([[0,0],
               [0,maxHeight-1],
               [maxWidth-1,maxHeight-1],
               [maxWidth-1,0]])
img=np.zeros((512,512),np.uint8)
# M = cv2.getPerspectiveTransform(a,b)
# top_view = cv2.warpPerspective(img,M,(maxWidth,maxHeight),flags=cv2.INTER_LINEAR)
cv2.waitKey(0)
# cv2.imshow("top view",top_view)
# cv2.waitKey(0)
# result = cv2.normalize(bounder, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
# markers = cv2.normalize(markers, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
# # cv2.imwrite(f"boundary.png", result) 
# cv2.imshow("markers",markers)
# cv2.imshow("mask",mask)
# # cv2.imshow("boundary",result)
# cv2.waitKey(0)
# list = np.where(mask==0)
# x,y = list
# # print(x)
# # print(y)
# # zipped = np.column_stack((x,y))
# zipped = np.c_[y,x]
# zipped = zipped.tolist()
# file = open("boundary.txt","w")
# # write = np.array_str(zipped)
# # file.write(zipped)
# # file.close()

# image=np.zeros((512,512),np.uint8)
# # image, centerOfCircle, radius, color, thickness
# for z in zipped:
#     cv2.circle(image,z,0,(255,255,255),-1)
# # plt.plot(x, y,"o")
# # plt.show()

# cv2.imshow("bound",image)
# cv2.waitKey(0)
# # print(zipped)
# # ax.axis('off')
# # ax.set_title('boundary')
# # plt.show()
# # print(bounder)
# # print(list)
# # print(X)
# # print(Y)

import glob
import torch
import tqdm
import cv2
from tqdm import tqdm_notebook
from preprocess import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as im
import pandas as pd
from constant import *
from datetime import datetime
import time

cuda = True
test = True

TOP_VIEW_IMAGE_DIMESNION = (416, 416) # inv map output-image size (w, h) = (x, y)

FRONT_VIEW_IMAGE_DIMESNION = (416, 416) # (w, h) = (x, y)

FRONT_VIEW_POINTS = [(0   , 100),# camera
					 (-600, 416),
         			 (416 , 100),
         			 (1016, 416)]

# Dependent constants
TOP_VIEW_POINTS = [(0         	       	       , 0			    ),
          	   (0         		       , TOP_VIEW_IMAGE_DIMESNION[1]),
          	   (TOP_VIEW_IMAGE_DIMESNION[0], 0         		    ),
        	   (TOP_VIEW_IMAGE_DIMESNION[0], TOP_VIEW_IMAGE_DIMESNION[1])]

M = cv2.getPerspectiveTransform( np.float32(FRONT_VIEW_POINTS), np.float32(TOP_VIEW_POINTS) )

# TOP_VIEW_CAR_COORDINATE = (TOP_VIEW_IMAGE_DIMESNION[0]//2, TOP_VIEW_IMAGE_DIMESNION[1] + CAR_BELOW_Y) # car coordinates on image
# model_name = "Unet_2.pt"
model_path = r"seg_2142_20_pc.pt"
input_shape = input_shape

# map_location=torch.device('cpu')

def max_zero(A):
    """Returns max length of continous zeros in an array
    Input: 1D array - column pixels of image
    """
    c = 1
    cmax = 1
    for i,val in enumerate(A):
        if A[i-1] == 0 and A[i] == 0:
            c += 1
        if A[i-1] == 0 and A[i] != 0:
            c = 1
        if c > cmax:
            cmax = c
    return cmax
def predict(model,frame):
    
    # model = torch.load("seg_2142_20_pc.pt",map_location=torch.device('cuda'))
    # model.eval()
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # model.to(dev)
    count = 0
    # cap = cv2.VideoCapture(r"C:\Users\Aman Sheikh\Desktop\WIN_20230319_16_17_35_Pro.mp4")
    frame_width = 416
    frame_height = 416

    size = (frame_width, frame_height)
    print(size)
    # save = cv2.VideoWriter('result_sevc.mp4', cv2.VideoWriter_fourcc(*'XVID'),30.0, size)
    # while cap.isOpened:
        
    i= float(time.time())
        # ret,img = cap.read()
        # if ret:
    # img = cv2.imread(frame)
    img = frame        
    # trial = np.zeroes(shape=(img.shape[0], img.shape[1],img.shape[2]), dtype=int)
    trial = np.full((img.shape[0], img.shape[1],img.shape[2]), 255)
    # print(image)
    
    image = tensorize_image(img, input_shape, cuda)
    # print(batch_test.shape[0])
    # cv2.imshow("batch",batch_test)
    print("ran")
    image = image.to(dev)
    print("ran1")
    output = model(image)
    print("ran2")
    out = torch.argmax(output, axis=1)
    print("ran2")
    

    outputs_list = out.cpu().detach().numpy()
    mask = np.squeeze(outputs_list, axis=0)

    mask_uint8 = mask.astype('uint8')
    mask_resize = cv2.resize(mask_uint8, ((img.shape[1]), (img.shape[0])), interpolation=cv2.INTER_CUBIC)
    # mask_ind = mask_resize == 1
    trial = trial.astype(np.uint8)
    trial[mask_resize == 1, :] = (0, 0, 0)
    trial = trial.astype(np.uint8)
    img[mask_resize == 1, :] = (255, 0, 125)
    img = img.astype(np.uint8)
    img = cv2.resize(img, (416, 416), interpolation=cv2.INTER_LINEAR)

    trial_img = cv2.resize(trial
                        , (416, 416),
                        interpolation=cv2.INTER_LINEAR)
    # cv2.imshow('MainPic', img)
    # cv2.waitKey(0)
    # save.write(img)
    # wrapTrial = cv2.warpPerspective(
    #     trial_img, M, TOP_VIEW_IMAGE_DIMESNION, flags=cv2.INTER_LINEAR)

    # cv2.imshow('Trial wrap', wrapTrial)
    # cv2.waitKey(0)

    '''trial_img = np.array(cv2.cvtColor(trial_img, cv2.COLOR_BGR2GRAY))
    indices2 = trial_img.shape[0] - np.array([max_zero(i) for i in trial_img.T])  # im.T slow compared to indices1
    maxv = np.argmin(indices2)          # Edges max value index

    # Left Edges
    ly = np.array(indices2[:maxv + 5])
    coordl = [[ind, val] for ind, val in enumerate(ly)]  # if ind%5==0
    xl = np.array([i[0] for i in coordl])
    yl = np.array([i[1] for i in coordl])
    yl = yl + ((yl / trial_img.shape[0]) * 10)
    rolling_mean_l = pd.Series(yl).rolling(window=10).mean()
    rolling_mean_l = rolling_mean_l.fillna(method='bfill')

    # Right Edges
    ry = indices2[maxv - 5:]
    coordr = [[maxv + ind, val] for ind, val in enumerate(ry)]  # if ind%5==0

    # trial_img = cv2.drawContours(trial_img,coordr,-1,(255,0,50),3)
    # print(coordr)
    # print("coordr")
    xr = np.array([i[0] for i in coordr])
    yr = np.array([i[1] for i in coordr])
    yr = yr + ((yr / trial_img.shape[0]) * 20)
    rolling_mean_r = pd.Series(yr).rolling(window=10).mean()
    rolling_mean_r = rolling_mean_r.fillna(method='bfill')
    pts = np.array([[i, val] for i, val in enumerate(rolling_mean_r)], np.int32)'''
    # print(rolling_mean_r)
    # print("rmr")
    # print(yr)
    # print("yr")
    # print(xr)
    # print("xr")
    # for i in range(0,maxv):
    #     try:
    #         trial_img = cv2.line(trial_img,(int(xr[i]),int(yr[i])),(int(xr[i+1]),int(yr[i+1])),(255, 255, 0))
    #         trial_img = cv2.line(trial_img,(int(xl[i]),int(yl[i])),(int(xl[i+1]),int(yl[i+1])),(255, 50, 0))
    #     except:
    #         print("reached final")


    # plt.figure(figsize=(20, 20))
    # plt.plot(xl, rolling_mean_l, linewidth=2)
    # plt.plot(xr, rolling_mean_r, linewidth=2)
    # 
    # plt.plot((int(trial_img.shape[1] * 25 / 100), maxv - 5), (trial_img.shape[0], indices2[maxv]))
    # plt.plot((int(trial_img.shape[1] * 75 / 100), maxv + 5), (trial_img.shape[0], indices2[maxv]))
    # print("maxv = "+str(maxv))////////
    # plt.axis('off')
    # plt.gca().invert_yaxis()
    # plt.show()
    '''# trial_img = np.array(cv2.cvtColor(trial_img, cv2.COLOR_GRAY2BGR))
    # white1 = np.full((img.shape[0], img.shape[1],img.shape[2]), 255)
    # white1 = white1.astype(np.uint8)
    # for i in range (len(coordl)):
    #     p = coordl[i]
    #     white1 = cv2.circle(white1, (p[0],p[1]), 1, (0, 255, 0), 1)
    # for i in range (len(coordr)):
    #     p = coordr[i]
    #     white1 = cv2.circle(white1, (p[0],p[1]), 1, (255, 0, 0), 1)

    # cv2.imshow('image1', white1)
    # # cv2.waitKey(0)

    # for i in range (len(coordl)):
    #     p = coordl[i]
    #     trial_img = cv2.circle(trial_img, (p[0],p[1]), 1, (0, 255, 0), 1)
    # for i in range (len(coordr)):
    #     p = coordr[i]
    #     trial_img = cv2.circle(trial_img, (p[0],p[1]), 1, (255, 0, 0), 1)
    # cv2.imshow('image2', trial_img)
    # cv2.waitKey(0)'''
    
    j= float(time.time())
    f = float(j-i)
    fps = float(1/f)
    print(fps)
    return trial_img

        # trial[mask_resize, :] = (255, 255, 255)

        # data = im.fromarray(trial)
        # data.save('gfg_dummy_pic.png')

        # print(mask_resize)
        # copy_img = img.copy()
        # img[mask_resize == 1, :] = (255, 0, 50)
        # new_img = cv2.resize(img
        #                      , (780, 540),
        #                      interpolation=cv2.INTER_NEAREST)
        # cv2.imshow('image', new_img)
        # cv2.waitKey(0)
        #
        # # print(img)
        # opac_image = (img / 2 + copy_img / 2).astype(np.uint8)
        # # print(opac_image)
        # # newimg = cv2.imread(opac_image)
        # new_img = cv2.resize(opac_image
        #                 , (780, 540),
        #            interpolation = cv2.INTER_NEAREST)
        # cv2.imshow('image',new_img)
        # cv2.waitKey(0)
        # # cv2.imwrite(os.path.join(predict_path, image.split("/")[-1]), opac_image)
        #
# if __name__ == "__main__":
#     predict(model)

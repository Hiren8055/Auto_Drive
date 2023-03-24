import cv2
# from tensorflow import keras
from matplotlib import pyplot as plt
import numpy as np
import time 
import os
import sys
import findBoundary
from json import dump

from a_star import AStarPlanner
import plotting
sys.path.append(r'D:\THINGS\Yolo_v4\darknet\build\darknet\x64')
import darknet2 as darknet
import torch
import math
# import tensorflow as tf
mask = np.zeros((416,416,3), dtype=np.uint8)
from localization import get_corrected_top_view
import control
import dataLog
import custom_test
import serial
# threading
from threading import Thread, enumerate
from queue import Queue

ARDUINO_CONNECTED = True #in constants
# np.set_printoptions(threshold=sys.maxsize)
prev_time_my = time.time()
def connect_arduino():
    #
    # if(ARDUINO_CONNECTED):
    arduino = None
    try:
        arduino = serial.Serial('/dev/ttyACM0', 115200,timeout=0.1)
        # arduino = serial.Serial('COM5', BAUD_RATE)# ,timeout=.1)
        arduino.flushInput()
        print("Connecting to: /dev/ttyACM0")
    except:
        try:
            arduino = serial.Serial('COM6', 115200,timeout=0.1)
            # arduino.flushInput()
            # arduino.flushOutput()
            print("com7")
        except:
            print("failed to connect to COM6")
    # time.sleep(2)
    return arduino

def send_data(arduino, data):
    print("st_angle_send",data)
    # arduino.flushInput()
    # arduino.flushOutput()
    arduino.write(str(data).encode())

def stop_car(arduino):   #bluetooth 
    arduino.flushOutput()
    time.sleep(1)
    arduino.write(str('b').encode())
    # time.sleep(1)
    #arduino.close()
    print("Stop command")
    
def select_cam_port():
    port = 10
    camera = cv2.VideoCapture(port)
    # print("Searching Camera port")
    
    while camera.isOpened() is not True:
        
        camera.release()
        port -= 1
        camera = cv2.VideoCapture(port)
        
        if port == -1:
            print("CAMERA PORT NOT FOUND")
            return 0

    camera.release()
    print("Camera running on port:", port)
    return port

curr_path = os.path.dirname(__file__)
sx = 1000
sy = 00
MS = 1/3    
# ratio = 
def set_saved_video(input_video, output_video, cap):
    # fps = int(input_video.get(cv2.CAP_PROP_FPS))
    # print("output_video",output_video.shape)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    size = (frame_width, frame_height)
    video = cv2.VideoWriter('filename.avi', 
                            cv2.VideoWriter_fourcc(*'MJPG'),
                            10, size)
    return video

class astar_input():
    def __init__(self,xvalues,yvalues,gx,gy):
        self.xvalues = xvalues
        self.yvalues = yvalues
        self.gx =gx 
        self.gy = gy
        
def terminate_thread(arduino, cap, video, tv_video, DATA, log_data, f):
    cap.release()
    print("release")
    video.release()
    tv_video.release()
    cv2.destroyAllWindows()
    DATA.append( {
    "log_data" : log_data
    } )
    dump(DATA, f, indent=4)
    f.close()
    print("LOGGING SUCCESSFULL")

def start(cap,network,class_names,scaled_topview_queue,termination1,arduino):
    ############################### ##############################
    # model = keras.models.load_model(curr_path+"/roadseg_epoch30_dataset867.h5",compile=False)
    # model = keras.models.load_model(curr_path+"/roadseg_epoch20_256_dataset867.h5",compile=False)
    # model = keras.models.load_model(curr_path+"/seg_867_25_L.pt",compile=False)
    # '''model = create_model()
    # model = model.load_weights("seg_867_25_L.pt")'''
    # model = tf.keras.models.load_model("seg_867_25_L.pt")
    ############################################################
    # cap = cv2.VideoCapture(curr_path+"/testing_video.mp4")
    # cap = cv2.VideoCapture(r"C:\Users\Aman Sheikh\Desktop\lane-segmentation\lane segmentation\raw2.mp4")
    # cap  = cv2.VideoCapture(r"C:\Users\Aman Sheikh\Desktop\Image Segmentation\data\data_road\testing\image_2\um_00000.png")
    # cap = cv2.VideoCapture(r"C:\Users\Aman Sheikh\Desktop\Solecthon\video.mp4")
    # cap = cv2.VideoCapture(r"C:\Users\Aman Sheikh\Desktop\WIN_20230319_16_19_35_Pro.mp4")
    # cap =  cv2.VideoCapture(2)
    global prev_time_my
    # arduino = None
    model = torch.load("seg_2142_20_pc.pt",map_location=torch.device('cuda'))
    model.eval()
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(dev)
    # cap = cv2.VideoCapture(2)
    # cap = cv2.VideoCapture(r"D:\Autonomous\logs\23-3-2023\test3.mp4")
    # cap = cv2.VideoCapture(r"C:\Users\Aman Sheikh\Desktop\WIN_20230319_16_14_39_Pro.mp4")
    # cap = cv2.VideoCapture(r"C:\Users\Aman Sheikh\Desktop\Segmentation\test23 (1).mp4")
    # cap = cv2.VideoCapture(r"C:\Users\Aman Sheikh\Downloads\endurance9.mp4")
    ########################################################
    i = 0
    j = 0
    f, log_file_name = dataLog.give_file()
    # video = set_saved_video(cap, log_file_name+".avi", cap)
    count = 0
    # if ARDUINO_CONNECTED:
    #     arduino = connect_arduino()
    while cap.isOpened():
        video_capture_time = time.time()
        
        ret, frame = cap.read()
        # cv2.imshow("fr",frame)
        # cv2.waitKey(0)
        if not ret:
            break
        
        if count == 1:    
            if ARDUINO_CONNECTED :
                print("Starting throttle command")
                time.sleep(1)  
                arduino.write(str('a').encode())
                time.sleep(5) # 5
                
        count+=1
        cv2.imwrite(r"D:\Autonomous\Segmentation_Pytorch (2)\logs\23-3-2023\output"+str(count)+".jpeg",frame)
        
        test_img = cv2.resize(frame,(416,416))
        frame_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        width = 416
        height = 416
        
        frame_resized_darknet = cv2.resize(frame_rgb, (width, height),
                                interpolation=cv2.INTER_LINEAR)
        frame_resized = cv2.resize(frame_rgb, (width, height),
                                interpolation=cv2.INTER_LINEAR)
        frame_resized = cv2.bitwise_or(frame_resized,mask)
        
        result = custom_test.predict(model,frame_resized)
        
        result = cv2.bitwise_not(result)
        
        img_for_detect = darknet.make_image(width, height, 3)
        darknet.copy_image_from_bytes(img_for_detect, frame_resized_darknet.tobytes())
        detection = darknet.detect_image(network, class_names, img_for_detect, thresh=.50)
        print("detections: ",detection)
        mybox = []
        for i  in range(len(detection)):
            x, y, w, h = detection[i][2][0],\
            detection[i][2][1],\
            detection[i][2][2],\
            detection[i][2][3]
            xmin, ymin, xmax, ymax = convertBack(
                float(x), float(y), float(w), float(h))
            pt1 = (xmin, ymin)
            pt2 = (xmax, ymax)
            p = np.array([[((xmax+xmin)//2), (ymax//1)]], dtype='float32')
            p = np.array([p])
            a = np.float32([(112.43,135),(298.1,135),(-230.9,416),(637.8,416)])
            b = np.float32([[0,0],
                   [416,0],
                   [0,416],
                   [416,416]])
            M = cv2.getPerspectiveTransform(a,b)
            pointsOut = cv2.perspectiveTransform(p, M)
            box = int(pointsOut[0][0][0]), int(pointsOut[0][0][1])
            mybox.append(box)

        xvalues,yvalues,boundAry = findBoundary.getBoundary(result)
        boundAry = get_corrected_top_view(boundAry)
        sorted(boundAry,key=lambda x: x[1])
        xvalues,yvalues = zip(*boundAry)
        xvalues,yvalues = list(xvalues),list(yvalues)
        mx = -sys.maxsize
        for i in range(len(yvalues)):
            if yvalues[i] > mx:
                mx = yvalues[i]
                gx = xvalues[i]
                gy = yvalues[i]
        similar_Y = []
        similar_Y_current = []
        for i in range(len(boundAry)-1):
            if boundAry[i][1] == boundAry[i+1][1]:
                similar_Y_current.append(boundAry[i])
            else:
                if len(similar_Y) < len(similar_Y_current):
                    similar_Y = similar_Y_current
                similar_Y_current.clear()
                
                
        n = len(similar_Y)
        print("Length of Boundary: ",n)
        # if n > 100:
        #     gx = similar_Y[n//2][0]
        #     gy = similar_Y[n//2][1]
        print("gx gy",gx,gy)

        mybox = get_corrected_top_view(mybox)
        for box in mybox:
            boundAry.append((box[0],box[1]))
            xvalues.append(box[0])
            yvalues.append(box[1])

        astar_io = astar_input(xvalues,yvalues,gx,gy)
        scaled_topview_queue.put(astar_io)
        cv2.imshow("frame",test_img)
        cv2.imshow("segmentation",result)
        if not termination1.empty():
            print("termincation")
            if termination1.get() == True:
                
                break
        if cv2.waitKey(1) == 27:
                
                # terminate_thread(arduino, cap, video, tv_video, DATA, log_data, f)
                terminate_thread(arduino, cap, f)
                break
        try:
            final_video_capture = 1/(time.time()-video_capture_time)
            print("video capture function time",final_video_capture)
        except:
            print("nothing executed")
    cap.release()
    cv2.destroyAllWindows()
    print("Perception process has stopped")
        
               
        
# def getSegmentedArea(test_img,model):
#     frame = np.expand_dims(test_img, axis = 0)
#     prediction = model.predict(frame)
#     prediction_image = prediction.reshape((416,416))
#     result = cv2.normalize(prediction_image, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#     return result


def getpath(scaled_topview_queue,termination2,arduino):
    # li = list(zip(xvalues,yvalues))
    # print(f"fi:{li}")
    global prev_time_my
    while True:
        # if not scaled_topview_queue.empty():
        start_time = time.time()
        astar_io = scaled_topview_queue.get()
        
        a_star = AStarPlanner(astar_io.xvalues, astar_io.yvalues, 27, 50)
        grid = a_star.get_grid(sx, sy, astar_io.gx, astar_io.gy,astar_io.xvalues,astar_io.yvalues)
        # print(grid)
        mn = -sys.maxsize-1
        g_x = 0
        g_y = 0
        for i in range(len(grid)):
            if grid[i][1] > mn:
                mn = grid[i][1]
                g_x = grid[i][0]
                g_y = grid[i][1]
        print(f"g_x: {g_x},g_y: {g_y}")
        rx, ry= a_star.planning(sx, sy, g_x, g_y,astar_io.xvalues,astar_io.yvalues)
        lines = list(zip(rx,ry))
        lines = lines[::-1]
        # arduino = None
        # if ARDUINO_CONNECTED:
        #     arduino = connect_arduino()
        carx = 1000
        cary = 0
        i = 0
        radius = 180
        allPointsInside = False
        angle = 0
        print("path: ",lines)
        try:
            if type(lines) is tuple:
                lines = [lines]
            dist = math.sqrt((cary-lines[i][1])**2+(carx-lines[i][0])**2)
            if len(lines) > 1:
                while dist < radius:
                    dist = math.sqrt((cary-lines[i][1])**2+(carx-lines[i][0])**2)
                    if dist > radius:    
                        break
                    i += 1
                    if i > len(lines):
                        allPointsInside = True
                        break
            if allPointsInside:
                i = -1
            angle = control.car_angle((carx,cary),lines[i],True)
            print("outside",angle)
            angle = math.floor(angle)
        except:
            angle = 0
            i = -1
            print("problem in pp logic angle")
        selected_point = lines[i]
        if angle > 50:
            angle = 50
        elif angle < -50:
            angle = -50 
        print(f"angle: {angle},dist: {dist}")
        top_view = np.zeros((1100, 1300, 3),dtype=np.uint8)
        boundary = list(zip(astar_io.xvalues,astar_io.yvalues))
        top_view = plotting.plot_cv2(rx,ry,top_view,(astar_io.gx,astar_io.gy),(sx,sy),radius,selected_point,boundary,grid)
        top_view = cv2.resize(top_view, (608, 608),interpolation=cv2.INTER_LINEAR) 
        # cv2.imshow("boundary",boundary)
        end = time.time()
        d = end - start_time
        fps = 1/d
        print(f"fps: {fps}")
        cv2.imshow("topview",top_view)
        if time.time() - prev_time_my >= MS:
            prev_time_my = time.time()
            if(ARDUINO_CONNECTED):

                serial_data = angle
                serial_writen_now = True
                
                send_data(arduino, serial_data)
        # if ARDUINO_CONNECTED:
        #     stop_car(arduino)
        if cv2.waitKey(1) == 27:
                # terminate_thread(arduino, cap, video, tv_video, DATA, log_data, f)
                terminate_thread(arduino)
                break
        if not termination2.empty():
            if termination2.get() == True:
                terminate_thread()
                break
        
            
    

def polygonArea(X, Y, n):
 
    # Initialize area
    area = 0.0
 
    # Calculate value of shoelace formula
    j = n - 1
    for i in range(0,n):
        area += (X[j] + X[i]) * (Y[j] - Y[i])
        j = i   # j is previous vertex to i
     
 
    # Return absolute value
    return int(abs(area/2.0))

def convertBack(x, y, w, h):
    """
    Converts detections output into x-y coordinates
    :x, y: position of bounding box
    :w, h: height and width of bounding box
    """
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def main():
    width = 416
    height = 416
    config_file = r"models\yolov4-3l-v4.cfg"
    data_file = r"models\yolov4-3l-v4.data"
    weights = r"models\yolov4-3l-v4.weights"
    network, class_names, class_colors   = darknet.load_network(
            config_file,
            data_file,
            weights,
            batch_size=1
        )
    
    cap = cv2.VideoCapture(r"D:\Autonomous\lane_segmentation\clg_roads.mp4")
    # cap = cv2.VideoCapture(1)
    scaled_topview_queue = Queue(maxsize=1)
    termination1 = Queue(maxsize=1)
    termination2 = Queue(maxsize=1)
    ardiuno =   None
    # planning_queue = Queue(maxsize=1)
    if ARDUINO_CONNECTED:
        ardiuno = connect_arduino()
    print("network loaded")
    try:
        t1 = Thread(target=start,args=(cap,network,class_names,scaled_topview_queue,termination1,ardiuno))
        t2 = Thread(target=getpath,args=(scaled_topview_queue,termination2,ardiuno))
        t1.start()
        t2.start()
        print("Threads started")
        while True:
            if not t2.is_alive():
                termination1.put(True)
                termination2.put(True)
                time.sleep(3)
                break
            time.sleep(2)
            
            
    except:
        cap.release()
        cv2.destroyAllWindows()
    print("program ends now")

if "__main__" == __name__ : 
    print("starting main")
    main()
    
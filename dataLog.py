import os
import datetime
# from constants import log_constants
from json import dump, load

test_cone_coordinates = [(1,2),(3,5)]
# log_folder = "/home/nvidia/Documents/VCET-Driverless/Solecthon/logs"
log_folder = "../logs"
def give_file():
	test_count = 0
	today = datetime.datetime.now()

	if not os.path.isdir(log_folder):
		os.mkdir(log_folder)

	day_folder = log_folder + "/" + str(today.day) + "-" + str(today.month) + "-" + str(today.year)

	if not os.path.isdir(day_folder):
		os.mkdir(day_folder)

	while os.path.isfile(day_folder + "/" +"test" + str(test_count) + ".json"):
		test_count = test_count + 1

	log_file_name = day_folder + "/" + "test" + str(test_count)	
	
	f = open(log_file_name + ".json", "w+")

	return f, log_file_name

def playback_video(path):
		cap = cv2.VideoCapture(path)
		while True:
			ret, frame = cap.read()
			cv2.imshow("frame",frame)
			if cv2.waitKey(1) == ord("q"):
				break
		cap.release()
		cv2.destroyAllWindows()


if __name__ == '__main__':
	import cv2
	inp = int(input("1. Playback of video from logs \n2. Logs \n"))
	if inp == 1:
		playback_video(r"C:\Autonomous\Solecthon-a_star\logs\24-12-2022\test3	.mp4")
	elif inp == 2:
		f, log_file_name = give_file()
		DATA = [log_constants()]
		DATA[0]["log_constants"]["CAM_PATH"] = "video.mp4"
		log_data = []

		for i in range(10):
			frame_data = {
					"time_stamp":datetime.datetime.now().astimezone().isoformat(),
					"frame_count":i,
					"steering":12,
					"left_box":test_cone_coordinates,
					"right_box":test_cone_coordinates,
					"lines":test_cone_coordinates
				}
			log_data.append(frame_data)
				
		DATA.append( {
				"log_data" : log_data
			} )
		dump(DATA, f, indent=4)

		f.close() 
def log_constants():
	return {
		"log_constants" : {
			"CAM_PATH" : CAM_PATH,  
			"BAUD_RATE" : BAUD_RATE, 
			"CAR_BELOW_Y" : CAR_BELOW_Y,
			"LIMIT_CONE" : LIMIT_CONE,
			"MIDPOINT_ONE_BOUNDARY" : MIDPOINT_ONE_BOUNDARY,
			"P" : P,
			"MAX_CONELESS_FRAMES" : MAX_CONELESS_FRAMES,
			"ARDUINO_CONNECTED" : ARDUINO_CONNECTED,
			"RATE" : RATE, 
			"TESTER" : TESTER,
			"WHICH_SYSTEM" : WHICH_SYSTEM,
			"TOP_VIEW_IMAGE_DIMESNION" : TOP_VIEW_IMAGE_DIMESNION,
			"FRONT_VIEW_POINTS" : FRONT_VIEW_POINTS,
			"Lookahead_Distance" : L
		}
	} 
import cv2
import numpy as np

class WebcamFeed(object):
	def __init__(self):
		self.video = cv2.VideoCapture(0)

	def __del__(self):
		self.video.release()

	def get_frame(self):
		success, frame = self.video.read()
		frame_flip = cv2.flip(frame,1)
		ret, jpeg = cv2.imencode('.jpg', frame_flip)
		return jpeg.tobytes()

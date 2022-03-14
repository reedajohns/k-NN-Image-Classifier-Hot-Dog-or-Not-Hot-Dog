# Class to provide prepressing pipeline

# Import required packages
import cv2

# Class
class ImPreprocessor:
	# Initialize (width and height to re-size)
	def __init__(self, width, height, inter=cv2.INTER_AREA):
		# Store width and height to re-size images to
		# Optionally define interpolation method
		self.width = width
		self.height = height
		self.inter = inter

	# Preprocessing funct
	def preprocess(self, image):
		# Return re-sized image (using cv2)
		return cv2.resize(image, (self.width, self.height), interpolation=self.inter)
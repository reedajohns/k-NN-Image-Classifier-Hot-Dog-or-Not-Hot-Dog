# Class to handle loading and preprocessing of image pipeline

# Import required packages
import cv2
import argparse
import os

import numpy as np


class ImDatasetLoader:
	# Initialize pipeline
	def __init__(self, preprocessors=[]):
		# Store preprocessors (if any)
		if len(preprocessors) == 0:
			# Set to None
			self.preprocessors = None
		else:
			# set to input
			self.preprocessors = preprocessors

	# Load images from paths and add preprocess step
	def load(self, image_paths, verbose = -1):
		# Create list(s) for image data and labels
		data = []
		labels = []

		# Loop over all image paths
		for (i, image_path) in enumerate(image_paths):
			# Load image and extract label
			# Assume that path is formatted as:
			# /path/to/dataset/{class}/{image}.jpg
			image = cv2.imread(image_path)
			label = image_path.split(os.path.sep)[-2]

			# Check to see if any preprocessing step, if so, apply
			if self.preprocessors is not None:
				# Perform all preprocessing steps could be more than 1)
				for preproc in self.preprocessors:
					image = preproc.preprocess(image)

			# Append to overall data and label lists
			data.append(image)
			labels.append(label)

			# Show progress update every nth image (if desired)
			if i > 0  and (i+1) % verbose == 0 and verbose > 0:
				print('----- Processed {} out of {} images.'.format(i+1, len(image_paths)))

		# Convert to numpy arrays
		data_np = np.array(data)
		labels_np = np.array(labels)

		# Return tuple
		return (data_np, labels_np)
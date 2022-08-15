import os
import matplotlib.pyplot as plt
import numpy as np

def getFileNames(file_type="csv", data_dir="./data"):

	'''
	This function returns list of found file paths in the given directory in the form of string.

	file_type - string
			File type extension abbreviation (without the dot).
	data_dir - string
			Directory path 
	'''
	data_files = os.listdir(data_dir)
	data_paths = []
	for file in data_files:
		if file[-len(file_type):] == file_type:
			data_paths.append("{}/{}".format(data_dir,file))
	return data_paths

def plotStats(peaks):
	
	fig, ax = plt.subplots(1, 3,figsize=(20,12))
	plt.style.use('bmh')

	ax[0].plot(peaks["Pn"],'.',markersize=12)
	ax[0].set_xlabel("Count")
	ax[0].set_ylabel("Pn")

	ax[1].hist(peaks["Pn"],bins=10,edgecolor='black', linewidth=1.2)
	ax[1].set_ylabel("Count")
	ax[1].set_xlabel("Pn")

	ax[2].plot(peaks.ref1,'.', label='ref1',markersize=12)
	ax[2].plot(peaks.ref2,'.', label='ref2',markersize=12)
	ax[2].plot(peaks.dataPoint,'.', label = 'dataPoint',markersize=12)
	ax[2].set_xlabel("Count")
	ax[2].set_ylabel("Value/Power")
	ax[2].legend()
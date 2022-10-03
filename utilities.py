import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xlsxwriter

def getFileNames(file_type="csv", data_dir="./data"):

	'''
	This function returns list of found file paths in the given directory in the form of a string.

	`file_type` - string - 

	File type extension abbreviation (without the dot).

	`data_dir` - string - 

	Directory path 
	'''
	data_files = os.listdir(data_dir)
	data_paths = []
	for file in data_files:
		if file[-len(file_type):] == file_type:
			data_paths.append("{}/{}".format(data_dir,file))	
	return data_paths

def plotStats(peaks):

	'''
	This functions plots found peaks.

	`peaks` - `pandas` data frame -

	Result of the `analyze_peaks` method of class `SwapCore`.
	'''
	
	fig, ax = plt.subplots(1, 3,figsize=(20,12))
	plt.style.use('bmh')

	ax[0].plot(peaks["Pn"],'.',markersize=12)
	ax[0].set_xlabel("Count")
	ax[0].set_ylabel("Pn")

	ax[1].hist(peaks["Pn"],bins=10,edgecolor='black', linewidth=1.2)
	ax[1].set_ylabel("Count")
	ax[1].set_xlabel("Pn")

	ax[2].plot(peaks.ref1,'.', label='ref1',markersize=12,color='slateblue')
	ax[2].plot(peaks.ref2,'.', label='ref2',markersize=12,color='dimgray')
	ax[2].plot(peaks.dataPoint,'.', label = 'dataPoint',markersize=12,color='darkred')
	ax[2].set_xlabel("Count")
	ax[2].set_ylabel("Value/Power")
	ax[2].legend()

def find_parameters(path,rows_to_skip = 2):
    '''
    This function returns a pandas data frame containing estimated peak width, average height and
	maximum peak distance of the given signal.

	`path` - string, required -  

	.csv file name path. 

    `rows_to_skip` - int, optional - 

	How many header lines to skip in the .csv file. 
    '''

    powermeter_df = pd.read_csv(path, skiprows=rows_to_skip, names = ['time', 'value'],index_col=False)

    values = np.array(powermeter_df.value)	
    values_avg = np.mean(values)

    peak_start = 0
    peak_end = 0
    going_through_peak = False
    peak_len = []

    for index,value in enumerate(values):
        if (not going_through_peak) and (value >= values_avg):
            going_through_peak = True
            peak_start = index
        if going_through_peak and (value < values_avg):
            going_through_peak = False
            peak_end = index
            peak_len.append(peak_end-peak_start)
	
    _max = max(peak_len)
    peak_len = [x for x in peak_len if x > 0.5*_max]
    peak_avg  = np.mean(peak_len)

    vals  = {"PeakWidth":(peak_avg//1),"maxPeakDist":(2.5*peak_avg//1),"avgHeight":values_avg}
    parameters_df = pd.DataFrame.from_dict(vals,orient='index')

    return parameters_df.T


def write_to_xlsx(signal,peaks,name,dir="results/"):

	'''
	This function creates and exports given signal values and found peak values to the
	.xlsx file. 
	'''
	
	name = name.split("/")[1][:-4] + "-results.xlsx"

	writer = pd.ExcelWriter(dir+name,engine='xlsxwriter')   
	workbook=writer.book
	worksheet=workbook.add_worksheet('results')
	writer.sheets['results'] = worksheet
	signal.to_excel(writer,sheet_name='results',startrow=0 , startcol=0)   
	peaks.to_excel(writer,sheet_name='results',startrow=0, startcol=4) 
	writer.save()
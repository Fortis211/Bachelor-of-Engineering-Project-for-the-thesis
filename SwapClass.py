import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


class SwapCore():

    def __init__(self,path,rows_to_skip=2):

        '''
        Class constructor, sets default parameters.

        `path` - string, required -  

        .csv file name path. 

        `rows_to_skip` - int, optional -

         How many header lines to skip in the .csv file.
        '''

        self.path = path
        self.rows_to_skip = rows_to_skip

        self.value_array = None
        self.time_array = None
        self.peaks = None

        # Parameters
        self.x_axis_norm = True
        self.min_height_peak = 1*10**(-6)
        self.horizontal_distance = 1 
        self.vertical_dist_threshold = None
        self.max_peak_distance = 8 

        self.number_of_peaks_per_signal = 3

        self.analysis_without_correction = False

        self.rolling_mean_signal = False
        self.roll_strength = None
        self.original_value_array = None

        # Loaded file
        self.powermeter_df = None

    def load_file(self):

        '''
        This method loads .csv file into pandas data-frame and then splits data into class fields as the
        numpy array.
        '''

        powermeter_df =  pd.read_csv(self.path, skiprows=self.rows_to_skip, names = ['time', 'value'],index_col=False)
        self.value_array = np.array(powermeter_df.value)

        if self.x_axis_norm == True:
            self.time_array = np.arange(len(powermeter_df.time))
        else:    
            self.time_array = np.array(powermeter_df.time)    
        
        if self.rolling_mean_signal == True:
            self.original_value_array = np.array(powermeter_df.value).copy()
            self.value_array = np.array(powermeter_df['value'].rolling(self.roll_strength,min_periods=1).mean())
            self.time_array = np.arange(len(self.value_array))

        self.powermeter_df = powermeter_df
       
    def find_and_group_peaks(self):

        '''
        This method is using `scipy.signal` find_peaks to find peaks that are matching given parameters
        and then splits them into list of lists accordingly for further processing.
        '''

        if (self.time_array is None) or (self.value_array is None):
            self.load_file()

        # Function 'find_peaks' from scipy.signal finds the peaks in the signal considering the given boundary conditions.
        peaks, peak_heights = find_peaks(self.value_array, height=self.min_height_peak, 
                                        distance=self.horizontal_distance, threshold=self.vertical_dist_threshold)

        # Divide the indices of the found peaks into groups of individually sent signals.
        signal_list, last = [[]], None

        for peak in peaks:
            if last is None or abs(last - peak) <= self.max_peak_distance:
                # [-1] refers to the last element
                signal_list[-1].append(peak)
            else:
                signal_list.append([peak])
            last = peak 
        
        return signal_list
    
    def find_peaks(self):

        '''
        This method returns all peaks found by `scipy.signal.find_peaks` function.
        Could be used for debugging. 
        '''

        if (self.time_array is None) or (self.value_array is None):
            self.load_file()

        # Function 'find_peaks' from scipy.signal finds the peaks in the signal considering the given boundary conditions.
        peaks, peak_heights = find_peaks(self.value_array, height=self.min_height_peak, 
                                        distance=self.horizontal_distance, threshold=self.vertical_dist_threshold)

        print(f"Found signals count (including ones with signal being higher or lower than both ref signals): {len(peaks)//3}")   
                          
        return peaks

    def ref_analysis(self):

        '''
        This method filters out found peaks provided by find_and_group_peaks method of this class
        by selecting and correcting found peaks indices. Found peaks are then further analyzed in the
        analyze_peaks method.
        '''

        signal_list = self.find_and_group_peaks()

        # Filter for signals with the correct amount of peaks.
        correct_signal_list = []
        for x in signal_list:
            if len(x) == self.number_of_peaks_per_signal:
                correct_signal_list.append(x)

        """ 
        2. Correction: Add missing datapoints to signal
        """
        # Create list 'points_number_list' to check the number of datapoints per signal
        points_number_list = []
        # Create list for the indices of correct peaks 
        new_peaks_correct = []
        # Create list of lists which each contain the indices of correct peaks for one single signal
        signal_correct_peaks = []

        # for each signal consisting of number_of_peaks_per_signal peaks
        for corr_sig in correct_signal_list:
            # a is the index of the first peak. Now check whether the 2 datapoints before should also be part of that 
            # signal -> remember that the reference peaks i.e. the first part of the signal is sent for the length of 
            # three time-gaps
            a = corr_sig[0] 
            # if the value is equal or bigger than the third part of its neighbor: add as part of the signal,
            # repeat for second neighbor
            if self.value_array[a-1] >= self.value_array[a]/3: 
                a = a-1
                if self.value_array[a-1] >= self.value_array[a]/3:
                    a = a-1
            else:
                a=corr_sig[0]

            # b is the index of the last peak. Check whether the 2 following datapoints should also be part of that signal
            # remember that the reference peaks i.e. the last part of the signal is sent for the length of three time-gaps
            b = corr_sig[-1]
            # if the value is equal or bigger than the third part of its neighbor: add as part of the signal,
            # repeat for second neighbor
            if self.value_array[b+1] >= self.value_array[b]/3:
                b = b+1
                if (b+1)<len(self.value_array):
                    if self.value_array[b+1] >= self.value_array[b]/3:
                        b = b+1
            else:
                b = corr_sig[-1]

            """
            3. Correction: Check if signal fulfills conditions for good time-gap alignment between sender and receiver.
            -Check the number of datapoints per signal, only consider those where the overall time-gap of sender 
            and receiver match. 
            -In the next step check, whether within the signal the time-gaps match by checking that the number of 
            datapoints in pause-state i.e. datapoints with values equal or smaller than 1/3*Min(signal peaks) 
            (thus 1/3 of the small reference) is correct as well as the number of datapoints in sending-state i.e.
            datapoints with values equal or bigger than 2/3*Min(signal peaks) (thus 2/3 of the small reference).
            Here the algorithm allows for maximal two datapoints which do not match the conditions i.e. one pause-state-
            and one sending-state datapoint.
            """
            points_number_list.append((b-a+1))
            # correct number of points in the overall signal
            if (b-a+1) == (self.number_of_peaks_per_signal*3 + (self.number_of_peaks_per_signal-1)*3):
                # array of indices of the middle datapoint of the peaks
                peaks_correct_new = np.arange(a+1, b+1, 6)
                # list of all datapoints in one signal 
                checker = np.arange(a, b+1, 1)

                # list of pause-state datapoints
                check_list1 = [x for x in self.value_array[checker] if x <= (2*np.min(self.value_array[peaks_correct_new])/6)]
                
                # list of sending-state datapoints
                check_list2 = [x for x in self.value_array[checker] if x >= (4*np.min(self.value_array[peaks_correct_new])/6)]

                # check if number of pause- or rather sending-state datapoints is correct allowing for max 2 errors
                if len(check_list1) in ( ((self.number_of_peaks_per_signal-1)*3), ((self.number_of_peaks_per_signal-1)*3) +1,
                    ((self.number_of_peaks_per_signal-1)*3) -1) and len(check_list2) in ( (self.number_of_peaks_per_signal*3), 
                   (self.number_of_peaks_per_signal*3) +1, (self.number_of_peaks_per_signal*3) -1):
            #optionally allow for more errors
            #, ((number_of_peaks_per_signal-1)*3)+2, ((number_of_peaks_per_signal-1)*3)-2 
            #, (number_of_peaks_per_signal*3)+2, (number_of_peaks_per_signal*3)-2 

                    # if the conditions are met: 
                    # -add the indices of this signal to the signal_correct_peaks list
                    # -add the indices of this signal to the continuous list of indices: new_peaks_correct
                    signal_correct_peaks.append(peaks_correct_new)
                    for peakilito in peaks_correct_new:
                        new_peaks_correct.append(peakilito)
        
        # print some details of the processing in order to get an idea if the measurement is a success.

        avg_points_per_signal = np.average(points_number_list)
        expected_avg_points_per_signal = (self.number_of_peaks_per_signal*3)+ ((self.number_of_peaks_per_signal-1)*3)

        print(f"Average number of points per signal: {avg_points_per_signal}. Should be: {expected_avg_points_per_signal}")

        self.peaks = new_peaks_correct

        return avg_points_per_signal, expected_avg_points_per_signal

    def ref_analysis_without_correction(self):

        '''
        This method filters out found peaks provided by `find_and_group_peaks` method of this class
        by selecting found peaks indices. Found peaks are then further analyzed in the
        `analyze_peaks` method. This method does not correct found peaks in any ways other than just selecting groups of
        set number of peaks per signal.
        '''
        signal_list = self.find_and_group_peaks()

        # Filter for signals with the correct amount of peaks.
        correct_signal_list = []
        for x in signal_list:
            if len(x) == self.number_of_peaks_per_signal:
                correct_signal_list.append(x)

        # rewriting nested list to one dimensional list
        signal_list_1d = []
        for signal in correct_signal_list:
            for peak in signal:
                signal_list_1d.append(peak)

        self.peaks = signal_list_1d
        print(f"Found signals count (including ones with signal being higher or lower than both ref signals): {len(signal_list_1d)//3}")
       
    def analyze_peaks(self):

        '''
        This method filters out incorrect signals, that is signals that are bigger or smaller than both
        reference signals.

        Returns pandas data-frame containing indices of the found peaks, grouped into columns.
        The 4th column named 'Pd' contains calculated imposition of the signals basing on the reference signals.

        Used formula - Pn = [MiddleDataPoint - Min(Ref1,Ref2)] / |Ref1 - Ref2|.
        '''

        if self.peaks is None:
            if self.analysis_without_correction == False:
                self.ref_analysis()
            if self.analysis_without_correction == True: #and self.rolling_mean_signal == False:
                self.ref_analysis_without_correction()
            #if self.analysis_without_correction == True and self.rolling_mean_signal == True:
            #    self.find_peaks()     

        ref1_peaks =  self.value_array[self.peaks[0::self.number_of_peaks_per_signal]]
        ref2_peaks = self.value_array[self.peaks[2::self.number_of_peaks_per_signal]]
        data_points = self.value_array[self.peaks[1::self.number_of_peaks_per_signal]]

        ref1_peaks_index =  self.time_array[self.peaks[0::self.number_of_peaks_per_signal]]
        ref2_peaks_index = self.time_array[self.peaks[2::self.number_of_peaks_per_signal]]
        data_points_index = self.time_array[self.peaks[1::self.number_of_peaks_per_signal]]

        a = {"ref1":ref1_peaks,"ref2":ref2_peaks,"dataPoint":data_points,
            "ref1_index":ref1_peaks_index,"ref2_index":ref2_peaks_index,"dataPoint_index":data_points_index}
 
        peaks_df = pd.DataFrame.from_dict(a, orient='index')
        peaks_df = peaks_df.transpose()

        # calculating signal imposition ( Pn = [MiddleDataPoint - Min(Ref1,Ref2)] / |Ref1 - Ref2| )
        peaks_df["Pn"] =(peaks_df["dataPoint"] - peaks_df[["ref1","ref2"]].min(axis=1)) / abs(peaks_df["ref1"] - peaks_df["ref2"]).dropna(how='all',axis=0)

        # Condition to filter out incorrect signals 
        condition = "((dataPoint  < ref1) & (dataPoint > ref2)) | ((dataPoint  > ref1) & (dataPoint < ref2))"
        

        #return peaks_df.query(condition)
        return peaks_df

    def plot(self,x_lim=None):

        '''
        This method creates 3 subplots with labeled peak markers found by `ref_analysis` or `ref_analysis_without_correction` method of this class.
        
         `x_lim` - List, tuple optional - Set the x limits of the axes for the second and third plot.
        '''
        peaks = self.analyze_peaks()

        fig, ax = plt.subplots(3, 1, sharey='col',figsize=(20,14))
        plt.style.use('bmh')

        plt.suptitle(self.path)

        if x_lim != False:
            ax[1].set_xlim(x_lim)
            ax[2].set_xlim(x_lim)
        
        for i in [0,1,2]:
            if i != 1:
                ax[i].plot(self.time_array, self.value_array,  label='Signal')
            else:
                ax[i].plot(self.time_array, self.value_array, '.', label='Signal')
            ax[i].plot(peaks.dataPoint_index, peaks.dataPoint,"x", markersize=6, mew=3, label='Bit-string-datapoint',color='darkred')
            ax[i].plot(peaks.ref1_index, peaks.ref1, "x", color="slateblue", markersize=6, mew=3, label='Ref 1')
            ax[i].plot(peaks.ref2_index, peaks.ref2, "x", color="dimgray", markersize=6, mew=3, label='Ref 2')    
            ax[i].set_xlabel("Time [a.u.]", fontsize=16)
            ax[i].set_ylabel("Value", rotation=90, fontsize=16)
            ax[i].legend(loc="upper right", prop={'size':11}, fontsize=11)

        plt.show()

    def set_parameters(self,x_axis_norm=True, min_height_peak=1*10**(-6), horizontal_distance=1, 
                            vertical_dist_threshold=None, max_peak_distance=8, number_of_peaks_per_signal=3,analysis_without_correction=True, 
                            rolling_mean_signal=True, roll_strength=25):

            '''
            This method allows to set parameters for the given signal, if called without arguments it will set default
            parameters that are also set in the class constructor.

            `x_axis_norm` -  Boolean True or False -

            The time value from the measurement can be optionally replaced by
            a sequence of integers of the same length for more clarity. 

            `min_height_peak`  - number or ndarray or sequence, optional -
           Required height of peaks. Either a number, `None`, an array matching  
           `x` or a 2-element sequence of the former. The first element is  
           always interpreted as the  minimal and the second, if supplied, as the maximal required height. 

           `horizontal_distance` - number, optional -

            Required minimal horizontal distance in samples between neighboring 
            peaks. Smaller peaks are removed first until the condition is 
            fulfilled for all remaining peaks.

            `vertical_dist_threshold` - number or ndarray or sequence, optional -

            Required threshold of peaks, the vertical distance to its neighboring  samples.

            `max_peak_distance` - integer number -

            The maximal distance between peaks belonging to the same signal 
            (same signal = same cycle as the DMD sends the signal in a loop).

           `number_of_peaks_per_signal` - integer number -

            The number of peaks per signal to be considered (more or rather less 
            means a wrong time-gap-alignment and the signal is neglected).

            `analysis_without_correction` - Boolean True or False, optional

            Setting to True will use ref_analysis_without_correction instead of ref_analysis.    

            `rolling_mean_signal` - Boolean True or False -

            Setting to True will apply provided `roll_strength` parameter on the signal, to use it, it is necessary to also make sure
            that  `analysis_without_correction` is set to True.

            `roll_strength` - positive integer -

            The amount of points to consider for signal averaging.

            '''
            self.x_axis_norm = x_axis_norm
            self.min_height_peak = min_height_peak
            self.horizontal_distance = horizontal_distance 
            self.vertical_dist_threshold = vertical_dist_threshold
            self.max_peak_distance = max_peak_distance 
            self.number_of_peaks_per_signal = number_of_peaks_per_signal
            self.analysis_without_correction = analysis_without_correction  
            self.rolling_mean_signal = rolling_mean_signal
            self.roll_strength = roll_strength    

    def calculate_time_interval(self):

        '''
        This method calculates time intervals between points in the given signal and prints out
        count of the calculated intervals, mean and std.
        '''
        sig = self.powermeter_df
        sig['interval'] = sig['time']
        sig = sig.set_index('time').diff()
        print('Time interval counts:')
        print(sig.interval.value_counts())
        mean = sig.aggregate('interval').mean()
        std = sig.aggregate('interval').std()
        print(f"Interval mean = {mean}, std = {std}")

    def compare_rolling_mean_with_original(self,x_lim=None):

        '''
        This method creates 2 plots with plotted original signal and averaged one for comparison purposes. Works only if 
        `analysis_without_correction`, `rolling_mean_signal` are set to `True` and `roll_strength` is provided.

        `x_lim` - List, tuple optional - 
        
        Set the x limits of the axes for the second plot.
        '''

        peaks = self.analyze_peaks()

        fig, ax = plt.subplots(2, 1, sharey='col',figsize=(20,14))
        plt.style.use('bmh')

        plt.suptitle(self.path)

        if x_lim != False:
            ax[1].set_xlim(x_lim)

        for i in [0,1]:

            ax[i].plot(self.time_array+self.roll_strength//2, self.original_value_array,  label='Original signal',color='tan')
            ax[i].plot(self.time_array, self.value_array,  label='Rolling mean signal')

            ax[i].plot(peaks.dataPoint_index, peaks.dataPoint,"x", markersize=6, mew=3, label='Bit-string-datapoint',color='darkred')
            ax[i].plot(peaks.ref1_index, peaks.ref1, "x" , markersize=6, mew=3, label='Ref 1',color='slateblue')
            ax[i].plot(peaks.ref2_index, peaks.ref2, "x", markersize=6, mew=3, label='Ref 2',color='dimgray')    

            ax[i].set_xlabel("Time [a.u.]", fontsize=16)
            ax[i].set_ylabel("Value", rotation=90, fontsize=16)
            ax[i].legend(loc="upper right", prop={'size':11}, fontsize=11)
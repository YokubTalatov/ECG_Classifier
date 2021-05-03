# -*- coding: utf-8 -*-
"""
DIAGNOSTICS OF THE CARDIOVASCULAR SYSTEM BASED ON NEURAL NETWORKS
Classify ECG-Signals to CAD-Symptoms by means of artificial neural networks

Skript containing classes ECGRecord and ECGDataset with all methods needed to run the classifier pipeline.

"""
# %% Imports and Settings
import numpy as np
import os
import pandas as pd
import biosppy
from scipy.signal import butter, sosfilt
from scipy import stats
from collections import Counter
from statsmodels.tsa.stattools import acovf
import math
import itertools

import wfdb
import pywt
import hrvanalysis
from hrvanalysis import get_time_domain_features, get_frequency_domain_features, get_geometrical_features, get_poincare_plot_features

from Skripts.HelperFunctions import *

# %% Class Definitions
class ECGRecord:
    '''
    Class for ECG Records.
    Input: Instance of class wfdb.Record
    Output: Instance of class ECGRecord
    Attributes:
        - name: name of the record
        - signal: raw ECG signal
        - len: length of sample
        - filtered_signal: ECG signal after filtering
        - features: computed features
    '''
    def __init__(self, recordname,record):
        '''
        Extract attributes from wfdb.record Class.
        Possible Attributes: [base_date, base_time, comments, fs,n_sig,sig_len,sig_name,units]
        '''
 
        self.name = recordname
        self.n_sig = record[1]['n_sig']
        if self.n_sig > 1:
            self.full_signal = record[0][:,0] # Just the first (Lead II) signal
        else:
            self.full_signal = record[0].flatten()
        self.full_len = record[1]['sig_len']
        self.freq = record[1]['fs']
        self.signal = self.full_signal[:min(self.full_len,30*60*self.freq)]
        self.signal_len = len(self.signal)
        self.filtered_signal = None
        self.features = None
        self.label= None
        
    def assign_label(self,label):
        '''
        Assigns target label to record sample
        '''
        self.label = label

        
class ECGDatabase:
    '''
    Class to represent the complete Database.
    Attributes:
        - records:      Dict with Instances of class ECGRecord
        - frequency:    sampling frequency
        - features:     Computed features/HRV Parameters
    '''
    
    def load_dataset(self, dirPath):
        '''
        Load Dataset from specified Path to internal Representation i.e. List of ECGRecord Instances
        Input:
            - dirPath: Directory of the Database with .dat files
            
        Output:
            - records: List with Instances of Class ECGDatabase
        '''
        
        # Checks if directory exist
        if not os.path.exists(dirPath):
            raise Exception(f"Directory does not exist: {dirPath}")
        
        #     
        record_names = [os.path.splitext(file)[0] for file in os.listdir(dirPath) if os.path.isfile(os.path.join(dirPath, file)) and file.endswith('.dat')]
        joinPath = lambda file: os.path.join(dirPath, file)
        record_list = list(map(wfdb.rdsamp, list(map(joinPath,record_names)))) # list with tupel for each record. first entry is signal, second is dict with information about record
        return [ECGRecord(recordname,record) for recordname, record in zip(record_names, record_list)]
    
    def __init__(self,  dirPath = None):
        '''
        Initialise ECGDatabase Instance. If direction dirPath is given, the dataset will be loaded from the speciefied direction, otherwise it will create an empty class.
        '''
        
#        self.records = []
#        self.record_names = []
#        self.freq = None
        
        if not dirPath == None:
            self.records = self.load_dataset(dirPath)
            self.record_names = [record.name for record in self.records]
            self.freq = self.records[0].freq
            self.n_records = len(self.records)

        
    
    def combine_datasets(self, ecg_databases):
        '''
        Combines multiple instances of class ECGDatabase to a new one.
        '''
        #self.records = [database.records for database in ecg_databases]
        self.records = []
        for database in ecg_databases:
            self.records += database.records
        self.record_names = [record.name for record in self.records]
        self.n_records = len(self.records)
        self.freq = ecg_databases[0].freq
        if not len(set([database.freq for database in ecg_databases])) == 1:
            print('Warning! Databases have different sampling frequencies!')
        
    
    def nofilter_signal(self, arg = None):
        '''
        Provides working even if no filtering is performed
        '''     
        for record in self.records:
            record.filtered_signal = record.signal[:]
    
    def filter_signal(self, arg = None):
        '''
        Performs frequency filtering methods to extract the raw ecg signal out of the biased measured version.
        Methods:
            - Remove baseline wandering with wavelet transformation
            - Remove power spectrum frequency 
            - Remove stochastical bias
        Attributes:
            - filtered_signal: signal after filtering 
        '''
        def filter_dwt(record):
            '''
            Performs baseline wander removal using wavelet deformation.
            '''
            coeffs = pywt.wavedec(record.signal, wavelet='db6', mode='sp1', level = 8) # sp1 is the mode for smooth edge correction
            cA, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
            cA[:] = 0
            record.wftfiltered_signal = pywt.waverec([cA, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1], wavelet= 'db6',mode= 'sp1')
            record.filtered_signal = record.wftfiltered_signal[:]
        
        def filter_powerline(record, powerline):
            '''
            Performs powerline interference removal using butterworth band stop filter.
            '''
            nyq = 0.5 * record.freq
            start = (powerline - 1) / nyq
            stop = (powerline + 1) / nyq
            sos = butter(4, [start, stop], btype = 'bandstop', analog = False, output = 'sos')
            record.butterfiltered_signal = sosfilt(sos, record.filtered_signal)
            record.filtered_signal = record.butterfiltered_signal[:]
            
        def filter_lowpass(record):
            nyq = 0.5 * record.freq
            high = min(0.99, 80 / nyq)
            sos = butter(4, high, btype = 'lowpass', analog = False, output = 'sos')
            record.lowpassfiltered_signal = sosfilt(sos, record.filtered_signal)
            record.filtered_signal = record.lowpassfiltered_signal[:]
        
        def filter_noise(record):
            coeffs = pywt.swt(record.filtered_signal, 'db6', trim_approx = True)
            std = 1.483 * np.median(coeffs[-1])
            tsh = std * math.sqrt(2 * math.log(record.signal_len))
            for i, swt_coeff in enumerate(coeffs[1:],1):
                coeffs[i] = [np.sign(CD) * (abs(CD)-tsh) if abs(CD) >= tsh else 0 for CD in swt_coeff ]
            record.noisefiltered_signal = pywt.iswt(coeffs, wavelet= 'db6')
            record.filtered_signal = record.noisefiltered_signal[:]
        
        def smoothing(record):
            record.smoothed_signal, smoother_params = biosppy.signals.tools.smoother(record.filtered_signal)
            record.filtered_signal = record.smoothed_signal[:]
        
        for record in self.records:
            filter_dwt(record)
            filter_powerline(record, powerline = 60)
            filter_lowpass(record)
            #filter_noise(record)
            smoothing(record)

            
    def compute_nnints(self, arg = None):  
        '''
        Computes NN-Intervals from ECG signal. Detects RPeaks, computes RR-Intervals and discard outliers to receive NN-intervals.
        Input: 
            -record: ECGRecord Instance
        Attributes:
            - rpeaks: List of indices of R-Peaks
            - rr_ints: List of interbeat interval durations 
            - nn_ints: List of normal interbeat inverval durations
        '''
                    
        def detect_rpeaks(record):
            '''
            Detects R-peaks in ecg signal.
            Input:
                - record: ECGRecord Instance
            Outpu:
                - rpeaks: List of Indices where rpeaks appear in ecg signal
            '''
            
            #record.rpeaks, = biosppy.ecg.hamilton_segmenter(signal=record.filtered_signal, sampling_rate=record.freq)
            record.rpeaks, = biosppy.ecg.engzee_segmenter(signal=record.filtered_signal, sampling_rate=record.freq)
            record.rpeaks, = biosppy.ecg.correct_rpeaks(signal=record.filtered_signal,
                                     rpeaks=record.rpeaks,
                                     sampling_rate=record.freq,
                                     tol=0.1)
            
        def compute_rrints(record):
            '''
            Computes interbeat interval duration.
            Input: 
                -record: ECGRecord Instance
            Ouput:
                - rr_ints: List of rr interval durations of given ecg signal
            '''
            
            record.rr_ints = []
            cnt = 0
            while (cnt < (len(record.rpeaks)-1)):
                record.rr_interval = (record.rpeaks[cnt+1] - record.rpeaks[cnt]) #Calculate distance between beats in # of samples
                ms_dist = ((record.rr_interval / record.freq) * 1000.0) #Convert sample distances to ms distances
                record.rr_ints.append(ms_dist) #Append to list
                cnt += 1


        def discard_outliers(record, winsize = 2, cutoff = 0.2):
            '''
            Remove RR-Intervals which are < 0.4s and > 2s.
            Calculate Avarage of neighboring intervals (20 to each side). If it lies outside of 20/30% interval of the mean remove it.
            Input:
                - record: ECGRecord Instance
                - winsize: Range in which the avarage is computed
                - cutoff: Deviation of avarage in which a interval is considered as outlier
            Output: 
                - List of interval durations without un/misdetected intervals
            '''
            
            record.nn_ints = [rr_int for rr_int in record.rr_ints if rr_int > 400 and rr_int < 2000]
            
            for idx, nn_int in enumerate(record.nn_ints[winsize:len(record.rr_ints)-winsize]):
                mvidx = [ix for ix in range(len(record.nn_ints)) if ix > idx-winsize and ix < idx+winsize and idx is not ix]
                mean = np.mean([record.nn_ints[i] for i in mvidx])
                if mean - (mean * cutoff) <= nn_int <= mean + (mean * cutoff):
                    record.nn_ints[idx] = mean
                    
        def preprocessing_rrints(record):
            '''
            Computes NN-ints from RR-ints:
                1) removes putliers outside range 300-2000 nan
                2) removes ectopic beats with Kamath method
                3) interpolates nans with linear interpolation
            Input: RR-Interval list
            Output: NN-Interval list
            '''
            record.nn_ints = hrvanalysis.preprocessing.get_nn_intervals(record.rr_ints,ectopic_beats_removal_method = 'malik', verbose = False)
            record.nn_ints = [nnint for nnint in record.nn_ints if str(nnint) != 'nan']
        for record in self.records:
            # Detect R-Peaks using segementer and correction algorithm
            detect_rpeaks(record)
            # Compute interbeat interval durations
            compute_rrints(record)
            # Remove Outliers
            #discard_outliers(record) # own version of preprocessing
            preprocessing_rrints(record)
        
        # Define class attributes for features
        self.rpeaks = [record.rpeaks for record in self.records]
        self.rr_ints = [record.rr_ints for record in self.records]
        self.nn_ints = [record.nn_ints for record in self.records]
                
    
    def compute_own_features(self,arg = None):
        '''
        Computes the features of the ecg signal preserved by the Project description.
        '''

        def compute_statistics_parameters(record):
            '''
            Computes HRV from the rr interval duration list.
            '''
            nnstats = stats.describe(record.nn_ints)
            
            record.mean = nnstats[2]
            record.std = np.sqrt(nnstats[3])
            record.rmsd = np.sqrt((1/(len(record.nn_ints) -1)) * sum(np.diff(record.nn_ints)**2))
            record.skew = nnstats[4]
            record.kurt = nnstats[5]

        
        def compute_correction_parameters(record):
            int_box = [nnint // 40 for nnint in record.nn_ints if str(nnint) != 'nan']
            int_diff = np.diff(int_box)
        
            record.DELp = 0
            record.DEL0 = 0
            record.DELn = 0   
            for diff in int_diff:
               if diff > 0: 
                   record.DELp += 1
               elif diff < 0: 
                   record.DELn += 1
               elif diff == 0: 
                   record.DEL0 += 1
               else: 
                   print(f'error in {record.name}: diff = {diff}') 
                   
            record.DELp /= len(int_diff)
            record.DEL0 /= len(int_diff)
            record.DELn /= len(int_diff)      
                   
                   
        def compute_autocorrelation_parameters(record):
            '''
            Compute Features which are derives from the autocorrelation function with K = 1
            '''
            
            record.cor = acovf(record.nn_ints, unbiased = True, fft = True, nlag = None)    
            record.cor /= record.cor[0]  
            record.r1 = record.cor[1] # Difference instead of first value?
        
            negatives = [neg[0] for neg in enumerate(record.cor) if neg[1] < 0]
            if len(negatives) > 0: 
                record.m0 = negatives[0]
            else:
                record.m0 = 0 # Maybe not the best to set an 'undefined' value to 0. Nan?
            
            
        # Iterate over all records in database
        for record in self.records:
            # Compute relevant statistics parameter from interval durations
            compute_statistics_parameters(record)
            #Compute Delta Parameters
            compute_correction_parameters(record)
            #Compute autocorrelation parameters
            compute_autocorrelation_parameters(record)

    def compute_hrv(self):
        '''
        Compute HRV parameters
        '''
        def compute_time_features(record):
            record.time_parameters = get_time_domain_features(record.nn_ints) # Dictionary with parameternames as keys
        def compute_frequency_features(record):
            record.frequency_parameters = get_frequency_domain_features(record.nn_ints) # Dictionary with parameternames as keys
        def merge_features(record):
            record.hrv_parameters = {**record.time_parameters, **record.frequency_parameters}
        
        for record in self.records:
            compute_time_features(record)
            compute_frequency_features(record)
            merge_features(record)
            
        self.hrv_parameters = [record.hrv_parameters for record in self.records]

    def create_df_own_features(self):
        '''
        Fill the DataFrame self.features with the already computed features
        '''
        feature_names = ["Label","AVNN","SDNN","RMSSD" ,"SKEW" ,"KURT","MO","AMO","M0","R1","DELp","DEL0","DELn"]
        features = pd.DataFrame(np.zeros([self.n_records, len(feature_names)]), index=self.record_names, columns=feature_names) 
        features['Label'] = [record.label for record in self.records]
        features['AVNN'] = [record.mean for record in self.records]
        features['SDNN'] = [record.std for record in self.records] 
        features["RMSSD"] = [record.rmsd for record in self.records]
        features['SKEW'] = [record.skew for record in self.records]
        features['KURT'] = [record.kurt for record in self.records]
        features['MO'] = [Counter(record.nn_ints).most_common(1)[0][0] for record in self.records]
        features['AMO'] = [Counter(record.nn_ints).most_common(1)[0][1]/len(record.nn_ints) * 100 for record in self.records]
        features['DELp'] = [record.DELp for record in self.records]
        features['DEL0'] = [record.DEL0 for record in self.records]
        features['DELn'] = [record.DELn for record in self.records]
        features['R1'] = [record.r1 for record in self.records]
        features['M0'] = [record.m0 for record in self.records] 
        
        return features
        
    def create_df_hrv_features(self):
        '''
        Fill the DataFrame self.features with the computed features
        '''
        feature_names = ["Label"] + list(self.hrv_parameters[0].keys())
        features = pd.DataFrame(np.zeros([self.n_records, len(feature_names)]), index=self.record_names, columns=feature_names)

        for key_parameter in self.hrv_parameters[0].keys():
            features[key_parameter] = [record.hrv_parameters[key_parameter] for record in self.records]
            
        return features
        
    def assign_features(self, feature_df1, feature_df2 = None):
        if feature_df2 is not None:
            feature_df1 = feature_df1.merge(feature_df2, left_index = True, right_index = True) 
            feature_df1 = feature_df1.drop(labels = ['Label_x', 'Label_y'], axis = 1)
            feature_df1.insert(0, 'Label', 0)
        self.features = feature_df1
        
    def extract_own_features(self):
        self.compute_nnints()
        self.compute_own_features()
        
    def extract_hrv_parameters(self):
        self.compute_nnints()
        self.compute_hrv()
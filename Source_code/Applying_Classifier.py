# -*- coding: utf-8 -*-
"""
DIAGNOSTICS OF THE CARDIOVASCULAR SYSTEM BASED ON NEURAL NETWORKS
Classify ECG-Signals to CAD-Symptoms by means of artificial neural networks

Requirements:
    Package Wfdb. Install by running the command on your command prompt: pip install wfdb
    Package pywt. Install by running the command on your command prompt: pip install PyWavelets
    Skript Preprcessing_Pipeline_Functions.py, Plotting_Function.py, (Training_Classifier)
    
    Make sure the directiory of the installed package is part of the environment variable.

Skript for HRV-Analysis consisting of signal preprocessing, feature extraction and classification of the dataset samples.
"""

#%% Importing Functions
import os
import pandas as pd
from joblib import load

from Skripts.Preprocessing_Pipeline_Functions import ECGDatabase
from Skripts.Plotting_Functions import *

#%% Loading Dataset
datasets = ['mit-bih-normal-sinus-rhythm-database-1.0.0','mit-bih-arrhythmia-database-1.0.0', 'RohmineEcgDatabase']    
# Directory of the Dataset
datasetname = datasets[2]
rootPath = 'C:/Users/Yokub/Documents/Study/IAESTE/ECGNetwork'
databasePath = os.path.join(rootPath, 'Database', datasetname) #'C:/Users/Yokub/Documents/Study/IAESTE/ECGNetwork/Database/RohmineEcgDatabase'
if not os.path.exists(os.path.join(rootPath, "Results_" + datasetname)):
            os.mkdir(os.path.join(rootPath, "Results_" + datasetname))
resultsPath = os.path.join(rootPath, "Results_" + datasetname)

# Load Dataset
ecg_data = ECGDatabase(databasePath)


#%% Signal Preprocessing - Filtering
ecg_data.filter_signal()

#No filtering
#ecg_data.nofilter_signal()

#%% Analyse Filtering via Plots
# Plot PowerlineSpectrum after filtering
if not os.path.exists(os.path.join(resultsPath, "Images_Powerlinespectrum")):
            os.mkdir(os.path.join(resultsPath, "Images_Powerlinespectrum"))
powerlinePath = os.path.join(resultsPath, "Images_Powerlinespectrum")
for record in ecg_data.records:
    plot_powerlinespectrum(record.signal, record.filtered_signal, freq = record.freq, dirPath = powerlinePath, picName= record.name + '_powerline_smoothed.png')
    
# Plot influence of filtering    
if not os.path.exists(os.path.join(resultsPath, "Images_Filtering")):
    os.mkdir(os.path.join(resultsPath, "Images_Filtering"))
filteringPath = os.path.join(resultsPath, "Images_Filtering")   
for record in ecg_data.records:
    #plot_filtered_signal(record.signal, record.wftfiltered_signal,freq = record.freq, dirPath = filteringPath, picName = record.name + '_wft.png', begin = 10, end = 15)
    #plot_filtered_signal(record.wftfiltered_signal, record.butterfiltered_signal,freq = record.freq, dirPath = filteringPath, picName = record.name + '_butter.png', begin = 10, end = 15)
    #plot_filtered_signal(record.butterfiltered_signal, record.noisefiltered_signal,freq = record.freq, dirPath = filteringPath, picName = record.name + '_noise.png', begin = 10, end = 15)
    #plot_powerlinespectrum(record.wftfiltered_signal, record.noisefiltered_signal, freq = record.freq, dirPath = powerlinePath, picName= record.name + '_noisefilter.png')
    plot_filtered_signal(record.signal, record.filtered_signal, freq = record.freq, dirPath = filteringPath, picName= record.name + '_filtering.png', begin = 10, end = 15)



#%% Signal Preprocessing - Feature Extraction
# Compute NN-Intervals
ecg_data.compute_nnints()

# HRV-Parameters
ecg_data.compute_hrv()
# Own-Parameters
ecg_data.compute_own_features()

# Merge both Parameters
hrv_df = ecg_data.create_df_hrv_features()
own_df = ecg_data.create_df_own_features()
ecg_data.assign_features(hrv_df, own_df) # not working. df contains just hrv

featurePath = os.path.join(resultsPath, 'features.csv')
ecg_data.features.to_csv(featurePath)

#%% Analyse NN-ints and HRV via Plots
# Checking RPeak Detection
if not os.path.exists(os.path.join(resultsPath, "Images_RPeaks")):
    os.mkdir(os.path.join(resultsPath, "Images_RPeaks"))
rpeaksPath = os.path.join(resultsPath, "Images_RPeaks")   
for record in ecg_data.records:
    plot_rpeak_detection(record.filtered_signal, record.rpeaks, record.freq, rpeaksPath, record.name + '_rpeak_detection.png')

# Comparing RR- with NN-Intervals
if not os.path.exists(os.path.join(resultsPath, "Images_HRV")):
            os.mkdir(os.path.join(resultsPath, "Images_HRV"))
hrvPath = os.path.join(resultsPath, "Images_HRV")
for record in ecg_data.records:
    plot_ints_timeseries(record.rr_ints,record.nn_ints, dirPath = hrvPath, picName= record.name + '_ints_timeseries_malik.png')

#%% Assigning Class
# Loading Dataset
featurePath = os.path.join(resultsPath, 'features.csv')
ecg_dataset = pd.read_csv(featurePath, header = 0, index_col = 0)
ecg_dataset.reset_index(drop=True, inplace = True)

for ind in ecg_dataset.index:
    ratio = ecg_dataset['lf_hf_ratio'][ind]
    if ratio < 0.9:
        class_nr = 1 #Parasympathius is overactivated
    elif ratio > 3.1:
        class_nr = 2 #Sympathius is overactivated
    else:
        class_nr = 0 # Balance of nervous system
    
    ecg_dataset['Label'][ind] = class_nr
    
labeledfeaturePath = os.path.join(resultsPath, 'features_labeled.csv')
ecg_dataset.to_csv(labeledfeaturePath)
# nearly no class 0 ->filtering maybe bad

#%% Classify
# Loading Dataset
labeledfeaturePath = os.path.join(resultsPath, 'features_labeled.csv')
ecg_dataset = pd.read_csv(labeledfeaturePath, header = 0, index_col = 0)
ecg_dataset.reset_index(drop=True, inplace = True)

# Loading the Features
X = ecg_dataset.drop(['Label'], axis = 1)

# Loading the Estimator
model_filename = 'model_all_parameters'
modelPath = os.path.join(rootPath, model_filename)
model = load(modelPath +'.joblib')

predictions = model.predict(X)


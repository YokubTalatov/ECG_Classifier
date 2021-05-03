# -*- coding: utf-8 -*-
"""
DIAGNOSTICS OF THE CARDIOVASCULAR SYSTEM BASED ON NEURAL NETWORKS
Classify ECG-Signals to CAD-Symptoms by means of artificial neural networks

Skript containing function to plot results.
"""
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter

    
def plot_filtered_signal(signal,filtered_signal, freq, dirPath, picName, begin = None, end = None):
    '''
    PLots the comparision of two signals in one fig.
    '''
    if begin is None:
        begin = 0
    if end is None:
        end = int(len(signal) / freq)
    
    fig, ax = plt.subplots(3, sharex = True, sharey = True)
    time = np.linspace(begin, end, (end - begin)*freq)
    ax[0].plot(time,signal[begin * freq:end * freq],'black', linewidth = 1, label = 'signal before filtering' )
    ax[0].legend(loc = 'upper right')
    ax[1].plot(time,filtered_signal[begin * freq:end * freq], 'g', linewidth = 1, label = 'signal after filtering')
    ax[1].legend(loc = 'upper right')
    ax[2].plot(time,signal[begin * freq:end * freq],'black', linewidth = 1, label = 'signal before filtering' )
    ax[2].plot(time,filtered_signal[begin * freq:end * freq], 'g', linewidth = 1, label = 'signal after filtering')
    ax[2].legend(loc = 'upper right')
    ax[2].set(xlabel = 'time in s', ylabel = 'amplitude')
    plt.grid()
    fig.savefig(os.path.join(dirPath, picName))
    plt.close(fig)

    
def plot_powerlinespectrum(signal, filtered_signal, freq, dirPath, picName):
    '''
    Plots Powerlinespectrum before and after filtering.
    '''
    fig, ax = plt.subplots(2, sharex = True, sharey = True)
    ax[0].psd(signal, 512, freq, label = 'before filtering')
    ax[0].legend(loc = 'upper right')
    ax[1].psd(filtered_signal, 512, freq, label = 'after filtering')
    ax[1].legend(loc = 'upper right')
    fig.savefig(os.path.join(dirPath,picName))
    plt.close(fig)

    
def plot_rpeak_detection(signal, rpeaks, freq, dirPath, picName):
    signal = np.array(signal)
    rpeaks = np.array(rpeaks, dtype = np.int_)
    r_indx = np.array([np.nan] * len(signal))
    r_indx[rpeaks] = signal[rpeaks]
    timeline = np.linspace(0, len(signal)/360, len(signal))
    
    ax1 = plt.subplot(211) # Plot full ecg signal
    ax1.plot(timeline,signal, 'b') # Plot full ecg signal
    #plt.plot(rpeaks, signal[rpeaks], 'ro') # Plot R-Peaks as points
    ax1.plot(timeline,r_indx, 'ro', markersize=1)
    ax1.set_title('ECG-signal with detected R-Peaks')
    ax1.set_ylabel('ECG (mV)')
    ax1.set_xlabel('Time (s)')
    
    ax2 = plt.subplot(212) # Plot detailed ecg signal
    ax2.plot(timeline[10 * freq:20 * freq],signal[10 * freq:20 * freq]) # Plot detailed ecg signal
    ax2.plot(timeline[10 * freq:20 * freq],r_indx[10 * freq:20 * freq], 'ro', markersize=1) # Plot R-Peaks of the detailed interval
    ax2.set_title('Detailed ECG-signal with detected R-Peaks ')
    ax2.set_ylabel('ECG (mV)')
    ax2.set_xlabel('Time (s)')
    
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(os.path.join(dirPath,picName))
    plt.clf()

        
def plot_ints_timeseries(rr_intervals, nn_intervals, dirPath, picName, autoscale = True, y_min = None, y_max = None):
    
    style.use("seaborn-darkgrid")
    plt.figure(figsize=(12, 8))
    plt.title("Interval time series")
    plt.ylabel("Interval duration (ms)", fontsize=15)
    plt.xlabel("Time (s)", fontsize=15)
    
    plt.plot(np.cumsum(rr_intervals) / 1000, rr_intervals, 'black', label='RR-Intervals')
    plt.plot(np.cumsum(rr_intervals) / 1000, nn_intervals, label = 'NN-Intervals')
    plt.legend(loc = 'upper right')


    if not autoscale:
        plt.ylim(y_min, y_max)

    plt.savefig(os.path.join(dirPath,picName))
    plt.clf()
    plt.close()
    
    
def plot_class_distribution(dataset, y_train, y_test):
    colors = sns.color_palette('colorblind')
    
    class_count1 = Counter(dataset['Label'])
    data1 = [class_count1[key] for key in sorted(class_count1.keys())]
    
    class_count2 = Counter(y_train)
    data2 = [class_count2[key] for key in sorted(class_count2.keys())]
    
    class_count3 = Counter(y_test)
    data3 = [class_count3[key] for key in sorted(class_count3.keys())]
    
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.invert_yaxis()   
    ax.yaxis.set_tick_params(labelsize=20)
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data1))
    
    category_names = ['class 0', 'class 1', 'class 2']
    data1_cum = np.array(data1).cumsum()
    data2_cum = np.array(data2).cumsum()
    data3_cum = np.array(data3).cumsum()

    for i, colname in enumerate(category_names):
        width1 = data1[i]
        start1 = data1_cum[i] - width1
        ax.barh(['Complete Dataset'], width1, left = start1, height = 0.5, color = colors[i], label = colname)
        ax.text(start1 + width1/2, 0, str(width1), ha='center', va = 'center', color = 'white', fontsize=25)
        
        width2 = data2[i]
        start2 = data2_cum[i] - width2
        ax.barh(['Train Dataset'], width2, left = start2, height = 0.5, color = colors[i], label = colname)
        ax.text(start2 + width2/2, 1, str(width2), ha='center', va = 'center', color = 'white', fontsize=25)
        
        width3 = data3[i]
        start3 = np.sum(data2) + data3_cum[i] - width3
        ax.barh(['Test Dataset'], width3, left = start3, height = 0.5, color = colors[i], label = colname)
        ax.text(start3 + width3/2, 2, str(width3), ha='center', va = 'center', color = 'white', fontsize=25)
    

    
    ax.legend(ncol = 3,bbox_to_anchor=(0, 1), loc='lower left', fontsize = 20)
    
    plt.show()
    
    
    
    
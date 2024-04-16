import pandas as pd
from rfpy_intf_utils import *
import numpy as np 
from scipy import signal, fft
import matplotlib.pyplot as plt
import itertools
from scipy import stats
import ot
from hyperopt import fmin, tpe, hp
from matplotlib.dates import date2num


def write_log(outname, output_folder):
    
    
    # Get the current script's file name
    script_filename = inspect.getframeinfo(inspect.currentframe()).filename

    # Read the content of the current script
    with open(script_filename, 'r') as original_file:
        script_content = original_file.read()

    # Write the script content to a new text file
    with open(output_folder+outname+".log", 'w') as copy_file:
        copy_file.write(script_content)

    # print("File copied successfully.")

## Station information
##------------------------------------------------------------------------------
## ./station/P_Data/ should contain the RfPy RFs
wfolder = "/media/pasan/OneTouch/NZ_RF_interferometry/"
# wfolder = "/home/pasan/Desktop/RF_PSD_Timelapse_Imaging/"

station = "RATZ"  
network = "NZ"

## Output file names for PSD, RF (with synthetic) and velocity pertubation (syn)
## computations
outname = "Test1" ## output file name prefix for pkl files with psds and rfs

## make folder to output files
output_folder = "{:s}/PSD/{:s}/".format(station, outname)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
   
write_log(outname, output_folder)

## 01. Gather rfs
## -----------------------------------------------------------------------------
## Use this at the begining to gather all RFs to one pkl file. This can be 
## omitted in the subsequent runs as the next steps will read this pkl file.
## This is the time consuming step. But once it is done, synthetic RF and PSD 
## computation is quick. Make sure to comment the following line out after
## running it once (unless you recompute RFs using different params)
# gather_rfs(station, network)  

## 02. Filter RFs based on baz, epicentral dist and snr
## -----------------------------------------------------------------------------
baz_band = [0,360]
epi_band = [0,90]
snr = 2

## 03. Time window of RFs (in sec) to use for PSD computation.
## ----------------------------------------------------------------------------- 
## None to use the entire RFs
time_window = [3,10]
# time_window = None

## 04. Define time binning parameters
## -----------------------------------------------------------------------------
bin_size_days = int(1*365)
overlap_pc = 0.95

## 05. Define the period band to compute the integrated power
## -----------------------------------------------------------------------------
spower_periods = [0.3,1.]
# spower_periods = [0.5,2]


## 06. Compute PSDS and bin them
## -----------------------------------------------------------------------------

nb_workers = 6 ## number of processes for parallel computations

use_phase_list = False

phase_list_fname = "phase_descriptors.txt"
 
binned_rfs, rf_data_df = compute_rf_psds(wfolder = wfolder, station = station, 
                network = network, baz_band = baz_band, epi_band= epi_band, 
                snr=snr, bin_size_days = bin_size_days, overlap_pc = overlap_pc,
                spower_periods = spower_periods, time_window = time_window,
                output_folder = output_folder, outname=outname, 
                compute_syn=False, syn_pertb_ts=None, 
                use_phase_list = False, phase_list_fname = phase_list_fname, 
                nb_workers = nb_workers,
                dump_results=True)

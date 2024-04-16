import pandas as pd
import numpy as np
import os 
from obspy import read, Trace, Stream
from obspy.core import UTCDateTime
from datetime import datetime
import itertools
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.interpolate import LinearNDInterpolator
from pyraysum import prs, Model, Geometry, Control
from datetime import timedelta
from spectrum import *
from pandarallel import pandarallel
from joblib import Parallel, delayed
import signal
import _pickle
from tqdm import tqdm

def find_rf_ids(wfolder, network, station):
    '''
    Function to iteratively look for receiver functions (ids) calculated by RfPy, based on the 
    network and station names.

    Inputs:
    network                 : str
    station                 : str

    Outputs:
    rf_ids                  : np.ndarray
    '''
    rf_ids = np.array([name for name in \
         os.listdir("{:s}/P_DATA/{:s}.{:s}/".format(wfolder, network, station))])
    return rf_ids

def read_metadata_to_df(wfolder, station, network):
    '''
    Function to read metadata of the receiver functions calculated by RfPy based on the network and 
    station names.

    Inputs:
    network                 : str
    station                 : str

    Outputs:
    rf_metadata_df          : pd.DataFrame
    '''

    rf_ids = find_rf_ids(wfolder, network, station)
    rf_metadata_arr = np.array([])
    print ("Reading RF metadata for {:s}.{:s}".format(network,station))
    for r, rf_id in enumerate(tqdm(rf_ids, total=len(rf_ids))):
        try:
            metadata = pd.read_pickle(
                "{:s}{:s}/P_DATA/{:s}.{:s}/{:s}/Meta_Data.pkl".format(wfolder, station, network, \
                                                                      station, rf_id))
            
            metadata_dict = vars(metadata)
            metadata_dict['id'] = rf_id

            rf_metadata_arr = np.append(rf_metadata_arr, metadata_dict)
        
        ## some rf folders with rf_id may not have any data inside.
        except (FileNotFoundError, EOFError, _pickle.UnpicklingError) as e:
            pass
        

    rf_metadata_df = pd.DataFrame.from_records(rf_metadata_arr)
    rf_metadata_df.sort_values(by="time").reset_index(drop=True)
    rf_metadata_df["date"] = rf_metadata_df.apply(lambda row: pd.Timestamp(row.time.date), axis=1)
    return rf_metadata_df

def read_RF_Data(wfolder, station, network, rf_id):
    '''
    Function to read the calculated receiver function based on id. The receiver function is trimmed 
    so that the positive times are returned

    Inputs:
    network                 : str
    station                 : str
    rf_id                   : str

    Outputs:
    rf_data_df              : pd.DataFrame
    '''

    rf_data_df = pd.read_pickle(
                 "{:s}{:s}/P_DATA/{:s}.{:s}/{:s}/RF_Data.pkl".format(wfolder, 
                                            station,  network, station, rf_id))

    rf_data_df.trim(starttime=rf_data_df[0].stats.starttime + 
              (rf_data_df[0].stats.endtime -  rf_data_df[0].stats.starttime)/2)

    return rf_data_df


def gather_rfs(wfolder, station, network):
    '''
    Function to gather all radial receiver functions to one dataframe and output as pickled file 
    with numpy arrays. 

    Inputs:
    network                 : str
    station                 : str

    Outputs:
    rf_data_df              : pickle
    '''

    rf_metadata_df = read_metadata_to_df(wfolder, station, network)
    rf_data_dict_arr = np.array([])
    
    print ("Gathering RF data for {:s}.{:s}".format(network,station))
    for _e, event in tqdm(rf_metadata_df.iterrows(), total=len(rf_metadata_df)):

        try:
            rf_data_wf = read_RF_Data(wfolder, station, network, event.id)
            # print (rf_data_wf.select(component="R")[0])
            rf_data = \
        {"id": event.id,
        "waveform_data_r": rf_data_wf.select(component="R")[0].data,
        "waveform_data_t": rf_data_wf.select(component="T")[0].data,
        "sr": rf_data_wf[0].stats.sampling_rate
        }
            rf_data_dict_arr = np.append(rf_data_dict_arr, rf_data)
            
        except EOFError:
            pass

    rf_data_df = pd.DataFrame.from_records(rf_data_dict_arr)

    rf_data_df = rf_data_df.merge(rf_metadata_df, on="id")

    rf_data_df.to_pickle("{:s}/{:s}/{:s}_RF_Data.pkl".format(wfolder, station, station))
                         
    print ("Gathered RFs for {:s}.{:s} saved to Pickle ....".format(network, station))

def create_time_bins_with_overlap_psd(rf_data_df, bin_size_days, overlap_pc):
    '''
    Function to bin the RF PSDs into overlapping time bins

    Inputs:
    rf_data_df              : pd.DataFrame
    bin_size_days           : int
    overlap_pc              : float

    Outputs:
    time_bins_df            : pd.DataFrame
    '''

    overlap_days = int(overlap_pc*bin_size_days)

    dates = pd.date_range(np.min(rf_data_df["date"]), \
                          np.max(rf_data_df["date"]), freq='D')
    
    time_bins = np.array([], dtype=np.datetime64)
    for d, date in enumerate(dates):
        
        step = (bin_size_days-overlap_days)*d
        if step + bin_size_days < len(dates):
            d1 = dates[step]
            dm = dates[step + int(bin_size_days/2)]
            d2 = dates[step + bin_size_days]
            time_bin =  [d1, dm, d2] 
            # bin_size = (time_bin[-1] - time_bin[0]).astype('timedelta64[D]')
            # if bin_size == bin_size_days:
            time_bins = np.append(time_bins, time_bin, axis=0)
    
    time_bins = time_bins.reshape((-1,3))
    # print (time_bins)
    
    time_bins_df = pd.DataFrame({'start': time_bins[:, 0], \
                                'mid': time_bins[:, 1], 'end': time_bins[:, 2]})
    return time_bins_df

def compute_mt_psd(data, sr):
    '''
    Function to compute RF PSD using multitaper method

    Inputs:
    data (rf time series)   : np.array
    sr (sampling rate)      : float

    Outputs:
    f (frequencies)         : np.array
    Sk (power spectrum)     : np.array
    '''
    NW = 2.5
    k = 3
    # print (data)
    f =  np.fft.fftfreq(len(data), d=1/sr)
    [tapers, eigen] = dpss(len(data), NW, k)
    
    ##  similar to RfPy deconvolution
    # Sk_complex = np.fft.fft(np.multiply(tapers.transpose(), data))
    # Sk = np.sum(np.real(Sk_complex*np.conjugate(Sk_complex)), axis=0)/sr
    # return f, Sk

    ## from https://stackoverflow.com/questions/62836233/ --- 
    ## ----multi-taper-spectral-analysis-with-spectrum-in-python
    Sk_complex, weights, eigenvalues = pmtm(data, e=eigen, v=tapers,NFFT=len(data), show=False)
    Sk = np.real(Sk_complex)**2
    Sk = np.mean(Sk * np.transpose(weights), axis=0)
    return f, Sk
    
def resample_psd_octave(f, Pxx):
    '''
    Function to resample PSDs into Octave. Adapted from following reference.
    *** McNamara, Daniel E., and Raymond P. Buland. 2004. 
    Ambiente Noise Levels in the Continental United States.
    Bulletin of the Seismological Society of America 94 (4): 1517–27.***

    Inputs:
    f (frequencies)         : np.array
    Pxx (power spectrum)    : np.array

    Outputs:
    op_centers (octave period centers)   : np.array
    psd_centers (psd at octave period centers)   : np.array
    '''
    
    ## choose frequencies > 0 as otherwise would result in inf periods
    Pxx = Pxx[f>0]
    f = f[f>0]
    
    ## compute periods and powers sort in ascending order
    T = 1/f[::-1]
    # print (T)
    Pxx = Pxx[::-1]
    
    ## min and max periods in spectrum
    T0 = T[0]
    T1 = T[-1]
    
    ## create increment array with len(T) and increment by 1/8 octave widths
    ## from the previous value
    inc = np.full(len(T), 2**(0.125)).cumprod()

    ## create Ts and Tl arrays and select those less than T1
    Ts = np.append(T0, T0*inc)
    Ts = Ts[Ts<T1]
    
    Tl = 2*Ts
    Tl = Tl[Tl<T1]
    
    ## create arrays of 1/8 octave period ranges and their mid points
    op_ranges = [[i,j] for (i,j) in zip(Ts, Tl)]
    op_centres = [np.sqrt(i*j) for (i,j) in zip(Ts, Tl)]
    
    ## initialise array to store mean power at op_centres
    psd_centres = np.zeros(len(op_centres))
    
    ## store mean power at op_centres
    for op, opr in enumerate(op_ranges):
        fmax = 1/opr[0]
        fmin = 1/opr[1]
        
        ## mask indices in f between fmin and fmax
        f_idx = (f>=fmin)*(f<=fmax)
        
        ## sort f in ascending as it was period sorted in ascending order
        psd_centres[op] = np.median(Pxx[::-1][f_idx])
        
    return op_centres, psd_centres


def compute_psd_in_octave(radial, transverse, sr, time_window):
    '''
    Function to compute RF PSD using multitaper method and resample them in 
    Octave periods

    Inputs:
    radial (rf time series radial)           : np.array
    transverse (rf time series transverse)   : np.array
    sr (sampling rate)                       : float
    time_window (rf time window for psd)     : np.array
    

    Outputs:
    cps_rad (center periods of radial rfs)   : np.array
    Pxx_cp_rad (power spectrum of at cps_rad): np.array
    cps_trans (center periods of trans rfs)  : np.array
    Pxx_cp_trans( power spectrum of at cps_trans): np.array
    '''
    
    if time_window is None:
        f_rad, Pxx_rad = \
            compute_mt_psd(radial, sr)
        f_trans, Pxx_trans = \
            compute_mt_psd(transverse, sr)
    else:
        f_rad, Pxx_rad = compute_mt_psd(radial[int(time_window[0]*sr):int(time_window[1]*sr)], sr)
        f_trans, Pxx_trans = \
                       compute_mt_psd(transverse[int(time_window[0]*sr):int(time_window[1]*sr)], sr)
            
    ## central periods and their powers
    cps_rad, Pxx_cp_rad = resample_psd_octave(f_rad, Pxx_rad)
    cps_trans, Pxx_cp_trans = resample_psd_octave(f_trans, Pxx_trans)
    
    return cps_rad, Pxx_cp_rad, cps_trans, Pxx_cp_trans

def bin_psds(rf_psd_df, bin_size_days, overlap_pc, spower_periods, compute_syn):
    '''
    Function to time bin the power spectrums

    Inputs:
    rf_psd_df (datframe with PSDs)                    : pd.DataFrame
    bin_size_days                                     : int
    overlap_pc                                        : float
    spower_periods (power band to integrate power)    : np.array
    compute_syn (whether synthetic RFs/PSDs computed) : bool
    
    Outputs:
    psd_bins_df (time-binned psd df)                  : pd.DataFrame
    '''
    
    time_bins_df = create_time_bins_with_overlap_psd(rf_psd_df, bin_size_days,overlap_pc)

    psd_bins_df = pd.DataFrame()
    
    for tb, time_bin in time_bins_df.iterrows():
        
        rf_psd_bin = rf_psd_df[rf_psd_df.date.between(pd.Timestamp(time_bin.start),\
                               pd.Timestamp(time_bin.end), inclusive='left')]

        if len(rf_psd_bin) > 0:
            if not compute_syn:
                ##true rf psds
                cp_psd_rad = np.array(np.unique(rf_psd_bin.CP_PSD_rad)[0])
                psd_rad = rf_psd_bin.PSD_rad_norm
                cp_psd_trans = rf_psd_bin.CP_PSD_trans
                psd_trans = rf_psd_bin.PSD_trans_norm
            
                ## period indices for spectral power integration
                pidx = np.argwhere((cp_psd_rad>=spower_periods[0])*(cp_psd_rad<=spower_periods[1]))

                ## computations
                mean_psd_rad = np.sum(psd_rad.values, axis=0)/len(psd_rad.values)
                ## spectral power
                spower_rad = np.sum(mean_psd_rad[pidx])
                mean_psd_trans = np.sum(psd_trans.values, axis=0)/len(psd_trans.values)
                spower_trans = np.sum(mean_psd_trans[pidx])
                
                    
            
                mean_psds_bin = pd.Series({"start": time_bin.start,
                                  "mid": time_bin.mid,
                                  "end": time_bin.end,
                                  "CP_PSD_rad": cp_psd_rad,
                                  "RM_PSD_rad": mean_psd_rad,
                                  "RM_PSD_tr": mean_psd_trans,
                                  "Spower_rad": spower_rad,
                                  "Spower_tr": spower_trans})
            
            
                psd_bins_df = psd_bins_df.append(mean_psds_bin, ignore_index=True)
               
            
            if  compute_syn:
                ##true rf psds
                cp_psd_rad = np.array(np.unique(rf_psd_bin.CP_PSD_rad)[0])
                psd_rad = rf_psd_bin.PSD_rad_norm
                cp_psd_trans = rf_psd_bin.CP_PSD_trans
                psd_trans = rf_psd_bin.PSD_trans_norm
            
                ## period indices for spectral power integration
                pidx = np.argwhere((cp_psd_rad>=spower_periods[0])*(cp_psd_rad<=spower_periods[1]))

                ## computations
                mean_psd_rad = np.sum(psd_rad.values, axis=0)/len(psd_rad.values)
                ## spectral power
                spower_rad = np.sum(mean_psd_rad[pidx])
                mean_psd_trans = np.sum(psd_trans.values, axis=0)/len(psd_trans.values)
                spower_trans = np.sum(mean_psd_trans[pidx])
                
                
                ##syn rf psds
                cp_psd_rad_syn = np.array(np.unique(rf_psd_bin.CP_PSD_rad_syn)[0])
                psd_rad_syn = rf_psd_bin.PSD_rad_norm_syn
                cp_psd_trans_syn = rf_psd_bin.CP_PSD_trans_syn
                psd_trans_syn = rf_psd_bin.PSD_trans_norm_syn
            
                ## period indices for spectral power integration
                pidx_syn = \
                np.argwhere((cp_psd_rad_syn>=spower_periods[0])*(cp_psd_rad_syn<=spower_periods[1]))

                ## computations
                mean_psd_rad_syn = np.sum(psd_rad_syn.values, axis=0)/len(psd_rad_syn.values)
                ## spectral power
                spower_rad_syn = np.sum(mean_psd_rad_syn[pidx_syn])
                mean_psd_trans_syn = np.sum(psd_trans_syn.values, axis=0)/len(psd_trans_syn.values)
                spower_trans_syn = np.sum(mean_psd_trans_syn[pidx_syn])
                    
            
                mean_psds_bin = pd.Series({"start": time_bin.start,
                                  "mid": time_bin.mid,
                                  "end": time_bin.end,
                                  "CP_PSD_rad": cp_psd_rad,
                                  "RM_PSD_rad": mean_psd_rad,
                                  "RM_PSD_tr": mean_psd_trans,
                                  "Spower_rad": spower_rad,
                                  "Spower_tr": spower_trans,
                                  "CP_PSD_rad_syn": cp_psd_rad_syn,
                                  "RM_PSD_rad_syn": mean_psd_rad_syn,
                                  "RM_PSD_tr_syn": mean_psd_trans_syn,
                                  "Spower_rad_syn": spower_rad_syn,
                                  "Spower_tr_syn": spower_trans_syn})        
            
                psd_bins_df = psd_bins_df.append(mean_psds_bin, ignore_index=True)   

    return psd_bins_df

def add_white_noise(trace, snr_db=40):
    '''
    Function to add white noise to synthetic RFs

    Inputs:
    trace (rf data)             : np.array
    snr_db (desired snr in db)  : float
    
    Outputs:
    trace (noise added rf data) : np.array
    '''
    
    ## Calculate the signal power
    signal_power = np.mean(trace.data**2)

    ## Calculate the noise power
    noise_power = signal_power / (10**(snr_db/10))

    ## Calculate the standard deviation of the noise
    noise_std = np.sqrt(noise_power)

    ## Generate random white noise
    noise = np.random.normal(scale=noise_std, size=len(trace.data))

    ## Add noise to the signal
    trace.data = trace.data + noise
    
    return trace
    
def compute_syn_rf(row, use_phase_list, phase_list_fname, Tmin, Tmax):
    '''
    Function to compute synthetic RFs using PyRaySum

    Inputs:
    row (data row from Dataframe)           : pd.Series
    use_phase_list (for fast computations)  : bool
    Tmin (minimum period to filter rfs)     : float
    Tmax (maximum period to filter rfs)     : float
    
    
    Outputs:
    rf_rad.data (radial synthetic data)     : np.array
    rf_tr.data (transverse synthetic data)  : np.array
    '''
    thickness = row["thickness_syn"]
    rho = row["rho_syn"]
    
    ## Vp not perturbed
    vp = row["vp_syn"]
    
    ## to perturb Vp as well
    # vp = row["vp_perturb_syn"]
    
    ## Vs perturbation
    vs = row["vs_perturb_syn"]
    
    ## VpVs perturbation
    # vpvs = row["vpvs_perturb_syn"]
    strike = row["strike_syn"]
    dip = row["dip_syn"]
    
    slow = row["slow"]
    baz = row["baz"]
    snr= row["snr"]

    mults = 2
    if use_phase_list:
        mults = 3
        phase_list = np.loadtxt("phase_descriptors.txt", dtype=str)
    geom = prs.Geometry(baz=baz, slow=slow)

    model = Model(thickness, rho=rho, vp=vp, vs=vs, strike=strike, dip=dip, flag=1, ani=0., \
                  trend=0., plunge=0.)   
    ctrl = prs.Control(verbose=0,
                       rot=1,
                       mults=mults,
                       dt=1/row["sr_syn"],
                       npts=row["npts_syn"],
                       align=1)
    if mults == 3:
        ctrl.set_phaselist(phase_list)
    
    # model.plot()
    result = prs.run(model, geom, ctrl, rf=True, mode='full')
    
    # result.calculate_rfs()
    
    result["rfs"][0].normalize(global_max=False)    

    result["rfs"][0].trim(starttime=result["rfs"][0][0].stats.starttime + 
                  (result["rfs"][0][0].stats.endtime -  result["rfs"][0][0].stats.starttime)/2,
                  endtime=result["rfs"][0][0].stats.starttime + 
                  (result["rfs"][0][0].stats.endtime -  result["rfs"][0][0].stats.starttime)/2 + 60)

    syn_phase_descriptors_rad = result["rfs"][0][0].stats.phase_descriptors
    syn_phase_times_rad =  result["rfs"][0][0].stats.phase_times
    syn_phase_descriptors_tr = result["rfs"][0][1].stats.phase_descriptors
    syn_phase_times_tr = result["rfs"][0][1].stats.phase_times
    
    # result["rfs"][0].normalize(global_max=False)   
    rf_rad = add_white_noise(result["rfs"][0][0], snr) 
    rf_tr = add_white_noise(result["rfs"][0][1], snr)
    
    
    result.filter('rfs', 'bandpass', freqmin=1/Tmax, freqmax=1/Tmin, zerophase=False, corners=1)
    
    rf_rad =result["rfs"][0][0].resample(20)
    rf_tr = result["rfs"][0][1].resample(20)

    return rf_rad.data, rf_tr.data, syn_phase_descriptors_rad, syn_phase_times_rad, syn_phase_descriptors_tr, syn_phase_times_tr

def compute_norm_psd(psd_arr, global_mean, global_std):
    
    '''
    Function to normalize RF PSDs at a station with the global mean
    '''
    # norm_psd = (psd_arr-global_mean)/global_std
    norm_psd = psd_arr/global_mean
    
    # print (psd_arr, norm_psd)
    return [norm_psd]
    
      
def compute_rf_psds(wfolder, station, network,
                    baz_band, epi_band, snr,
                    bin_size_days, overlap_pc,
                    spower_periods, time_window, output_folder, outname,
                    syn_pertb_ts=None,compute_syn=False, nb_workers=2, 
                    use_phase_list = False, phase_list_fname = None,
                    dump_results=False):
    '''
    Main unction to compute RF psds and bin them
    '''
    
    
    rf_data_df = pd.read_pickle(wfolder+"{:s}/{:s}_RF_Data.pkl".format(station, 
                                                                      station))
    rf_data_df['date']= pd.to_datetime(rf_data_df['date']).dt.date
    rf_data_df = rf_data_df[(rf_data_df.baz.between(baz_band[0], baz_band[1], inclusive='both')) & \
                  (rf_data_df.gac.between(epi_band[0], epi_band[1], inclusive='both')) & \
                  (rf_data_df.snr >=snr)]                                                                  

    
    
    ## compute synthetic RFS
    if compute_syn:
        rf_data_df = rf_data_df.merge(syn_pertb_ts, on="date")
        
        print ("\nComputing synthetic RFS...")
        pandarallel.initialize(progress_bar = False, nb_workers = nb_workers, verbose=0)
        rf_data_df[["syn_rf_rad", "syn_rf_trans", "syn_ph_des_rad", \
            "syn_ph_times_rad", "syn_ph_des_tr", "syn_ph_times_tr"]] = \
                rf_data_df.iloc[::].parallel_apply(lambda row: compute_syn_rf(row, use_phase_list, \
                    phase_list_fname), axis=1, result_type='expand')
        print ("\nSynthetic RF computation complete....")

    ## compute observed RF PSDs
    print ("\nComputing observed RF PSDs...")
    pandarallel.initialize(progress_bar = False, nb_workers = nb_workers, verbose=0)
    rf_data_df[["CP_PSD_rad", "PSD_rad", "CP_PSD_trans", "PSD_trans"]] =  \
               rf_data_df.iloc[::].parallel_apply(lambda row: \
               compute_psd_in_octave(row["waveform_data_r"], row["waveform_data_t"], row["sr"], \
               time_window), axis=1, result_type='expand')
    
    print ("\nObserved RF PSD computation complete....")
    
    ## global mean of psds for normalization
    gb_mean_psd_rad = np.array(rf_data_df["PSD_rad"].tolist()).mean()
    gb_mean_psd_tr = np.array(rf_data_df["PSD_trans"].tolist()).mean()
    gb_std_psd_rad = np.array(rf_data_df["PSD_rad"].tolist()).std()
    gb_std_psd_tr = np.array(rf_data_df["PSD_trans"].tolist()).std()
    
    pandarallel.initialize(progress_bar = False, nb_workers = nb_workers, verbose=0)
    rf_data_df["PSD_rad_norm"] = rf_data_df.parallel_apply(lambda row: \
        compute_norm_psd(row["PSD_rad"], gb_mean_psd_rad, gb_std_psd_rad), axis=1, result_type='expand')
    pandarallel.initialize(progress_bar = False, nb_workers = nb_workers, verbose=0)
    rf_data_df["PSD_trans_norm"] = rf_data_df.parallel_apply(lambda row: \
        compute_norm_psd(row["PSD_trans"], gb_mean_psd_tr, gb_std_psd_tr), axis=1, result_type='expand')
                                                              
    if compute_syn:
        print ("\nComputing synthetic RF PSDs...")
        pandarallel.initialize(progress_bar = False, nb_workers = nb_workers, verbose=0)
        rf_data_df[["CP_PSD_rad_syn", "PSD_rad_syn", "CP_PSD_trans_syn", "PSD_trans_syn"]] =  \
               rf_data_df.iloc[::].parallel_apply(lambda row: \
               compute_psd_in_octave(row["syn_rf_rad"], row["syn_rf_trans"], row["sr_syn"], time_window), \
               axis=1, result_type='expand')
        print ("\nSynthetic RF PSD computation complete....")

    ## global mean of psds for normalization
        syn_gb_mean_psd_rad = np.array(rf_data_df["PSD_rad_syn"].tolist()).mean()
        syn_gb_mean_psd_tr = np.array(rf_data_df["PSD_trans_syn"].tolist()).mean()
        syn_gb_std_psd_rad = np.array(rf_data_df["PSD_rad_syn"].tolist()).std()
        syn_gb_std_psd_tr = np.array(rf_data_df["PSD_trans_syn"].tolist()).std()
    
    
        pandarallel.initialize(progress_bar = False, nb_workers = nb_workers, verbose=0)
        rf_data_df["PSD_rad_norm_syn"] = rf_data_df.parallel_apply(lambda row: \
            compute_norm_psd(row["PSD_rad_syn"], syn_gb_mean_psd_rad, syn_gb_std_psd_rad), axis=1, result_type='expand')
        pandarallel.initialize(progress_bar = False, nb_workers = nb_workers, verbose=0)
        rf_data_df["PSD_trans_norm_syn"] = rf_data_df.parallel_apply(lambda row: \
            compute_norm_psd(row["PSD_trans_syn"], syn_gb_mean_psd_tr, syn_gb_std_psd_tr), axis=1, result_type='expand')


    rf_data_df["date"]= pd.to_datetime(rf_data_df["date"])

                        
    print ("Binning PSDs...")
    binned_psds = bin_psds(rf_data_df, bin_size_days, overlap_pc, spower_periods, compute_syn)
    print ("Binning PSDS complete....")

    
    
    if dump_results:
        print ("Dumping files to disk...")
        rf_data_df.to_pickle("{:s}/{:s}_RF.pkl".format(output_folder, outname))

        binned_psds.to_pickle("{:s}/{:s}_PSD.pkl".format(output_folder, outname))
    
        print ("Dumping files to disk complete....")
        
    
    return binned_psds, rf_data_df        
    
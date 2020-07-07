import os
import numpy as np
import essentia.standard as ess
import vamp
import simmusic
from simmusic.dtw import dtw
from joblib import load
from simmusic import constants
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from collections import Counter
import csv 
import re
from tempfile import NamedTemporaryFile

#PARAMETERS
HPCP_SIZE = 120 
FS = 44100 # sampling rate
H = 512  # Hop size
N = 4096 # FFT size
M = 4096 # window soze
dist_function = 'euclidean'

spectrum = ess.Spectrum(size=N)
window = ess.Windowing(size=M, type='hann')
spectralPeaks = ess.SpectralPeaks()


def hz2cents(pitch_hz, tonic=440.0):
    '''
    Converts Pitch values in Hz to cents. Similar function in simmusic.utilities gives error
    Parameters
    ----------
    pitch_hz : float
        Pitch value in Hz
    tonic : float
        base tonic for conversion to cents

    Returns
    -------
    pitch_cents : float
        pitch values in cents
    '''
    return 1200 * np.log2(pitch_hz/ float(tonic))

def hpcp_extract(audio):
    '''
    Extracts HPCP using globally defined constants and functions for the provided audio
    Parameters
    ----------
    audio : numpy.ndarray
        audio data 

    Returns
    -------
    hpcps = numpy.ndarray
        A 2D array of the extracted HPCPs
    '''
    hpcp = ess.HPCP(size=HPCP_SIZE)
    hpcps = []

    for frame in ess.FrameGenerator(audio, frameSize=M, hopSize=H, startFromZero=False):          
        mX = spectrum(window(frame))
        spectralPeaks_freqs, spectralPeaks_mags = spectralPeaks(mX) 
        hpcp_vals = hpcp(spectralPeaks_freqs, spectralPeaks_mags)
        hpcp_vals = np.array(hpcp_vals)
        hpcps.append(hpcp_vals)
    return np.array(hpcps)

def pitch_extract(audio):
    '''
    Extracts pitch using vamp plugin for smoothedpitch track
    Parameters
    ----------
    audio : numpy.ndarray
        audio data 

    Returns
    -------
    pitch = numpy.ndarray
        Pitch values in Hz
    '''
    params = {'outputunvoiced':2, 'lowampsuppression':0.001}
    pitch = vamp.collect(audio, FS, 'pyin:pyin', 'smoothedpitchtrack', parameters=params, step_size=H, block_size=N)['vector'][1]
    pitch[np.where(pitch < 0)] = 0.0
    return pitch

def get_annotation(annotationFile):
    '''
    get onset and offsets of reference audio from the provided annotation file
    '''
    data = list(csv.reader(open(annotationFile)))
    data = np.array(data)
    data = data.T[0]
    data = [re.split('\t',i) for i in data]
    data = [np.float(i[0]) for i in data]
    return data

def get_index_from_time(time):
    '''
    Get array index from time value.
    '''
    index= (time*FS/H).astype(int)
    return index
    
def get_std_time_ticks_dtw(ref_time_ticks, path ):
    '''
    Get student onset offsets from DTW alignment and annotated reference onset-offsets.S
    '''
    std_time_ticks_dtw=[]
    for t in ref_time_ticks:
        ind_ref= np.int(t*FS/H)
        ind_std=[i[1] for i in path if i[0] == ind_ref][-1*(ref_time_ticks.index(t)%2)]
        std_time_tick_dtw = ind_std*H/FS
        std_time_ticks_dtw.append(std_time_tick_dtw)
    return std_time_ticks_dtw

def get_segment(time_ticks):
    
    segment=[]
    for start, end in zip(time_ticks[::2], time_ticks[1::2]):
        segment.append([start,end])
    return np.asarray(segment)

def pitch_histogram(pitch, bin_size, tonic=440, show=False):
    voiced_pitch= pitch[np.where(pitch>constants.PITCH_MIN_VOCAL)]
    pitch_cents = hz2cents(voiced_pitch, tonic=tonic)
    pitch_cents = bin_size * np.round(pitch_cents/bin_size)
    pitch_hist = np.histogram(pitch_cents, bins=np.arange(-2400,2401, bin_size))
    if(show):
        pitch_hist = plt.hist(pitch_cents, bins=np.arange(-2400,2401, bin_size))
    return pitch_hist

def hist_feature_extract(ref_pitch, std_pitch):
    if std_pitch.size and (np.mean(std_pitch)>0):
        # For octave correction, we assume that reference pitch is always non empty numpy array.
        # Std pitch maybe empty array or array with all zeros, so we check for this case before 
        # applying octave correction

        pitch_mean_ratio = np.mean(ref_pitch) / np.mean(std_pitch)
        # Octave correction factor for pitch in Hz is the the value among 0.5, 1 and 2 closest to
        # the ratio of mean_ref_pitch to mean_std_pitch
        octave_factors = np.array([0.5, 1, 2])
        oct_factor = octave_factors[np.argmin(np.abs(octave_factors - pitch_mean_ratio))]
        std_pitch = std_pitch*oct_factor

    bin_sizes = [10,20,50,100]
    features={}
    for bs in bin_sizes:
        ref_hist = pitch_histogram(ref_pitch, bin_size=bs)
        std_hist = pitch_histogram(std_pitch, bin_size=bs)
        if np.all(std_hist[0]==0):   # if no voiced pitch detected, make distance maximum
            hist_cos = 1
        else:
            hist_cos = cosine(ref_hist[0], std_hist[0])
           
        features.update({'pitch_hist_cos_{0}'.format(bs) : hist_cos})       
    return features

def features_extract(ref_annotation_file, path, ref_pitch, std_pitch):
    ref_time_ticks = get_annotation(ref_annotation_file)
    std_time_ticks_dtw = get_std_time_ticks_dtw(ref_time_ticks, path)

    ref_segs = get_segment(ref_time_ticks)
    std_segs = get_segment(std_time_ticks_dtw)

    weights =[]
    features= Counter()
    for ref_seg, std_seg in zip(ref_segs, std_segs):
        ref_seg_ind = get_index_from_time(ref_seg)
        std_seg_ind = get_index_from_time(std_seg)
        
        seg_weight = ref_seg_ind[1]-ref_seg_ind[0]  #longer duration notes given more weightage
        weights.append(seg_weight)
        ref_seg_pitch = ref_pitch[ref_seg_ind[0]:ref_seg_ind[1]]
        std_seg_pitch = std_pitch[std_seg_ind[0]:std_seg_ind[1]]
        seg_features = {k:v*seg_weight for k,v in hist_feature_extract(ref_seg_pitch, std_seg_pitch).items()}
        features = features+Counter(seg_features)
        
    features = {k:v/sum(weights) for k,v in dict(features).items()}
    
    return features

def visualize(ref_audio, std_audio, ref_pitch, std_pitch, ref_time_ticks, std_time_ticks_dtw):
    fs = FS
    ref_segs = get_segment(ref_time_ticks)
    std_segs = get_segment(std_time_ticks_dtw)

    figure = plt.figure()
    ax1 = plt.subplot(2,1,1)
    ax2 = plt.subplot(2,1,2)
    
    for ref_seg, std_seg in zip(ref_segs, std_segs):
        ref_seg_ind = get_index_from_time(ref_seg)
        std_seg_ind = get_index_from_time(std_seg)
   
        ref_seg_pitch = ref_pitch[ref_seg_ind[0]:ref_seg_ind[1]]
        std_seg_pitch = std_pitch[std_seg_ind[0]:std_seg_ind[1]]
        features = hist_feature_extract(ref_seg_pitch, std_seg_pitch)
        
        score = sum([int(key.split('_')[-1])*value for key, value in features.items()])/sum([int(key.split('_')[-1]) for key in features.keys()])
        
        seg_start_sample =(ref_seg*fs).astype(int)[0]
        seg_end_sample =(ref_seg*fs).astype(int)[1]
        
        plt.xlim((0, len(std_audio)/fs))
        ax1.plot(np.linspace(0,len(ref_audio)/44100, len(ref_audio)),ref_audio, color=(1,0,0,0.3))
        ax1.axvspan(seg_start_sample/fs, seg_end_sample/fs, ymin=0, ymax=1, color ='blue', alpha=0.5)
        
        plt.xlim((0, len(std_audio)/fs))
        ax2.plot(np.linspace(0,len(std_audio)/44100, len(std_audio)),std_audio, color=(1,0,0,0.3) )
        x = np.arange(seg_start_sample, seg_end_sample)
        y = ref_audio[seg_start_sample:seg_end_sample]
        seg_start_sample =(std_seg*fs).astype(int)[0]
        seg_end_sample =(std_seg*fs).astype(int)[1]
        ax2.axvspan(seg_start_sample/fs, seg_end_sample/fs, ymin=0, ymax=1, color =(0.1,1-score,0), alpha=0.5)
    
    temp_fig = NamedTemporaryFile(delete=False, suffix='.png')
    figure.savefig(temp_fig.name)
    plt.close(figure)

    with open(temp_fig.name, "rb") as img_file:
        pngbytes = img_file.read()
    os.unlink(temp_fig.name)
    return pngbytes

def get_features_from_student_audio(ref_audio_file, std_audio_file, ref_annotation_file):
    '''
    If a full assessment is not desired, this function can be used to extract features from audio 
    and save them on the disk. 
    '''
    ref_audio = ess.MonoLoader(filename=ref_audio_file)()
    ref_hpcp = hpcp_extract(ref_audio)
    ref_pitch = pitch_extract(ref_audio) 

    std_audio = ess.MonoLoader(filename=std_audio_file)()
    std_hpcp = hpcp_extract(std_audio)
    std_pitch = pitch_extract(std_audio)

    cost, pathlen, mapping, matrix = dtw.dtw_vector(ref_hpcp, std_hpcp)
    path = np.array(mapping).T

    features = features_extract(ref_annotation_file, path, ref_pitch, std_pitch)
    return features

def assess_singing(ref_audio_file, std_audio_file, ref_annotation_file):
    '''
    Assess student performance based on pretrained models.
    '''
    ref_audio = ess.MonoLoader(filename=ref_audio_file)()
    ref_hpcp = hpcp_extract(ref_audio)
    ref_pitch = pitch_extract(ref_audio) 

    std_audio = ess.MonoLoader(filename=std_audio_file)()
    std_hpcp = hpcp_extract(std_audio)
    std_pitch = pitch_extract(std_audio)

    cost, pathlen, mapping, matrix = dtw.dtw_vector(ref_hpcp, std_hpcp)
    path = np.array(mapping).T

    features = features_extract(ref_annotation_file, path, ref_pitch, std_pitch)

    ref_time_ticks = get_annotation(ref_annotation_file)
    std_time_ticks_dtw = get_std_time_ticks_dtw(ref_time_ticks, path)

    pngbytes = visualize(ref_audio, std_audio, ref_pitch, std_pitch, ref_time_ticks, std_time_ticks_dtw)

    #models_dir = os.path.join(simmusic.__path__[0], 'extractors', 'notes_singing_models')
    models_dir = 'notes_singing_models'
    reg_model = load(os.path.join(models_dir,'regression.joblib'))
    KNN_clf = load(os.path.join(models_dir,'KNN.joblib'))
    KNN_pass_fail_clf = load(os.path.join(models_dir,'KNN_pass_fail.joblib'))

    X = [v for k, v in features.items()]
    regression_score = reg_model.predict([X])[0]
    knn_grade = KNN_clf.predict([X])[0]
    result = KNN_pass_fail_clf.predict([X])[0]
    if result == 1:
        result = 'Pass'
    elif result == 0:
        result = 'Fail'
    print('regression score : {0} , KNN_grade : {1} ,  Result : {2} '.format(regression_score, knn_grade, result))

    feedback = {}
    feedback['grade'] = knn_grade 
    feedback['score'] = regression_score 
    feedback['pass_fail']= result
    feedback['png']= pngbytes

    return feedback

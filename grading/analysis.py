import essentia.standard as ess
import librosa
import numpy as np
import matplotlib.pyplot as plt
import itertools
from simmusic.dtw import dtw
import vamp
import utils
import csv, re
from collections import Counter
from scipy.spatial.distance import cosine
import modular_dtw
import distance_metrics

FRAME_SIZE = 4096
WINDOW_SIZE = 4096
HOP_SIZE = 512
SAMPLING_RATE = 44100.0
#PITCH_EXTRACTOR = ess.PitchMelodia(frameSize=FRAME_SIZE, hopSize=HOP_SIZE)
# PITCH_EXTRACTOR = ess.PitchYinProbabilistic(frameSize=FRAME_SIZE, hopSize=HOP_SIZE, outputUnvoiced='zero')
PITCH_EXTRACTOR = 'pyin_vamp'

class Analysis():
    def __init__(self, *args, **kwargs):

        # SET PARAMS
        self.M = FRAME_SIZE # frame size
        self.N = WINDOW_SIZE 
        self.H = HOP_SIZE # hop size
        self.fs = SAMPLING_RATE # sampling rate
        
        # Initialize essentia algorithms with parameters
        self.spectrum = ess.Spectrum(size=self.N)
        self.window = ess.Windowing(size=self.M, type='hann')
        self.spectralPeaks = ess.SpectralPeaks()
        self.pitch_extractor = getattr(self,PITCH_EXTRACTOR)
        self.energy = ess.Energy()
        #hpcp = ess.HPCP(size=hpcp_size, normalized=normalize_method )
        # self.ref_audio_file = 'reference_tracks/510_1_MA_ref.mp3'
        # self.std_audio_file = 'submissions/49_recording-0-2019-01-07T06-17-05-300Z-0.wav'

    def load_audio(self, audio_file):
        return ess.MonoLoader(filename=audio_file)()

    def modulo(self, a, b, m):
        return min((a-b)%m, (b-a)%m)

    def short_time_energy(self, audio):
        short_time_energy=[]
        for frame in ess.FrameGenerator(audio, frameSize=self.M, hopSize=self.H, startFromZero=False):          
            ste = self.energy(self.window(frame))
            short_time_energy.append(ste)
        
        short_time_energy = np.array(short_time_energy, dtype='float32')
        if np.mean(short_time_energy) > 0:
            short_time_energy = short_time_energy/np.mean(short_time_energy)
        return short_time_energy         

    def energy_mask(self, audio, energy_threshold=0.1):
        short_time_energy = self.short_time_energy(audio)
        short_time_energy[np.where(short_time_energy < energy_threshold)]= 0.0
        short_time_energy[np.where(short_time_energy > energy_threshold)]= 1.0
        energy_mask = short_time_energy
        # energy_mask = np.interp(np.linspace(0, len(short_time_energy), num=len(audio)), np.arange(len(short_time_energy)), short_time_energy)
        return np.array(energy_mask, dtype='float32')        

    def normalized_hpcp_extract(self, audio, hpcp_size = 12):
        hpcp = ess.HPCP(size=hpcp_size, normalized='none' )
        hpcps = []
        # weights = [np.exp(-1*self.modulo(0,i, hpcp_size)) for i in range(hpcp_size)]

        for frame in ess.FrameGenerator(audio, frameSize=self.M, hopSize=self.H, startFromZero=True):          
            mX = self.spectrum(self.window(frame))
            spectralPeaks_freqs, spectralPeaks_mags = self.spectralPeaks(mX) 
            hpcp_vals = hpcp(spectralPeaks_freqs, spectralPeaks_mags)
            hpcp_vals = np.array(hpcp_vals)
            if np.linalg.norm(hpcp_vals) !=0:
                hpcp_vals = hpcp_vals/np.linalg.norm(hpcp_vals)

            # spill_hpcp_vals = np.zeros(hpcp_size)
            # for i, v in enumerate(hpcp_vals):
            #     spill_hpcp_vals += v*np.roll(weights, i)
            # hpcps.append(spill_hpcp_vals)
            hpcps.append(hpcp_vals)
        return np.array(hpcps, dtype='float32')
        # return super().__init__(*args, **kwargs)

    def hpcp_extract(self, audio, normalize_method ='none', hpcp_size = 120):
        hpcp = ess.HPCP(size=hpcp_size, normalized=normalize_method )
        hpcps = []
        # weights = [np.power(-1*self.modulo(0,i, hpcp_size),2) for i in range(hpcp_size)]

        for frame in ess.FrameGenerator(audio, frameSize=self.M, hopSize=self.H, startFromZero=True):          
            mX = self.spectrum(self.window(frame))
            spectralPeaks_freqs, spectralPeaks_mags = self.spectralPeaks(mX) 
            hpcp_vals = hpcp(spectralPeaks_freqs, spectralPeaks_mags)
            hpcp_vals = np.array(hpcp_vals)
            #print("HPCP before : ", hpcp_vals, hpcp_size)
            # weights = [np.exp(-1*self.modulo(0,i, hpcp_size)) for i in range(hpcp_size)]
            # spill_hpcp_vals = np.zeros(hpcp_size)
            # for i, v in enumerate(hpcp_vals):
            #     spill_hpcp_vals += v*np.roll(weights, i)
            
            #print("HPCP after : ", spill_hpcp_vals, hpcp_size)

            hpcps.append(hpcp_vals)
            # hpcps.append(spill_hpcp_vals)
        return np.array(hpcps)
        # return super().__init__(*args, **kwargs)
    

    def nnls_chroma_extract(self, audio):
        chroma= vamp.collect(audio, SAMPLING_RATE, 'nnls-chroma:nnls-chroma', 'chroma', step_size=HOP_SIZE, block_size=FRAME_SIZE)['matrix'][1]
        # chroma = librosa.feature.chroma_stft(y=audio, sr=self.fs, n_fft=self.N, hop_length=self.H)
        return np.array(chroma, dtype='float32')

    def mfcc_extract(self, audio):
        mfcc = ess.MFCC(numberCoefficients = 12)
        mfccs = []

        for frame in ess.FrameGenerator(audio, frameSize=self.M, hopSize=self.H, startFromZero=True):          
            mX = self.spectrum(self.window(frame))
            mfcc_bands, mfcc_coeffs = mfcc(mX)
            mfccs.append(mfcc_coeffs)            
        return np.array(mfccs)

    def pitch_extractor(self, audio):
        pitch, confidence= self.pitch_extractor(audio)
        return pitch, confidence

    def pyin_vamp(self, audio):
        params = {'outputunvoiced':2, 'lowampsuppression':0.001}
        
        pitch = vamp.collect(audio, SAMPLING_RATE, 'pyin:pyin', 'smoothedpitchtrack', parameters=params, step_size=HOP_SIZE, block_size=FRAME_SIZE)['vector'][1]
        pitch[np.where(pitch < 0)] = 0.0
        
        confidence = vamp.collect(audio, SAMPLING_RATE, 'pyin:pyin', 'voicedprob', parameters=params, step_size=HOP_SIZE, block_size=FRAME_SIZE)['vector'][1]

        return pitch, confidence

    def pyin_essentia(self, audio):
        pitch, confidence = ess.PitchYinProbabilistic(frameSize=FRAME_SIZE, hopSize=HOP_SIZE, outputUnvoiced='zero')(audio)

        return pitch, confidence

    def audio_align(self, series_ref, series_std, dist_function=distance_metrics.euclidean):
        matrix = np.array(modular_dtw.cost_matrix(series_ref, series_std, dist_function))
        path = modular_dtw.path(matrix)
        cost = np.array(matrix[-1][-1])
        # if len(series_ref.shape) == 1:
        #     cost, pathlen, mapping, matrix = dtw.dtw(series_ref, series_std)
        # if len(series_ref.shape) == 2:
        #     cost, pathlen, mapping, matrix = dtw.dtw_vector(series_ref, series_std)
        # path = np.array(mapping).T
        return cost, path, matrix

    def get_annotation(self, annotationFile):
        data = list(csv.reader(open(annotationFile)))
        data = np.array(data)
        data = data.T[0]
        data = [re.split('\t',i) for i in data]
        #print(data)
        data = [np.float(i[0]) for i in data]
        return data

    def get_index_from_time(self, time):
        index= (time*self.fs/self.H).astype(int)
        #ind_std=[i[1] for i in path if i[0] == ind_ref][0]
        return index
        
    def get_std_time_ticks_dtw(self, ref_time_ticks, path ):
        std_time_ticks_dtw=[]
        for t in ref_time_ticks:
            ind_ref= np.int(t*self.fs/self.H)
            #-1*(ref_time_ticks.index(t)%2) this selects first matching frame at beginning of
            # segment and last matching frame at end of segment
            ind_std=[i[1] for i in path if i[0] == ind_ref][-1*(ref_time_ticks.index(t)%2)]
            std_time_tick_dtw = ind_std*self.H/self.fs
            std_time_ticks_dtw.append(std_time_tick_dtw)
        return std_time_ticks_dtw
    
    def get_segment(self, time_ticks):
        segment=[]
        for start, end in zip(time_ticks[::2], time_ticks[1::2]):
            segment.append([start,end])
        return np.asarray(segment)


    def pitch_histogram(self, pitch, bin_size, tonic=440, show=False):
        voiced_pitch= pitch[np.where(pitch>60)]
        #print(len(voiced_pitch))
        pitch_cents = utils.hz2cents(voiced_pitch, tonic=tonic)
        pitch_cents = bin_size * np.round(pitch_cents/bin_size)
        pitch_hist = np.histogram(pitch_cents, bins=np.arange(-2400,2401, bin_size))
        if(show):
            pitch_hist = plt.hist(pitch_cents, bins=np.arange(-2400,2401, bin_size))
            #plt.show()
        return pitch_hist

    def hist_feature_extract(self, ref_pitch, std_pitch):
        pitch_mean_diff = np.mean(ref_pitch) / np.mean(std_pitch)
        octave_factors = np.array([0.5, 1, 2])
        oct_factor = octave_factors[np.argmin(np.abs(octave_factors - pitch_mean_diff))]
        print(oct_factor)

        # print(utils.hz2cents(np.mean(ref_pitch), np.mean(std_pitch)))
        bin_sizes = [10,20,50,100]
        features={}
        for bs in bin_sizes:
            # print(bs)
            ### TODO Try out octave correction
            # std_hist_0 = std_hist = self.pitch_histogram(std_pitch, bin_size=bs)
            # std_hist_plus_1 = std_hist = self.pitch_histogram(std_pitch+1200, bin_size=bs)
            # std_hist_minus_1 = std_hist = self.pitch_histogram(std_pitch-1200, bin_size=bs)
            
            
            ref_hist = self.pitch_histogram(ref_pitch, bin_size=bs)
            std_hist = self.pitch_histogram(std_pitch, bin_size=bs)
            if np.all(std_hist[0]==0):   # if no voiced pitch detected, make distance maximum
                hist_cos = 1
            else:
                hist_cos = cosine(ref_hist[0], std_hist[0])
                ##
                # hist_cos_0= cosine(ref_hist[0], std_hist_0[0])
                # hist_cos_plus_1= cosine(ref_hist[0], std_hist_plus_1[0])
                # hist_cos_minus_1= cosine(ref_hist[0], std_hist_minus_1[0])
                # hist_cos = min(hist_cos_0, hist_cos_plus_1, hist_cos_minus_1)
            features.update({'pitch_hist_cos_{0}'.format(bs) : hist_cos})         

        # print(features)
        return features

    def features_extract(self, segmentAnnotationFile, dtw_path, ref_pitch, std_pitch):
        ref_time_ticks = self.get_annotation(segmentAnnotationFile)
        std_time_ticks_dtw = self.get_std_time_ticks_dtw(ref_time_ticks, dtw_path)

        ref_segs = self.get_segment(ref_time_ticks)
        std_segs = self.get_segment(std_time_ticks_dtw)

        weights =[]
        features= Counter()
        for ref_seg, std_seg in zip(ref_segs, std_segs):
            #print(seg)
            ref_seg_ind = self.get_index_from_time(ref_seg)
            std_seg_ind = self.get_index_from_time(std_seg)
            # print(ref_seg)
            # print(ref_seg_ind)
            # print(ref_seg_ind[1]-ref_seg_ind[0])
            seg_weight = ref_seg_ind[1]-ref_seg_ind[0]
            weights.append(seg_weight)
            ref_seg_pitch = ref_pitch[ref_seg_ind[0]:ref_seg_ind[1]]
            std_seg_pitch = std_pitch[std_seg_ind[0]:std_seg_ind[1]]
            # print(len(ref_seg_pitch), len(std_seg_pitch))
            # print(np.mean(ref_seg_pitch), np.mean(std_seg_pitch), '\n')
            seg_features = {k:v*seg_weight for k,v in self.hist_feature_extract(ref_seg_pitch, std_seg_pitch).items()}
            features = features+Counter(seg_features)
            # print(seg_features)
            # print(features)
            # plt.plot(utils.hz2cents(ref_seg_pitch))
            # plt.plot(utils.hz2cents(std_seg_pitch))
            # plt.show()
        # print(dict(features))
        features = {k:v/sum(weights) for k,v in dict(features).items()}
        #print(features)
        return features
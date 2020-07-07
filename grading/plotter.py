import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from analysis import HOP_SIZE, SAMPLING_RATE
import os
from pathlib import Path


PLOTS_SAVE_PATH = 'plotsdump'
os.makedirs(PLOTS_SAVE_PATH, exist_ok=True)


def plot_hpcp(hpcp_vector, hop_size= HOP_SIZE, fs = SAMPLING_RATE, save_file_name=None):
    if not save_file_name:
        save_file_name = 'HPCP'
    else:
        save_file_name += '_HPCP'
    hpcp_size = int(hpcp_vector[0,:].size)
    numFrames = int(hpcp_vector[:,0].size)
    #print(numFrames)
    #print(hpcp_size, numFrames)
    frmTime = hop_size*np.arange(numFrames)/float(fs) 
    fig = plt.figure()   
    plt.pcolormesh(frmTime, np.arange(hpcp_size), np.transpose(hpcp_vector))
    plt.ylabel('spectral bins')
    plt.title('HPCP')
    #plt.show()
    plt.savefig(os.path.join(PLOTS_SAVE_PATH,save_file_name))
    plt.close(fig)
    return

def plot_pitch_contour(pitch_contour, hop_size= HOP_SIZE, fs = SAMPLING_RATE, save_file_name=None):
    if not save_file_name:
        save_file_name = 'pitch'
    else:
        save_file_name += '_pitch'
    #hpcp_size = int(hpcp_vector[0,:].size)
    numFrames = int(pitch_contour.size)
    #print(numFrames)
    #print(numFrames)
    frmTime = hop_size*np.arange(numFrames)/float(fs) 
    fig = plt.figure()   
    #plt.pcolormesh(frmTime, np.arange(hpcp_size), np.transpose(hpcp_vector))
    plt.plot(frmTime, pitch_contour)
    plt.ylabel('pitch')
    plt.xlabel('time')
    plt.title('Pitch')
    #plt.show()
    plt.savefig(os.path.join(PLOTS_SAVE_PATH,save_file_name))
    plt.close(fig)
    return

def plot_dtw_alignment(ref_audio, std_audio, path, hop_size=HOP_SIZE, fs = SAMPLING_RATE, save_file_name=None):
    if not save_file_name:
        save_file_name = 'align'
    else:
        save_file_name += '_align'
    
    #fig = plt.figure(figsize=(16, 8))
    fig = plt.figure()
    #fig.suptitle(std_filename)
    # Plot ref_audio
    plt.subplot(2, 1, 1)
    plt.plot(np.linspace(0,len(ref_audio)/fs, len(ref_audio)), ref_audio)
    ax1 = plt.gca()

    # Plot std_audio
    plt.subplot(2, 1, 2)
    plt.plot(np.linspace(0,len(std_audio)/fs, len(std_audio)), std_audio)
    ax2 = plt.gca()

    plt.tight_layout()

    trans_figure = fig.transFigure.inverted()
    lines = []
    arrows = 100
    points_idx = np.int16(np.round(np.linspace(0, path.shape[0] - 1, arrows)))

    # for tp1, tp2 in zip((wp[points_idx, 0]) * hop_size, (wp[points_idx, 1]) * hop_size):
    for tp1, tp2 in path[points_idx] * hop_size / fs:
        # get position on axis for a given index-pair
        coord1 = trans_figure.transform(ax1.transData.transform([tp1, 0]))
        coord2 = trans_figure.transform(ax2.transData.transform([tp2, 0]))

        # draw a line
        line = matplotlib.lines.Line2D((coord1[0], coord2[0]),
                                       (coord1[1], coord2[1]),
                                       transform=fig.transFigure,
                                       color='r')
        lines.append(line)

    fig.lines = lines
    plt.savefig('plotsdump/'+save_file_name)
    plt.close(fig)
    return

def plot_dtw_alignment_segments(ref_audio, std_audio, ref_time_ticks, std_time_ticks_dtw, hop_size=HOP_SIZE, fs = SAMPLING_RATE, save_file_name=None):
    if not save_file_name:
        save_file_name = 'align_segments'
    else:
        save_file_name += '_align_segments'
    
    #fig = plt.figure(figsize=(16, 8))
    fig = plt.figure()
    #fig.suptitle(std_filename)
    # Plot ref_audio
    plt.subplot(2, 1, 1)
    plt.plot(np.linspace(0,len(ref_audio)/fs, len(ref_audio)), ref_audio)
    ax1 = plt.gca()

    # Plot std_audio
    plt.subplot(2, 1, 2)
    plt.plot(np.linspace(0,len(std_audio)/fs, len(std_audio)), std_audio)
    ax2 = plt.gca()

    plt.tight_layout()

    trans_figure = fig.transFigure.inverted()
    lines = []
    # arrows = 100
    # points_idx = np.int16(np.round(np.linspace(0, path.shape[0] - 1, arrows)))

    for i in range(len(ref_time_ticks)) :
        # get position on axis for a given index-pair
        tp1=ref_time_ticks[i]
        tp2=std_time_ticks_dtw[i]
        coord1 = trans_figure.transform(ax1.transData.transform([tp1, 0]))
        coord2 = trans_figure.transform(ax2.transData.transform([tp2, 0]))

        # draw a line
        line1 = matplotlib.lines.Line2D((coord1[0], coord2[0]),
                                       (coord1[1], coord2[1]),
                                       transform=fig.transFigure,
                                       color='k')
        lines.append(line1)  

    # for tp1, tp2 in zip((wp[points_idx, 0]) * hop_size, (wp[points_idx, 1]) * hop_size):
    # for tp1, tp2 in path[points_idx] * hop_size / fs:
    #     # get position on axis for a given index-pair
    #     coord1 = trans_figure.transform(ax1.transData.transform([tp1, 0]))
    #     coord2 = trans_figure.transform(ax2.transData.transform([tp2, 0]))

    #     # draw a line
    #     line = matplotlib.lines.Line2D((coord1[0], coord2[0]),
    #                                    (coord1[1], coord2[1]),
    #                                    transform=fig.transFigure,
    #                                    color='r')
    #     lines.append(line)

    fig.lines = lines
    plt.savefig('plotsdump/'+save_file_name)
    plt.close(fig)
    return

def performance_visualize(analysis,ref_audio,std_audio,ref_pitch, std_pitch, 
                            ref_time_ticks, std_time_ticks_dtw):
    fs = analysis.fs
    ref_segs = analysis.get_segment(ref_time_ticks)
    std_segs = analysis.get_segment(std_time_ticks_dtw)

    figure = plt.figure()
    ax1 = plt.subplot(2,1,1)
    ax2 = plt.subplot(2,1,2)
    #ax = figure.add_subplot(111)
    
    for ref_seg, std_seg in zip(ref_segs, std_segs):
        ref_seg_ind = analysis.get_index_from_time(ref_seg)
        std_seg_ind = analysis.get_index_from_time(std_seg)
   
        ref_seg_pitch = ref_pitch[ref_seg_ind[0]:ref_seg_ind[1]]
        std_seg_pitch = std_pitch[std_seg_ind[0]:std_seg_ind[1]]
        features = analysis.hist_feature_extract(ref_seg_pitch, std_seg_pitch)
        # print(ref_seg)
        # print(features)
        #score = features['pitch_hist_cos_100']
        score = sum([int(key.split('_')[-1])*value for key, value in features.items()])/(190.0)
        #print("Segment Scores : %s  " % list(features.values()))
        seg_start_sample =(ref_seg*fs).astype(int)[0]
        seg_end_sample =(ref_seg*fs).astype(int)[1]
        # plt.subplot(2,1,1)
        plt.xlim((0, len(std_audio)/fs))
        ax1.plot(np.linspace(0,len(ref_audio)/44100, len(ref_audio)),ref_audio, color=(1,0,0,0.3))
        ax1.axvspan(seg_start_sample/fs, seg_end_sample/fs, ymin=0, ymax=1, color ='blue', alpha=0.5)
        #print(seg_start_sample, seg_end_sample)
        # plt.subplot(2,1,2)
        plt.xlim((0, len(std_audio)/fs))
        ax2.plot(np.linspace(0,len(std_audio)/44100, len(std_audio)),std_audio, color=(1,0,0,0.3) )
        x = np.arange(seg_start_sample, seg_end_sample)
        y = ref_audio[seg_start_sample:seg_end_sample]
        seg_start_sample =(std_seg*fs).astype(int)[0]
        seg_end_sample =(std_seg*fs).astype(int)[1]
        ax2.axvspan(seg_start_sample/fs, seg_end_sample/fs, ymin=0, ymax=1, color =(0.1,1-score,0), alpha=0.5)

        #plt.plot(x, y, color =(0.1,1-score,0))
    return figure
    #plt.show()

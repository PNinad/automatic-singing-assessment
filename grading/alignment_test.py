from glob import glob
import pandas as pd
import settings
import mcclient
import operator
import utils
import os, csv
from analysis import Analysis
import matplotlib.pyplot as plt
import plotter
from pathlib import Path
import numpy as np
import gc
import fastdtw
from scipy.spatial.distance import cosine, euclidean, cityblock
import distance_metrics
import modular_dtw

def alignment_eval(sound_id, std_time_ticks, std_time_ticks_dtw):
    segments_ground_truth= analysis.get_segment(std_time_ticks)
    segments_algorithm=analysis.get_segment(std_time_ticks_dtw)
    if(len(segments_algorithm) != len(segments_ground_truth)):
        print(sound_id, len(segments_algorithm), len(segments_ground_truth))

    intersection = [[max(a[0], b[0]), min(a[1], b[1])] for a, b in zip (segments_algorithm, segments_ground_truth)]

    precision = sum([((i[1]-i[0])/ (alg[1]- alg[0])) if (alg[1] > alg[0] and i[1] > i[0]) else 0.0 for i, alg in zip(intersection, segments_algorithm) ])/len(intersection)

    recall = sum([((i[1]-i[0])/ (gt[1]- gt[0])) if (gt[1] > gt[0] and i[1] > i[0]) else 0.0 for i, gt in zip(intersection, segments_ground_truth) ])/len(intersection)
    
    if precision==0 and recall==0:
        f_measure = 0.0
    else:
        f_measure = 2*precision*recall/ (precision+recall)
    # duration=sum(segments_ground_truth[:,1] - segments_ground_truth[:,0])
    # eval_duration=sum(segments_algorithm[:,1] - segments_algorithm[:,0])
    # abs_error= np.abs(np.array(std_time_ticks_dtw) - np.array(std_time_ticks))
    

    results={
        'sound_id':sound_id,
        # 'annotated_time_ticks':np.round(std_time_ticks, decimals=3),
        # 'estimated_time_ticks':np.round(std_time_ticks_dtw, decimals=3),
        # 'ann_voiced_duration':np.round(duration, decimals=3),
        # 'est_voiced_duration':np.round(eval_duration, decimals=3),
        # 'total_error':cityblock(std_time_ticks_dtw, std_time_ticks),
        'precision':precision,
        'recall':recall,
        'f-measure':f_measure
        # 'max_error':np.round(np.max(abs_error), decimals=3),
        # '1st_quartile_error':np.round(np.percentile(abs_error,25,interpolation='nearest'), decimals=3),
        # 'median_error':np.round(np.median(abs_error),decimals=3),
        # '3rd_quartile_error':np.round(np.percentile(abs_error,75,interpolation='nearest'), decimals=3),
        # 'avg_error':np.round(np.average(abs_error),decimals=3),
        # 'ME_segment_detection':mir_eval.segment.detection(segments_ground_truth, segments_algorithm),
        # 'ME_segment_deviation':mir_eval.segment.deviation(segments_ground_truth, segments_algorithm)
    }
    
    return results

HPCP_SIZE = 120               # parameters for HPCP
HPCP_NORMALIZE = 'unitMax'
ENERGY_MASK_PERCENT = 5
distance_metrics.params['alpha'] = 0.0
distance_metrics.params['hpcp_size'] = HPCP_SIZE
DIST_FUNCTION = 'euclidean'         # euclidean, cosine
series = 'nnls_chroma'   # hpcp, mfcc, pitch, energy, energy_mask, multi_hpcp_energy_mask, nnls_chroma, multi_hpcp_mfcc
# series = 'multi_hpcp_mfcc'

print(series)

analysis = Analysis()
context_id = settings.CONTEXT_ID
exercises = mcclient.get_full_context(context_id)['exercises']
sub_ref_map={}
for ex in exercises:
    for sub in ex['submissions']:
        for s in sub['sounds']:
            sub_ref_map[os.path.basename(s['download_url'])] = {'exercise_name': ex['name'], 'version': s['version']}
# print(sub_ref_map)

annotated_files = glob('submissions/*.csv')
# to test single file comment above line and uncomment below
# annotated_files = ['submissions/783_std_annotation.csv']
audio_files = glob('submissions/*.wav')

def get_dist_function(DIST_FUNCTION):
    if DIST_FUNCTION == 'euclidean':
        dist_function = distance_metrics.euclidean
    elif DIST_FUNCTION == 'cosine':
        dist_function = distance_metrics.cosine
    return dist_function

alignment_results = []
print(len(annotated_files))
for f in annotated_files:
    sound_id = os.path.basename(f).split('_')[0]
    # print(sound_id)
    for af in audio_files:
        # print(af)
        if sound_id in os.path.basename(af)[:4]:
            audio_file = os.path.basename(af)
            # print(sub_ref_map[audio_file])
            break
    # print(audio_file)
    try:
        exercise = utils.get_exercise_by_name(sub_ref_map[audio_file]['exercise_name'], exercises)
        version = sub_ref_map[audio_file]['version']
        ref = utils.get_reference_track(exercise, version)
        ref = utils.file_download(ref['download_url'], filetype='reference_tracks')
        # print(ref)
        std = os.path.join('submissions', audio_file)


        ref_audio = analysis.load_audio(ref)
        std_audio = analysis.load_audio(std)

        
        if series == 'hpcp':
            params = {'normalize_method':HPCP_NORMALIZE,'hpcp_size':HPCP_SIZE, 'dist_function':DIST_FUNCTION}
            ref_series = analysis.hpcp_extract(ref_audio,params['normalize_method'],params['hpcp_size'])
            std_series = analysis.hpcp_extract(std_audio,params['normalize_method'],params['hpcp_size'])
        
        if series =='pitch':
            params = {'dist_function':DIST_FUNCTION}
            ref_series = analysis.pitch_extractor(ref_audio)[0]
            std_series = analysis.pitch_extractor(std_audio)[0]

        if series =='energy':
            params = {'dist_function':DIST_FUNCTION}
            ref_series = analysis.short_time_energy(ref_audio)
            std_series = analysis.short_time_energy(std_audio)
        
        if series =='energy_mask':
            params = {'dist_function':DIST_FUNCTION}
            ref_series = analysis.energy_mask(ref_audio)
            std_series = analysis.energy_mask(std_audio)
        
        if series =='mfcc':
            params = {'dist_function':DIST_FUNCTION}
            ref_series = analysis.mfcc_extract(ref_audio)
            std_series = analysis.mfcc_extract(std_audio)

        if series == 'nnls_chroma':
            params = {'dist_function':DIST_FUNCTION}
            ref_series = analysis.nnls_chroma_extract(ref_audio)
            std_series = analysis.nnls_chroma_extract(std_audio)
        
        if series == 'multi_hpcp_mfcc':
            params = {'normalize_method':HPCP_NORMALIZE,'hpcp_size':HPCP_SIZE, 'dist_function':DIST_FUNCTION,'mfcc_percent':5}
            dist_function = get_dist_function(DIST_FUNCTION)
            ref_series = analysis.hpcp_extract(ref_audio,params['normalize_method'],params['hpcp_size'])
            std_series = analysis.hpcp_extract(std_audio,params['normalize_method'],params['hpcp_size'])

            # ref_series = analysis.normalized_hpcp_extract(ref_audio,hpcp_size=params['hpcp_size'])
            # std_series = analysis.normalized_hpcp_extract(std_audio,hpcp_size=params['hpcp_size'])
            # ref_len = len(ref_series)
            # std_len = len(std_series)
            cost_1, path_1, matrix_1 = analysis.audio_align(ref_series, std_series, dist_function)

            # params = {'dist_function':DIST_FUNCTION}
            # dist_function = get_dist_function('euclidean')
            ref_series = analysis.mfcc_extract(ref_audio)
            std_series = analysis.mfcc_extract(std_audio)
            cost_2, path_2, matrix_2 = analysis.audio_align(ref_series, std_series, dist_function)

            mfcc_ratio = params['mfcc_percent']/100.0
            matrix = matrix_1*(1 - mfcc_ratio) + matrix_2*mfcc_ratio
            path = modular_dtw.path(matrix)

        if series == 'multi_hpcp_energy_mask':
            params = {'normalize_method':HPCP_NORMALIZE,'hpcp_size':HPCP_SIZE, 'dist_function':DIST_FUNCTION,'energy_mask_percent':ENERGY_MASK_PERCENT}
            dist_function = get_dist_function(DIST_FUNCTION)
            ref_series = analysis.hpcp_extract(ref_audio,params['normalize_method'],params['hpcp_size'])
            std_series = analysis.hpcp_extract(std_audio,params['normalize_method'],params['hpcp_size'])

            # ref_series = analysis.normalized_hpcp_extract(ref_audio,hpcp_size=params['hpcp_size'])
            # std_series = analysis.normalized_hpcp_extract(std_audio,hpcp_size=params['hpcp_size'])
            # ref_len = len(ref_series)
            # std_len = len(std_series)
            cost_1, path_1, matrix_1 = analysis.audio_align(ref_series, std_series, dist_function)

            # params = {'dist_function':DIST_FUNCTION}
            dist_function = get_dist_function('euclidean')
            ref_series = analysis.energy_mask(ref_audio)
            std_series = analysis.energy_mask(std_audio)
            cost_2, path_2, matrix_2 = analysis.audio_align(ref_series, std_series, dist_function)

            energy_mask_ratio = params['energy_mask_percent']/100.0
            matrix = matrix_1*(1 - energy_mask_ratio) + matrix_2*energy_mask_ratio
            path = modular_dtw.path(matrix)


        ## DTW
        if 'multi' not in series:
            dist_function = get_dist_function(DIST_FUNCTION)
            cost, path, matrix = analysis.audio_align(ref_series, std_series, dist_function)
        ## DTW
        # cost, path, matrix = analysis.audio_align(ref_series, std_series, dist_function)

        refAnnotationFile = os.path.join('reference_tracks',exercise['name']+'_'+version+'_ref_segments.csv')
        ref_time_ticks = analysis.get_annotation(refAnnotationFile)
        stdAnnotationFile = f
        std_time_ticks = analysis.get_annotation(stdAnnotationFile)
        std_time_ticks_dtw = analysis.get_std_time_ticks_dtw(ref_time_ticks, path)

        # print(std_time_ticks)
        # print(std_time_ticks_dtw)

        results = alignment_eval(sound_id, std_time_ticks, std_time_ticks_dtw)
        print("Sound_id {0} Results precision :{1} recall: {2} f-measure: {3}".format(sound_id, results['precision'], results['recall'], results['f-measure']))
        alignment_results.append(results)

    except:
        print(sound_id)

result_save_filename = '{0}_'.format(series)+'_'.join(['{0}'.format(v) for _,v in params.items()])+'.csv'
keys = alignment_results[0].keys()
with open(result_save_filename, 'w') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(alignment_results)
    # print('{0}_'.format(series)+'_'.join(['{0}'.format(v) for _,v in params.items()]))
    #print(series+v for v in params.values())


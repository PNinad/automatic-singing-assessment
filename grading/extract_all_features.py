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
from scipy.spatial.distance import cosine, euclidean

#######

# Parameters for audio alignement , feature extraction
HPCP_SIZE = 120                # parameters for HPCP
HPCP_NORMALIZE = 'unitMax'
REF_ENERGY_THRESH = 0.00 # threshold for masking energy
STD_ENERGY_THRESH = 0.00
MOV_AVG_SIZE = 5

analysis = Analysis()               #initialize analysis object
image_save_path ='performance_feedback'
features_save_file = 'features_save_file_test.csv' #save_file for extracted features
report_file='Ninad_Hindustani_training_grading_report.csv'
#grade_df = pd.read_csv(report_file, names=['sound_id', 'grade'])

def mod_dist(x, y, dist_function=euclidean, max_shift=int(HPCP_SIZE/12)):
    dist=[(np.exp(analysis.modulo(0,i,HPCP_SIZE)/HPCP_SIZE)*dist_function(np.roll(x,i),y)) for i in range(-1*max_shift, max_shift+1)]
    # if np.argmin(dist) != max_shift:
    #     print(np.argmin(dist)-max_shift)
    return min(dist)

def dist_cosine(x, y):
    if (np.linalg.norm(x)==0 and np.linalg.norm(y) == 0):
        dist = 0        
    elif (np.linalg.norm(x)==0 and np.linalg.norm(y) != 0):
        dist = 1
    elif (np.linalg.norm(x)!=0 and np.linalg.norm(y)==0):
        dist = 1
    else:
        dist= cosine(x, y)

    
    return dist

def dist_cos_W(x, y, alpha=0.8):
    # if(alpha <= 0 or alpha > 1):
    #     return dist_cosine(x,y)
    mod_x = np.linalg.norm(x) 
    mod_y = np.linalg.norm(y)
    max_shift = int(HPCP_SIZE/6)
      
    if (mod_x==0 and mod_y == 0):
        dist = 0        
    elif (mod_x==0 and mod_y != 0):
        dist = 1
    elif (mod_x!=0 and mod_y==0):
        dist = 1
    else:
        # 1-(i/6)**1.7 
        #num =[np.cos((np.pi/2)*(analysis.modulo(0,i,HPCP_SIZE)/(HPCP_SIZE/2)))*np.dot(np.roll(x,i),y) for i in range(-1*max_shift, max_shift+1)]
        num =[(alpha**analysis.modulo(0,i,HPCP_SIZE))*np.dot(np.roll(x,i),y) for i in range(-1*max_shift, max_shift+1)]
        #num= np.dot(x,y)
        num = max(num)
        dist = 1 - num/(mod_x*mod_y)    
    return dist

def dist_combination(x, y, weights=[0.8,0.2], functions=[euclidean, dist_cos_W]):
    '''
    function should return value between 0 an 1
    '''
    dist = np.zeros(2)
    ref_hpcp = x[:-1]
    std_hpcp = y[:-1]
    dist[0] = dist_cos_W(ref_hpcp, std_hpcp)

    ref_energy = x[-1]
    std_energy = y[-1]
    dist[1] = euclidean(ref_energy, std_energy)
    # weights = weights/sum(weights)
    dist = sum([w*d for w, d in zip(weights, dist)])
    return dist

def pitch_hist_cos_dist(x,y):
    if x ==0 and y == 0:
        dist = 0        
    elif x==0 and y != 0:
        dist = 1
    elif x!=0 and y==0:
        dist = 1
    else:
        dist= utils.hz2cents(np.abs(x-y),x)/1200 
    return dist



def extract_submission_features(exercise, submission):
    version = submission['version']
    segmentAnnotationFile = os.path.join('reference_tracks',exercise['name']+'_'+version+'_ref_segments.csv')
    #print(segmentAnnotationFile)

    reference_track = utils.get_reference_track(exercise, version)
    ref_audio_file = utils.file_download(download_url = reference_track['download_url'], filetype='reference_tracks')
    # print(ref_audio_file)

    ref_audio = analysis.load_audio(audio_file=ref_audio_file)
    
    # thresholding
    # ref_energy_mask = analysis.energy_mask(ref_audio, REF_ENERGY_THRESH)
    # # ref_audio = ref_audio * ref_energy_mask
    # ref_energy = analysis.short_time_energy(ref_audio)
    # ref_energy = ref_energy/np.max(ref_energy)
    # mean_ref_energy = np.mean(ref_energy)
    # ref_energy[np.where(ref_energy > 0.01*mean_ref_energy)] = 1
    # ref_energy[np.where(ref_energy <= 0.01*mean_ref_energy)] = 0

    ref_hpcp_vector = analysis.hpcp_extract(audio=ref_audio, normalize_method =HPCP_NORMALIZE,hpcp_size=HPCP_SIZE)
    # ref_hpcp_vector = analysis.hpcp_extract(audio=ref_audio,hpcp_size=HPCP_SIZE)


    #ref_hpcp_vector = np.array([ref_hpcp_vector[i] * ref_energy[i] for i in range(len(ref_hpcp_vector))], dtype='float32')
    
    # ref_hpcp_vector[np.where(ref_energy < REF_ENERGY_THRESH)]=0
    # # ref_hpcp_vector = np.array([(np.convolve(ref_hpcp_vector[:,i], np.ones(MOV_AVG_SIZE), 'valid')/MOV_AVG_SIZE) for i in range(HPCP_SIZE)]).T


    ref_pitch, ref_conf = analysis.pitch_extractor(ref_audio)
    
    # ref_pitch_mask = np.zeros(len(ref_pitch))
    # ref_pitch_mask[np.where(ref_pitch > 0)] = 1
    # ref_pitch_mask = ref_pitch_mask[:len(ref_hpcp_vector)]

    #ref_hpcp_vector = np.array([ref_hpcp_vector[i] * ref_energy_mask[i] for i in range(len(ref_hpcp_vector))], dtype='float32')
    #ref_hpcp_vector = ref_hpcp_vector * ref_energy_mask

    submission_sound =submission['sounds'][0]
    sound_id = str(submission_sound['id'])
    std_audio_file = utils.file_download(download_url = submission_sound['download_url'], filetype='submissions')
    print(std_audio_file)

    std_audio = analysis.load_audio(audio_file=std_audio_file)
    std_pitch, std_conf = analysis.pitch_extractor(std_audio)
    
    # std_energy_mask = analysis.energy_mask(std_audio, STD_ENERGY_THRESH)
    # std_audio = std_audio * std_energy_mask
    # std_energy = analysis.short_time_energy(std_audio)
    # std_energy = std_energy/np.max(std_energy)
    # mean_std_energy = np.mean(std_energy)
    # std_energy[np.where(std_energy > 0.01*mean_std_energy)] = 1
    # std_energy[np.where(std_energy <= 0.01*mean_std_energy)] = 0

    std_hpcp_vector = analysis.hpcp_extract(audio=std_audio, normalize_method =HPCP_NORMALIZE, hpcp_size=HPCP_SIZE)
    # std_hpcp_vector = analysis.hpcp_extract(audio=std_audio,hpcp_size=HPCP_SIZE)


    #std_hpcp_vector = np.array([std_hpcp_vector[i] * std_energy[i] for i in range(len(std_hpcp_vector))], dtype='float32')
    # std_hpcp_vector[np.where(std_energy < STD_ENERGY_THRESH)]=0
    
    # std_pitch_mask = np.zeros(len(std_pitch))
    # std_pitch_mask[np.where(std_pitch > 0)] = 1
    # std_pitch_mask = std_pitch_mask[:len(std_hpcp_vector)]

    # std_hpcp_vector = np.array([std_hpcp_vector[i] * std_pitch_mask[i] for i in range(len(std_hpcp_vector))], dtype='float32')
    
    #std_hpcp_vector = np.array([(np.convolve(std_hpcp_vector[:,i], np.ones(MOV_AVG_SIZE), 'valid')/MOV_AVG_SIZE) for i in range(HPCP_SIZE)]).T


    
    
    # std_pitch_mask = np.zeros(len(std_pitch))
    # std_pitch_mask[np.where(std_pitch > 0)] = 1
    # std_pitch_mask = std_pitch_mask[:len(std_hpcp_vector)]

    #std_hpcp_vector = np.array([std_hpcp_vector[i] * std_energy_mask[i] for i in range(len(std_hpcp_vector))], dtype='float32')
    #std_hpcp_vector = std_hpcp_vector * std_energy_mask   
    
    #plotter.plot_pitch_contour(ref_pitch, save_file_name=Path(std_audio_file).stem)

    #cost, path, matrix = analysis.audio_align(ref_energy, std_energy)
    cost, path, matrix = analysis.audio_align(ref_hpcp_vector, std_hpcp_vector)
    #cost, path = fastdtw.fastdtw(ref_hpcp_vector, std_hpcp_vector, dist=dist_cosine)
    # ref_ser =np.append(ref_hpcp_vector.T, [ref_energy], axis=0).T
    # std_ser =np.append(std_hpcp_vector.T, [std_energy], axis=0).T

    # cost, path = fastdtw.fastdtw(ref_ser, std_ser, dist=dist_combination)

    #cost, path = fastdtw.fastdtw(ref_hpcp_vector, std_hpcp_vector, dist=dist_cos_W)
    #path = np.array(path)

    plotter.plot_dtw_alignment(ref_audio, std_audio, path, save_file_name=Path(std_audio_file).stem)


    ref_time_ticks = analysis.get_annotation(segmentAnnotationFile)
    std_time_ticks_dtw = analysis.get_std_time_ticks_dtw(ref_time_ticks, path)

    figure = plotter.performance_visualize(analysis, ref_audio, std_audio, ref_pitch, std_pitch, ref_time_ticks, std_time_ticks_dtw)
    figure.savefig(fname=os.path.join(image_save_path,sound_id ))
    plt.close()
    try:
        features = analysis.features_extract(segmentAnnotationFile, path, ref_pitch, std_pitch)
        if sound_id in grades:
            features['grade']=grades[sound_id]
        else:
            features['grade']='0'
        features['sound_id']= sound_id
        #print(features)
        all_features.append(features)
        
    except:
        print(sound_id)

    gc.collect()
    return


with open(report_file) as f:
    reader = csv.reader(f)
    grades = dict(reader)
    f.close()

context_id = settings.CONTEXT_ID
exercises = mcclient.get_full_context(context_id)['exercises'] 
# TODO replace with json data 
exercises.sort(key=operator.itemgetter('name'))
# for exercise in exercises:
#     print(exercise['name'])

#exercise=exercises[0]

all_features=[]
for exercise in exercises[0:5]:   
    print(exercise['name'])
    submissions = utils.get_submissions_in_exercise(exercise)
    print(len(submissions), submissions[0].keys())
    for submission in submissions:
        try:
            extract_submission_features(exercise, submission)
        except:
            print(submission['sounds'][0]['id'])
        
        
    
# Save extracted features to csv
keys = all_features[0].keys()
with open(features_save_file, 'w') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(all_features)


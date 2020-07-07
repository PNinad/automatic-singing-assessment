# import pandas as pd
import settings
import mcclient
import operator
import utils
import os, csv
# from analysis import Analysis
# import matplotlib.pyplot as plt
# import plotter
# from pathlib import Path
import numpy as np
import gc
# import fastdtw
# from scipy.spatial.distance import cosine, euclidean
from tqdm import tqdm
import notes_singing

def extract_submission_features(exercise, submission_sound, plots_path = None):
    version = submission_sound['version']
    ref_annotation_file = os.path.join('reference_tracks',exercise['name']+'_'+version+'_ref_segments.csv')
    reference_track = utils.get_reference_track(exercise, version)
    ref_audio_file = utils.file_download(download_url = reference_track['download_url'], filetype='reference_tracks')

    std_audio_file = utils.file_download(download_url = submission_sound['download_url'], filetype='submissions')
    
    features = notes_singing.get_features_from_student_audio(ref_audio_file, std_audio_file, ref_annotation_file, plots_path = plots_path, method='MAST')
    features['file_name'] = std_audio_file

    return features 

def plot_all_performance_visualizations(plots_path=None):
    if plots_path is None:
        print('Please provide plots_path')
        return
    
    context_id = settings.CONTEXT_ID
    exercises = mcclient.get_full_context(context_id)['exercises'] 
    # TODO replace with json data 
    exercises.sort(key=operator.itemgetter('name'))

    for ex in tqdm(exercises):
        print(ex['name'])
        submissions = utils.get_submissions_in_exercise(ex)
        print(len(submissions), submissions[0].keys())
        for s in tqdm(submissions):
            submission_sound_files = s['sounds']
            for submission_sound in submission_sound_files:
                sound_id = str(submission_sound['id'])
                try :
                    features = extract_submission_features(ex, submission_sound, plots_path=plots_path)
                except:
                    print('Error processing' + sound_id)
                gc.collect()
    return

def extract_all_features(features_save_file, grade_report_file, plots_path = None):
    context_id = settings.CONTEXT_ID
    exercises = mcclient.get_full_context(context_id)['exercises'] 
    # TODO replace with json data 
    exercises.sort(key=operator.itemgetter('name'))

    with open(grade_report_file) as f:
        reader = csv.reader(f)
        grades = dict(reader)

    all_features=[]
    
    for ex in tqdm(exercises):
        print(ex['name'])
        submissions = utils.get_submissions_in_exercise(ex)
        print(len(submissions), submissions[0].keys())
        for s in tqdm(submissions):
            submission_sound_files = s['sounds']
            for submission_sound in submission_sound_files:
                sound_id = str(submission_sound['id'])
                if sound_id in grades:
                    try :
                        features = extract_submission_features(ex, submission_sound, plots_path=plots_path)
                        features['grade'] = grades[sound_id]
                        # print('\n'+ sound_id +'\n')
                        # print(features)

                        all_features.append(features)
                    except:
                        print('Error processing' + sound_id)
                    # features['grade']=grades[sound_id]
                gc.collect()            
    
    keys = all_features[0].keys()
    with open(features_save_file, 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(all_features)
    return

def test_one_file(ex, submission):
    return

if __name__ == "__main__":
    # main()
    # features_save_file = 'Ninad_MAST_model_features.csv'
    # plots_path = './notes_singing_plots/'
    grade_report_file = './HT_MTG_grades.csv'
    features_save_file = './HT_MTG_MAST_tryndelete.csv'
    # extract_all_features(features_save_file, grade_report_file, plots_path=plots_path)
    extract_all_features(features_save_file, grade_report_file)
    # absolute_error = train_test_linear_regr_model(features_save_file)
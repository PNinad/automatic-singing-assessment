import numpy as np
import os, json, csv
from glob import glob
import pickle
import essentia.standard as ess
# from simmusic.extractors import notes_singing   
import notes_singing
import tempfile
# import subprocess
# import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import gc
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error as MAE
import pandas as pd
from sklearn import linear_model
from joblib import dump, load

# from PIL import Image
#PARAMETERS
# HPCP_SIZE = 120 
# FS = 44100 # sampling rate
# H = 512  # Hop size
# N = 4096 # FFT size
# M = 4096 # window soze
# dist_function = 'euclidean'

# spectrum = ess.Spectrum(size=N)
# window = ess.Windowing(size=M, type='hann')
# spectralPeaks = ess.SpectralPeaks()

dataset_root = './MASTgraded'  # 
plots_path = './plots'
REF_ANNOTATION_TEMPFILE = tempfile.NamedTemporaryFile(delete=False)


def get_exercises(dataset_root):
    return glob(os.path.join(dataset_root, '*mel*'))

def get_reference_path(exercise_path):
    return glob(os.path.join(exercise_path, 'reference', '*.wav'))[0]

def get_student_performances(exercise_path):
    return glob(os.path.join(exercise_path, 'performances', '*.wav'))

def get_student_performance(performance):
    return 

def get_ref_segments(reference_path):
    ref_segments_path = reference_path[:-4]+ '.trans'
    ref_segments=[]
    with open(ref_segments_path) as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            ref_segments.append([row[0], row[1]])
    return ref_segments

def get_ref_trans_file_path(reference_path):
    return reference_path[:-4]+ '.trans'

def get_ref_annotation(exercise_path):
    return glob(os.path.join(exercise_path, 'reference', '*annotation.csv'))[0]

def ref_trans_to_ref_annotation(ref_trans_file_path, ref_annotation_outfile = REF_ANNOTATION_TEMPFILE.name):
    annotations=[]
    with open(ref_trans_file_path) as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            annotations.append([row[0], 'New Point'])
            annotations.append([row[1], 'New Point'])
    # print(annotations)
    
    with open(ref_annotation_outfile, 'w') as f:
        csvwriter = csv.writer(f)
        for row in annotations:
            csvwriter.writerow(row)
    return ref_annotation_outfile

def performance_assess(exercise, performance):
    ref_audio_file = get_reference_path(exercise)
    ref_trans = get_ref_trans_file_path(ref_audio_file)
    ref_annotation_file = ref_trans_to_ref_annotation(ref_trans)
    # ref_annotation_file = get_ref_annotation(exercise)
    std_audio_file = performance

    feedback = notes_singing.assess_singing(ref_audio_file, std_audio_file, ref_annotation_file)
    return feedback

def get_num_lines_in_file(filepath):
    with open(filepath, 'r') as f:
        line_count = sum(1 for _line in f)
    return line_count
    # return int(subprocess.check_output('wc -l {}'.format(file), shell=True).split()[0])

def get_feedback(ex, selected_perf):
    feedback = performance_assess(ex, selected_perf)
    print(feedback['grade'])
    return

def is_graded(perf):
    jsonpath = perf[:-4] + '.json'
    # print(jsonpath)
    with open(jsonpath) as f:
        ann = json.load(f)
    return 'similarity' in ann['overall grade'][0].keys()

def get_ground_truth(perf):
    jsonpath = perf[:-4] + '.json'
    # print(jsonpath)
    with open(jsonpath) as f:
        ann = json.load(f)
    return np.mean(np.fromiter(ann['overall grade'][0]['similarity'].values(), dtype=float))

def performance_feature_extract(exercise, performance):
    ref_audio_file = get_reference_path(exercise)
    ref_trans = get_ref_trans_file_path(ref_audio_file)
    ref_annotation_file = ref_trans_to_ref_annotation(ref_trans)
    # ref_annotation_file = get_ref_annotation(exercise)
    std_audio_file = performance
    features = notes_singing.get_features_from_student_audio(ref_audio_file, std_audio_file, ref_annotation_file,method='MAST')
    features['file_name'] = std_audio_file
    # print(features)
    return features    


def extract_all_features(features_save_file):
    exercises = get_exercises(dataset_root) 
    all_features=[]
    
    for ex in tqdm(exercises):
        # print(ex)
        performances = get_student_performances(ex)
        for p in performances:
            print(p)
            if is_graded(p):
                features = performance_feature_extract(ex, p)
                features['grade'] = get_ground_truth(p)
                # print(features)
                all_features.append(features)
            gc.collect()            
    
    keys = all_features[0].keys()
    with open(features_save_file, 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(all_features)
    return

def train_test_linear_regr_model(features_csv_file, model_save_path=None):
    # Split the data into training/testing sets
    df = pd.read_csv(features_csv_file) 
    print(df['grade'].value_counts())
    features = list(df.columns) 
    X = df.drop(columns=['grade', 'file_name']).values
    # features = ['pitch_hist_cos_10', 'pitch_hist_cos_20', 'pitch_hist_cos_50','pitch_hist_cos_100']
    # features = ['pitch_hist_cos_50']
    # X= df[features].values
    y = df['grade'].values

    # from sklearn.preprocessing import PolynomialFeatures #experimental
    # poly = PolynomialFeatures(degree=2)
    # X = poly.fit_transform(X)
    
    splitRatio4Test=0.10
    numSamples=X.shape[0]
    print('numSamples : ' + str(numSamples))
    numTestSamples=round(numSamples*splitRatio4Test)

    #Running repeated number of random experiments
    _MAE=[]
    gt_array = []
    pred_array = []
    abs_error_per_file = np.empty(shape=[0, ])
    # tested_inds = []
    #Run the tests and print estimation results for each file in log-file: 'resultsPerFile.csv'
    for k in range(0,numSamples,numTestSamples):
        #splitting
        test_indices =list(range(k,min(k+numTestSamples,numSamples)))
        train_indices=np.delete(range(numSamples),test_indices)
        # print('TRAIN : ' + str(train_indices))
        # print('TEST : ' + str(test_indices))

        X_train = X[train_indices]
        X_test = X[test_indices]

        # Split the targets into training/testing sets
        Y_train = y[train_indices]
        Y_test = y[test_indices]
        # tested_inds.extend(test_indices)

        # Create linear regression object
        regr = linear_model.LinearRegression()
        # regr = linear_model.LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')

        # Train the model using the training sets
        regr.fit(X_train, Y_train)
        # print(regr.coef_)
        # Make predictions using the testing set
        Y_pred = regr.predict(X_test)
        #applying hard limits: 1-4 for prediction results
        # Y_pred = np.round(Y_pred)

        minVal=np.ones(Y_pred.shape)
        maxVal=minVal*4
        Y_pred = np.maximum(Y_pred,minVal)
        Y_pred = np.minimum(Y_pred,maxVal)

        

        # Compute mean squared error
        _MAE.append(MAE(Y_test, Y_pred))
        gt_array.extend(Y_test)
        pred_array.extend(Y_pred)

        abs_error_per_file=np.append(abs_error_per_file,np.abs(Y_pred-Y_test))

    print('Mean absolute error for automatic grading: ',np.mean(_MAE))
    # plt.scatter(gt_array, pred_array, marker='o', alpha=0.1)
    # plt.xlabel("Average human grade (true grade)")
    # plt.ylabel("Predicted grade")
    # plt.title("Predicted grades versus true/target grades")
    # plt.show()
    print('Std_dev of absolute error for automatic grading: ',np.std(np.abs(np.array(gt_array)-np.array(pred_array))))
    
    print(features)
    print(regr.coef_)
    print(regr.intercept_)
    abs_err_hist = plt.hist(abs_error_per_file,40)
    # plt.xlabel('Absolute error')
    # plt.ylabel('Number of occurence')
    # plt.xlim([0,4])
    # plt.show()
    
    absolute_error = np.abs(np.array(gt_array)-np.array(pred_array))
    # print(len(absolute_error[absolute_error<1.0]) , len(absolute_error)) 
    # print('Here are large errors', np.where(absolute_error > 1.0)[0],  absolute_error[absolute_error > 1.0])
    large_error = 0.5
    # print(df.iloc[np.where(absolute_error > large_error)[0]], np.array(pred_array)[np.where(absolute_error > large_error)[0]])
    print('\n \n')

    if model_save_path is not None:    
        regr_save = linear_model.LinearRegression()
        regr_save.fit(X, y)
        dump(regr_save, os.path.join(model_save_path,features_csv_file[:-4]+'.joblib'))
    return absolute_error

def model_identity_test(model_path, feat_extract_method):
    reg_model = load(model_path)
    exercises = get_exercises(dataset_root) 
    gt_array=[]
    pred_array=[]
    pred_grade_array=[]
    abs_error_per_file=[]
    for ex in tqdm(exercises):
        print(ex)
        ref_audio_file = get_reference_path(ex)
        ref_trans = get_ref_trans_file_path(ref_audio_file)
        ref_annotation_file = ref_trans_to_ref_annotation(ref_trans)
        std_audio_file = get_reference_path(ex)
        print(ref_audio_file, std_audio_file, ref_annotation_file)
        features = notes_singing.get_features_from_student_audio(ref_audio_file, std_audio_file, ref_annotation_file,method=feat_extract_method)
        print(features)
        X = [v for k, v in features.items()]
        regression_score = reg_model.predict([X])[0]
        pred_array.append(regression_score)
        gt_score = 4.0
        gt_array.append(gt_score)
    print(gt_array, pred_array)



def main():
    exercises = get_exercises(dataset_root) 
    gt_array=[]
    pred_array=[]
    pred_grade_array=[]
    abs_error_per_file=[]
    for ex in tqdm(exercises):
        print(ex)
        # f1 = glob(os.path.join(ex, 'reference','*.txt' ))[0]
        # f2 = glob(os.path.join(ex, 'reference','*.trans'))[0]
        # f1_lines = get_num_lines_in_file(f1)
        # f2_lines = get_num_lines_in_file(f2)
        # print(ex)
        # print('No of lines in test file : ', f1_lines)
        # print('No of lines in trans file : ', f2_lines)
        
        performances = get_student_performances(ex)
        for p in performances:
            if is_graded(p):
                print(p)
                # try:
                feedback = performance_assess(ex, p)
                print(feedback['score'])
                # print(feedback['png'])
                # t = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                
                pred_array.append(feedback['score'])
                # print(pred_array)
                
                pred_grade_array.append(feedback['grade'])
                # print(pred_grade_array)
                
                gt_score = get_ground_truth(p)
                print(gt_score)
                gt_array.append(gt_score)
                abs_error_per_file.append(abs(feedback['score'] - gt_score))
                # print(gt_array)
                ## PLOTTING ###
                plot_outfile = os.path.join(plots_path, os.path.basename(p)[:-4]+'.png')
                with open(plot_outfile, 'wb') as f:
                    f.write(feedback['png'])
                plt.show()
                # print(len(pred_array), len(gt_array))

            gc.collect()
                # except:
                #     print('Error in ',p)
                # selected_perf = performances[0]
                # print(selected_perf)

    plt.scatter(gt_array, pred_array, marker='o', alpha=0.1)
    plt.show()
    # plt.scatter(gt_array, pred_grade_array, marker='o', alpha=0.1)
    # plt.show()
    print(MAE(gt_array, pred_array))
    print(np.mean(abs_error_per_file), np.std(abs_error_per_file), len(abs_error_per_file))

    plt.hist(abs_error_per_file,40)
    plt.xlabel('Absolute error')
    plt.ylabel('Number of occurence')
    plt.xlim([0,4])
    # plt.savefig(dataset_root_dir+'absErrorHist.png')
    plt.show()
    return

if __name__ == "__main__":
    # main()
    # features_save_file = './HT_MTG_MAST_tryndelete.csv'
    # # features_save_file = './MAST_data_MAST_features.csv'
    # # # # features_save_file = './Ninad_data_Ninad_features.csv'
    # # # features_save_file = './MAST_data_both_features.csv'
    # # extract_all_features(features_save_file)
    # absolute_error = train_test_linear_regr_model(features_save_file, model_save_path='.')
    
    # all_features_save_files = glob('*.csv')
    # for f in all_features_save_files:
    #     print(f)
    #     try:
    #         absolute_error = train_test_linear_regr_model(f)
    #     except:
    #         pass
    model_identity_test(model_path='./MAST_data_MAST_features.joblib', feat_extract_method='MAST')
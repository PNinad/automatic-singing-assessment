from settings import *
import logging
# LOG_FILENAME = 'grader_gui_logs.log'
# logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)
LOGGER =logging.getLogger()
# from numpy import finfo
# EPS = finfo(float).eps 

def get_exercise_list_by_context_id(context_id):
    # Redundant
    return mcclient.get_full_context(context_id)['exercises']

def get_exercise_by_name(exercise_name, exercise_list):
    for exercise in exercise_list:    
        if exercise['name'] == exercise_name :
            return exercise

def get_submissions_in_exercise(exercise):
    return exercise['submissions']

def get_submission_sound_ids(submissions_list):
    for submission in submissions_list:
        sound_ids = [submission['sounds'][0]['id']]
        return sound_ids

def get_student_submission_by_sound_id(sound_id, submissions_list):
    for submission in submissions_list:
        if submission['sounds'][0]['id'] == sound_id :
            return submission['sounds'][0]

def get_backing_track(exercise, version):
    for backing_track in exercise['backing_tracks']:
        if backing_track['version'] == version :
            return backing_track

def get_reference_track(exercise, version):
    for reference_track in exercise['reference_tracks']:
        if reference_track['version'] == version :
            return reference_track

def is_downloaded(download_url) :
    is_downloaded = False
    for root, dirs, files in os.walk(DOWNLOAD_PATH):
        if os.path.basename(download_url) in files:
            is_downloaded = True
            return is_downloaded 
    return is_downloaded

def get_audio_file(download_url, download_path=DOWNLOAD_PATH ):
    if is_downloaded(download_url):
        return os.path.abspath(os.path.basename(download_url))

def file_download(download_url, filetype):
    fname = os.path.basename(download_url)
    save_path = os.path.join(DOWNLOAD_PATH, filetype, fname)
    #print(save_path)
    if os.path.isfile(save_path):
        LOGGER.info("File already exists ")
        return save_path
    
    basepath = mcclient.HOSTNAME
    url= 'https://'+basepath+download_url
    #print(url) 
    
    urllib.request.urlretrieve(url, save_path)
    return save_path

def hz2cents(pitch_hz, tonic=440):
    return 1200 * np.log2(pitch_hz/ float(tonic))





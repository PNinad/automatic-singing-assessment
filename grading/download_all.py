import settings
import utils
import mcclient
import json

#context_id = settings.CONTEXT_ID
exercises = mcclient.get_full_context(settings.CONTEXT_ID)['exercises']
with open('data_{}.json'.format(settings.CONTEXT_ID), 'w') as fp:
    json.dump(exercises, fp)
#print(len(exercises), type(exercises))

for exercise in exercises:
    for backing_track in exercise['backing_tracks']:
        utils.file_download(download_url = backing_track['download_url'], filetype='backing_tracks')
    
    for reference_track in exercise['reference_tracks']:
        utils.file_download(download_url = reference_track['download_url'], filetype='reference_tracks')
    
    submissions = utils.get_submissions_in_exercise(exercise)
    #print(len(submissions), type(submissions))
    for submission in submissions:
        sound_files = submission['sounds']
        # print(sound_files)
        for sound_file in sound_files:
            # print(sound_file['download_url'])
            utils.file_download(download_url = sound_file['download_url'],  filetype='submissions')


import notes_singing
import os
import matplotlib.pyplot as plt

test_data_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'data/notes_singing'))

def demo_notes_singing():
    std_audio_file = os.path.join(test_data_dir, 'std_audio.wav')
    ref_audio_file = os.path.join(test_data_dir, 'ref_audio.mp3')
    ref_annotation_file = os.path.join(test_data_dir, 'ref_annotation_file.csv')
    feedback = notes_singing.assess_singing(ref_audio_file, std_audio_file, ref_annotation_file)
    print(feedback['grade'])
    print(feedback['score'])
    print(feedback['pass_fail'])
    plt.show(feedback['png'])

demo_notes_singing()

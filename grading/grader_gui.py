import tkinter as tk
import settings
import mcclient
import operator
import utils
import numpy as np
import os
from tkinter.filedialog import askopenfilename
from pydub import AudioSegment 
import pydub.playback
import threading
from tkinter import simpledialog
import pandas as pd
import csv
import logging
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import plotter
from PIL import Image, ImageTk
import json

LOG_FILENAME = 'grader_gui_logs.log'
logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)
LOGGER =logging.getLogger()
#LOGGER.info("This is a log)")
PLOTS_PATH = 'performance_feedback_plots_HT_MTG'
# PLOTS_PATH = './notes_singing_plots'

class grader_gui(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.label1 = tk.Label(self, text="MusicCritic grading tool")
        self.label1.pack()

        self.reference_track = ""
        self.backing_track = ""
        self.student_track =""
        self.report_filename = None
        self.grades_data = {}


        #exercises = ["1","2","3"]
        self.context_id = settings.CONTEXT_ID
        try:
            with open('data_{}.json'.format(self.context_id), 'r') as jsondata:
                self.exercises = json.load(jsondata) 
        except FileNotFoundError:
            self.exercises = mcclient.get_full_context(self.context_id)['exercises']
            with open('data_{}.json'.format(settings.CONTEXT_ID), 'w') as fp:
                json.dump(exercises, fp)
        self.exercises.sort(key=operator.itemgetter('name'))

        #LOGGER.info(self.exercises[0]['name'])
        self.fig = tk.LabelFrame(self,text="Figure" )
        self.fig.pack(fill= tk.BOTH)
        
        image_file = './Figure_1.png'
        load = Image.open(image_file)
        render = ImageTk.PhotoImage(load)
        self.img = tk.Label(self.fig, image=render)
        self.img.image = render
        self.img.place(x=0, y=0)
        self.img.pack()
        # # x = np.linspace(0,5)
        # # figure = plt.figure()
        # #self.ax = self.figure.add_subplot(111)
        # figure = plt.figure()
        # line = FigureCanvasTkAgg(figure, self.fig)
        # line.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
        # # ax.plot(x, x)

    ##########################################
        top = tk.LabelFrame(self, text="")
        top.pack(fill=tk.BOTH)

        #self.Label()
        list_exercises = tk.LabelFrame(top, text="""Exercises:""")
        list_exercises.pack(side= tk.LEFT)
    
        self.exercise_listbox = tk.Listbox(list_exercises, width=50, selectmode=tk.SINGLE)
        self.exercise_listbox.bind('<<ListboxSelect>>', self.select_exercise)
        for item in self.exercises:
            self.exercise_listbox.insert(tk.END, item['name'])
        self.exercise_listbox.pack()

        #submissions = ["1","2","3"]
        """
        submisssions = load_exercise_submissions(exercise_name)
        """
        self.exercise = utils.get_exercise_by_name(self.exercises[0]['name'], self.exercises)
        self.submissions = utils.get_submissions_in_exercise(self.exercise)

        list_submitted = tk.LabelFrame(top, text="""Submissions:""")
        list_submitted.pack(side= tk.RIGHT)
        self.scrollbar = tk.Scrollbar(list_submitted)
        self.scrollbar.pack( side = tk.RIGHT, fill = tk.Y )

        self.submissions_listbox = tk.Listbox(list_submitted, yscrollcommand = self.scrollbar.set, width=50, selectmode=tk.SINGLE)
        self.submissions_listbox.bind('<<ListboxSelect>>', self.load_submission)

        self.populate_submissions_listbox(self.submissions)


        
        self.submissions_listbox.selection_set(0)
        self.submissions_listbox.pack()
        #submissions_listbox.pack( side = LEFT, fill = BOTH )
        self.scrollbar.config( command = self.submissions_listbox.yview )
    
        buttons = tk.LabelFrame(self, text="""Actions:""")
        buttons.pack(side=tk.LEFT)
        #tk.Label(buttons, text="""Grade :""").pack()
        self.b1 = tk.Button(buttons, text="Start new session",width=20, command=self.new_session)
        self.b1.pack()
        self.b2 = tk.Button(buttons, text="Load session",width=20, command=self.load_session)
        self.b2.pack()
        self.b3 = tk.Button(buttons, text="Play Reference",width=20, command=self.play_reference)
        self.b3.pack()
        self.b4 = tk.Button(buttons, text="Play Student",width=20, command=self.play_student)
        self.b4.pack()
        self.b5 = tk.Button(buttons, text="Save",width=20, command=self.save)
        self.b5.pack()
        self.b6 = tk.Button(buttons, text="Next",width=20, command=self.save_and_next(self.submissions_listbox))
        self.b6.pack()
        self.b7 = tk.Button(buttons, text="Save and exit",width=20, command=self.exit)
        self.b7.pack()


        self.grade = tk.IntVar()
        #grade.set(1) 
        radio_select = tk.LabelFrame(self, text="""Grade :""")
        radio_select.pack()
        #tk.Label(radio_select, text="""Grade :""").pack()
        tk.Radiobutton(radio_select, text="1", variable=self.grade, value=1).pack()
        tk.Radiobutton(radio_select, text="2", variable=self.grade, value=2).pack()
        tk.Radiobutton(radio_select, text="3", variable=self.grade, value=3).pack()
        tk.Radiobutton(radio_select, text="4", variable=self.grade, value=4).pack()
    
        # <create the rest of your GUI here>
    def new_session(self, message=""):
        #global report_filename
        filename = simpledialog.askstring("New_Session", message+" Report file name:",
                                    parent=self, initialvalue="report.csv")
        if os.path.isfile(filename):
            self.new_session(message="FILE EXISTS, try new filename ")
        # else:
        self.report_filename = filename
        open(self.report_filename, 'w').close()
        # print(report_filename)
        self.load_data(self.report_filename)
        #load_data(open(filename, 'a+'))
        return 

    def load_session(self):
        #LOGGER.info('Load session')
        # global report_filename
        filename = askopenfilename(initialdir=settings.DOWNLOAD_PATH,
                            filetypes =(("csv file", "*.csv"),("All Files","*.*")),
                            title = "Choose a file."
                            )
        self.report_filename = filename
        print(self.report_filename)
        self.load_data(self.report_filename)
        #load_data(open(filename, 'r+'))
        return 

    def load_data(self, filename):
        #global grades_data
        # if(filebuffer.read()==""):
        #     col_names = ["filename", "grade"]
        #     report_df = pd.DataFrame(columns = col_names)
        # #report_filename = filename
        # else:
        with open(filename) as f:
            reader = csv.reader(f)
            self.grades_data = dict(reader)
        #LOGGER.info(print(self.grades_data))
            #report_df = pd.read_csv(filebuffer)
        #print(report_df)
        return

    def select_exercise(self, event):
        #global exercise, submissions
        exercise_name = self.exercise_listbox.get(tk.ANCHOR)
        #LOGGER.info('Select exer', exercise_name)
        self.exercise = utils.get_exercise_by_name(exercise_name, self.exercises)
        self.submissions = utils.get_submissions_in_exercise(self.exercise)
        self.populate_submissions_listbox(self.submissions)
        return 

    def play_reference(self):
        
        sound_id = self.submissions_listbox.get(tk.ANCHOR)
        #LOGGER.info(sound_id)
        ref_audio = AudioSegment.from_file(self.reference_track)
        backing_audio = AudioSegment.from_file(self.backing_track)
        #std_audio = AudioSegment.from_file(student_track)
        ref_mix = ref_audio.overlay(backing_audio)
        #std_mix = std_audio.overlay(backing_audio)
        while True:
            try:
                t_ref = threading.Thread(target = pydub.playback.play(ref_mix))
                t_ref.start()
                raise KeyboardInterrupt
            except KeyboardInterrupt:
                print("Stopping playing")
                break #to exit out of loop, back to main program
        return

    def play_student(self):
        sound_id = self.submissions_listbox.get(tk.ANCHOR)
        #LOGGER.info(sound_id)
        #ref_audio = AudioSegment.from_file(reference_track)
        backing_audio = AudioSegment.from_file(self.backing_track)
        std_audio = AudioSegment.from_file(self.student_track)
        #ref_mix = ref_audio.overlay(backing_audio)
        std_mix = std_audio.overlay(backing_audio)
        while True:
            try:
                t_std = threading.Thread(target = pydub.playback.play(std_mix))
                t_std.start()
                raise KeyboardInterrupt
            except KeyboardInterrupt:
                print("Stopping playing")
                break #to exit out of loop, back to main program
        return

    def save(self):
        #global grades_data
        #LOGGER.info(self.grade.get())
        sound_id = self.submissions_listbox.get(tk.ANCHOR)
        # if sound_id not in grades_data:
        #     grades_data.append({sound_id : grade.get()})
        # else:
        #     grades_data.update({sound_id : grade.get()})
        if sound_id is not '':
            self.grades_data[str(sound_id)] = str(self.grade.get())
            self.mark_graded(sound_id=sound_id, curr_selection= tk.ANCHOR)
            #print("this works")
        LOGGER.info(self.grades_data)
        #self.list_next(self.submissions_listbox)
        # if report_filename is not None:
        #     with open(report_filename,  as f)

        return

    def save_file(self):
        try:
            #print("it is here")
            with open(self.report_filename, 'w') as csv_file:
                writer = csv.writer(csv_file)
                for key, value in self.grades_data.items():
                    writer.writerow([key, value])
        except e:
            LOGGER.error("error saving file")

        return

    def exit(self):
        try:
            self.save_file()
        except:
            return
        self.parent.destroy()
        #raise KeyboardInterrupt
        return

    def list_next(self, listBox):
        curr_selection = int(listBox.curselection()[-1])
        #LOGGER.info(curr_selection)
        next_selection = curr_selection + 1
        listBox.activate(next_selection)
        #LOGGER.info(curr_selection)
        listBox.selection_set(next_selection)
        listBox.event_generate("<<ListboxSelect>>")
        return

    def load_submission(self, event):
        #global student_track, backing_track, reference_track
        sound_id = self.submissions_listbox.get(tk.ANCHOR)
        #LOGGER.info(" download files", sound_id)
        #submissions = utils.get_submissions_in_exercise(exercise)
        submission = utils.get_student_submission_by_sound_id(sound_id, self.submissions)
        #LOGGER.info(submission)
        if submission is not None:
            #print(submissions_listbox.)
            #sub_ind = [item for item in submissions_listbox]
            #submissions_listbox.itemconfig(sub_ind, {'bg':'red'})
            version = submission['version']
            self.backing_track = utils.get_backing_track(self.exercise, version)
            self.reference_track = utils.get_reference_track(self.exercise, version)
            #LOGGER.info(version, self.backing_track['download_url'], self.reference_track['download_url'])
            self.student_track = utils.file_download(download_url = submission['download_url'], filetype='submissions')
            self.backing_track = utils.file_download(download_url = self.backing_track['download_url'], filetype='backing_tracks')
            self.reference_track = utils.file_download(download_url = self.reference_track['download_url'], filetype='reference_tracks')
            #play(reference_track, backing_track, student_track)
            # image_file = os.path.join('performance_feedback',str(sound_id)+'.png')
            image_file = os.path.join(PLOTS_PATH,os.path.basename(self.student_track)[:-4]+'.png')
            render = ImageTk.PhotoImage(Image.open(image_file))
            self.img.configure(image=render)
            self.img.image = render
            # figure = self.visualizer(version)
            # line = FigureCanvasTkAgg(figure, self.fig)
            # line.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)

        return

    def save_and_next(self, listBox):
        # save()
        # list_next(listBox)
        # load_submission(event)
        return

    def populate_submissions_listbox(self, submissions):
        self.submissions_listbox.delete(0, tk.END)
        for item in submissions:
            self.submissions_listbox.insert(tk.END, item['sounds'][0]['id'])
            # colour code when grading done
            sound_id = self.submissions_listbox.get(tk.END)
            self.mark_graded(sound_id, tk.END)
    
    def mark_graded(self,sound_id, curr_selection):
        if str(sound_id) in self.grades_data.keys():
            self.submissions_listbox.itemconfig(curr_selection, {'bg':'red'})

    def visualizer(self, version):
        HPCP_NORMALIZE = 'unitMax'
        HPCP_SIZE = 120
        from analysis import Analysis
        analysis = Analysis()

        segmentAnnotationFile = os.path.join('reference_tracks',self.exercise['name']+'_'+version+'_ref_segments.csv')
        print(segmentAnnotationFile)

        ref_audio_file = self.reference_track
        ref_audio = analysis.load_audio(audio_file=ref_audio_file)
        ref_hpcp_vector = analysis.hpcp_extract(audio=ref_audio, normalize_method =HPCP_NORMALIZE,hpcp_size=HPCP_SIZE)
        ref_pitch, ref_conf = analysis.pitch_extractor(ref_audio)

        std_audio_file = self.student_track
        std_audio = analysis.load_audio(audio_file=std_audio_file)
        std_hpcp_vector = analysis.hpcp_extract(audio=std_audio, normalize_method= HPCP_NORMALIZE, hpcp_size=HPCP_SIZE)
        std_pitch, std_conf = analysis.pitch_extractor(std_audio)

        cost, path, matrix = analysis.audio_align(ref_hpcp_vector, std_hpcp_vector)

        ref_time_ticks = analysis.get_annotation(segmentAnnotationFile)
        std_time_ticks_dtw = analysis.get_std_time_ticks_dtw(ref_time_ticks, path)

        figure = plotter.performance_visualize(analysis, ref_audio, std_audio, ref_pitch, std_pitch, ref_time_ticks, std_time_ticks_dtw)
        return figure

if __name__ == "__main__":
    #download all files
    root = tk.Tk()
    grader_gui(root).pack(side="top", fill="both", expand=True)
    root.mainloop()

This piece of software is a GUI tool developed to grade exercises from the MusicCritic contexts using the research client mcclient. 

Steps:
1)Set the download path, mcclient TOKEN and hostname in settings.py

2) If you want to download all soundfiles beforehand, run:
python3 download_all.py

This step is not essential as the GUI tool is capable of downloading relevant data as per requirement

3) Run:
python3 grader_gui.py

4) Start new session or load previous session. Specify file name for the grade report.

5) Click on exercise name and submission id in the gui list to load a submission.

6) play reference and student version. Select appropriate grade by clicking radio button.

7) Click save to record the grade. The graded submissions will appear red in the submissions list.

8) Click exit to Save the grades in your report file and exit.

import argparse
import mcclient
import numpy as np
import urllib.request
#import utils
import os

DOWNLOAD_PATH = '.'
MC_CLIENT_TOKEN = "your mcclient token here"

CONTEXT_ID = 12 # sargamsangeet dataset
# CONTEXT_ID = 6 # Hindustani Training dataset

mcclient.set_token(MC_CLIENT_TOKEN)
mcclient.set_hostname('staging.musiccritic.upf.edu')

dirs = ['submissions','backing_tracks','reference_tracks']
for d in dirs:
    os.makedirs(d, exist_ok=True)






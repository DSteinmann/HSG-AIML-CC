# --- Kaggle Submission (Requires API Key) ---
from kaggle.api.kaggle_api_extended import KaggleApi
import os
import pandas as pd

api = KaggleApi()
api.authenticate()  # Ensure your kaggle.json is in the correct location (~/.kaggle/)

kaggle_competition = '8-860-1-00-coding-challenge-2025'  # Your competition name
kaggle_message = 'track_1'  # Your submission message
output_csv_path = 'track_1.csv' # Output CSV

try:
  #Kaggle may throw errors if it thinks the file already exist
  #this is a quick a dirty solution for it
    api.competition_submissions(kaggle_competition) #list submissions to check if the file is already there
    api.competition_submit(output_csv_path, kaggle_message, kaggle_competition) #resubmit
    print("Submission successful!")
except Exception as e:
    print(f"Submission failed: {e}")
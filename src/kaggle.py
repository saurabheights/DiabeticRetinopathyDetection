import os

from configuration import kaggle_params
import subprocess

def submit_solution(filepath):
    u = kaggle_params['kaggle_username']
    p = kaggle_params['kaggle_password']
    msg = kaggle_params['kaggle_submission_message']
    bashCommand = f"kg submit {filepath} -u {u} -p {p} -c diabetic-retinopathy-detection -m {msg}"
    out = os.popen(bashCommand).read()
    return out

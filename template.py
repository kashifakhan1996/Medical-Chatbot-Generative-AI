import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,format='[%(asctime)s]:%(message)s:')

list_of_paths = {
    'src/__init__.py',
    'src/helper.py',
    'src/prompt.py',
    '.env',
    'setup.py',
    'requirements.txt',
    'app.py',
    'research/trials.ipynb'

}

for filepath in list_of_paths:
    filepath = Path(filepath)
    filedir,filename=os.path.split(filepath)

    if filedir !='':
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"Create Dicrectory; {filedir} for the file {filename}")
    if (not os.path.exists(filepath) or os.path.getsize(filepath) is 0):
        with open(filepath,'w') as f:
            pass
            logging.info('create empty file: {filepath}')
    else:
        logging.info(f'{filepath} already exists')
# this file will create the folder structure of the project

# 1. Imports 
import os
from pathlib import Path
import logging

# 2. Defining logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 3. This list defines the folder structure
project_name = "VeriLLM"

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/_init _. py",
    f"src/{project_name}/components/_init _. py",
    f"src/{project_name}/utils/_init _. py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/config/_init _. py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/_init _. py",
    f"src/{project_name}/entity/_init _. py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/constants/_init _. py",
    "config/config.yaml",
    "params.yaml",
    "schema.yaml",
    "main.py",
    "app.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb",
    "templates/index.html",
    "test.py"
]

# 4. Create the folder structure
for filepath in list_of_files:
    # Path will convert the / or \ to the format according to operating sys
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating file directory; {filedir} for file: {filename}")

    # Create an empty file if it does not exist or is empty
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as fp:
            pass
            logging.info(f"Creating empty file: {filepath}")

    else:
        logging.info(f"File already exists: {filepath}, skipping file creation.")
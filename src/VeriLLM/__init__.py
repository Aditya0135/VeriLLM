import os
import sys
import logging

LOG_FORMAT = "[%(asctime)s: [%(levelname)s  || %(name)s || %(message)s]"
LOG_DIR = os.path.join("artifacts", "logs")
LOG_FILEPATH = os.path.join(LOG_DIR, "running_logs.log")
logger = logging.getLogger(__name__)

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, handlers=[
    logging.FileHandler(LOG_FILEPATH),
    logging.StreamHandler(sys.stdout)
])

logger = logging.getLogger(__name__)
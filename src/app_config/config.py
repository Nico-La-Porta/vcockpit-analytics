import os
import dotenv

dotenv.load_dotenv()

# Set a default environment if none is provided
APP_ENV = os.getenv('APP_ENV')

if APP_ENV == 'dev':
    from .config_dev import *
elif APP_ENV == 'test':
    from .config_test import *
elif APP_ENV == 'prod':
    from .config_prod import *
else:
    DEBUG = os.getenv('DEBUG', 'false')
    DEBUG = DEBUG.lower() in ["true", "1", "yes"]

PATH = {
    'base_root': os.getenv('BASE_ROOT'),
    'data': os.getenv('DATA'),
    'logs_json': os.getenv('LOGS_JSON'),
    'study_folder': os.getenv('STUDY_FOLDER'),
    'study_pickle_file_path': os.getenv('STUDY_PICKLE_FILE_PATH'),
}

METADATA = {
    'scenario': os.getenv('SCENARIO'),
    'users': os.getenv('USERS')
}
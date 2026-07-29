import os

# Ensure `src` is importable without a real .env file: config.py reads these at
# import time, so they must be set before any `src.*` module is collected.
os.environ.setdefault('DEBUG', 'false')
os.environ.setdefault('BASE_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault('DATA', os.path.join(os.environ['BASE_ROOT'], 'data'))
os.environ.setdefault('LOGS_JSON', os.path.join(os.environ['DATA'], 'logs_json'))
os.environ.setdefault('STUDY_PICKLE_FILE_PATH', 'study.pkl')
os.environ.setdefault('SCENARIO', 'all')
os.environ.setdefault('USERS', 'all')

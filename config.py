import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    RESULTS_FOLDER = os.path.join(BASE_DIR, 'results')
    LIB_DIR = os.path.join(BASE_DIR, 'lib')          # DLL 存放位置
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024           # 500MB
    SEND_FILE_MAX_AGE_DEFAULT = 0
    TZ = 'Asia/Shanghai'
    DB_HOST = '10.3.11.32'
    DB_PORT = 3306
    DB_USER = 'root'
    DB_PASSWORD = '111111'
    DB_NAME = 'ai_track_analysis'
    MODEL_PATH = os.path.join(BASE_DIR, 'best.pt')
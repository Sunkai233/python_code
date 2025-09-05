from waitress import serve
from app import app
import os

if __name__ == '__main__':
    os.environ['FLASK_ENV'] = 'production'
    os.environ['SECRET_KEY'] = 'your-production-secret-key-change-this'

    print("OSS文件管理系统启动中...")
    print(f"访问地址: http://117.72.172.99:5000")

    serve(app, host='0.0.0.0', port=5000, threads=4)
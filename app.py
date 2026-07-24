#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
# 修复 Windows 控制台编码
if sys.platform == 'win32':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    except Exception:
        pass

from flask import Flask
from config import Config
from exts import cors
from core.utils import create_directories
# 不再启动时初始化追踪系统
# from service.analysis_service import init_tracking_system as _init_tracking
from blueprints.main import bp as main_bp
from blueprints.api import bp as api_bp

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # 设置时区
    os.environ['TZ'] = app.config.get('TZ', 'Asia/Shanghai')

    # 初始化 CORS
    cors.init_app(app, resources={
        r"/api/*": {"origins": "*"},
        r"/results/*": {"origins": "*"},
        r"/video/*": {"origins": "*"}
    })

    # 创建必要目录（uploads, results）
    create_directories(app)

    # 不在此处初始化追踪系统，改为在第一次分析时懒加载
    # with app.app_context():
    #     _init_tracking()   # 注释掉

    # 注册蓝图
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp)

    return app

if __name__ == '__main__':
    from core.utils import get_local_ip
    app = create_app()
    local_ip = get_local_ip()
    port = 5000
    print("=" * 60)
    print("AI视频追踪分析系统")
    print(f"访问地址: http://{local_ip}:{port}")
    app.run(host='0.0.0.0', port=5003, debug=False)  
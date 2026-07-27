#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import signal
import threading

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
from blueprints.main import bp as main_bp
from blueprints.api import bp as api_bp

shutdown_event = threading.Event()

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    os.environ['TZ'] = app.config.get('TZ', 'Asia/Shanghai')

    cors.init_app(app, resources={
        r"/api/*": {"origins": "*"},
        r"/results/*": {"origins": "*"},
        r"/video/*": {"origins": "*"}
    })

    create_directories(app)

    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp)

    return app

def handle_shutdown(signum=None, frame=None):
    print("\n收到终止信号，正在关闭服务...")
    
    shutdown_event.set()
    
    try:
        from blueprints.api import _reset_realtime_state
        _reset_realtime_state()
    except Exception as e:
        print(f"重置实时状态失败: {e}")
    
    print("服务已关闭")
    os._exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    from core.utils import get_local_ip
    app = create_app()
    local_ip = get_local_ip()
    port = 5003
    
    print("=" * 60)
    print("AI视频追踪分析系统")
    print(f"访问地址: http://{local_ip}:{port}")
    print("按 Ctrl+C 停止服务")
    print("=" * 60)
    
    try:
        app.run(host='0.0.0.0', port=port, debug=False)
    except KeyboardInterrupt:
        handle_shutdown()
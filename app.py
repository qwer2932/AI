#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import signal
import threading
import socket
import time

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
_server = None
_server_thread = None


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

    # 注册蓝图
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp)

    return app


def cleanup_resources():
    """清理所有资源"""
    try:
        from blueprints.api import _reset_realtime_state
        _reset_realtime_state()
    except Exception as e:
        print(f"重置实时状态失败: {e}")
    
    print("资源清理完成")


def handle_shutdown(signum=None, frame=None):
    print("\n收到终止信号，正在关闭服务...")
    
    # 设置关闭事件
    shutdown_event.set()


def wait_for_port_release(port, timeout=5):
    """等待端口释放"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('0.0.0.0', port))
            sock.close()
            return True
        except OSError:
            sock.close()
            time.sleep(0.1)
    return False


def run_server():
    """在后台线程中运行服务器"""
    global _server
    try:
        _server.serve_forever()
    except Exception as e:
        print(f"服务器线程异常: {e}")


if __name__ == '__main__':
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    from core.utils import get_local_ip
    app = create_app()
    local_ip = get_local_ip()
    port = 5003
    
    print("=" * 60)
    print("AI视频追踪分析系统")
    print("按 Ctrl+C 停止服务")
    print("=" * 60)
    
    # 检查端口是否被占用
    if not wait_for_port_release(port, timeout=2):
        print(f"警告: 端口 {port} 可能仍被占用，尝试继续启动...")
        # 尝试强制释放端口
        import subprocess
        try:
            result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True)
            lines = result.stdout.split('\n')
            for line in lines:
                if f':{port}' in line and 'LISTENING' in line:
                    parts = line.strip().split()
                    pid = parts[-1]
                    print(f"发现占用端口的进程 PID: {pid}，正在终止...")
                    subprocess.run(['taskkill', '/F', '/PID', pid], capture_output=True)
                    print(f"已终止进程 {pid}")
                    time.sleep(1)
                    break
        except Exception as e:
            print(f"强制释放端口失败: {e}")
    
    try:
        from werkzeug.serving import make_server
        _server = make_server('0.0.0.0', port, app, threaded=True)
        # 设置端口复用
        _server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # 在后台线程中启动服务器
        _server_thread = threading.Thread(target=run_server, daemon=True)
        _server_thread.start()
        
        print(f"服务启动成功: http://0.0.0.0:{port}")
        print("等待客户端连接，按 Ctrl+C 停止...")
        
        # 主线程等待关闭信号
        while not shutdown_event.is_set():
            try:
                shutdown_event.wait(timeout=0.5)
            except Exception:
                break
        
        # 收到关闭信号，开始清理
        print("\n正在关闭服务...")
        
        # 清理资源（停止实时线程、释放摄像头等）
        cleanup_resources()
        
        # 停止服务器
        if _server is not None:
            try:
                _server.shutdown()
                print("服务器已停止")
            except Exception as e:
                print(f"停止服务器异常: {e}")
        
        # 等待服务器线程结束
        if _server_thread is not None:
            _server_thread.join(timeout=3)
        
        print("服务已关闭")
        sys.exit(0)
        
    except Exception as e:
        print(f"服务启动失败: {e}")
        cleanup_resources()
        sys.exit(1)

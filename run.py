#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动脚本
"""

import socket
from app import app, init_tracking_system

def get_local_ip():
    """获取本机局域网IP地址"""
    try:
        # 创建一个socket连接来获取本机IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

if __name__ == '__main__':
    print("=" * 60)
    print("AI视频追踪分析系统")
    print("=" * 60)
    
    # 初始化追踪系统
    init_tracking_system()
    
    # 获取本机IP地址
    local_ip = get_local_ip()
    port = 5000
    
    print("启动Web服务器...")
    print("=" * 60)
    print("访问地址:")
    print(f"  本机访问: http://localhost:{port}")
    print(f"  局域网访问: http://{local_ip}:{port}")
    print("=" * 60)
    print("局域网访问说明:")
    print("1. 确保防火墙允许5000端口访问")
    print("2. 其他设备需要与服务器在同一局域网")
    print("3. 在其他设备浏览器中输入上述局域网地址")
    print("=" * 60)
    print("按 Ctrl+C 停止服务器")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=port, debug=False)



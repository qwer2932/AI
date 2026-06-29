#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试HTTP视频访问
"""

import requests
import os

def test_http_video():
    video_filename = "f16c6fd4-20c9-4798-955d-ffeb6296d1a1_tracked.mp4"
    
    print("测试HTTP视频访问")
    print("=" * 50)
    
    # 测试API路径
    api_url = f"http://localhost:5000/api/video/results/{video_filename}"
    print(f"API URL: {api_url}")
    
    try:
        response = requests.head(api_url, timeout=10)
        print(f"API状态码: {response.status_code}")
        print(f"Content-Type: {response.headers.get('Content-Type')}")
        print(f"Content-Length: {response.headers.get('Content-Length')}")
        print(f"Accept-Ranges: {response.headers.get('Accept-Ranges')}")
        print(f"Access-Control-Allow-Origin: {response.headers.get('Access-Control-Allow-Origin')}")
        
        if response.status_code == 200:
            print("API访问成功")
        else:
            print("API访问失败")
    except Exception as e:
        print(f"API访问异常: {e}")
    
    print()
    
    # 测试直接路径
    direct_url = f"http://localhost:5000/videos/{video_filename}"
    print(f"直接URL: {direct_url}")
    
    try:
        response = requests.head(direct_url, timeout=10)
        print(f"直接状态码: {response.status_code}")
        print(f"Content-Type: {response.headers.get('Content-Type')}")
        print(f"Content-Length: {response.headers.get('Content-Length')}")
        print(f"Accept-Ranges: {response.headers.get('Accept-Ranges')}")
        print(f"Access-Control-Allow-Origin: {response.headers.get('Access-Control-Allow-Origin')}")
        
        if response.status_code == 200:
            print("直接访问成功")
        else:
            print("直接访问失败")
    except Exception as e:
        print(f"直接访问异常: {e}")
    
    print()
    
    # 测试部分内容请求
    print("测试部分内容请求")
    try:
        headers = {'Range': 'bytes=0-1023'}
        response = requests.get(api_url, headers=headers, timeout=10)
        print(f"部分内容状态码: {response.status_code}")
        print(f"实际内容长度: {len(response.content)}")
        
        if response.status_code in [200, 206]:
            print("部分内容请求成功")
        else:
            print("部分内容请求失败")
    except Exception as e:
        print(f"部分内容请求异常: {e}")

if __name__ == "__main__":
    test_http_video()



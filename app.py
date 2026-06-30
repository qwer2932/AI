#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask后端API - 支持前端视频追踪分析
"""

import os
import sys
# 修复 Windows 下 stdout 默认 GBK 编码无法输出 Unicode 字符的问题
if sys.platform == 'win32':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    except Exception:
        pass

import cv2
import json
import uuid
import time
import threading
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from tracking_system import TrackingSystem
from database import DatabaseManager

# 创建Flask应用
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # 禁用文件缓存

# 启用CORS支持
CORS(app, resources={
    r"/api/*": {"origins": "*"},
    r"/results/*": {"origins": "*"},
    r"/video/*": {"origins": "*"}
})

# 设置时区
import os
os.environ['TZ'] = 'Asia/Shanghai'

# 创建必要的目录
def create_directories():
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

create_directories()

# 全局变量
tracking_system = None
db_manager = None
analysis_status = {
    'status': 'idle',
    'is_processing': False,
    'progress': 0,
    'current_frame': 0,
    'total_frames': 0,
    'message': '等待中...'
}
task_status = {}  # 存储每个任务的状态
pause_requests = {}  # 存储暂停请求 {analysis_id: True/False}

# 任务状态管理 - 支持多个并发任务
task_status = {}

def init_tracking_system():
    """初始化追踪系统"""
    global tracking_system, db_manager
    try:
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'best.pt')
        print(f"模型路径: {model_path}")
        print(f"模型文件存在: {os.path.exists(model_path)}")
        tracking_system = TrackingSystem(model_path)
        print("追踪系统初始化成功")
        
        # 初始化数据库管理器（可选，失败不影响主流程）
        print("正在初始化数据库管理器...")
        try:
            db_manager = DatabaseManager(
                host="localhost",
                port=3306,
                user="root",
                password="111111",
                database="ai_track_analysis"
            )
            print("数据库系统初始化成功")
            print(f"db_manager 对象: {db_manager}")
            print(f"db_manager 是否为 None: {db_manager is None}")
        except Exception as db_e:
            print(f"⚠ 数据库初始化失败（已跳过，分析结果将仅保存到本地文件）: {db_e}")
            db_manager = None
    except Exception as e:
        print(f"系统初始化失败: {e}")
        import traceback
        traceback.print_exc()

def allowed_file(filename):
    """检查文件类型是否允许"""
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_behavior(tracking_result, video_info):
    """
    使用后置推理分析装配步骤
    根据每帧检测到的物体位置关系，判断当前处于哪个装配步骤

    Args:
        tracking_result: 包含 tracking_data, per_frame_detections 等
        video_info: 视频信息

    Returns:
        行为分析结果（7个装配步骤统计）
    """
    from step_inference import StepInference

    fps = video_info.get('fps', 25)
    per_frame_detections = tracking_result.get('per_frame_detections', [])
    
    print(f"[DEBUG] 分析帧数: {len(per_frame_detections)}")
    if per_frame_detections:
        print(f"[DEBUG] 第一帧检测数: {len(per_frame_detections[0]['detections'])}")
        # 打印第一帧检测到的类别
        classes = set(d.get('class_name') for d in per_frame_detections[0]['detections'])
        print(f"[DEBUG] 第一帧检测到的类别: {classes}")

    inference = StepInference(proximity_threshold=0.30, warmup_frames=30)

    # 逐帧推理
    for frame_data in per_frame_detections:
        inference.infer_step(
            frame_shape=(video_info.get('height', 1080),
                         video_info.get('width', 1920), 3),
            detections=frame_data['detections']
        )
    
    # 获取统计摘要
    step_summary = inference.get_summary(fps=fps)
    print(f"[DEBUG] 行为分析结果: {step_summary}")

    # 构建 track_behaviors
    track_behaviors = {}
    for track_id, steps in step_summary.items():
        # 排除 _total，只计算各步骤的帧数
        step_values = {k: v for k, v in steps.items() if k != '_total'}
        total_time = steps.get('_total', sum(step_values.values()))
        track_behaviors[str(track_id)] = {
            'total_time': float(total_time),
            **step_values
        }

    # 按总时间排序，取前3
    sorted_tracks = sorted(
        track_behaviors.items(),
        key=lambda x: x[1]['total_time'],
        reverse=True
    )[:3]

    return {
        'track_behaviors': track_behaviors,
        'top_tracks': [
            {**{'track_id': int(track_id)}, **{k: float(v) for k, v in behavior.items()}}
            for track_id, behavior in sorted_tracks
        ]
    }


def run_analysis(analysis_id, filepath, original_filename=None):
    """在后台运行分析"""
    print(f"=== 开始分析 {analysis_id} ===")
    print(f"=== 文件路径: {filepath} ===")
    try:
        global analysis_status, tracking_system, db_manager, task_status, pause_requests
        
        print(f"db_manager 状态: {db_manager is not None}")
        
        if not tracking_system:
            print("重新初始化追踪系统...")
            init_tracking_system()
        
        # 初始化任务状态
        task_status[analysis_id] = {
            'status': 'processing',
            'is_processing': True,
            'progress': 0,
            'current_frame': 0,
            'total_frames': 0,
            'message': '正在加载模型...'
        }
        
        # 更新全局状态（保持向后兼容）
        analysis_status['status'] = 'processing'
        analysis_status['message'] = '正在加载模型...'
        
        # 检查暂停和终止状态
        def check_pause_or_stop():
            if analysis_id in pause_requests:
                status = pause_requests[analysis_id]
                if status == 'stop':
                    print(f"分析 {analysis_id} 被终止")
                    return 'stop'
                elif status == True:
                    print(f"分析 {analysis_id} 被暂停")
                    return 'pause'
            return False
        
        # 修改进度回调函数，添加暂停和终止检查
        def progress_callback_with_pause(frame, total, msg, aid):
            status = check_pause_or_stop()
            if status == 'stop':
                # 终止分析
                raise Exception("分析已被用户终止")
            elif status == 'pause':
                # 等待恢复
                while check_pause_or_stop() == 'pause':
                    time.sleep(0.1)
                print(f"分析 {analysis_id} 已恢复")
            update_progress(frame, total, msg, aid)
        
        # 执行追踪分析
        result = tracking_system.analyze_video(
            filepath, 
            analysis_id,
            progress_callback=progress_callback_with_pause
        )
        
        # 执行行为分析（使用后置推理）
        analysis_status['message'] = '正在分析行为数据...'
        behavior_result = analyze_behavior(
            result,          # 包含 tracking_data + per_frame_detections
            result['video_info']
        )
        
        # 如果没有提供原始文件名，使用文件路径中的文件名
        if not original_filename:
            original_filename = os.path.basename(filepath)
        
        # 合并结果
        final_result = {
            **result,
            'behavior_analysis': behavior_result,
            'analysis_id': analysis_id,
            'timestamp': datetime.now().isoformat(),
            'filename': os.path.basename(filepath),
            'original_filename': original_filename
        }
        
        # 保存到数据库
        print(f"准备保存到数据库...")
        print(f"db_manager 是否为 None: {db_manager is None}")
        if db_manager:
            try:
                print(f"正在保存分析结果到MySQL: {analysis_id}")
                success = db_manager.save_analysis_result(final_result)
                if success:
                    print(f"✓ 分析结果已保存到数据库: {analysis_id}")
                else:
                    print(f"✗ 数据库保存失败: {analysis_id}")
            except Exception as e:
                print(f"✗ 数据库保存异常: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("✗ db_manager 为 None，无法保存到数据库")
        
        # 保存结果文件（保持向后兼容）
        result_file = os.path.join(app.config['RESULTS_FOLDER'], f"{analysis_id}.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)
        
        # 更新任务状态为完成
        if analysis_id in task_status:
            task_status[analysis_id].update({
                'status': 'completed',
                'is_processing': False,
                'progress': 100,
                'current_frame': task_status[analysis_id]['total_frames'],
                'total_frames': task_status[analysis_id]['total_frames'],
                'message': '分析完成'
            })
        
        # 更新全局状态为完成
        analysis_status = {
            'status': 'completed',
            'is_processing': False,
            'progress': 100,
            'current_frame': analysis_status['total_frames'],
            'total_frames': analysis_status['total_frames'],
            'message': '分析完成'
        }
        
        print(f"=== 分析完成: {analysis_id} ===")
        
        # 清理任务状态（延迟清理，给前端时间获取最终状态）
        import threading
        def cleanup_task():
            import time
            time.sleep(30)  # 30秒后清理
            if analysis_id in task_status:
                del task_status[analysis_id]
                print(f"已清理任务状态: {analysis_id}")
        
        cleanup_thread = threading.Thread(target=cleanup_task)
        cleanup_thread.daemon = True
        cleanup_thread.start()
    
    except Exception as e:
        print(f"=== 分析失败: {e} ===")
        import traceback
        traceback.print_exc()
        
        # 检查是否是用户终止
        if "分析已被用户终止" in str(e):
            print(f"分析 {analysis_id} 已被用户终止")
            # 更新任务状态为已终止
            if analysis_id in task_status:
                task_status[analysis_id].update({
                    'status': 'stopped',
                    'is_processing': False,
                    'message': '分析已终止'
                })
        else:
            # 更新任务状态为错误
            if analysis_id in task_status:
                task_status[analysis_id].update({
                    'status': 'error',
                    'is_processing': False,
                    'progress': 0,
                    'current_frame': 0,
                    'total_frames': 0,
                    'message': f'分析失败: {str(e)}'
                })
        
        # 更新全局状态为错误
        analysis_status = {
            'status': 'error',
            'is_processing': False,
            'progress': 0,
            'current_frame': 0,
            'total_frames': 0,
            'message': f'分析失败: {str(e)}'
        }

def update_progress(current_frame, total_frames, message, analysis_id=None):
    """更新分析进度"""
    global analysis_status, task_status
    progress = int((current_frame / total_frames) * 100) if total_frames > 0 else 0
    
    # 更新全局状态
    analysis_status.update({
        'progress': progress,
        'current_frame': current_frame,
        'total_frames': total_frames,
        'message': message
    })
    
    # 如果提供了analysis_id，也更新对应的任务状态
    if analysis_id and analysis_id in task_status:
        task_status[analysis_id].update({
            'progress': progress,
            'current_frame': current_frame,
            'total_frames': total_frames,
            'message': message
        })

# API路由
@app.route('/')
def index():
    """主页"""
    return send_from_directory('static', 'index.html')

@app.route('/test')
def test_page():
    """视频访问测试页面"""
    return send_from_directory('static', 'test.html')

@app.route('/video-test')
def video_test_page():
    """视频访问测试页面"""
    return send_from_directory('static', 'video_test.html')

@app.route('/browser-test')
def browser_test_page():
    """浏览器兼容性测试页面"""
    return send_from_directory('static', 'browser_test.html')

@app.route('/video/<filename>')
def serve_video_universal(filename):
    """通用视频服务（支持所有浏览器）"""
    try:
        file_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
        if not os.path.exists(file_path):
            return jsonify({'error': '视频文件不存在'}), 404
        
        # 获取文件大小
        file_size = os.path.getsize(file_path)
        
        # 检查Range请求
        range_header = request.headers.get('Range')
        if range_header:
            # 处理范围请求
            byte_start = 0
            byte_end = file_size - 1
            
            if range_header.startswith('bytes='):
                range_match = range_header[6:].split('-')
                if range_match[0]:
                    byte_start = int(range_match[0])
                if range_match[1]:
                    byte_end = int(range_match[1])
            
            # 确保范围有效
            byte_start = max(0, byte_start)
            byte_end = min(file_size - 1, byte_end)
            
            if byte_start > byte_end:
                return jsonify({'error': 'Invalid range'}), 416
            
            content_length = byte_end - byte_start + 1
            
            def generate():
                with open(file_path, 'rb') as f:
                    f.seek(byte_start)
                    remaining = content_length
                    while remaining > 0:
                        chunk_size = min(1024 * 1024, remaining)  # 1MB chunks
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        remaining -= len(chunk)
                        yield chunk
            
            response = app.response_class(
                generate(),
                206,  # Partial Content
                {
                    'Content-Type': 'video/mp4',
                    'Content-Length': str(content_length),
                    'Content-Range': f'bytes {byte_start}-{byte_end}/{file_size}',
                    'Accept-Ranges': 'bytes',
                    'Cache-Control': 'public, max-age=3600',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
                    'Access-Control-Allow-Headers': 'Range, Content-Range',
                    'X-Content-Type-Options': 'nosniff',
                }
            )
            return response
        else:
            # 完整文件请求
            def generate():
                with open(file_path, 'rb') as f:
                    while True:
                        chunk = f.read(1024 * 1024)  # 1MB chunks
                        if not chunk:
                            break
                        yield chunk
            
            response = app.response_class(
                generate(),
                200,
                {
                    'Content-Type': 'video/mp4',
                    'Content-Length': str(file_size),
                    'Accept-Ranges': 'bytes',
                    'Cache-Control': 'public, max-age=3600',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
                    'Access-Control-Allow-Headers': 'Range, Content-Range',
                    'X-Content-Type-Options': 'nosniff',
                }
            )
            return response
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/<path:filename>')
def static_files(filename):
    """静态文件服务"""
    return send_from_directory('static', filename)

@app.route('/videos/<path:filename>')
def serve_video(filename):
    """直接服务视频文件"""
    try:
        # 检查文件是否存在
        file_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
        if not os.path.exists(file_path):
            return jsonify({'error': '视频文件不存在'}), 404
        
        # 获取文件大小
        file_size = os.path.getsize(file_path)
        
        # 检查Range请求
        range_header = request.headers.get('Range')
        if range_header:
            # 处理范围请求
            byte_start = 0
            byte_end = file_size - 1
            
            if range_header.startswith('bytes='):
                range_match = range_header[6:].split('-')
                if range_match[0]:
                    byte_start = int(range_match[0])
                if range_match[1]:
                    byte_end = int(range_match[1])
            
            # 确保范围有效
            byte_start = max(0, byte_start)
            byte_end = min(file_size - 1, byte_end)
            
            if byte_start > byte_end:
                return jsonify({'error': 'Invalid range'}), 416
            
            content_length = byte_end - byte_start + 1
            
            def generate():
                with open(file_path, 'rb') as f:
                    f.seek(byte_start)
                    remaining = content_length
                    while remaining > 0:
                        chunk_size = min(1024 * 1024, remaining)  # 1MB chunks for better performance
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        remaining -= len(chunk)
                        yield chunk
            
            response = app.response_class(
                generate(),
                206,  # Partial Content
                {
                    'Content-Type': 'video/mp4',
                    'Content-Length': str(content_length),
                    'Content-Range': f'bytes {byte_start}-{byte_end}/{file_size}',
                    'Accept-Ranges': 'bytes',
                    'Cache-Control': 'public, max-age=3600',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
                    'Access-Control-Allow-Headers': 'Range, Content-Range',
                    'X-Content-Type-Options': 'nosniff',
                }
            )
            return response
        else:
            # 完整文件请求
            def generate():
                with open(file_path, 'rb') as f:
                    while True:
                        chunk = f.read(8192)  # 8KB chunks
                        if not chunk:
                            break
                        yield chunk
            
            response = app.response_class(
                generate(),
                200,
                {
                    'Content-Type': 'video/mp4',
                    'Content-Length': str(file_size),
                    'Accept-Ranges': 'bytes',
                    'Cache-Control': 'public, max-age=3600',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
                    'Access-Control-Allow-Headers': 'Range, Content-Range',
                    'X-Content-Type-Options': 'nosniff',
                }
            )
            return response
            
    except Exception as e:
        print(f"视频服务错误: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """视频上传接口"""
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': '没有文件被上传'})
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'success': False, 'error': '没有选择文件'})
        
        if file and allowed_file(file.filename):
            # 保存原始文件名（包含中文字符）
            original_filename = file.filename
            
            # 生成安全的文件名用于存储
            filename = secure_filename(file.filename)
            # 添加时间戳避免重名
            timestamp = int(time.time())
            name, ext = os.path.splitext(filename)
            filename = f"{name}_{timestamp}{ext}"
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # 生成分析ID
            analysis_id = str(uuid.uuid4())
            
            return jsonify({
                'success': True,
                'filename': filename,
                'original_filename': original_filename,
                'analysis_id': analysis_id,
                'filepath': filepath
            })
        else:
            return jsonify({'success': False, 'error': '不支持的文件格式'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/analyze', methods=['POST'])
def analyze_video():
    """视频分析接口"""
    print("=== 收到分析请求 ===")
    try:
        data = request.get_json()
        print(f"接收到的数据: {data}")
        
        filename = data.get('filename')
        original_filename = data.get('original_filename')
        analysis_id = data.get('analysis_id')  # 使用前端传递的analysis_id
        
        print(f"文件名: {filename}")
        print(f"原始文件名: {original_filename}")
        print(f"分析ID: {analysis_id}")
        print(f"数据类型: filename={type(filename)}, original_filename={type(original_filename)}, analysis_id={type(analysis_id)}")
        
        if not filename:
            print("错误: 缺少文件名")
            return jsonify({'success': False, 'error': '缺少文件名'})
        
        if not analysis_id:
            print("错误: 缺少分析ID")
            return jsonify({'success': False, 'error': '缺少分析ID'})
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f"文件路径: {filepath}")
        if not os.path.exists(filepath):
            print("错误: 文件不存在")
            return jsonify({'success': False, 'error': '文件不存在'})
        
        print("文件存在，开始分析...")
        
        # 重置分析状态
        global analysis_status
        analysis_status = {
            'status': 'processing',
            'is_processing': True,
            'progress': 0,
            'current_frame': 0,
            'total_frames': 0,
            'message': '开始分析...'
        }
        
        # 在后台线程中执行分析，传递原始文件名
        print("启动后台分析线程...")
        import threading
        thread = threading.Thread(
            target=run_analysis,
            args=(analysis_id, filepath, original_filename)
        )
        thread.daemon = True
        thread.start()
        print(f"后台分析线程已启动: {analysis_id}")
        
        return jsonify({
            'success': True,
            'analysis_id': analysis_id,
            'message': '分析已开始'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/status')
def get_status():
    """获取分析状态"""
    return jsonify(analysis_status)

@app.route('/api/status/<analysis_id>')
def get_task_status(analysis_id):
    """获取指定任务的状态"""
    global task_status
    if analysis_id in task_status:
        return jsonify(task_status[analysis_id])
    else:
        return jsonify({'error': '任务不存在'}), 404

@app.route('/api/pause/<analysis_id>', methods=['POST'])
def pause_analysis(analysis_id):
    """暂停分析"""
    global pause_requests
    try:
        pause_requests[analysis_id] = True
        print(f"暂停分析请求: {analysis_id}")
        return jsonify({'success': True, 'message': '分析已暂停'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/resume/<analysis_id>', methods=['POST'])
def resume_analysis(analysis_id):
    """继续分析"""
    global pause_requests
    try:
        pause_requests[analysis_id] = False
        print(f"继续分析请求: {analysis_id}")
        return jsonify({'success': True, 'message': '分析已继续'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stop/<analysis_id>', methods=['POST'])
def stop_analysis(analysis_id):
    """终止分析"""
    global pause_requests, task_status
    try:
        # 设置终止标志
        pause_requests[analysis_id] = 'stop'
        
        # 更新任务状态为已终止
        if analysis_id in task_status:
            task_status[analysis_id].update({
                'status': 'stopped',
                'is_processing': False,
                'message': '分析已终止'
            })
        
        print(f"终止分析请求: {analysis_id}")
        return jsonify({'success': True, 'message': '分析已终止'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/result/<analysis_id>')
def get_result(analysis_id):
    """获取分析结果"""
    try:
        print(f"=== 获取分析结果: {analysis_id} ===")
        
        # 优先从数据库获取
        if db_manager:
            print(f"从数据库获取结果: {analysis_id}")
            result = db_manager.get_analysis_by_id(analysis_id)
            if result:
                print(f"✓ 数据库中找到结果: {analysis_id}")
                return jsonify(result)
            else:
                print(f"✗ 数据库中未找到结果: {analysis_id}")
        else:
            print("✗ 数据库管理器未初始化")
        
        # 如果数据库中没有，尝试从文件获取（向后兼容）
        result_file = os.path.join(app.config['RESULTS_FOLDER'], f"{analysis_id}.json")
        print(f"尝试从文件获取结果: {result_file}")
        if os.path.exists(result_file):
            print(f"✓ 找到结果文件: {result_file}")
            with open(result_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
            return jsonify(result)
        else:
            print(f"✗ 结果文件不存在: {result_file}")
        
        print(f"✗ 未找到分析结果: {analysis_id}")
        return jsonify({'error': '结果不存在'}), 404
    
    except Exception as e:
        print(f"✗ 获取分析结果异常: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/video/<filename>')
def get_video(filename):
    """获取视频文件"""
    try:
        response = send_from_directory(app.config['UPLOAD_FOLDER'], filename)
        response.headers['Content-Type'] = 'video/mp4'
        response.headers['Accept-Ranges'] = 'bytes'
        return response
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/results/<filename>', methods=['GET', 'HEAD', 'OPTIONS'])
def serve_result_video_direct(filename):
    """直接服务结果视频文件（使用统一流式服务）"""
    # 处理OPTIONS请求
    if request.method == 'OPTIONS':
        response = app.response_class()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, HEAD, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Range, Content-Range'
        return response
    
    # 直接调用统一的视频服务
    return serve_video_universal(filename)

@app.route('/api/video/results/<filename>')
def get_result_video(filename):
    """获取结果视频（支持 mp4 和 avi 格式）"""
    file_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
    if not os.path.exists(file_path):
        return jsonify({'error': '视频文件不存在'}), 404
    
    # 根据文件扩展名设置 mimetype
    if filename.endswith('.avi'):
        mimetype = 'video/x-msvideo'
    else:
        mimetype = 'video/mp4'
    
    # 返回完整文件，不支持 Range 请求（避免 AVI 格式问题）
    return send_file(
        file_path, 
        mimetype=mimetype,
        as_attachment=False,
        download_name=filename,
        conditional=False  # 禁用条件请求
    )

@app.route('/api/history')
def get_history():
    """获取分析历史记录"""
    try:
        if not db_manager:
            return jsonify({'error': '数据库未初始化'}), 500
        
        # 获取查询参数
        days = request.args.get('days', None, type=int)  # 天数过滤参数
        page = request.args.get('page', 1, type=int)  # 页码，默认第1页
        per_page = request.args.get('per_page', 10, type=int)  # 每页条数，默认10条
        
        print(f"=== 历史记录API调用 ===")
        print(f"days: {days}, page: {page}, per_page: {per_page}")
        
        # 如果指定了天数，获取最近N天的记录
        if days:
            history, total = db_manager.get_analysis_history_by_days_paginated(days, page, per_page)
        else:
            # 获取所有记录
            history, total = db_manager.get_analysis_history_all_paginated(page, per_page)
        
        print(f"返回记录数: {len(history)}, 总记录数: {total}")
        if history:
            print(f"第一条记录ID: {history[0].get('analysis_id', 'N/A')}")
            print(f"第一条记录时间: {history[0].get('created_at', 'N/A')}")
            print(f"最后一条记录ID: {history[-1].get('analysis_id', 'N/A')}")
            print(f"最后一条记录时间: {history[-1].get('created_at', 'N/A')}")
        
        return jsonify({
            'success': True,
            'data': history,
            'total': total,
            'page': page,
            'per_page': per_page,
            'total_pages': (total + per_page - 1) // per_page
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/history/<analysis_id>')
def get_history_detail(analysis_id):
    """获取历史记录详情"""
    try:
        if not db_manager:
            return jsonify({'error': '数据库未初始化'}), 500
        
        result = db_manager.get_analysis_by_id(analysis_id)
        if not result:
            return jsonify({'error': '记录不存在'}), 404
        
        return jsonify({
            'success': True,
            'data': result
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/history/<analysis_id>', methods=['DELETE'])
def delete_history(analysis_id):
    """删除历史记录"""
    try:
        if not db_manager:
            return jsonify({'error': '数据库未初始化'}), 500
        
        success = db_manager.delete_analysis(analysis_id)
        if success:
            return jsonify({'success': True, 'message': '删除成功'})
        else:
            return jsonify({'success': False, 'error': '删除失败'}), 500
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/statistics')
def get_statistics():
    """获取统计信息"""
    try:
        if not db_manager:
            return jsonify({'error': '数据库未初始化'}), 500
        
        stats = db_manager.get_statistics()
        return jsonify({
            'success': True,
            'data': stats
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/tracks/<analysis_id>')
def get_tracks(analysis_id):
    """获取指定分析的所有追踪ID列表"""
    try:
        if not db_manager:
            return jsonify({'error': '数据库未初始化'}), 500
        
        result = db_manager.get_analysis_by_id(analysis_id)
        if not result:
            return jsonify({'error': '分析记录不存在'}), 404
        
        # 从行为分析数据中提取所有追踪ID
        behavior_analysis = result.get('behavior_analysis', {})
        track_behaviors = behavior_analysis.get('track_behaviors', {})
        
        # 按总时间排序
        sorted_tracks = sorted(
            track_behaviors.items(),
            key=lambda x: x[1]['total_time'],
            reverse=True
        )
        
        tracks = []
        for track_id, behavior in sorted_tracks:
            # 确保track_id是整数
            try:
                track_id_int = int(track_id)
            except (ValueError, TypeError):
                track_id_int = track_id
                
            tracks.append({
                'track_id': track_id_int,
                'total_time': float(behavior['total_time']),
                'value_ratio': float(behavior['value_ratio']),
                'non_value_ratio': float(behavior['non_value_ratio']),
                'walking_ratio': float(behavior['walking_ratio']),
                'waiting_ratio': float(behavior['waiting_ratio'])
            })
        
        return jsonify({
            'success': True,
            'data': {
                'tracks': tracks,
                'total_count': len(tracks)
            }
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analysis/<analysis_id>')
def get_analysis_detail(analysis_id):
    """获取分析详情（用于批量分析）"""
    try:
        if not db_manager:
            return jsonify({'error': '数据库未初始化'}), 500
        
        result = db_manager.get_analysis_by_id(analysis_id)
        if not result:
            return jsonify({'error': '分析记录不存在'}), 404
        
        # 提取追踪数据用于批量分析
        tracking_data = result.get('tracking_data', {})
        tracks = []
        
        for track_id, track_data in tracking_data.items():
            tracks.append({
                'track_id': int(track_id) if str(track_id).isdigit() else track_id,
                'frames': track_data.get('frames', []),
                'bboxes': track_data.get('bboxes', []),
                'class_ids': track_data.get('class_ids', [])
            })
        
        return jsonify({
            'success': True,
            'analysis_id': analysis_id,
            'filename': result.get('original_filename', ''),
            'tracks': tracks,
            'video_info': result.get('video_info', {}),
            'created_at': result.get('created_at', '')
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/batch-analysis', methods=['POST'])
def batch_analysis():
    """批量分析接口 - 计算线平衡率"""
    try:
        if not db_manager:
            return jsonify({'error': '数据库未初始化'}), 500
        
        data = request.get_json()
        analysis_ids = data.get('analysis_ids', [])
        
        if not analysis_ids:
            return jsonify({'error': '请提供分析ID列表'}), 400
        
        print(f"开始批量分析，分析ID: {analysis_ids}")
        
        # 获取所有分析数据
        analysis_results = []
        failed_ids = []
        
        for analysis_id in analysis_ids:
            try:
                print(f"正在获取分析数据: {analysis_id}")
                result = db_manager.get_analysis_by_id(analysis_id)
                if result:
                    analysis_results.append({
                        'analysis_id': analysis_id,
                        'filename': result.get('original_filename', ''),
                        'tracks': result.get('tracking_data', {}),
                        'video_info': result.get('video_info', {})
                    })
                    print(f"成功获取分析数据: {analysis_id}")
                else:
                    print(f"未找到分析数据: {analysis_id}")
                    failed_ids.append(analysis_id)
            except Exception as e:
                print(f"获取分析数据失败 {analysis_id}: {e}")
                failed_ids.append(analysis_id)
        
        if failed_ids:
            print(f"以下分析ID获取失败: {failed_ids}")
        
        if not analysis_results:
            return jsonify({
                'success': False, 
                'error': '没有可用的分析数据，请检查分析ID是否正确'
            }), 400
        
        print(f"获取到 {len(analysis_results)} 个分析结果")
        
        # 计算线平衡率
        balance_result = calculate_line_balance(analysis_results)
        
        return jsonify({
            'success': True,
            'data': balance_result
        })
    
    except Exception as e:
        print(f"批量分析失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

def calculate_line_balance(analysis_results):
    """计算线平衡率"""
    workstations = []  # 所有工位数据
    video_details = []  # 每个视频的详细信息
    
    # 遍历每个视频（每个视频代表一个工位的完整工作周期）
    for i, analysis in enumerate(analysis_results):
        print(f"处理分析 {i+1}/{len(analysis_results)}: {analysis['analysis_id']}")
        
        tracking_data = analysis['tracks']
        video_info = analysis['video_info']
        
        # 计算每个ID的总出现时间（帧数）
        track_times = {}
        for track_id, track_data in tracking_data.items():
            frames = track_data.get('frames', [])
            track_times[track_id] = len(frames)
        
        # 找到当前视频出现时间最长的ID（该工位的工作人员）
        max_frames = 0
        max_time_id = None
        for track_id, frames in track_times.items():
            if frames > max_frames:
                max_frames = frames
                max_time_id = track_id
        
        # 获取视频实际时长（整个工作周期时长）
        video_duration = video_info.get('duration', 0)
        total_frames = video_info.get('total_frames', max_frames)
        
        print(f"视频 {analysis['filename']} 信息: duration={video_duration}, total_frames={total_frames}")
        
        # 工位工作周期时长 = 视频时长
        workstation_cycle_time = video_duration
        
        # 计算该工位工作人员的各类时间
        value_frames = 0
        walking_frames = 0
        waiting_frames = 0
        non_value_frames = 0
        
        if max_time_id and max_time_id in tracking_data:
            track_data = tracking_data[max_time_id]
            class_ids = track_data.get('class_ids', [])
            
            for class_id in class_ids:
                if class_id == 0:  # Value_Action (增值操作)
                    value_frames += 1
                elif class_id == 1:  # Walking (步行)
                    walking_frames += 1
                elif class_id == 2:  # Waiting (等待)
                    waiting_frames += 1
                elif class_id == 4:  # Non_Value_Action (非增值)
                    non_value_frames += 1
        
        # 按比例计算各类时间
        value_time = (value_frames / total_frames) * video_duration if total_frames > 0 else 0
        walking_time = (walking_frames / total_frames) * video_duration if total_frames > 0 else 0
        waiting_time = (waiting_frames / total_frames) * video_duration if total_frames > 0 else 0
        non_value_time = (non_value_frames / total_frames) * video_duration if total_frames > 0 else 0
        
        print(f"工位 {max_time_id} 计算结果: value_time={value_time:.2f}, walking_time={walking_time:.2f}")
        
        # 为每个工位分配唯一的工位ID（基于视频索引）
        workstation_id = i + 1  # 工位ID从1开始
        
        # 保存工位数据
        if max_time_id:
            workstations.append({
                'track_id': workstation_id,  # 使用唯一的工位ID
                'cycle_time': workstation_cycle_time,
                'value_time': value_time,
                'walking_time': walking_time,
                'waiting_time': waiting_time,
                'non_value_time': non_value_time,
                'filename': analysis['filename'],
                'analysis_id': analysis['analysis_id'],
                'total_frames': total_frames,
                'value_frames': value_frames,
                'walking_frames': walking_frames,
                'waiting_frames': waiting_frames,
                'non_value_frames': non_value_frames
            })
        
        # 保存每个视频的详细信息
        video_details.append({
            'filename': analysis['filename'],
            'analysis_id': analysis['analysis_id'],
            'workstation_id': workstation_id,  # 使用唯一的工位ID
            'cycle_time': workstation_cycle_time,
            'value_time': value_time,
            'walking_time': walking_time,
            'waiting_time': waiting_time,
            'non_value_time': non_value_time,
            'efficiency': (value_time / workstation_cycle_time * 100) if workstation_cycle_time > 0 else 0,
            'video_duration': video_duration,
            'total_frames': total_frames,
            'value_frames': value_frames,
            'walking_frames': walking_frames,
            'waiting_frames': waiting_frames,
            'non_value_frames': non_value_frames
        })
    
    # 找出所有工位中增值时间（作业时间）最长的是瓶颈工位
    max_value_time = 0
    bottleneck_id = None
    for workstation in workstations:
        if workstation['value_time'] > max_value_time:
            max_value_time = workstation['value_time']
            bottleneck_id = workstation['track_id']
    
    # 计算线平衡率
    total_value_time = sum(w['value_time'] for w in workstations)
    workstation_count = len(workstations)
    
    balance_rate = (total_value_time / (max_value_time * workstation_count) * 100) if (workstation_count > 0 and max_value_time > 0) else 0
    
    print(f"线平衡率计算结果: balance_rate={balance_rate:.2f}%, total_value_time={total_value_time:.2f}, bottleneck_value_time={max_value_time:.2f}")
    print(f"瓶颈工位ID: {bottleneck_id}, 工位数量: {workstation_count}")
    
    result = {
        'global_result': {
            'balance_rate': round(balance_rate, 2),
            'total_value_time': round(total_value_time, 2),
            'workstation_count': workstation_count,
            'bottleneck_value_time': round(max_value_time, 2),
            'bottleneck_id': bottleneck_id,
            'total_cycle_time': round(sum(w['cycle_time'] for w in workstations), 2),
            'video_count': len(analysis_results)
        },
        'workstations': workstations,
        'video_details': video_details
    }
    
    print(f"返回结果: {result}")
    return result

@app.route('/api/track/<analysis_id>/<track_id>')
def get_track_detail(analysis_id, track_id):
    """获取指定追踪ID的详细信息"""
    try:
        if not db_manager:
            return jsonify({'error': '数据库未初始化'}), 500
        
        result = db_manager.get_analysis_by_id(analysis_id)
        if not result:
            return jsonify({'error': '分析记录不存在'}), 404
        
        print(f"获取追踪详情: analysis_id={analysis_id}, track_id={track_id}")
        print(f"分析结果键: {list(result.keys())}")
        
        # 从行为分析数据中获取指定ID的详细信息
        behavior_analysis = result.get('behavior_analysis', {})
        track_behaviors = behavior_analysis.get('track_behaviors', {})
        
        print(f"行为分析数据键: {list(behavior_analysis.keys())}")
        print(f"追踪行为数据键: {list(track_behaviors.keys())}")
        
        track_id_int = int(track_id)
        
        # 尝试不同的键格式查找
        behavior = None
        if track_id_int in track_behaviors:
            behavior = track_behaviors[track_id_int]
            print(f"找到行为数据 (int key): {track_id_int}")
        elif str(track_id_int) in track_behaviors:
            behavior = track_behaviors[str(track_id_int)]
            print(f"找到行为数据 (str key): {str(track_id_int)}")
        elif track_id in track_behaviors:
            behavior = track_behaviors[track_id]
            print(f"找到行为数据 (original key): {track_id}")
        else:
            print(f"未找到追踪ID {track_id} 的行为数据")
            print(f"可用的追踪ID: {list(track_behaviors.keys())}")
        
        if behavior is None:
            return jsonify({'error': f'追踪ID {track_id} 不存在，可用ID: {list(track_behaviors.keys())}'}), 404
        
        # 获取原始追踪数据
        tracking_data = result.get('tracking_data', {})
        track_data = tracking_data.get(str(track_id_int), {})
        
        track_detail = {
            'track_id': track_id_int,
            'total_time': behavior['total_time'],
            'value_time': behavior['value_time'],
            'non_value_time': behavior['non_value_time'],
            'walking_time': behavior['walking_time'],
            'waiting_time': behavior['waiting_time'],
            'value_ratio': behavior['value_ratio'],
            'non_value_ratio': behavior['non_value_ratio'],
            'walking_ratio': behavior['walking_ratio'],
            'waiting_ratio': behavior['waiting_ratio'],
            'frames': track_data.get('frames', []),
            'bboxes': track_data.get('bboxes', []),
            'class_ids': track_data.get('class_ids', [])
        }
        
        return jsonify({
            'success': True,
            'data': track_detail
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    import socket
    
    def get_local_ip():
        """获取本机局域网IP地址"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"
    
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

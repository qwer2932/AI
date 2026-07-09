#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API 蓝图 - 所有后端接口
"""

import os
import json
import uuid
import time
import threading
import shutil
import cv2
from flask import Blueprint, request, jsonify, send_from_directory, send_file, current_app, Response
from werkzeug.utils import secure_filename

# 导入 core.state 模块（不要直接导入变量）
import core.state
from service.analysis_service import run_analysis, update_progress, init_tracking_system
from service.balance_service import calculate_line_balance
from core.utils import allowed_file

bp = Blueprint('api', __name__, url_prefix='/api')


@bp.route('/upload', methods=['POST'])
def upload_video():
    """上传视频文件（单个）"""
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': '没有文件被上传'})
        file = request.files['video']
        if file.filename == '':
            return jsonify({'success': False, 'error': '没有选择文件'})
        if file and allowed_file(file.filename):
            original_filename = file.filename
            filename = secure_filename(file.filename)
            timestamp = int(time.time())
            name, ext = os.path.splitext(filename)
            filename = f"{name}_{timestamp}{ext}"
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
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


@bp.route('/analyze', methods=['POST'])
def analyze_video():
    """启动视频分析"""
    print("=== 收到分析请求 ===")
    try:
        data = request.get_json()
        filename = data.get('filename')
        original_filename = data.get('original_filename')
        analysis_id = data.get('analysis_id')
        if not filename or not analysis_id:
            return jsonify({'success': False, 'error': '缺少文件名或分析ID'})
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': '文件不存在'})
        # 更新全局状态（使用 core.state 访问）
        core.state.analysis_status.update({
            'status': 'processing',
            'is_processing': True,
            'progress': 0,
            'current_frame': 0,
            'total_frames': 0,
            'message': '开始分析...'
        })
        thread = threading.Thread(target=run_analysis, args=(analysis_id, filepath, original_filename))
        thread.daemon = True
        thread.start()
        return jsonify({'success': True, 'analysis_id': analysis_id, 'message': '分析已开始'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/status', methods=['GET'])
def get_status():
    """获取全局分析状态（兼容旧接口）"""
    return jsonify(core.state.analysis_status)


@bp.route('/status/<analysis_id>', methods=['GET'])
def get_task_status(analysis_id):
    """获取指定任务状态"""
    if analysis_id in core.state.task_status:
        return jsonify(core.state.task_status[analysis_id])
    else:
        return jsonify({'error': '任务不存在'}), 404


@bp.route('/pause/<analysis_id>', methods=['POST'])
def pause_analysis(analysis_id):
    """暂停分析"""
    core.state.pause_requests[analysis_id] = True
    return jsonify({'success': True, 'message': '分析已暂停'})


@bp.route('/resume/<analysis_id>', methods=['POST'])
def resume_analysis(analysis_id):
    """继续分析"""
    core.state.pause_requests[analysis_id] = False
    return jsonify({'success': True, 'message': '分析已继续'})


@bp.route('/stop/<analysis_id>', methods=['POST'])
def stop_analysis(analysis_id):
    """终止分析"""
    core.state.pause_requests[analysis_id] = 'stop'
    if analysis_id in core.state.task_status:
        core.state.task_status[analysis_id].update({
            'status': 'stopped',
            'is_processing': False,
            'message': '分析已终止'
        })
    return jsonify({'success': True, 'message': '分析已终止'})


@bp.route('/result/<analysis_id>', methods=['GET'])
def get_result(analysis_id):
    """获取单个分析结果（优先数据库，其次JSON文件）"""
    try:
        if core.state.db_manager:
            result = core.state.db_manager.get_analysis_by_id(analysis_id)
            if result:
                return jsonify(result)
        # fallback to file
        result_file = os.path.join(current_app.config['RESULTS_FOLDER'], f"{analysis_id}.json")
        if os.path.exists(result_file):
            with open(result_file, 'r', encoding='utf-8') as f:
                return jsonify(json.load(f))
        return jsonify({'error': '结果不存在'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/video/<filename>', methods=['GET'])
def get_video(filename):
    """获取原始上传视频"""
    try:
        response = send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)
        response.headers['Content-Type'] = 'video/mp4'
        response.headers['Accept-Ranges'] = 'bytes'
        return response
    except Exception as e:
        return jsonify({'error': str(e)})


# -------------------- 历史记录相关接口（需数据库支持）--------------------

@bp.route('/history', methods=['GET'])
def get_history():
    """获取历史记录列表（分页），支持日期筛选"""
    if not core.state.db_manager:
        return jsonify({
            'success': False,
            'error': '数据库未初始化，请检查MySQL服务',
            'data': [],
            'total': 0,
            'page': 1,
            'per_page': 10,
            'total_pages': 0
        }), 200
    try:
        days = request.args.get('days', type=int)
        date = request.args.get('date', type=str)
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        
        # 优先使用日期筛选
        if date:
            history, total = core.state.db_manager.get_analysis_history_by_date_paginated(date, page, per_page)
        elif days:
            history, total = core.state.db_manager.get_analysis_history_by_days_paginated(days, page, per_page)
        else:
            history, total = core.state.db_manager.get_analysis_history_all_paginated(page, per_page)
            
        return jsonify({
            'success': True,
            'data': history,
            'total': total,
            'page': page,
            'per_page': per_page,
            'total_pages': (total + per_page - 1) // per_page if total > 0 else 0
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'data': [],
            'total': 0,
            'page': 1,
            'per_page': 10,
            'total_pages': 0
        }), 200


@bp.route('/history/<analysis_id>', methods=['GET'])
def get_history_detail(analysis_id):
    """获取单条历史记录详情"""
    if not core.state.db_manager:
        return jsonify({'success': False, 'error': '数据库未初始化'}), 200
    try:
        result = core.state.db_manager.get_analysis_by_id(analysis_id)
        if not result:
            return jsonify({'success': False, 'error': '记录不存在'}), 404
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 200


def _delete_result_file(analysis_id: str) -> int:
    """删除 results 目录下指定分析ID的相关文件（JSON + tracked 视频）。

    Returns:
        int: 删除成功的文件数量
    """
    deleted_count = 0
    try:
        results_folder = current_app.config['RESULTS_FOLDER']
        candidates = [
            os.path.join(results_folder, f"{analysis_id}.json"),
            os.path.join(results_folder, f"{analysis_id}_tracked.mp4")
        ]
        for result_file in candidates:
            if os.path.exists(result_file):
                try:
                    os.remove(result_file)
                    deleted_count += 1
                except Exception as e:
                    print(f"删除结果文件失败 {result_file}: {e}")
    except Exception as e:
        print(f"删除结果文件失败 {analysis_id}: {e}")
    return deleted_count


@bp.route('/history/<analysis_id>', methods=['DELETE'])
def delete_history(analysis_id):
    """删除历史记录"""
    try:
        if core.state.db_manager and not core.state.db_manager.delete_analysis(analysis_id):
            return jsonify({'success': False, 'error': '删除失败'}), 200

        deleted_files = _delete_result_file(analysis_id)
        return jsonify({'success': True, 'message': f'删除成功，已移除 {deleted_files} 个结果文件'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 200


@bp.route('/history/batch', methods=['DELETE'])
def batch_delete_history():
    """批量删除历史记录"""
    try:
        data = request.get_json(silent=True) or {}
        analysis_ids = data.get('analysis_ids', [])
        if not isinstance(analysis_ids, list) or len(analysis_ids) == 0:
            return jsonify({'success': False, 'error': '请提供分析ID列表'}), 400

        if core.state.db_manager:
            for analysis_id in analysis_ids:
                try:
                    core.state.db_manager.delete_analysis(analysis_id)
                except Exception:
                    pass

        deleted_count = 0
        for analysis_id in analysis_ids:
            deleted_count += _delete_result_file(analysis_id)

        return jsonify({'success': True, 'message': f'已删除 {len(analysis_ids)} 条记录，其中 {deleted_count} 个结果文件已移除'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 200


@bp.route('/history/all', methods=['DELETE'])
def delete_all_history():
    """删除全部历史记录，并清空结果文件夹内容"""
    try:
        results_folder = current_app.config['RESULTS_FOLDER']
        if os.path.exists(results_folder):
            for entry in os.listdir(results_folder):
                entry_path = os.path.join(results_folder, entry)
                if os.path.isdir(entry_path) and not os.path.islink(entry_path):
                    shutil.rmtree(entry_path)
                else:
                    os.remove(entry_path)

        if core.state.db_manager:
            db_deleted = core.state.db_manager.delete_all_analysis()
            if not db_deleted:
                return jsonify({'success': False, 'error': '删除失败'}), 200

        return jsonify({'success': True, 'message': '已删除全部历史记录并清空结果文件夹'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 200


@bp.route('/statistics', methods=['GET'])
def get_statistics():
    """获取统计信息"""
    if not core.state.db_manager:
        return jsonify({
            'success': False,
            'error': '数据库未初始化',
            'data': {
                'total_analyses': 0,
                'total_tracks': 0,
                'last_analysis': None,
                'daily_stats': []
            }
        }), 200
    try:
        stats = core.state.db_manager.get_statistics()
        return jsonify({'success': True, 'data': stats})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 200


@bp.route('/tracks/<analysis_id>', methods=['GET'])
def get_tracks(analysis_id):
    """获取指定分析的所有追踪ID列表"""
    if not core.state.db_manager:
        return jsonify({'success': False, 'error': '数据库未初始化'}), 200
    try:
        result = core.state.db_manager.get_analysis_by_id(analysis_id)
        if not result:
            return jsonify({'success': False, 'error': '分析记录不存在'}), 404
        behavior = result.get('behavior_analysis', {})
        track_behaviors = behavior.get('track_behaviors', {})
        tracks = []
        for tid, beh in sorted(track_behaviors.items(), key=lambda x: x[1]['total_time'], reverse=True):
            try:
                tid_int = int(tid)
            except:
                tid_int = tid
            tracks.append({
                'track_id': tid_int,
                'total_time': beh['total_time'],
                'value_ratio': beh.get('value_ratio', 0),
                'non_value_ratio': beh.get('non_value_ratio', 0),
                'walking_ratio': beh.get('walking_ratio', 0),
                'waiting_ratio': beh.get('waiting_ratio', 0)
            })
        return jsonify({'success': True, 'data': {'tracks': tracks, 'total_count': len(tracks)}})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 200


@bp.route('/analysis/<analysis_id>', methods=['GET'])
def get_analysis_detail(analysis_id):
    """获取分析详情（用于批量分析）"""
    if not core.state.db_manager:
        return jsonify({'success': False, 'error': '数据库未初始化'}), 200
    try:
        result = core.state.db_manager.get_analysis_by_id(analysis_id)
        if not result:
            return jsonify({'success': False, 'error': '分析记录不存在'}), 404
        tracking_data = result.get('tracking_data', {})
        tracks = []
        for tid, td in tracking_data.items():
            tracks.append({
                'track_id': int(tid) if str(tid).isdigit() else tid,
                'frames': td.get('frames', []),
                'bboxes': td.get('bboxes', []),
                'class_ids': td.get('class_ids', [])
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
        return jsonify({'success': False, 'error': str(e)}), 200


@bp.route('/batch-analysis', methods=['POST'])
def batch_analysis():
    """批量分析线平衡率"""
    if not core.state.db_manager:
        return jsonify({'success': False, 'error': '数据库未初始化'}), 200
    try:
        data = request.get_json()
        analysis_ids = data.get('analysis_ids', [])
        if not analysis_ids:
            return jsonify({'success': False, 'error': '请提供分析ID列表'}), 400
        analysis_results = []
        failed = []
        for aid in analysis_ids:
            try:
                res = core.state.db_manager.get_analysis_by_id(aid)
                if res:
                    analysis_results.append({
                        'analysis_id': aid,
                        'filename': res.get('original_filename', ''),
                        'video_info': res.get('video_info', {}),
                        'behavior_analysis': res.get('behavior_analysis', {})
                    })
                else:
                    failed.append(aid)
            except Exception as e:
                failed.append(aid)
        if not analysis_results:
            return jsonify({'success': False, 'error': '没有可用的分析数据'}), 400
        balance_result = calculate_line_balance(analysis_results)
        return jsonify({'success': True, 'data': balance_result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 200


@bp.route('/track/<analysis_id>/<track_id>', methods=['GET'])
def get_track_detail(analysis_id, track_id):
    """获取指定追踪ID的详细信息（行为比例）"""
    if not core.state.db_manager:
        return jsonify({'success': False, 'error': '数据库未初始化'}), 200
    try:
        result = core.state.db_manager.get_analysis_by_id(analysis_id)
        if not result:
            return jsonify({'success': False, 'error': '分析记录不存在'}), 404
        behavior = result.get('behavior_analysis', {})
        track_behaviors = behavior.get('track_behaviors', {})
        tid_int = int(track_id)
        beh = None
        if tid_int in track_behaviors:
            beh = track_behaviors[tid_int]
        elif str(tid_int) in track_behaviors:
            beh = track_behaviors[str(tid_int)]
        if beh is None:
            return jsonify({'success': False, 'error': f'追踪ID {track_id} 不存在'}), 404
        tracking_data = result.get('tracking_data', {})
        td = tracking_data.get(str(tid_int), {})
        detail = {
            'track_id': tid_int,
            'total_time': beh['total_time'],
            'value_time': beh.get('value_time', 0),
            'non_value_time': beh.get('non_value_time', 0),
            'walking_time': beh.get('walking_time', 0),
            'waiting_time': beh.get('waiting_time', 0),
            'value_ratio': beh.get('value_ratio', 0),
            'non_value_ratio': beh.get('non_value_ratio', 0),
            'walking_ratio': beh.get('walking_ratio', 0),
            'waiting_ratio': beh.get('waiting_ratio', 0),
            'frames': td.get('frames', []),
            'bboxes': td.get('bboxes', []),
            'class_ids': td.get('class_ids', [])
        }
        return jsonify({'success': True, 'data': detail})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 200


@bp.route('/video/results/<filename>', methods=['GET'])
def get_result_video(filename):
    """返回处理后的视频文件（兼容流式）"""
    file_path = os.path.join(current_app.config['RESULTS_FOLDER'], filename)
    if not os.path.exists(file_path):
        return jsonify({'error': '视频文件不存在'}), 404
    # 根据扩展名设置MIME类型
    if filename.endswith('.avi'):
        mimetype = 'video/x-msvideo'
    else:
        mimetype = 'video/mp4'
    return send_file(file_path, mimetype=mimetype, as_attachment=False, download_name=filename, conditional=False)


# -------------------- RTSP视频流代理接口 --------------------

def gen_frames(rtsp_url):
    """生成RTSP视频流的帧数据"""
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"无法打开RTSP流: {rtsp_url}")
        return
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()


@bp.route('/rtsp/stream', methods=['GET'])
def rtsp_stream():
    """RTSP视频流代理 - 将RTSP流转换为MJPEG流供浏览器播放"""
    try:
        ip = request.args.get('ip')
        port = request.args.get('port', '554')
        username = request.args.get('username', 'admin')
        password = request.args.get('password', '')
        path = request.args.get('path', '/Streaming/Channels/101')
        
        if not ip:
            return jsonify({'error': '请提供摄像头IP地址'}), 400
        
        rtsp_url = f"rtsp://{username}:{password}@{ip}:{port}{path}"
        print(f"尝试连接RTSP流: rtsp://{username}:***@{ip}:{port}{path}")
        
        return Response(gen_frames(rtsp_url),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"RTSP流错误: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/rtsp/test', methods=['GET'])
def rtsp_test():
    """测试RTSP连接"""
    try:
        ip = request.args.get('ip')
        port = request.args.get('port', '554')
        username = request.args.get('username', 'admin')
        password = request.args.get('password', '')
        path = request.args.get('path', '/Streaming/Channels/101')
        
        if not ip:
            return jsonify({'success': False, 'error': '请提供摄像头IP地址'}), 400
        
        rtsp_url = f"rtsp://{username}:{password}@{ip}:{port}{path}"
        print(f"测试RTSP连接: rtsp://{username}:***@{ip}:{port}{path}")
        
        cap = cv2.VideoCapture(rtsp_url)
        success = cap.isOpened()
        
        if success:
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            return jsonify({
                'success': True,
                'message': 'RTSP连接成功',
                'video_info': {
                    'fps': fps,
                    'width': width,
                    'height': height
                }
            })
        else:
            cap.release()
            return jsonify({'success': False, 'error': '无法连接到RTSP流，请检查IP、端口和认证信息'}), 500
    except Exception as e:
        print(f"RTSP测试错误: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
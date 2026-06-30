import os
import json
import time
import threading
from datetime import datetime
from core.state import tracking_system, db_manager, analysis_status, task_status, pause_requests
from core.tracking_system import TrackingSystem
from models import DatabaseManager
from core.step_inference import StepInference
from config import Config

def init_tracking_system():
    global tracking_system, db_manager
    try:
        tracking_system = TrackingSystem(Config.MODEL_PATH)
        print("追踪系统初始化成功")
        try:
            db_manager = DatabaseManager(
                host=Config.DB_HOST,
                port=Config.DB_PORT,
                user=Config.DB_USER,
                password=Config.DB_PASSWORD,
                database=Config.DB_NAME
            )
            print("数据库系统初始化成功")
        except Exception as db_e:
            print(f"⚠ 数据库初始化失败: {db_e}")
            db_manager = None
    except Exception as e:
        print(f"系统初始化失败: {e}")
        import traceback
        traceback.print_exc()

def analyze_behavior(tracking_result, video_info, fps):
    """
    后置推理分析装配步骤，返回以秒为单位的统计
    """
    per_frame_detections = tracking_result.get('per_frame_detections', [])
    inference = StepInference(proximity_threshold=0.30, warmup_frames=30)
    for frame_data in per_frame_detections:
        inference.infer_step(
            frame_shape=(video_info.get('height', 1080),
                         video_info.get('width', 1920), 3),
            detections=frame_data['detections']
        )
    step_summary = inference.get_summary(fps=fps)  # 返回帧数
    # 转换为秒
    track_behaviors = {}
    for track_id, steps in step_summary.items():
        total_frames = steps.pop('_total', 0)
        total_time = total_frames / fps if fps > 0 else 0
        step_times = {k: v / fps if fps > 0 else 0 for k, v in steps.items()}
        track_behaviors[str(track_id)] = {
            'total_time': float(total_time),
            **step_times
        }
    # 排序取前3
    sorted_tracks = sorted(track_behaviors.items(), key=lambda x: x[1]['total_time'], reverse=True)[:3]
    return {
        'track_behaviors': track_behaviors,
        'top_tracks': [
            {**{'track_id': int(tid)}, **{k: float(v) for k, v in beh.items()}}
            for tid, beh in sorted_tracks
        ]
    }

def update_progress(current_frame, total_frames, message, analysis_id=None):
    progress = int((current_frame / total_frames) * 100) if total_frames > 0 else 0
    analysis_status.update({
        'progress': progress,
        'current_frame': current_frame,
        'total_frames': total_frames,
        'message': message
    })
    if analysis_id and analysis_id in task_status:
        task_status[analysis_id].update({
            'progress': progress,
            'current_frame': current_frame,
            'total_frames': total_frames,
            'message': message
        })

def run_analysis(analysis_id, filepath, original_filename=None):
    global analysis_status, tracking_system, db_manager, task_status, pause_requests
    print(f"=== 开始分析 {analysis_id} ===")
    try:
        if not tracking_system:
            init_tracking_system()
        task_status[analysis_id] = {
            'status': 'processing',
            'is_processing': True,
            'progress': 0,
            'current_frame': 0,
            'total_frames': 0,
            'message': '正在加载模型...'
        }
        analysis_status['status'] = 'processing'
        analysis_status['message'] = '正在加载模型...'

        def check_pause_or_stop():
            if analysis_id in pause_requests:
                status = pause_requests[analysis_id]
                if status == 'stop':
                    return 'stop'
                elif status == True:
                    return 'pause'
            return False

        def progress_callback(frame, total, msg, aid):
            status = check_pause_or_stop()
            if status == 'stop':
                raise Exception("分析已被用户终止")
            elif status == 'pause':
                while check_pause_or_stop() == 'pause':
                    time.sleep(0.1)
                print(f"分析 {analysis_id} 已恢复")
            update_progress(frame, total, msg, aid)

        result = tracking_system.analyze_video(filepath, analysis_id, progress_callback)
        fps = result['video_info']['fps']
        analysis_status['message'] = '正在分析行为数据...'
        behavior_result = analyze_behavior(result, result['video_info'], fps)

        if not original_filename:
            original_filename = os.path.basename(filepath)

        final_result = {
            **result,
            'behavior_analysis': behavior_result,
            'analysis_id': analysis_id,
            'timestamp': datetime.now().isoformat(),
            'filename': os.path.basename(filepath),
            'original_filename': original_filename
        }

        if db_manager:
            try:
                db_manager.save_analysis_result(final_result)
                print(f"✓ 分析结果已保存到数据库: {analysis_id}")
            except Exception as e:
                print(f"✗ 数据库保存异常: {e}")

        result_file = os.path.join('results', f"{analysis_id}.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)

        if analysis_id in task_status:
            task_status[analysis_id].update({
                'status': 'completed',
                'is_processing': False,
                'progress': 100,
                'message': '分析完成'
            })
        analysis_status.update({
            'status': 'completed',
            'is_processing': False,
            'progress': 100,
            'message': '分析完成'
        })
        print(f"=== 分析完成: {analysis_id} ===")

        def cleanup():
            time.sleep(30)
            if analysis_id in task_status:
                del task_status[analysis_id]
        threading.Thread(target=cleanup, daemon=True).start()

    except Exception as e:
        print(f"=== 分析失败: {e} ===")
        import traceback
        traceback.print_exc()
        if "分析已被用户终止" in str(e):
            if analysis_id in task_status:
                task_status[analysis_id].update({
                    'status': 'stopped',
                    'is_processing': False,
                    'message': '分析已终止'
                })
        else:
            if analysis_id in task_status:
                task_status[analysis_id].update({
                    'status': 'error',
                    'is_processing': False,
                    'progress': 0,
                    'message': f'分析失败: {str(e)}'
                })
        analysis_status.update({
            'status': 'error',
            'is_processing': False,
            'progress': 0,
            'message': f'分析失败: {str(e)}'
        })
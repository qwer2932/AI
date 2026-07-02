#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析服务 - 核心业务逻辑
"""

import os
import json
import time
import threading
from datetime import datetime

import core.state
from core.tracking_system import TrackingSystem
from models import DatabaseManager
from core.step_inference import StepInference
from config import Config


def init_tracking_system():
    """
    初始化追踪系统和数据库连接
    """
    try:
        # 初始化追踪系统
        core.state.tracking_system = TrackingSystem(Config.MODEL_PATH)
        print("追踪系统初始化成功")

        # 初始化数据库连接
        try:
            print(f"正在连接数据库: {Config.DB_HOST}:{Config.DB_PORT} 用户 {Config.DB_USER}")
            db = DatabaseManager(
                host=Config.DB_HOST,
                port=Config.DB_PORT,
                user=Config.DB_USER,
                password=Config.DB_PASSWORD,
                database=Config.DB_NAME
            )
            core.state.db_manager = db
            print("数据库系统初始化成功")
            print(f"db_manager 对象地址: {id(core.state.db_manager)}")
        except Exception as db_e:
            print(f"⚠ 数据库初始化失败: {db_e}")
            import traceback
            traceback.print_exc()
            core.state.db_manager = None

    except Exception as e:
        print(f"系统初始化失败: {e}")
        import traceback
        traceback.print_exc()


def extract_cycles_from_step_sequence(per_frame_step_maps, fps):
    """
    从每帧的步骤映射序列中提取所有循环。
    固定 RobotPick 为循环开始，RobotReturn 为循环结束。
    只有同时包含 RobotPick 且以 RobotReturn 结束的循环才标记为完整。
    缺失步骤时间记为0。
    """
    step_order = ["RobotPick", "Scan", "RobotFix", "HandTighten", "ElectricGun", "RobotReturn"]
    # 构建时间序列
    step_sequence = []
    for frame_idx, step_map in enumerate(per_frame_step_maps, start=1):
        step = None
        for track_id, s in step_map.items():
            if s is not None:
                step = s
                break
        if step:
            step_sequence.append((frame_idx, step))

    if not step_sequence:
        return []

    cycles = []
    current_cycle = None  # {start_frame, step_frames: {step: start_frame}}
    expected_idx = None

    for frame_idx, step in step_sequence:
        if step == "RobotPick":
            # 遇到 RobotPick，强制开始新循环
            if current_cycle is not None:
                # 保存当前循环为不完整（因为被新 RobotPick 打断）
                end_frame = frame_idx - 1
                step_durations = {}
                for s in step_order:
                    if s in current_cycle['step_frames']:
                        start = current_cycle['step_frames'][s]
                        next_s = step_order[(step_order.index(s) + 1) % len(step_order)]
                        if next_s in current_cycle['step_frames']:
                            end = current_cycle['step_frames'][next_s] - 1
                        else:
                            end = end_frame
                        duration = end - start + 1
                        step_durations[s] = duration
                    else:
                        step_durations[s] = 0
                total_frames = end_frame - current_cycle['start_frame'] + 1
                if total_frames > 0:
                    step_sec = {k: (v / fps if v is not None else 0.0) for k, v in step_durations.items()}
                    # 检查是否所有步骤时间都接近0
                    has_any_step = any(v > 0.1 for v in step_sec.values())
                    if has_any_step:
                        # 被新 RobotPick 打断，循环不完整
                        cycles.append({
                            'start_frame': current_cycle['start_frame'],
                            'end_frame': end_frame,
                            'total_frames': total_frames,
                            'total_time': total_frames / fps,
                            'steps': step_sec,
                            'complete': False   # 明确标记不完整
                        })
            # 开始新循环
            current_cycle = {
                'start_frame': frame_idx,
                'step_frames': {'RobotPick': frame_idx}
            }
            expected_idx = 1  # 期望 Scan
        else:
            if current_cycle is not None:
                if step == step_order[expected_idx]:
                    current_cycle['step_frames'][step] = frame_idx
                    if step == "RobotReturn":
                        # 到达 RobotReturn，完整循环结束
                        end_frame = frame_idx
                        step_durations = {}
                        for s in step_order:
                            if s in current_cycle['step_frames']:
                                start = current_cycle['step_frames'][s]
                                next_s = step_order[(step_order.index(s) + 1) % len(step_order)]
                                if next_s in current_cycle['step_frames']:
                                    end = current_cycle['step_frames'][next_s] - 1
                                else:
                                    end = end_frame
                                duration = end - start + 1
                                step_durations[s] = duration
                            else:
                                step_durations[s] = 0
                        total_frames = end_frame - current_cycle['start_frame'] + 1
                        step_sec = {k: (v / fps if v is not None else 0.0) for k, v in step_durations.items()}
                        has_any_step = any(v > 0.001 for v in step_sec.values())
                        if has_any_step:
                            cycles.append({
                                'start_frame': current_cycle['start_frame'],
                                'end_frame': end_frame,
                                'total_frames': total_frames,
                                'total_time': total_frames / fps,
                                'steps': step_sec,
                                'complete': True   # 到达 RobotReturn，完整
                            })
                        # 重置循环
                        current_cycle = None
                        expected_idx = None
                    else:
                        expected_idx = (expected_idx + 1) % len(step_order)
                else:
                    # 步骤不是期望的步骤，忽略（但如果是 RobotPick 已在上面处理）
                    pass
            else:
                # 没有活动循环，且当前不是 RobotPick
                # 说明上一个循环刚结束（或视频开始处），现在遇到非 RobotPick
                if step in step_order:
                    current_cycle = {
                        'start_frame': frame_idx,
                        'step_frames': {step: frame_idx}
                    }
                    idx = step_order.index(step)
                    expected_idx = (idx + 1) % len(step_order)
                else:
                    current_cycle = None
                    expected_idx = None

    # 视频结束，如果还有活动循环，保存为不完整
    if current_cycle is not None:
        end_frame = step_sequence[-1][0] if step_sequence else 0
        step_durations = {}
        for s in step_order:
            if s in current_cycle['step_frames']:
                start = current_cycle['step_frames'][s]
                next_s = step_order[(step_order.index(s) + 1) % len(step_order)]
                if next_s in current_cycle['step_frames']:
                    end = current_cycle['step_frames'][next_s] - 1
                else:
                    end = end_frame
                duration = end - start + 1
                step_durations[s] = duration
            else:
                step_durations[s] = 0
        total_frames = end_frame - current_cycle['start_frame'] + 1
        if total_frames > 0:
            step_sec = {k: (v / fps if v is not None else 0.0) for k, v in step_durations.items()}
            has_any_step = any(v > 0.001 for v in step_sec.values())
            if has_any_step:
                cycles.append({
                    'start_frame': current_cycle['start_frame'],
                    'end_frame': end_frame,
                    'total_frames': total_frames,
                    'total_time': total_frames / fps,
                    'steps': step_sec,
                    'complete': False   # 视频结束未到达 RobotReturn，不完整
                })

    return cycles


def analyze_behavior(tracking_result, video_info, fps):
    """
    后置推理分析装配步骤，返回以秒为单位的统计
    同时从 per_frame_step_maps 中提取循环数据
    """
    from core.step_inference import StepInference
    per_frame_detections = tracking_result.get('per_frame_detections', [])
    inference = StepInference(proximity_threshold=0.30, warmup_frames=30)

    for frame_data in per_frame_detections:
        inference.infer_step(
            frame_shape=(video_info.get('height', 1080),
                         video_info.get('width', 1920), 3),
            detections=frame_data['detections']
        )

    step_summary = inference.get_summary(fps=fps)
    track_behaviors = {}
    for track_id, steps in step_summary.items():
        total_frames = steps.pop('_total', 0)
        total_time = total_frames / fps if fps > 0 else 0
        step_times = {k: v / fps if fps > 0 else 0 for k, v in steps.items()}
        track_behaviors[str(track_id)] = {
            'total_time': float(total_time),
            **step_times
        }

    # 提取循环数据
    per_frame_step_maps = tracking_result.get('per_frame_step_maps', [])
    cycles = extract_cycles_from_step_sequence(per_frame_step_maps, fps) if per_frame_step_maps else []

    # 排序取前3（保留兼容）
    sorted_tracks = sorted(track_behaviors.items(), key=lambda x: x[1]['total_time'], reverse=True)[:3]
    return {
        'track_behaviors': track_behaviors,
        'top_tracks': [
            {**{'track_id': int(tid)}, **{k: float(v) for k, v in beh.items()}}
            for tid, beh in sorted_tracks
        ],
        'cycles': cycles,
        'cycle_data': cycles[0] if cycles else {}
    }


def update_progress(current_frame, total_frames, message, analysis_id=None):
    """更新全局进度"""
    progress = int((current_frame / total_frames) * 100) if total_frames > 0 else 0
    core.state.analysis_status.update({
        'progress': progress,
        'current_frame': current_frame,
        'total_frames': total_frames,
        'message': message
    })
    if analysis_id and analysis_id in core.state.task_status:
        core.state.task_status[analysis_id].update({
            'progress': progress,
            'current_frame': current_frame,
            'total_frames': total_frames,
            'message': message
        })


def run_analysis(analysis_id, filepath, original_filename=None):
    """
    后台运行完整分析流程
    """
    print(f"=== 开始分析 {analysis_id} ===")
    try:
        # 如果追踪系统未初始化，则初始化
        if core.state.tracking_system is None:
            init_tracking_system()

        # 初始化任务状态
        core.state.task_status[analysis_id] = {
            'status': 'processing',
            'is_processing': True,
            'progress': 0,
            'current_frame': 0,
            'total_frames': 0,
            'message': '正在加载模型...'
        }
        core.state.analysis_status['status'] = 'processing'
        core.state.analysis_status['message'] = '正在加载模型...'

        # 定义暂停/终止检查函数
        def check_pause_or_stop():
            if analysis_id in core.state.pause_requests:
                status = core.state.pause_requests[analysis_id]
                if status == 'stop':
                    return 'stop'
                elif status == True:
                    return 'pause'
            return False

        # 进度回调函数（包含暂停和终止检查）
        def progress_callback(frame, total, msg, aid):
            status = check_pause_or_stop()
            if status == 'stop':
                raise Exception("分析已被用户终止")
            elif status == 'pause':
                while check_pause_or_stop() == 'pause':
                    time.sleep(0.1)
                print(f"分析 {analysis_id} 已恢复")
            update_progress(frame, total, msg, aid)

        # 执行追踪分析
        result = core.state.tracking_system.analyze_video(filepath, analysis_id, progress_callback)
        fps = result['video_info']['fps']

        # 行为分析
        core.state.analysis_status['message'] = '正在分析行为数据...'
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

        # 保存到数据库（如果 db_manager 可用）
        if core.state.db_manager is not None:
            try:
                core.state.db_manager.save_analysis_result(final_result)
                print(f"✓ 分析结果已保存到数据库: {analysis_id}")
            except Exception as e:
                print(f"✗ 数据库保存异常: {e}")
        else:
            print("✗ db_manager 为 None，无法保存到数据库")

        # 保存 JSON 文件（备用）
        result_file = os.path.join('results', f"{analysis_id}.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)

        # 更新任务状态为完成
        if analysis_id in core.state.task_status:
            core.state.task_status[analysis_id].update({
                'status': 'completed',
                'is_processing': False,
                'progress': 100,
                'message': '分析完成'
            })
        core.state.analysis_status.update({
            'status': 'completed',
            'is_processing': False,
            'progress': 100,
            'message': '分析完成'
        })
        print(f"=== 分析完成: {analysis_id} ===")

        # 延迟清理任务状态
        def cleanup():
            time.sleep(30)
            if analysis_id in core.state.task_status:
                del core.state.task_status[analysis_id]
        threading.Thread(target=cleanup, daemon=True).start()

    except Exception as e:
        print(f"=== 分析失败: {e} ===")
        import traceback
        traceback.print_exc()

        if "分析已被用户终止" in str(e):
            if analysis_id in core.state.task_status:
                core.state.task_status[analysis_id].update({
                    'status': 'stopped',
                    'is_processing': False,
                    'message': '分析已终止'
                })
        else:
            if analysis_id in core.state.task_status:
                core.state.task_status[analysis_id].update({
                    'status': 'error',
                    'is_processing': False,
                    'progress': 0,
                    'message': f'分析失败: {str(e)}'
                })
        core.state.analysis_status.update({
            'status': 'error',
            'is_processing': False,
            'progress': 0,
            'message': f'分析失败: {str(e)}'
        })
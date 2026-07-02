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

# 导入 core.state 模块（不要直接导入变量）
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
    从每帧的步骤映射序列中提取所有循环（包括不完整的循环）。
    每帧 step_map 格式: {track_id: step_name} 或 {}
    由于只有一个实际工人，我们将所有track的步骤按时间顺序合并。
    循环定义：从 RobotPick 开始，到 RobotReturn 结束（如果视频结束前未完成，视为不完整循环）。
    """
    step_order = ["RobotPick", "Scan", "RobotFix", "HandTighten", "ElectricGun", "RobotReturn"]
    step_sequence = []  # 元素为 (帧索引, step_name)
    for frame_idx, step_map in enumerate(per_frame_step_maps, start=1):
        step = None
        for track_id, s in step_map.items():
            if s is not None:
                step = s
                break
        if step:
            step_sequence.append((frame_idx, step))

    cycles = []
    current_cycle = {'start_frame': None, 'end_frame': None, 'steps': {step: None for step in step_order}}
    expected_idx = 0
    cycle_start_frame = None
    step_start_frame = None

    for frame_idx, step in step_sequence:
        if step not in step_order:
            continue

        # 如果当前是期望的下一步
        if step == step_order[expected_idx]:
            if expected_idx == 0:
                # 开始新循环
                cycle_start_frame = frame_idx
                current_cycle['start_frame'] = cycle_start_frame
                current_cycle['steps'] = {s: None for s in step_order}
                step_start_frame = frame_idx
            else:
                # 计算上一个步骤的持续时间（帧数）
                prev_step = step_order[expected_idx - 1]
                if current_cycle['steps'][prev_step] is None:
                    current_cycle['steps'][prev_step] = frame_idx - step_start_frame

            step_start_frame = frame_idx

            # 如果已经是最后一个步骤（RobotReturn），则完成一个循环
            if expected_idx == len(step_order) - 1:
                # 最后一个步骤的持续时间
                current_cycle['steps'][step] = frame_idx - step_start_frame + 1  # 包含当前帧
                current_cycle['end_frame'] = frame_idx
                # 计算循环总帧数（跨度）
                total_frames = current_cycle['end_frame'] - current_cycle['start_frame'] + 1
                # 转换为秒
                step_sec = {k: (v / fps if v is not None else 0.0) for k, v in current_cycle['steps'].items()}
                cycles.append({
                    'start_frame': current_cycle['start_frame'],
                    'end_frame': current_cycle['end_frame'],
                    'total_frames': total_frames,
                    'total_time': total_frames / fps,
                    'steps': step_sec,
                    'complete': True
                })
                # 重置，准备下一个循环
                expected_idx = 0
                current_cycle = {'start_frame': None, 'end_frame': None, 'steps': {}}
                cycle_start_frame = None
                continue  # 当前帧已处理，跳过下面的更新

            expected_idx += 1

        else:
            # 如果步骤不匹配期望，但可能是噪声；如果是 RobotPick，则重新开始循环
            if step == step_order[0]:
                # 如果有未完成的循环（已记录了一些步骤），先保存为不完整循环
                if current_cycle['start_frame'] is not None and not current_cycle.get('complete', False):
                    # 结束当前不完整循环，以当前帧的前一帧作为结束
                    # 但更合理的是以前一个步骤结束帧作为结束，但这里简单处理，记录当前循环的起始和已记录步骤
                    # 为了简单，我们放弃之前未完成的循环，重新开始（因为新的RobotPick意味着前一个循环被丢弃）
                    pass  # 我们直接覆盖，因为可能旧循环不完整，以新的为准
                # 重新开始循环
                cycle_start_frame = frame_idx
                current_cycle = {
                    'start_frame': cycle_start_frame,
                    'end_frame': None,
                    'steps': {s: None for s in step_order},
                    'complete': False
                }
                step_start_frame = frame_idx
                expected_idx = 1  # 下一个期望是 Scan
                continue  # 当前帧作为 RobotPick 已处理，但我们需要记录当前步骤的时间，所以仍需处理，但为了逻辑清晰，用 continue 并手动设置

            # 其他不匹配步骤，可能是噪声，忽略

    # 循环结束后，如果有未完成的循环（已开始但未结束），也保存
    if current_cycle.get('start_frame') is not None and not current_cycle.get('complete', False):
        # 补充缺失步骤的时间为0，结束帧为最后一帧
        # 计算已记录步骤的时间，总跨度到视频结束
        last_frame = per_frame_step_maps[-1]  # 最后一个步骤的帧索引（但这里没有，我们使用当前循环中记录的最后帧）
        # 用当前循环中最后记录的步骤的帧作为结束（临时）
        # 简单处理：使用最后一个步骤的帧作为结束帧（如果有），否则使用 start_frame
        end_frame = frame_idx if 'frame_idx' in locals() else cycle_start_frame
        # 查找已记录步骤中最大的帧数（这里我们没有保存每步骤的帧，但可用步骤起始帧，但为了简便，我们把结束帧设为 start_frame + 一些假设）
        # 更合理的是取已记录步骤中的最大帧，但这里我们没有记录，所以用当前循环中最后一个步骤的帧（即上次记录的 step_start_frame）
        # 由于我们没有保存步骤帧，我们可以近似：如果已记录步骤，使用最后一个步骤的起始帧+持续时间
        # 简单起见，将未完成循环视为从 start_frame 到当前帧（视频最后处理的帧）
        end_frame = frame_idx if 'frame_idx' in locals() else cycle_start_frame
        # 更新 steps 中缺失的步骤为 None（表示未出现）
        # 计算总跨度
        total_frames = end_frame - current_cycle['start_frame'] + 1
        step_sec = {}
        for k in step_order:
            val = current_cycle['steps'].get(k, None)
            step_sec[k] = (val / fps if val is not None else 0.0)
        cycles.append({
            'start_frame': current_cycle['start_frame'],
            'end_frame': end_frame,
            'total_frames': total_frames,
            'total_time': total_frames / fps,
            'steps': step_sec,
            'complete': False
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

    step_summary = inference.get_summary(fps=fps)  # 返回帧数
    # 转换为秒（保留原始track统计，可能仍用于其他）
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
        'cycles': cycles,          # 所有提取到的循环列表
        'cycle_data': cycles[0] if cycles else {}   # 第一个循环（兼容旧前端）
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
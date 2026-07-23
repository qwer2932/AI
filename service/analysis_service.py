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
from collections import defaultdict

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
        # 初始化追踪系统（进一步降低置信度阈值以检测更多物体）
        core.state.tracking_system = TrackingSystem(Config.MODEL_PATH, conf_threshold=0.2, iou_threshold=0.45)
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
    从每帧步骤映射中提取装配循环。
    规则：
    1. 只有 RobotPick 才能开始一个新循环，但连续的 RobotPick 帧合并为同一个步骤，不重复创建循环。
    2. 没有活动循环时，所有非 RobotPick 步骤均被忽略。
    3. 有活动循环时，所有步骤（包括 RobotReturn）都被累积到当前循环中。
    4. 循环只在遇到一个新的 RobotPick（即当前不在 RobotPick 步骤时）或视频结束时保存。
    5. 循环完整性：包含全部 6 个步骤（均有有效帧）才标记为完整。
    """
    from collections import defaultdict

    step_order = ["RobotPick", "Scan", "RobotFix", "HandTighten", "ElectricGun", "RobotReturn"]
    TIMEOUT_FRAMES = int(fps * 2)  # 2秒超时，用于合并连续相同步骤

    cycles = []
    total_frames = len(per_frame_step_maps)

    # 当前循环状态
    cycle_start = None
    cycle_step_counts = defaultdict(int)  # 已累积的步骤帧数
    current_step = None        # 当前正在累积的步骤名
    step_start = None          # 当前步骤起始帧
    step_last_seen = None      # 当前步骤最后出现帧

    def close_step():
        """结束当前步骤，将帧数累加到循环"""
        nonlocal current_step, step_start, step_last_seen, cycle_step_counts
        if current_step is not None and step_start is not None and step_last_seen is not None:
            duration = step_last_seen - step_start + 1
            if duration > 0:
                cycle_step_counts[current_step] += duration
        current_step = None
        step_start = None
        step_last_seen = None

    def save_cycle(end_frame, complete):
        """保存当前循环并重置"""
        nonlocal cycle_start, cycle_step_counts
        if cycle_start is not None:
            total_frames_cycle = end_frame - cycle_start + 1
            step_sec = {k: v / fps for k, v in cycle_step_counts.items()}
            has_any = any(v > 0.001 for v in step_sec.values())
            if has_any:
                cycles.append({
                    'start_frame': cycle_start,
                    'end_frame': end_frame,
                    'total_frames': total_frames_cycle,
                    'total_time': total_frames_cycle / fps,
                    'steps': step_sec,
                    'complete': complete
                })
            cycle_start = None
            cycle_step_counts = defaultdict(int)

    for frame_idx, step_map in enumerate(per_frame_step_maps, start=1):
        # 提取本帧步骤
        step = None
        if step_map:
            for s in step_map.values():
                if s is not None:
                    step = s
                    break

        # ---------- 超时检测：如果当前步骤超过 TIMEOUT_FRAMES 未出现，结束它 ----------
        if current_step is not None and step_last_seen is not None:
            if frame_idx - step_last_seen > TIMEOUT_FRAMES:
                close_step()

        # ---------- 处理 RobotPick ----------
        if step == "RobotPick":
            # 如果当前正在累积的步骤就是 RobotPick，则只是延续，不开启新循环
            if current_step == "RobotPick":
                step_last_seen = frame_idx
                continue
            # 否则，遇到新的 RobotPick，开始新循环
            # 结束当前步骤
            if current_step is not None:
                close_step()
            # 保存之前的循环（如果有）
            if cycle_start is not None:
                all_steps_present = all(s in cycle_step_counts for s in step_order)
                save_cycle(frame_idx - 1, complete=all_steps_present)
            # 开始新循环
            cycle_start = frame_idx
            cycle_step_counts = defaultdict(int)
            current_step = "RobotPick"
            step_start = frame_idx
            step_last_seen = frame_idx
            continue

        # ---------- 如果没有活动循环，忽略所有其他步骤 ----------
        if cycle_start is None:
            continue

        # ---------- 处理其他步骤（包括 RobotReturn） ----------
        if step is not None and step in step_order:
            # 如果当前正在累积的步骤与 step 相同，则更新最后出现帧
            if current_step == step:
                step_last_seen = frame_idx
            else:
                # 不同步骤，结束当前步骤，开始新步骤
                if current_step is not None:
                    close_step()
                current_step = step
                step_start = frame_idx
                step_last_seen = frame_idx
        # step 为 None，不处理

    # 视频结束，保存当前循环
    if current_step is not None:
        close_step()
    if cycle_start is not None:
        all_steps_present = all(s in cycle_step_counts for s in step_order)
        save_cycle(total_frames, complete=all_steps_present)

    return cycles


def analyze_behavior(tracking_result, video_info, fps):
    """
    直接从 per_frame_step_maps 统计步骤信息，不再重新运行推理。
    """
    from collections import defaultdict

    per_frame_step_maps = tracking_result.get('per_frame_step_maps', [])

    # 统计每个 track_id 的步骤出现帧数
    track_step_counts = defaultdict(lambda: defaultdict(int))
    for frame_idx, step_map in enumerate(per_frame_step_maps):
        if step_map:
            for track_id, step in step_map.items():
                if step is not None:
                    track_step_counts[track_id][step] += 1

    # 构建 track_behaviors
    track_behaviors = {}
    for track_id, step_counts in track_step_counts.items():
        total_frames = sum(step_counts.values())
        total_time = total_frames / fps if fps > 0 else 0
        step_times = {step: count / fps for step, count in step_counts.items()}
        track_behaviors[str(track_id)] = {
            'total_time': float(total_time),
            **step_times
        }

    # 提取循环数据
    cycles = extract_cycles_from_step_sequence(per_frame_step_maps, fps) if per_frame_step_maps else []

    # 计算指标
    analysis_stats = calculate_analysis_stats(per_frame_step_maps, cycles, fps, video_info)

    # 排序取前3
    sorted_tracks = sorted(track_behaviors.items(), key=lambda x: x[1]['total_time'], reverse=True)[:3]

    return {
        'track_behaviors': track_behaviors,
        'top_tracks': [
            {**{'track_id': int(tid)}, **{k: float(v) for k, v in beh.items()}}
            for tid, beh in sorted_tracks
        ],
        'cycles': cycles,
        'cycle_data': cycles[0] if cycles else {},
        'analysis_stats': analysis_stats
    }


def calculate_analysis_stats(per_frame_step_maps, cycles, fps, video_info):
    """
    计算三个分析指标：
    1. 标准化执行符合率：六个环节是否都出现
    2. 操作时间与理论时间比
    3. 等待时间与总时间比
    """
    # 理论时间（秒）
    THEORETICAL_TIMES = {
        "RobotPick": 6,
        "Scan": 2,
        "RobotFix": 10,
        "HandTighten": 7,
        "ElectricGun": 15,
        "RobotReturn": 4
    }
    ALL_STEPS = ["RobotPick", "Scan", "RobotFix", "HandTighten", "ElectricGun", "RobotReturn"]

    # 1. 标准化执行符合率：每个循环独立计算，统计出现的步骤实例数 / 总步骤槽位数
    total_step_slots = len(cycles) * 6  # 每个循环6个步骤
    appeared_step_count = 0
    for cycle in cycles:
        steps = cycle.get('steps', {})
        for step_name, duration in steps.items():
            if duration > 0.1:  # 时间大于 0.1 秒才算出现
                appeared_step_count += 1

    compliance_rate = (appeared_step_count / total_step_slots * 100) if total_step_slots > 0 else 0

    # 2. 操作时间与理论时间比
    total_time = video_info.get('duration', 0)
    one_cycle_theoretical_time = sum(THEORETICAL_TIMES.values())
    total_theoretical_time = len(cycles) * one_cycle_theoretical_time
    time_ratio = (total_time / total_theoretical_time * 100) if total_theoretical_time > 0 else 0

    # 3. 等待时间与总时间比
    operation_time = 0
    for cycle in cycles:
        steps = cycle.get('steps', {})
        for step_name, duration in steps.items():
            operation_time += duration
    wait_time = max(0, total_time - operation_time)
    wait_ratio = (wait_time / total_time * 100) if total_time > 0 else 0

    return {
        'compliance_rate': round(compliance_rate, 1),
        'time_ratio': round(time_ratio, 1),
        'wait_ratio': round(wait_ratio, 1)
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

        # 执行追踪分析（每5帧处理一次）
        result = core.state.tracking_system.analyze_video(
            filepath, analysis_id, progress_callback, process_every_n=5
        )
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

        # 保存到数据库
        if core.state.db_manager is not None:
            try:
                core.state.db_manager.save_analysis_result(final_result)
                print(f"✓ 分析结果已保存到数据库: {analysis_id}")
            except Exception as e:
                print(f"✗ 数据库保存异常: {e}")
        else:
            print("✗ db_manager 为 None，无法保存到数据库")

        # 保存 JSON 文件（备用）
        results_dir = Config.RESULTS_FOLDER
        try:
            os.makedirs(results_dir, exist_ok=True)
        except Exception:
            pass
        result_file = os.path.join(results_dir, f"{analysis_id}.json")
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
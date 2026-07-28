#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时视频流处理器
负责：摄像头取流、YOLO追踪、步骤推理、步骤链管理、符合率统计
"""
import time
import threading
import cv2
import logging
from collections import deque

import core.state
from config import Config
from service.UniviewService import UNIVIEW_DLL, get_uniview_stream

logger = logging.getLogger(__name__)

# ==================== 全局状态 ====================
_realtime_auto_started = False
_realtime_thread = None
_realtime_stop_event = threading.Event()
_realtime_cap = None
_realtime_frame_lock = threading.Lock()
_realtime_push_frame_lock = threading.Lock()
_realtime_last_pushed_frame = None
_realtime_push_interval = 0.033  # 约30fps
_frame_counter = 0

# 步骤链状态
_step_chain = {
    'steps': ['RobotPick', 'Scan', 'RobotFix', 'HandTighten', 'ElectricGun', 'RobotReturn'],
    'current_index': -1,          # 当前高亮步骤索引（-1表示空闲）
    'max_active_index': -1,       # 当前循环中最高到达的索引（用于常亮）
    'current_step': None,         # 当前步骤名称（None表示空闲）
    'last_update': 0,             # 最后更新时间
}

# 统计上下文（每个工人）
_person_context = {}  # {pid: {'seen': set(), 'cycles': 0, 'last_step': None}}

# ==================== 步骤链管理 ====================
def update_step_chain(step_name, frame_count=None):
    """更新步骤链状态"""
    global _step_chain
    if step_name is None:
        # 空闲状态，但不清除常亮
        _step_chain['current_step'] = None
        _step_chain['current_index'] = -1
        _step_chain['last_update'] = time.time()
        return

    # 获取步骤索引
    try:
        idx = _step_chain['steps'].index(step_name)
    except ValueError:
        logger.warning(f"未知步骤: {step_name}")
        return

    # 更新当前步骤
    _step_chain['current_step'] = step_name
    _step_chain['current_index'] = idx
    _step_chain['last_update'] = time.time()

    # 更新常亮索引（只增不减）
    if idx > _step_chain['max_active_index']:
        _step_chain['max_active_index'] = idx

    # 如果当前步骤是 RobotPick，重置常亮索引（新的循环开始）
    if step_name == 'RobotPick':
        _step_chain['max_active_index'] = 0  # RobotPick 本身常亮

def reset_step_chain():
    """重置步骤链"""
    global _step_chain
    _step_chain['current_index'] = -1
    _step_chain['max_active_index'] = -1
    _step_chain['current_step'] = None
    _step_chain['last_update'] = time.time()
    # 同时重置统计
    core.state.realtime_status['completed_steps'] = 0
    core.state.realtime_status['total_cycles'] = 0
    core.state.realtime_status['compliance_rate'] = 0.0
    global _person_context
    _person_context.clear()

def get_step_chain():
    """获取步骤链状态"""
    return _step_chain.copy()

# ==================== 实时流核心逻辑 ====================
def _ensure_realtime_started():
    """确保实时分析线程已启动（只执行一次）"""
    global _realtime_auto_started, _realtime_thread, _realtime_stop_event
    if _realtime_auto_started:
        return
    _realtime_auto_started = True
    _realtime_stop_event.clear()
    logger.info("启动实时分析线程")

    # 检测宇视SDK
    source_type = 'camera'
    if UNIVIEW_DLL is not None and get_uniview_stream is not None:
        try:
            cap = get_uniview_stream()
            if cap and cap.is_running():
                source_type = 'uniview'
                logger.info("使用宇视SDK取流")
            else:
                logger.warning("宇视取流对象未就绪，回退到OpenCV")
        except Exception as e:
            logger.error(f"宇视取流初始化失败: {e}，回退到OpenCV")
    else:
        logger.info("宇视SDK未加载，使用OpenCV摄像头")

    core.state.realtime_status['source_type'] = source_type
    core.state.realtime_status['is_running'] = True
    core.state.realtime_status['current_step'] = 'Idle'
    core.state.realtime_status['updated_at'] = int(time.time())
    # 初始化统计字段
    core.state.realtime_status['completed_steps'] = 0
    core.state.realtime_status['total_cycles'] = 0
    core.state.realtime_status['compliance_rate'] = 0.0

    _realtime_thread = threading.Thread(target=_realtime_analysis_loop, daemon=True)
    _realtime_thread.start()
    logger.info("实时分析线程已启动")

def _realtime_capture_loop():
    """采集线程：持续读取视频帧"""
    global _realtime_cap, _frame_counter
    logger.info("采集线程启动")

    source_type = core.state.realtime_status.get('source_type', 'camera')

    if source_type == 'uniview':
        if get_uniview_stream is None:
            logger.error("宇视取流服务不可用，回退到摄像头0")
            core.state.realtime_status['source_type'] = 'camera'
            _realtime_capture_loop()
            return
        try:
            cap = get_uniview_stream()
            if cap is None:
                logger.error("宇视取流失败，回退到摄像头0")
                core.state.realtime_status['source_type'] = 'camera'
                _realtime_capture_loop()
                return
            while not _realtime_stop_event.is_set():
                frame = cap.get_frame()
                if frame is not None:
                    with _realtime_frame_lock:
                        core.state._last_frame = frame
                        _frame_counter += 1
                else:
                    time.sleep(0.001)
            return
        except Exception as e:
            logger.error(f"宇视取流异常: {e}，回退到摄像头0")
            core.state.realtime_status['source_type'] = 'camera'
            _realtime_capture_loop()
            return

    # OpenCV 取流
    camera_index = core.state.realtime_status.get('camera_index', 0)
    logger.info(f"使用OpenCV摄像头 index={camera_index}")
    _realtime_cap = cv2.VideoCapture(camera_index)
    if not _realtime_cap.isOpened():
        logger.error(f"无法打开摄像头 {camera_index}")
        core.state.realtime_status['error'] = f'无法打开摄像头 {camera_index}'
        return

    while not _realtime_stop_event.is_set():
        ret, frame = _realtime_cap.read()
        if ret:
            with _realtime_frame_lock:
                core.state._last_frame = frame.copy()
                _frame_counter += 1
        else:
            time.sleep(0.01)

    if _realtime_cap:
        _realtime_cap.release()
        _realtime_cap = None
    logger.info("采集线程结束")

def _realtime_inference_loop():
    """推理线程：检测、追踪、步骤推理、编码推送"""
    global _realtime_last_pushed_frame, _person_context
    logger.info("推理线程启动 (完整版)")

    # 初始化步骤推理器
    step_inference = None
    try:
        from core.step_inference import StepInference
        step_inference = StepInference(
            proximity_threshold=0.20,
            warmup_frames=30,
            handtighten_window=10,
            handtighten_ratio=0.7,
            electric_shrink_window=5,
            electric_shrink_ratio=0.70,
            idle_timeout=0
        )
        logger.info("StepInference 初始化成功")
    except Exception as e:
        logger.error(f"StepInference初始化失败: {e}")

    # 懒加载追踪系统
    if core.state.tracking_system is None:
        try:
            from core.tracking_system import TrackingSystem
            core.state.tracking_system = TrackingSystem(Config.MODEL_PATH, conf_threshold=0.2, iou_threshold=0.45)
            logger.info("追踪系统懒加载成功")
        except Exception as e:
            logger.error(f"追踪系统初始化失败: {e}")

    frame_count = 0
    last_time = time.time()

    # 清空之前的统计上下文（新线程）
    _person_context.clear()
    core.state.realtime_status['completed_steps'] = 0
    core.state.realtime_status['total_cycles'] = 0
    core.state.realtime_status['compliance_rate'] = 0.0
    reset_step_chain()

    while not _realtime_stop_event.is_set():
        frame = None
        with _realtime_frame_lock:
            if core.state._last_frame is not None:
                frame = core.state._last_frame.copy()
        if frame is None:
            time.sleep(0.001)
            continue

        frame_count += 1
        current_time = time.time()
        fps = frame_count / (current_time - last_time) if (current_time - last_time) > 0 else 0

        tracked_objects = []
        step_map = {}
        infer_start = time.time()

        try:
            if core.state.tracking_system and frame is not None:
                _, _, tracked_objects = core.state.tracking_system.detect_and_track(frame)
                if step_inference and tracked_objects:
                    step_dets = []
                    for obj in tracked_objects:
                        x1, y1, x2, y2 = map(float, obj['bbox'])
                        step_dets.append({
                            'class_name': core.state.tracking_system.class_names.get(int(obj['class_id']), f"class_{int(obj['class_id'])}"),
                            'track_id': int(obj['track_id']),
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(obj['confidence']),
                        })
                    step_map = step_inference.infer_step(frame.shape, step_dets) or {}
        except Exception as e:
            logger.error(f"推理异常: {e}")

        infer_ms = (time.time() - infer_start) * 1000

        # 提取当前步骤和置信度
        current_step = 'Idle'
        confidence = 0
        track_id = None
        if step_map:
            for tid, step in step_map.items():
                if step is not None:
                    current_step = step
                    for obj in tracked_objects:
                        if int(obj['track_id']) == tid:
                            confidence = float(obj['confidence'])
                            track_id = tid
                    break

        # ========== 统计标准化执行符合率 ==========
        for pid, step in step_map.items():
            if pid not in _person_context:
                _person_context[pid] = {'seen': set(), 'cycles': 0, 'last_step': None}
            ctx = _person_context[pid]

            if step is not None:
                # 只有步骤发生变化时才处理
                if step != ctx['last_step']:
                    if step == 'RobotPick':
                        # 每次 RobotPick 都视为新循环开始
                        ctx['cycles'] += 1
                        # 清空已见步骤集合，开始新循环
                        ctx['seen'].clear()
                        # 将 RobotPick 加入已见（作为新循环的第一个步骤）
                        ctx['seen'].add(step)
                    else:
                        # 非 RobotPick：如果是新步骤，则加入并累计
                        if step not in ctx['seen']:
                            ctx['seen'].add(step)
                            core.state.realtime_status['completed_steps'] += 1
                    ctx['last_step'] = step
            else:
                ctx['last_step'] = None

        # 计算总循环数
        total_cycles = sum(c['cycles'] for c in _person_context.values())
        core.state.realtime_status['total_cycles'] = total_cycles

        # 计算符合率
        completed = core.state.realtime_status['completed_steps']
        if total_cycles > 0:
            compliance = (completed / (total_cycles * 6)) * 100
        else:
            compliance = 0.0
        core.state.realtime_status['compliance_rate'] = round(compliance, 2)

        # ========== 更新步骤链 ==========
        step_for_chain = current_step if current_step != 'Idle' else None
        update_step_chain(step_for_chain, frame_count)

        # ========== 绘制并编码帧 ==========
        try:
            if core.state.tracking_system:
                tracked_frame = core.state.tracking_system._draw_tracks(frame.copy(), tracked_objects, step_map=step_map)
                with _realtime_push_frame_lock:
                    _, buffer = cv2.imencode('.jpg', tracked_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    _realtime_last_pushed_frame = buffer.tobytes()
        except Exception as e:
            logger.error(f"编码帧失败: {e}")
            try:
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                with _realtime_push_frame_lock:
                    _realtime_last_pushed_frame = buffer.tobytes()
            except Exception as e2:
                logger.error(f"编码原始帧失败: {e2}")

        # 更新实时状态
        core.state.realtime_status.update({
            'is_running': True,
            'current_step': current_step,
            'confidence': confidence,
            'fps': fps,
            'infer_ms': infer_ms,
            'track_id': track_id,
            'updated_at': int(time.time()),
        })

    logger.info("推理线程结束")

def _realtime_analysis_loop():
    """主循环：启动采集和推理线程"""
    try:
        capture_thread = threading.Thread(target=_realtime_capture_loop, daemon=True)
        capture_thread.start()
        inference_thread = threading.Thread(target=_realtime_inference_loop, daemon=True)
        inference_thread.start()
        capture_thread.join()
        inference_thread.join()
    except Exception as e:
        logger.error(f"实时循环异常: {e}")
        core.state.realtime_status['error'] = str(e)
    finally:
        core.state.realtime_status['is_running'] = False
        _realtime_stop_event.clear()
        logger.info("实时循环结束")

# ==================== 对外接口 ====================
def ensure_started():
    """外部调用，确保实时流已启动"""
    _ensure_realtime_started()

def get_status():
    """获取实时状态（包含统计）"""
    return core.state.realtime_status

def get_step_chain_status():
    """获取步骤链状态"""
    return get_step_chain()

def reset_step_chain_status():
    """重置步骤链"""
    reset_step_chain()
    return True

def get_stream_generator():
    """返回视频流生成器"""
    ensure_started()
    def gen():
        frame_count = 0
        while not _realtime_stop_event.is_set():
            time.sleep(_realtime_push_interval)
            frame_bytes = None
            with _realtime_push_frame_lock:
                if _realtime_last_pushed_frame is not None:
                    frame_bytes = _realtime_last_pushed_frame
            if frame_bytes is None:
                continue
            frame_count += 1
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    return gen()
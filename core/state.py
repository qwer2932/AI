tracking_system = None
db_manager = None
_last_frame = None
_last_tracked_frame = None

analysis_status = {
    'status': 'idle',
    'is_processing': False,
    'progress': 0,
    'current_frame': 0,
    'total_frames': 0,
    'message': '等待中...'
}

task_status = {}
pause_requests = {}

realtime_status = {
    'is_running': False,
    'current_step': 'Idle',
    'confidence': 0,
    'fps': 0,
    'infer_ms': 0,
    'track_id': None,
    'error': None,
    'updated_at': 0,
    'step_history': [],
    'source_type': 'camera',
    'camera_index': 0,
    'completed_steps': 0,
    'total_cycles': 0,
    'compliance_rate': 0.0,
    'cycle_times': [],
}

# 装配步骤顺序（6 步真实视频驱动的步骤链）
STEP_CHAIN_ORDER = ['RobotPick', 'Scan', 'RobotFix', 'HandTighten', 'ElectricGun', 'RobotReturn']

# 实时步骤链状态：随每帧检测结果实时更新
step_chain = {
    'order': list(STEP_CHAIN_ORDER),           # 步骤链顺序
    'current_index': -1,                       # 当前步骤在链中的索引；-1 = 未开始
    'current_step': None,                      # 当前检测到的步骤名
    'last_step': None,                         # 上一检测到的步骤
    'cycle_count': 0,                          # 已完成循环数（RobotReturn→RobotPick 计 1 次）
    'in_cycle': False,                         # 是否处于一个循环中
    'step_frame_counts': [0] * 6,              # 各步累计帧数（所有循环累加）
    'cycle_frame_counts': [0] * 6,             # 当前循环中各步已累计帧数
    'step_completed_in_cycle': [False] * 6,    # 当前循环中各步是否已完成
    'step_started_at': [0] * 6,                # 各步本次循环起始时间戳（0=未开始）
    'max_active_index': -1,                    # 当前循环中已激活过的最高步骤索引（用于常亮）
    'last_step_change_at': 0,                  # 上次步骤变化时间戳
    'last_step_change_frame': 0,               # 上次步骤变化时的帧号
    'frame_count': 0,                          # 步骤链累计处理帧数
    'step_history': [],                        # 步骤切换历史 [{step, prev, frame, ts, cycle}, ...]
    'updated_at': 0,
}


def reset_step_chain():
    """重置步骤链状态（启动新一轮实时分析时调用）"""
    step_chain['current_index'] = -1
    step_chain['current_step'] = None
    step_chain['last_step'] = None
    step_chain['cycle_count'] = 0
    step_chain['in_cycle'] = False
    step_chain['step_frame_counts'] = [0] * 6
    step_chain['cycle_frame_counts'] = [0] * 6
    step_chain['step_completed_in_cycle'] = [False] * 6
    step_chain['step_started_at'] = [0] * 6
    step_chain['max_active_index'] = -1
    step_chain['last_step_change_at'] = 0
    step_chain['last_step_change_frame'] = 0
    step_chain['frame_count'] = 0
    step_chain['step_history'] = []


def update_step_chain(new_step, frame_count=None, timestamp=None):
    """
    根据当前帧的步骤检测结果更新步骤链状态。

    Args:
        new_step: 当前帧检测到的步骤名（必须是 STEP_CHAIN_ORDER 中的一个），None 表示无检测
        frame_count: 当前帧号（可选）
        timestamp: 当前时间戳（可选，默认 time.time()）

    行为：
        - 累计各步帧数
        - 检测步骤切换：记录历史、更新 last_step
        - 检测循环完成：last_step == RobotReturn 且 new_step == RobotPick → cycle_count += 1
        - 第一个有效步骤出现时自动开启 in_cycle
    """
    import time as _time
    if timestamp is None:
        timestamp = _time.time()
    if frame_count is None:
        frame_count = step_chain['frame_count'] + 1

    step_chain['frame_count'] = frame_count
    step_chain['updated_at'] = int(timestamp)

    # 累计当前步帧数（仅对有效步骤）
    if new_step in STEP_CHAIN_ORDER:
        idx = STEP_CHAIN_ORDER.index(new_step)
        step_chain['current_step'] = new_step
        step_chain['current_index'] = idx
        step_chain['step_frame_counts'][idx] += 1
        if step_chain['in_cycle']:
            step_chain['cycle_frame_counts'][idx] += 1
        
        # 更新 max_active_index：只要检测到步骤就更新，哪怕只有1帧（用于常亮）
        # 确保当前步骤及之前的步骤都常亮
        if idx > step_chain['max_active_index']:
            step_chain['max_active_index'] = idx
        # 如果当前步骤索引小于max_active_index，说明之前已经激活过更高步骤，
        # 但当前步骤也需要被标记为已激活（虽然max_active_index不需要更新）
        # 由于前端使用 idx <= max_active_index 判断常亮，这里只需要确保max_active_index正确即可

    prev = step_chain['last_step']

    # 步骤切换检测（仅在确实发生变化时记录）
    if new_step != prev and new_step in STEP_CHAIN_ORDER:
        # 检测到第一个步骤开始（RobotPick），清空所有状态重新开始循环
        if new_step == 'RobotPick':
            step_chain['cycle_count'] += 1
            step_chain['cycle_frame_counts'] = [0] * 6
            step_chain['step_completed_in_cycle'] = [False] * 6
            step_chain['step_started_at'] = [0] * 6
            step_chain['max_active_index'] = -1
            print(f"【循环重置】检测到第一个步骤开始，新循环 #{step_chain['cycle_count']}")
        
        # 检测到最后一个步骤结束（上一个是 RobotReturn），清空所有状态
        elif prev == 'RobotReturn':
            step_chain['cycle_frame_counts'] = [0] * 6
            step_chain['step_completed_in_cycle'] = [False] * 6
            step_chain['step_started_at'] = [0] * 6
            step_chain['max_active_index'] = -1
            print(f"【循环重置】检测到最后一个步骤结束，等待新循环")

        # 标记上一步在当前循环中已完成（如果是同一循环内）
        if prev in STEP_CHAIN_ORDER and step_chain['in_cycle']:
            prev_idx = STEP_CHAIN_ORDER.index(prev)
            step_chain['step_completed_in_cycle'][prev_idx] = True

        # 新步骤的起始时间
        if new_step in STEP_CHAIN_ORDER:
            idx = STEP_CHAIN_ORDER.index(new_step)
            if step_chain['step_started_at'][idx] == 0:
                step_chain['step_started_at'][idx] = timestamp

        # 进入循环
        step_chain['in_cycle'] = True

        # 记录历史（最近 200 条，避免无限增长）
        step_chain['step_history'].append({
            'step': new_step,
            'prev': prev,
            'frame': frame_count,
            'ts': timestamp,
            'cycle': step_chain['cycle_count'],
        })
        if len(step_chain['step_history']) > 200:
            step_chain['step_history'] = step_chain['step_history'][-200:]

        # 更新上次变化信息
        step_chain['last_step'] = new_step
        step_chain['last_step_change_at'] = timestamp
        step_chain['last_step_change_frame'] = frame_count
    elif new_step is None and prev in STEP_CHAIN_ORDER:
        # 当前帧未检测到步骤，但保留 last_step（不立即清空），避免抖动
        # 仅清空 current_step 字段（last_step 留给前端做"上次执行步骤"展示）
        step_chain['current_step'] = None
        step_chain['current_index'] = -1
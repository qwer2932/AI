#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
装配环节后置推理模块
根据每帧检测到的物体位置关系，判断当前处于哪个装配步骤

循环流程：
1. RobotPick: 机械手、人、悬挂出现，并在画面右侧
2. Scan: 扫码枪出现
3. RobotFix: 悬挂靠近车
4. HandTighten: 电枪的框大小基本不变，位置匀速移动或基本不动，且上一个行为是机械手固定悬挂到车身
5. ElectricGun: 车型、机械手、人、电枪全部出现，且上一个行为是手预紧螺母，电枪的框的大小从基本不变变成突然变化，且位置移动
6. RobotReturn: 机械手再次靠近悬挂（悬挂出现），本次循环结束
"""

import numpy as np
from collections import defaultdict, deque


# 装配步骤定义（严格顺序）
STEPS = {
    "RobotPick":     1,  # 机械手取悬挂
    "Scan":          2,  # 扫描条码
    "RobotFix":      3,  # 机械手固定悬挂到车身
    "HandTighten":   4,  # 手预紧螺母
    "ElectricGun":   5,  # 电枪打螺母
    "RobotReturn":   6,  # 机械手回位
}

STEP_NAMES = {v: k for k, v in STEPS.items()}


class StepInference:
    """
    装配步骤推理器
    根据每帧中各类别物体的位置关系和状态变化，判断当前装配步骤
    """

    def __init__(self, proximity_threshold=0.30, warmup_frames=30,
                 handtighten_window=10, handtighten_ratio=0.7,
                 electric_shrink_window=5, electric_shrink_ratio=0.70,
                 idle_timeout=0):
        """
        Args:
            proximity_threshold: 物体中心点距离阈值（占图像宽/高的比例）
            warmup_frames: 预热帧数。前 N 帧解除独立步骤的顺序 gating，
                           允许从视频中任意位置开始判断
            handtighten_window: HandTighten 判定滑动窗口大小（帧数）
            handtighten_ratio: HandTighten 窗口内稳定帧占比阈值
            electric_shrink_window: ElectricGun 面积缩小判定的滑动窗口大小（帧数）
            electric_shrink_ratio: ElectricGun 面积缩小阈值（相对于 HandTighten 时的面积）
            idle_timeout: 空闲超时帧数。连续多少帧无步骤判定后，重置当前步骤为 None。
                          设为 0 表示禁用超时（保留原始行为）。
        """
        self.proximity_threshold = proximity_threshold
        self.warmup_frames = warmup_frames
        # 滑动窗口参数
        self.handtighten_window = handtighten_window
        self.handtighten_ratio = handtighten_ratio
        self.electric_shrink_window = electric_shrink_window
        self.electric_shrink_ratio = electric_shrink_ratio
        self.idle_timeout = idle_timeout

        self.frame_count = 0
        self.robot_home_position = None
        self.robot_at_body = False
        self.step_frame_counts = defaultdict(lambda: defaultdict(int))
        self.last_step = {}
        # 追踪每个 person 的当前步骤
        self._current_step = {}
        # 追踪电枪的框中心位置历史（用于判断位置快速移动）
        self._gun_bbox_history = defaultdict(list)  # {person_id: [(center_x, center_y, area, frame_count), ...]}
        self._gun_size_stable = {}  # {person_id: True/False} 是否大小基本不变
        self._suspension_on_right = {}  # {person_id: True/False} 悬挂是否在右侧
        # 上一帧 机械手和悬挂 是否已经靠近在一起（已不再使用，保留为兼容）
        self._arm_susp_together = defaultdict(bool)
        # 上一帧 person / arm 中心 x（用于判断是否向画面右侧移动 → RobotReturn）
        self._prev_person_x = {}
        self._prev_arm_x = {}
        # person 最近 N 帧的 x 位置历史（用于累积位移判断）
        self._person_x_history = defaultdict(list)  # {pid: [x1, x2, ...]}
        # 机械臂 x 位置历史
        self._arm_x_history = defaultdict(list)    # {pid: [x1, x2, ...]}  机械臂也按pid跟踪（实际可能只有一个）
        self.X_HISTORY_LEN = 5   # 累积几帧的位移
        self.X_DISP_THRESHOLD = 3  # 累积位移 > N 像素 → 向右移动（降低阈值，让 RobotReturn 更易触发）
        # RobotReturn 阶段：一旦触发后，整段直到 RobotPick 都算 RobotReturn
        self._in_robot_return_phase = defaultdict(bool)
        # HandTighten 确认帧数（保留原字段，实际改用滑动窗口）
        self._handtighten_frames = defaultdict(int)
        self.HANDTIGHTEN_CONFIRM = 8   # 保留原常量（兼容，实际使用滑动窗口）
        # HandTighten 触发后冷却帧数（避免抖动被重复识别为 HandTighten）
        self._handtighten_cooldown = defaultdict(int)
        self.HANDTIGHTEN_COOLDOWN = 30
        # HandTighten 滑动窗口（存储每帧是否满足"稳定"条件）
        self._handtighten_stable_window = defaultdict(lambda: deque(maxlen=handtighten_window))
        # ElectricGun 触发帧数
        self._electricgun_triggered = defaultdict(bool)
        # ElectricGun 持续期到期帧（此属性已废弃，改为人物移动判断，但保留以防兼容）
        self._electricgun_active_until = defaultdict(int)
        self.ELECTRICGUN_DURATION = 45  # 不再使用，保留常量
        # ElectricGun 触发后冷却帧数
        self._electricgun_cooldown = defaultdict(int)
        self.ELECTRICGUN_COOLDOWN = 20
        # ElectricGun 激活状态：打螺母中。{person_id: 激活起始帧号}，0 表示未激活
        # 触发后整段电枪出现在人/车附近的期间都算 ElectricGun 步骤
        self._electricgun_active = defaultdict(int)
        # 记录 person 是否曾进入过 HandTighten（用于 ElectricGun 的前提判断）
        self._has_seen_handtighten = defaultdict(bool)
        # 记录 person 是否曾进入过 ElectricGun（用于 RobotReturn 的前提判断，本轮已出现 ElectricGun 才允许触发 RobotReturn）
        self._has_seen_electricgun = defaultdict(bool)
        # 悬挂（susp）最近 N 帧出现历史（用于 RobotPick 的"悬挂左成"短促判定的容错）
        self._susp_recent_frames = defaultdict(lambda: deque(maxlen=10))  # 最近 10 帧
        self.SUSP_RECENT_WINDOW = 10  # 悬挂容错窗口（帧数）
        # 悬挂最近一次出现的中心位置（供 RobotPick 在 susp 缺失时回退使用）
        self._last_susp_pos = {}
        # 悬挂最近一次出现在取料区（pick zone）的帧号
        self._susp_in_pick_zone_recent = defaultdict(int)
        # 记录 HandTighten 触发时的电枪面积（作为 ElectricGun 触发的基准）
        self._handtighten_gun_area = {}
        # RobotFix 阶段标记
        self._in_fix_phase = defaultdict(bool)
        # ElectricGun 面积缩小的滑动窗口（存储每帧面积比值）
        self._electric_shrink_window_data = defaultdict(lambda: deque(maxlen=electric_shrink_window))

        # 空闲超时计数器
        self._idle_counter = defaultdict(int)

    def reset(self):
        """重置推理器状态（新的分析任务时调用）"""
        self.step_frame_counts.clear()
        self.last_step.clear()
        self.frame_count = 0
        self.robot_home_position = None
        self.robot_at_body = False
        self._current_step.clear()
        self._gun_bbox_history.clear()
        self._gun_size_stable.clear()
        self._suspension_on_right.clear()
        self._arm_susp_together.clear()
        self._prev_person_x.clear()
        self._prev_arm_x.clear()
        self._person_x_history.clear()
        self._arm_x_history.clear()
        self._in_fix_phase.clear()
        self._in_robot_return_phase.clear()
        self._handtighten_frames.clear()
        self._handtighten_cooldown.clear()
        self._handtighten_stable_window.clear()
        self._electricgun_triggered.clear()
        self._electricgun_cooldown.clear()
        self._electricgun_active.clear()
        self._electricgun_active_until.clear()
        self._has_seen_handtighten.clear()
        self._has_seen_electricgun.clear()
        self._susp_recent_frames.clear()
        self._last_susp_pos.clear()
        self._susp_in_pick_zone_recent.clear()
        self._electric_shrink_window_data.clear()
        self._idle_counter.clear()

    def _get_center(self, bbox):
        """从边界框 [x1, y1, x2, y2] 获取中心点"""
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    def _susp_recently_in_pick_zone(self, pid):
        """检查 pid 在容错窗口内是否曾出现 susp 在取料区（pick zone）的悬挂"""
        return self.frame_count - self._susp_in_pick_zone_recent.get(pid, 0) <= self.SUSP_RECENT_WINDOW

    def _get_bbox_area(self, bbox):
        """获取边界框面积"""
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    def _is_point_in_bbox(self, point, bbox):
        """判断点 (x, y) 是否在边界框 [x1, y1, x2, y2] 内"""
        x, y = point
        x1, y1, x2, y2 = bbox
        return x1 <= x <= x2 and y1 <= y <= y2

    def _bbox_intersect(self, bbox1, bbox2):
        """判断两个边界框 [x1, y1, x2, y2] 是否有交集"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        x_overlap = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
        y_overlap = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
        return x_overlap > 0 and y_overlap > 0

    def _is_near(self, pos1, pos2, frame_shape):
        """判断两个位置是否足够近"""
        h, w = frame_shape[:2]
        dist = np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
        norm_dist = dist / np.sqrt(w ** 2 + h ** 2)
        return norm_dist < self.proximity_threshold

    def _is_near_any(self, pos, targets, frame_shape):
        """判断位置是否接近任意一个目标"""
        for t in targets:
            t_center = self._get_center(t['bbox'])
            if self._is_near(pos, t_center, frame_shape):
                return True
        return False

    def _is_on_right_side(self, pos, frame_shape):
        """判断位置是否在画面右侧（x > 60% 宽度）"""
        w = frame_shape[1] if len(frame_shape) > 1 else frame_shape[0]
        return pos[0] > w * 0.6

    def _find_by_class(self, detections, class_name):
        """查找指定类别的检测结果"""
        return [d for d in detections if d.get('class_name') == class_name]

    def infer_step(self, frame_shape, detections):
        """
        推理当前帧中各 person 的装配步骤

        Args:
            frame_shape: (H, W, C) 帧尺寸
            detections: list[dict]，每帧检测结果

        Returns:
            dict: {person_track_id: step_name, ...}
        """
        self.frame_count += 1
        h, w = frame_shape[:2]

        # 分类提取
        persons   = self._find_by_class(detections, 'person')
        arms      = self._find_by_class(detections, 'mechanical_arm')
        guns      = self._find_by_class(detections, 'electric_gun')
        scanners  = self._find_by_class(detections, 'scanner')
        susp      = self._find_by_class(detections, 'suspension_assembly')
        cars      = (self._find_by_class(detections, '310C') +
                     self._find_by_class(detections, 'E262C'))

        # 判断悬挂是否存在（本帧）
        susp_present = len(susp) > 0

        # 更新机械臂初始位置（仅取第一帧）
        if self.frame_count == 1 and arms:
            self.robot_home_position = self._get_center(arms[0]['bbox'])

        # 更新机械臂是否曾到过车身位置
        if arms and cars:
            arm_pos = self._get_center(arms[0]['bbox'])
            if self._is_near_any(arm_pos, cars, frame_shape):
                self.robot_at_body = True

        result = {}

        for person in persons:
            pid = person['track_id']
            person_pos = self._get_center(person['bbox'])

            # 获取当前步骤
            current_step = self._current_step.get(pid)
            current_idx = STEPS.get(current_step, 0) if current_step else 0

            # ========== 计算人的移动 ==========
            cur_px = person_pos[0]
            prev_px = self._prev_person_x.get(pid, cur_px)
            self._prev_person_x[pid] = cur_px

            hist = self._person_x_history[pid]
            hist.append(cur_px)
            if len(hist) > self.X_HISTORY_LEN:
                hist.pop(0)

            if len(hist) >= self.X_HISTORY_LEN:
                person_cumulative_dx = hist[-1] - hist[0]
                person_moving_right = person_cumulative_dx > self.X_DISP_THRESHOLD
            else:
                person_cumulative_dx = 0
                person_moving_right = False

            # ========== 计算机械臂的移动 ==========
            arm_present = len(arms) > 0
            arm_moving_right = False
            if arm_present:
                arm_bbox = arms[0]['bbox']
                arm_center = self._get_center(arm_bbox)
                arm_cur_x = arm_center[0]
                prev_arm_x = self._prev_arm_x.get(pid, arm_cur_x)
                self._prev_arm_x[pid] = arm_cur_x

                arm_hist = self._arm_x_history[pid]
                arm_hist.append(arm_cur_x)
                if len(arm_hist) > self.X_HISTORY_LEN:
                    arm_hist.pop(0)
                self._arm_x_history[pid] = arm_hist

                if len(arm_hist) >= self.X_HISTORY_LEN:
                    arm_cumulative_dx = arm_hist[-1] - arm_hist[0]
                    arm_moving_right = arm_cumulative_dx > self.X_DISP_THRESHOLD
                else:
                    arm_cumulative_dx = 0
                    arm_moving_right = False

            # ========== 强制结束条件（优先级最高） ==========
            # 1. RobotFix 结束：悬挂消失
            if current_step == "RobotFix" and not susp_present:
                self._current_step[pid] = None
                current_step = None

            # 2. RobotPick 结束：悬挂消失
            if current_step == "RobotPick" and not susp_present:
                self._current_step[pid] = None
                current_step = None

            # ========== RobotReturn 阶段处理 ==========
            # 进入条件：机械手向右移动且存在，且 上一个动作为 ElectricGun
            # 退出条件：悬挂出现 且 机械手停止向右移动
            if self._in_robot_return_phase.get(pid, False):
                # 检查是否应该退出
                if susp_present and not arm_moving_right:
                    self._in_robot_return_phase[pid] = False
                    self._current_step[pid] = None
                    # 不 continue，让后续代码尝试触发 RobotPick
                else:
                    # 仍在回位阶段
                    self.step_frame_counts[pid]["RobotReturn"] += 1
                    result[pid] = "RobotReturn"
                    continue

            # 进入 RobotReturn 条件（优化：上个动作为 ElectricGun）
            if not self._in_robot_return_phase.get(pid, False):
                if arm_moving_right and arm_present:
                    # 检查上一帧步骤是否为 ElectricGun
                    if self.last_step.get(pid) == "ElectricGun":
                        self._in_robot_return_phase[pid] = True
                        self._current_step[pid] = None
                        self.last_step[pid] = "RobotReturn"
                        self._has_seen_handtighten[pid] = False
                        self._electricgun_active[pid] = 0
                        self._electricgun_active_until[pid] = 0
                        if pid in self._handtighten_gun_area:
                            del self._handtighten_gun_area[pid]
                        self._gun_size_stable[pid] = False
                        self._handtighten_frames[pid] = 0
                        self.step_frame_counts[pid]["RobotReturn"] += 1
                        result[pid] = "RobotReturn"
                        continue

            # ========== ElectricGun 激活期间处理 ==========
            eg_active = self._electricgun_active.get(pid, 0) > 0
            if eg_active:
                # 检查结束条件：人物向右移动
                if person_moving_right:
                    # 结束 ElectricGun
                    self._electricgun_active[pid] = 0
                    self._current_step[pid] = None
                    # 不 continue，让后续代码检测新步骤（可能触发 RobotReturn 或 RobotPick）
                else:
                    # 仍在工作中，强制 ElectricGun
                    self.step_frame_counts[pid]["ElectricGun"] += 1
                    result[pid] = "ElectricGun"
                    self._current_step[pid] = "ElectricGun"
                    continue

            # ========== 检查电枪是否在人物框右侧（条件A或B） ==========
            gun_on_right_of_person = False
            if guns:
                person_bbox = person['bbox']
                for gun in guns:
                    gun_bbox = gun['bbox']
                    gun_center = self._get_center(gun_bbox)
                    intersects = self._bbox_intersect(gun_bbox, person_bbox)
                    gun_on_right_center = gun_center[0] > person_pos[0]
                    gun_entirely_right = gun_bbox[0] > person_bbox[2]
                    if (intersects and gun_on_right_center) or gun_entirely_right:
                        gun_on_right_of_person = True
                        break

            # ========== HandTighten → ElectricGun 切换 ==========
            # 如果当前是 HandTighten，人开始向右移动 → 切换为 ElectricGun（人开始工作）
            if current_step == "HandTighten" and person_moving_right and len(guns) > 0 and not gun_on_right_of_person:
                self._electricgun_active[pid] = self.frame_count
                self._electricgun_active_until[pid] = self.frame_count + self.ELECTRICGUN_DURATION  # 保留，但不会使用
                result[pid] = "ElectricGun"
                self._current_step[pid] = "ElectricGun"
                continue

            # 更新电枪框中心位置历史
            if guns:
                for gun in guns:
                    if self._is_near(person_pos, self._get_center(gun['bbox']), frame_shape):
                        gun_center = self._get_center(gun['bbox'])
                        gun_area = self._get_bbox_area(gun['bbox'])
                        self._gun_bbox_history[pid].append((gun_center[0], gun_center[1], gun_area, self.frame_count))
                        if len(self._gun_bbox_history[pid]) > 30:
                            self._gun_bbox_history[pid] = self._gun_bbox_history[pid][-30:]

            # 检查悬挂是否在右侧（用于 RobotPick 判断）
            if susp_present:
                susp_pos = self._get_center(susp[0]['bbox'])
                self._suspension_on_right[pid] = self._is_on_right_side(susp_pos, frame_shape)
                susp_x_ratio = susp_pos[0] / w
            else:
                self._suspension_on_right[pid] = False
                susp_x_ratio = None

            # 步骤判断
            in_warmup = self.frame_count <= self.warmup_frames
            detected_step = None

            # 区域判定：悬挂在右侧 1/5 算 RobotPick，否则 RobotFix
            # 更新悬挂（susp）出现历史的滑动窗口
            self._susp_recent_frames[pid].append(1 if susp else 0)
            susp_recent_any = any(self._susp_recent_frames[pid])  # 最近窗口内是否出现过

            if susp:
                susp_pos = self._get_center(susp[0]['bbox'])
                susp_x_ratio = susp_pos[0] / w
            else:
                susp_x_ratio = None

            was_in_pick_zone = self._suspension_on_right.get(pid, False)
            if susp_x_ratio is not None and susp_x_ratio > 0.80:
                susp_in_pick_zone = True
            elif susp_x_ratio is not None and susp_x_ratio > 0.75 and was_in_pick_zone:
                susp_in_pick_zone = True
            else:
                susp_in_pick_zone = False

            self._suspension_on_right[pid] = susp_in_pick_zone
            if susp_in_pick_zone:
                # 记录本次 susp 在取料区出现的帧号（用于 RobotPick 容错判定）
                self._susp_in_pick_zone_recent[pid] = self.frame_count
            # 记录最近一次出现 susp 的中心位置（供 susp 缺失帧回退使用）
            if susp:
                self._last_susp_pos[pid] = self._get_center(susp[0]['bbox'])

            # Step 1: RobotPick
            # 条件：arms + persons + (susp 当前帧 或 最近 SUSP_RECENT_WINDOW 帧内出现过) + 其他条件满足
            if arms and persons and (susp or susp_recent_any) and detected_step is None:
                # 计算 arm 与 susp 的位置关系
                arm_pos = self._get_center(arms[0]['bbox'])
                susp_pos = self._get_center(susp[0]['bbox']) if susp else self._last_susp_pos.get(pid)
                if susp_pos is not None and self._is_near(arm_pos, susp_pos, frame_shape):
                    # 额外要求：当前帧 susp 在右侧 1/5，或最近窗口内 susp 出现在右侧 1/5
                    if susp_in_pick_zone or self._susp_recently_in_pick_zone(pid):
                        detected_step = "RobotPick"
                        self._in_robot_return_phase[pid] = False   # 强制退出 RobotReturn
                        if susp:
                            self._last_susp_pos[pid] = susp_pos

            # Step 3: RobotFix - 悬挂在画面其他区域（机械手已持有悬挂向车移动）
            if susp_present and not susp_in_pick_zone and detected_step is None:
                # 只有在非 ElectricGun 和非 RobotReturn 阶段才触发
                if not eg_active and not self._in_robot_return_phase.get(pid, False):
                    detected_step = "RobotFix"

            # Step 2: Scan - 扫码枪出现在人手边（独立步骤）
            if detected_step is None and scanners and persons:
                scanner_pos = self._get_center(scanners[0]['bbox'])
                if self._is_near(person_pos, scanner_pos, frame_shape):
                    detected_step = "Scan"

            # Step 4: HandTighten - 电枪在人/车附近，框大小基本不变（独立步骤）
            handtighten_triggered_this_frame = False
            if self._handtighten_cooldown[pid] > 0:
                self._handtighten_cooldown[pid] -= 1
            if guns and cars and self._handtighten_cooldown[pid] == 0 and not eg_active:
                # 人必须在车内
                person_in_car = False
                for car in cars:
                    if self._is_point_in_bbox(person_pos, car['bbox']):
                        person_in_car = True
                        break
                if not person_in_car:
                    self._handtighten_stable_window[pid].clear()
                else:
                    # 检查电枪是否在人手边且在车附近
                    gun_near_person = False
                    for gun in guns:
                        if (self._is_near(person_pos, self._get_center(gun['bbox']), frame_shape) and
                                self._is_near_any(self._get_center(gun['bbox']), cars, frame_shape)):
                            gun_near_person = True
                            break

                    if gun_near_person:
                        # 检查电枪框位置是否基本稳定（2帧移动距离 < 8像素）
                        history = self._gun_bbox_history.get(pid, [])
                        if len(history) >= 3:
                            cx1, cy1 = history[-3][0], history[-3][1]
                            cx2, cy2 = history[-2][0], history[-2][1]
                            cx3, cy3 = history[-1][0], history[-1][1]
                            d1 = np.sqrt((cx2-cx1)**2 + (cy2-cy1)**2)
                            d2 = np.sqrt((cx3-cx2)**2 + (cy3-cy2)**2)
                            if d1 < 8 and d2 < 8:
                                self._handtighten_stable_window[pid].append(1)
                            else:
                                self._handtighten_stable_window[pid].append(0)
                            win = self._handtighten_stable_window[pid]
                            if len(win) >= self.handtighten_window:
                                stable_ratio = sum(win) / self.handtighten_window
                                if stable_ratio >= self.handtighten_ratio:
                                    if detected_step is None:
                                        detected_step = "HandTighten"
                                        handtighten_triggered_this_frame = True
                                        # 记录 HandTighten 触发时的枪面积（作为 ElectricGun 基准）
                                        for gun in guns:
                                            if (self._is_near(person_pos, self._get_center(gun['bbox']), frame_shape) and
                                                    self._is_near_any(self._get_center(gun['bbox']), cars, frame_shape)):
                                                self._handtighten_gun_area[pid] = self._get_bbox_area(gun['bbox'])
                                                break
                                    self._handtighten_frames[pid] = 0
                                    self._has_seen_handtighten[pid] = True
                                    self._handtighten_cooldown[pid] = self.HANDTIGHTEN_COOLDOWN
                                    self._handtighten_stable_window[pid].clear()
                    else:
                        self._handtighten_stable_window[pid].clear()

            # Step 4b: HandTighten 补充触发 - 电枪在人物框右侧（有/无交集）
            if guns and detected_step is None and self._handtighten_cooldown[pid] == 0 and not eg_active:
                if gun_on_right_of_person:
                    detected_step = "HandTighten"
                    handtighten_triggered_this_frame = True
                    for gun in guns:
                        gun_bbox = gun['bbox']
                        gun_center = self._get_center(gun_bbox)
                        if gun_center[0] > person_pos[0] or gun_bbox[0] > person['bbox'][2]:
                            self._handtighten_gun_area[pid] = self._get_bbox_area(gun_bbox)
                            break
                    self._has_seen_handtighten[pid] = True
                    self._handtighten_cooldown[pid] = self.HANDTIGHTEN_COOLDOWN
                    self._handtighten_stable_window[pid].clear()

            # Step 5: ElectricGun - 若尚未激活，且满足缩小条件，触发激活
            if not eg_active and detected_step is None and guns and persons and not self._in_robot_return_phase.get(pid, False) and not gun_on_right_of_person:
                gun_near_person = False
                for gun in guns:
                    if self._is_near(person_pos, self._get_center(gun['bbox']), frame_shape):
                        gun_near_person = True
                        break

                if gun_near_person:
                    ref_area = self._handtighten_gun_area.get(pid, 0)
                    if ref_area > 0:
                        history = self._gun_bbox_history.get(pid, [])
                        if len(history) >= self.electric_shrink_window:
                            recent_areas = [h[2] for h in history[-self.electric_shrink_window:]]
                            avg_area = np.mean(recent_areas)
                            if avg_area / ref_area <= self.electric_shrink_ratio:
                                # 触发 ElectricGun
                                self._electricgun_active[pid] = self.frame_count
                                self._electricgun_active_until[pid] = self.frame_count + self.ELECTRICGUN_DURATION
                                detected_step = "ElectricGun"
                else:
                    if self._electricgun_active.get(pid, 0) > 0 and self.frame_count > self._electricgun_active_until.get(pid, 0):
                        self._electricgun_active[pid] = 0

            # Step 6: RobotReturn - 独立触发（备用），仅在非阶段内且悬挂出现且机械手靠近时触发
            if detected_step is None and susp_present and arms and not self._in_robot_return_phase.get(pid, False):
                arm_pos = self._get_center(arms[0]['bbox'])
                susp_pos = self._get_center(susp[0]['bbox'])
                if self._is_near(arm_pos, susp_pos, frame_shape):
                    detected_step = "RobotReturn"
                    self._current_step[pid] = None
                    self.last_step[pid] = "RobotReturn"
                    self.step_frame_counts[pid]["RobotReturn"] += 1
                    self._in_fix_phase[pid] = False
                    result[pid] = "RobotReturn"
                    continue

            # 如果本帧触发了 HandTighten 但被其他步骤抢先，记录 HandTighten 历史
            if handtighten_triggered_this_frame and detected_step != "HandTighten":
                self._has_seen_handtighten[pid] = True

            # ---------- 更新步骤状态（增加空闲超时处理） ----------
            if detected_step:
                if self.idle_timeout > 0:
                    self._idle_counter[pid] = 0
                self._current_step[pid] = detected_step
                self.step_frame_counts[pid][detected_step] += 1
                self.last_step[pid] = detected_step
            else:
                if self.idle_timeout > 0 and self._current_step.get(pid) is not None:
                    self._idle_counter[pid] += 1
                    if self._idle_counter[pid] >= self.idle_timeout:
                        self._current_step[pid] = None
                elif self._current_step.get(pid) is not None:
                    self.step_frame_counts[pid][self._current_step[pid]] += 1

            result[pid] = self._current_step.get(pid)

        return result

    def get_summary(self, fps=25):
        """
        获取分析结束后的行为统计摘要

        Args:
            fps: 视频帧率，用于将帧数转换为秒数

        Returns:
            dict: {person_id: {'step': frames, ...}, ...}
            注意：返回的是帧数，不是秒数，便于前端统一处理
        """
        summary = {}
        for pid, step_counts in self.step_frame_counts.items():
            pid_summary = {}
            total_frames = sum(step_counts.values())
            for step_name, frames in step_counts.items():
                pid_summary[step_name] = frames
            pid_summary['_total'] = total_frames
            summary[pid] = pid_summary
        return summary
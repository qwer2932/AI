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
from collections import defaultdict


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

    def __init__(self, proximity_threshold=0.30, warmup_frames=30):
        """
        Args:
            proximity_threshold: 物体中心点距离阈值（占图像宽/高的比例）
            warmup_frames: 预热帧数。前 N 帧解除独立步骤的顺序 gating，
                           允许从视频中任意位置开始判断
        """
        self.proximity_threshold = proximity_threshold
        self.warmup_frames = warmup_frames
        self.frame_count = 0
        self.robot_home_position = None
        self.robot_at_body = False
        self.step_frame_counts = defaultdict(lambda: defaultdict(int))
        self.last_step = {}
        # 追踪每个 person 的当前步骤
        self._current_step = {}
        # 追踪电枪的框大小历史（用于判断大小变化）
        self._gun_bbox_history = defaultdict(list)  # {person_id: [(area, frame_count), ...]}
        self._gun_size_stable = {}  # {person_id: True/False} 是否大小基本不变
        self._suspension_on_right = {}  # {person_id: True/False} 悬挂是否在右侧
        # 上一帧 机械手和悬挂 是否已经靠近在一起（已不再使用，保留为兼容）
        self._arm_susp_together = defaultdict(bool)
        # HandTighten 确认帧数
        self._handtighten_frames = defaultdict(int)
        self.HANDTIGHTEN_CONFIRM = 15  # HandTighten 需要连续多少帧才确认
        # HandTighten 触发后冷却帧数（避免抖动被重复识别为 HandTighten）
        self._handtighten_cooldown = defaultdict(int)
        self.HANDTIGHTEN_COOLDOWN = 30
        # ElectricGun 触发帧数
        self._electricgun_triggered = defaultdict(bool)
        # ElectricGun 持续期到期帧（在此帧之前每帧都算 ElectricGun）
        self._electricgun_active_until = defaultdict(int)
        self.ELECTRICGUN_DURATION = 30  # 持续约 1.2 秒（25fps）
        # ElectricGun 触发后冷却帧数
        self._electricgun_cooldown = defaultdict(int)
        self.ELECTRICGUN_COOLDOWN = 20
        # ElectricGun 激活状态：打螺母中。{person_id: 激活起始帧号}，0 表示未激活
        # 触发后整段电枪出现在人/车附近的期间都算 ElectricGun 步骤
        self._electricgun_active = defaultdict(int)
        # 记录 person 是否曾进入过 HandTighten（用于 ElectricGun 的前提判断）
        self._has_seen_handtighten = defaultdict(bool)

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
        self._in_fix_phase.clear()
        self._handtighten_frames.clear()
        self._handtighten_cooldown.clear()
        self._electricgun_triggered.clear()
        self._electricgun_cooldown.clear()
        self._electricgun_active.clear()
        self._electricgun_active_until.clear()
        self._has_seen_handtighten.clear()

    def _get_center(self, bbox):
        """从边界框 [x1, y1, x2, y2] 获取中心点"""
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    def _get_bbox_area(self, bbox):
        """获取边界框面积"""
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

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

        # 调试：每 100 帧打印一次各类别计数
        if self.frame_count % 100 == 1:
            print(f"[STEP-DEBUG] frame={self.frame_count} persons={len(persons)} "
                  f"arms={len(arms)} guns={len(guns)} scanners={len(scanners)} "
                  f"susp={len(susp)} cars={len(cars)} in_warmup={self.frame_count <= self.warmup_frames}")
            # 同时打印本帧所有 detection 的 class_name
            all_classes = [d.get('class_name') for d in detections]
            print(f"[STEP-DEBUG] frame={self.frame_count} all_classes={all_classes}")

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

            # 更新电枪框大小历史
            if guns:
                for gun in guns:
                    if self._is_near(person_pos, self._get_center(gun['bbox']), frame_shape):
                        gun_area = self._get_bbox_area(gun['bbox'])
                        self._gun_bbox_history[pid].append((gun_area, self.frame_count))
                        # 只保留最近30帧的历史
                        if len(self._gun_bbox_history[pid]) > 30:
                            self._gun_bbox_history[pid] = self._gun_bbox_history[pid][-30:]

            # 检查悬挂是否在右侧
            if susp:
                susp_pos = self._get_center(susp[0]['bbox'])
                self._suspension_on_right[pid] = self._is_on_right_side(susp_pos, frame_shape)

            # 步骤判断
            # 设计原则：
            #   - 独立步骤（RobotPick/Scan/RobotFix/HandTighten/RobotReturn）：
            #       没有任何顺序 gating，可从视频任意位置被触发
            #   - 链式步骤（ElectricGun）：
            #       需要"曾经过 HandTighten"作为前提（因为它依赖"电枪先稳定再突变"的历史）
            #   - 预热期（前 N 帧）：仅对链式步骤生效，
            #       让"假设之前已经发生过 HandTighten"成立，从而可以从中间开始
            in_warmup = self.frame_count <= self.warmup_frames
            detected_step = None

            # 区域判定阈值：悬挂在画面最右侧 1/5（x 归一化坐标 > 0.8）时算 RobotPick
            # 其他区域时（人拿着悬挂去车身边）算 RobotFix
            # 容忍短暂漂移：即使当前帧 x 落在 0.75-0.80 区间，只要上一帧在右侧 1/5，仍算 RobotPick
            # 这样"连续出现"的悬挂 + 任何一帧在右侧 1/5 → 整段都算取悬挂
            if susp:
                _, w = frame_shape[:2]
                susp_pos = self._get_center(susp[0]['bbox'])
                susp_x_ratio = susp_pos[0] / w  # 0~1
            else:
                susp_x_ratio = None

            was_in_pick_zone = self._suspension_on_right.get(pid, False)
            if susp_x_ratio is not None and susp_x_ratio > 0.80:
                susp_in_pick_zone = True
            elif susp_x_ratio is not None and susp_x_ratio > 0.75 and was_in_pick_zone:
                # 短漂移到 0.75-0.80 区间：上一帧在 1/5 内 → 仍算 pick zone
                susp_in_pick_zone = True
            else:
                susp_in_pick_zone = False

            # 记录本帧是否在 pick zone（供下一帧参考）
            self._suspension_on_right[pid] = susp_in_pick_zone

            # Step 1: RobotPick - 机械手和悬挂靠近 + 悬挂在画面右侧 1/5 区域
            if arms and susp and persons and susp_in_pick_zone and detected_step is None:
                arm_pos = self._get_center(arms[0]['bbox'])
                susp_pos = self._get_center(susp[0]['bbox'])
                if self._is_near(arm_pos, susp_pos, frame_shape):
                    detected_step = "RobotPick"

            # Step 3: RobotFix - 悬挂在画面其他区域（机械手已持有悬挂向车移动）
            if susp and not susp_in_pick_zone and detected_step is None:
                # 只要悬挂在画面中（机械手持有）就算 RobotFix
                detected_step = "RobotFix"

            # Step 2: Scan - 扫码枪出现在人手边（独立步骤）
            if detected_step is None and scanners and persons:
                scanner_pos = self._get_center(scanners[0]['bbox'])
                if self._is_near(person_pos, scanner_pos, frame_shape):
                    detected_step = "Scan"

            # Step 4: HandTighten - 电枪在人/车附近，框大小基本不变（独立步骤）
            #         触发的副作用是把 has_seen_handtighten 置位，为 ElectricGun 铺路
            #         冷却期内不重复触发（避免抖动被反复识别）
            #         ElectricGun 激活期间不重复触发 HandTighten
            handtighten_triggered_this_frame = False
            if self._handtighten_cooldown[pid] > 0:
                self._handtighten_cooldown[pid] -= 1
            eg_active = self._electricgun_active.get(pid, 0) > 0
            if guns and cars and self._handtighten_cooldown[pid] == 0 and not eg_active:
                # 检查是否有电枪在人手边且在车附近
                gun_near_person = False
                for gun in guns:
                    if (self._is_near(person_pos, self._get_center(gun['bbox']), frame_shape) and
                            self._is_near_any(self._get_center(gun['bbox']), cars, frame_shape)):
                        gun_near_person = True
                        break

                if gun_near_person:
                    # 检查电枪框大小是否基本不变（变化 < 20%）
                    history = self._gun_bbox_history.get(pid, [])
                    if len(history) >= 5:
                        areas = [h[0] for h in history[-5:]]
                        max_area = max(areas)
                        min_area = min(areas)
                        if max_area > 0 and (max_area - min_area) / max_area < 0.20:
                            self._gun_size_stable[pid] = True
                            self._handtighten_frames[pid] += 1
                            if self._handtighten_frames[pid] >= self.HANDTIGHTEN_CONFIRM:
                                if detected_step is None:
                                    detected_step = "HandTighten"
                                    handtighten_triggered_this_frame = True
                                self._handtighten_frames[pid] = 0
                                self._has_seen_handtighten[pid] = True
                                self._handtighten_cooldown[pid] = self.HANDTIGHTEN_COOLDOWN
                        else:
                            self._gun_size_stable[pid] = False
                            self._handtighten_frames[pid] = 0
                    elif len(history) >= 2:
                        self._gun_size_stable[pid] = True

            # Step 5: ElectricGun - 电枪打螺母
            #   触发条件（任一满足即进入"打螺母中"状态）：
            #     a) 已处于激活状态 + 当前帧电枪在人/车附近 → 持续算 ElectricGun
            #     b) 当前帧电枪在人/车附近 + 电枪框近 5 帧变化 > 30% → 进入激活状态
            #   不再要求 arms 存在；不再依赖 _has_seen_handtighten 硬性条件（电枪工作本身是明确信号）
            #   但保留"预热期内"作为兜底
            if detected_step is None and guns and persons:
                gun_near_person = False
                for gun in guns:
                    if self._is_near(person_pos, self._get_center(gun['bbox']), frame_shape):
                        gun_near_person = True
                        break

                if gun_near_person:
                    is_active = self._electricgun_active.get(pid, 0) > 0
                    if is_active:
                        # 已在打螺母中 → 直接算 ElectricGun（不再要求电枪必须还在突变）
                        detected_step = "ElectricGun"
                        # 持续到当前循环结束：电枪消失/离开人手都算退出
                        self._electricgun_active_until[pid] = self.frame_count + 5  # 留 5 帧 buffer
                    else:
                        # 未激活：判断电枪框是否在工作（大小变化 > 30%）
                        can_activate = in_warmup or self._has_seen_handtighten.get(pid, False)
                        if can_activate:
                            history = self._gun_bbox_history.get(pid, [])
                            if len(history) >= 5:
                                areas = [h[0] for h in history[-5:]]
                                max_area = max(areas)
                                min_area = min(areas)
                                if max_area > 0 and (max_area - min_area) / max_area > 0.30:
                                    # 电枪在工作 → 进入激活状态
                                    self._electricgun_active[pid] = self.frame_count
                                    self._electricgun_active_until[pid] = self.frame_count + self.ELECTRICGUN_DURATION
                                    detected_step = "ElectricGun"
                                    self._has_seen_handtighten[pid] = True  # 触发即把 has_seen 置位
                else:
                    # 电枪不在人手边 → 检查激活状态是否到期
                    if self._electricgun_active.get(pid, 0) > 0 and self.frame_count > self._electricgun_active_until.get(pid, 0):
                        self._electricgun_active[pid] = 0

            # Step 6: RobotReturn - 机械手再次靠近悬挂（独立步骤）
            if detected_step is None and susp and arms:
                arm_pos = self._get_center(arms[0]['bbox'])
                susp_pos = self._get_center(susp[0]['bbox'])
                if self._is_near(arm_pos, susp_pos, frame_shape):
                    detected_step = "RobotReturn"
                    # 循环结束，重置步骤，下次从 RobotPick 开始
                    self._current_step[pid] = None
                    self.last_step[pid] = "RobotReturn"
                    self.step_frame_counts[pid]["RobotReturn"] += 1
                    # 重置 fix 阶段标记
                    self._in_fix_phase[pid] = False
                    result[pid] = "RobotReturn"
                    continue

            # 如果本帧触发了 HandTighten 但被其他步骤抢先，记录 HandTighten 历史
            if handtighten_triggered_this_frame and detected_step != "HandTighten":
                self._has_seen_handtighten[pid] = True

            # 更新步骤状态
            if detected_step:
                self._current_step[pid] = detected_step
                self.step_frame_counts[pid][detected_step] += 1
                self.last_step[pid] = detected_step
            elif current_step:
                # 继续保持当前步骤
                self.step_frame_counts[pid][current_step] += 1

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
                pid_summary[step_name] = frames  # 返回帧数
            pid_summary['_total'] = total_frames
            summary[pid] = pid_summary
        return summary

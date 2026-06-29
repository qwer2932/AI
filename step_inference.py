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

    def __init__(self, proximity_threshold=0.30):
        """
        Args:
            proximity_threshold: 物体中心点距离阈值（占图像宽/高的比例）
        """
        self.proximity_threshold = proximity_threshold
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
        # HandTighten 确认帧数
        self._handtighten_frames = defaultdict(int)
        self.HANDTIGHTEN_CONFIRM = 15  # HandTighten 需要连续多少帧才确认
        # ElectricGun 触发帧数
        self._electricgun_triggered = defaultdict(bool)

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
        self._handtighten_frames.clear()
        self._electricgun_triggered.clear()

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
            detected_step = None

            # Step 1: RobotPick - 机械手、人、悬挂出现，并在画面右侧
            if current_idx < STEPS["RobotPick"]:
                if arms and susp and persons:
                    arm_pos = self._get_center(arms[0]['bbox'])
                    susp_pos = self._get_center(susp[0]['bbox'])
                    if (self._is_near(arm_pos, susp_pos, frame_shape) and
                            self._suspension_on_right.get(pid, False)):
                        detected_step = "RobotPick"

            # Step 2: Scan - 扫码枪出现
            elif current_idx < STEPS["Scan"]:
                if scanners and persons:
                    scanner_pos = self._get_center(scanners[0]['bbox'])
                    if self._is_near(person_pos, scanner_pos, frame_shape):
                        detected_step = "Scan"

            # Step 3: RobotFix - 悬挂靠近车
            elif current_idx < STEPS["RobotFix"]:
                if susp and cars:
                    susp_pos = self._get_center(susp[0]['bbox'])
                    if self._is_near_any(susp_pos, cars, frame_shape):
                        detected_step = "RobotFix"

            # Step 4: HandTighten - 电枪框大小基本不变，位置移动缓慢，上一步是RobotFix
            elif current_idx < STEPS["HandTighten"]:
                if guns and cars:
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
                            # 大小变化小于20%认为是"基本不变"
                            if max_area > 0 and (max_area - min_area) / max_area < 0.20:
                                self._gun_size_stable[pid] = True
                                self._handtighten_frames[pid] += 1
                                # 需要连续多帧满足条件
                                if self._handtighten_frames[pid] >= self.HANDTIGHTEN_CONFIRM:
                                    detected_step = "HandTighten"
                                    self._handtighten_frames[pid] = 0
                            else:
                                self._gun_size_stable[pid] = False
                                self._handtighten_frames[pid] = 0
                        elif len(history) >= 2:
                            # 刚开始几帧，先标记为稳定
                            self._gun_size_stable[pid] = True

            # Step 5: ElectricGun - 车型、机械手、人、电枪全部出现，上一步是HandTighten
            # 电枪框的大小从基本不变变成突然变化，且位置移动
            elif current_idx < STEPS["ElectricGun"]:
                if cars and arms and persons and guns:
                    # 检查所有物体都在
                    gun_near_person = False
                    for gun in guns:
                        if self._is_near(person_pos, self._get_center(gun['bbox']), frame_shape):
                            gun_near_person = True
                            break

                    if gun_near_person and self._electricgun_triggered.get(pid, False):
                        detected_step = "ElectricGun"
                        self._electricgun_triggered[pid] = False

                    # 检查电枪框大小是否突然变化
                    if self._gun_size_stable.get(pid, False):
                        history = self._gun_bbox_history.get(pid, [])
                        if len(history) >= 10:
                            # 前5帧 vs 后5帧的大小变化
                            early_areas = [h[0] for h in history[-10:-5]]
                            late_areas = [h[0] for h in history[-5:]]
                            early_avg = sum(early_areas) / len(early_areas)
                            late_avg = sum(late_areas) / len(late_areas)
                            # 大小变化超过50%认为是"突然变化"
                            if early_avg > 0 and abs(late_avg - early_avg) / early_avg > 0.50:
                                self._electricgun_triggered[pid] = True

            # Step 6: RobotReturn - 机械手再次靠近悬挂（悬挂出现），本次循环结束
            elif current_idx < STEPS["RobotReturn"]:
                if susp and arms:
                    arm_pos = self._get_center(arms[0]['bbox'])
                    susp_pos = self._get_center(susp[0]['bbox'])
                    if self._is_near(arm_pos, susp_pos, frame_shape):
                        detected_step = "RobotReturn"
                        # 循环结束，重置步骤，下次从 RobotPick 开始
                        self._current_step[pid] = None
                        self.last_step[pid] = "RobotReturn"
                        self.step_frame_counts[pid]["RobotReturn"] += 1
                        result[pid] = "RobotReturn"
                        continue

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

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
        self.X_HISTORY_LEN = 5   # 累积几帧的位移
        self.X_DISP_THRESHOLD = 5  # 累积位移 > N 像素 → 向右移动
        # RobotReturn 阶段：一旦触发后，整段直到 RobotPick 都算 RobotReturn
        self._in_robot_return_phase = defaultdict(bool)
        # HandTighten 确认帧数
        self._handtighten_frames = defaultdict(int)
        self.HANDTIGHTEN_CONFIRM = 8   # HandTighten 需要连续多少帧才确认（降低门槛）
        # HandTighten 触发后冷却帧数（避免抖动被重复识别为 HandTighten）
        self._handtighten_cooldown = defaultdict(int)
        self.HANDTIGHTEN_COOLDOWN = 30
        # ElectricGun 触发帧数
        self._electricgun_triggered = defaultdict(bool)
        # ElectricGun 持续期到期帧（在此帧之前每帧都算 ElectricGun）
        self._electricgun_active_until = defaultdict(int)
        self.ELECTRICGUN_DURATION = 45  # 持续约 1.8 秒（25fps），窗口加长
        # ElectricGun 触发后冷却帧数
        self._electricgun_cooldown = defaultdict(int)
        self.ELECTRICGUN_COOLDOWN = 20
        # ElectricGun 激活状态：打螺母中。{person_id: 激活起始帧号}，0 表示未激活
        # 触发后整段电枪出现在人/车附近的期间都算 ElectricGun 步骤
        self._electricgun_active = defaultdict(int)
        # 记录 person 是否曾进入过 HandTighten（用于 ElectricGun 的前提判断）
        self._has_seen_handtighten = defaultdict(bool)
        # 记录 HandTighten 触发时的电枪面积（作为 ElectricGun 触发的基准）
        self._handtighten_gun_area = {}
        # RobotFix 阶段标记
        self._in_fix_phase = defaultdict(bool)

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
        self._in_fix_phase.clear()
        self._in_robot_return_phase.clear()
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

            # 计算人是否向画面右侧移动（累积位移）
            cur_px = person_pos[0]
            prev_px = self._prev_person_x.get(pid, cur_px)
            self._prev_person_x[pid] = cur_px

            # 累积 x 历史（每帧都追加当前位置）
            hist = self._person_x_history[pid]
            hist.append(cur_px)
            if len(hist) > self.X_HISTORY_LEN:
                hist.pop(0)

            # 累积位移 = 最新位置 - 最早位置（跨越 X_HISTORY_LEN 帧）
            if len(hist) >= self.X_HISTORY_LEN:
                cumulative_dx = hist[-1] - hist[0]
                person_moving_right = cumulative_dx > self.X_DISP_THRESHOLD
            else:
                cumulative_dx = 0
                person_moving_right = False

            # --- HandTighten → ElectricGun 切换 ---
            # 如果当前是 HandTighten，人开始向右移动 → 切换为 ElectricGun（人开始工作）
            if current_step == "HandTighten" and person_moving_right and len(guns) > 0:
                self._electricgun_active[pid] = self.frame_count
                self._electricgun_active_until[pid] = self.frame_count + self.ELECTRICGUN_DURATION
                result[pid] = "ElectricGun"
                self._current_step[pid] = "ElectricGun"
                continue

            # --- RobotReturn 阶段触发判定
            # 如果 person 持续向画面右侧移动（累积 5 帧位移 > 5 像素）+ 机械手在画面中
            # → 进入 RobotReturn 阶段（整段直到 RobotPick 都算 RobotReturn）
            arm_present = len(arms) > 0  # 机械手是否在画面中（不要求它也向右移动）

            # 调试日志：每 50 帧打印一次
            if self.frame_count % 50 == 0:
                print(f"[MOV-DEBUG] frame={self.frame_count} pid={pid} "
                      f"p_dx={cur_px - prev_px:.1f} cum_dx={cumulative_dx:.1f} "
                      f"person_moving={person_moving_right} arm_present={arm_present} "
                      f"current_step={current_step}")

            block_handtighten_this_frame = False
            block_electricgun_this_frame = False
            if person_moving_right and arm_present:
                # RobotReturn 只能在 ElectricGun 未激活时触发（枪工作期间是 ElectricGun，不是回位）
                eg_active = self._electricgun_active.get(pid, 0) > 0
                if current_step in (None, "HandTighten") and not eg_active:
                    print(f"[ROBOT-RETURN-PHASE-START] frame={self.frame_count} pid={pid} "
                          f"cum_dx={cumulative_dx:.1f} current_step={current_step}")
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
                    result[pid] = "RobotReturn"
                    continue

                # 更新电枪框中心位置历史（用于判断位置快速移动）
            if guns:
                for gun in guns:
                    if self._is_near(person_pos, self._get_center(gun['bbox']), frame_shape):
                        gun_center = self._get_center(gun['bbox'])
                        gun_area = self._get_bbox_area(gun['bbox'])
                        self._gun_bbox_history[pid].append((gun_center[0], gun_center[1], gun_area, self.frame_count))
                        # 只保留最近30帧的历史
                        if len(self._gun_bbox_history[pid]) > 30:
                            self._gun_bbox_history[pid] = self._gun_bbox_history[pid][-30:]
                        # 调试：电枪被人检测到
                        if self.frame_count % 100 == 1:
                            print(f"[GUN-NEAR] frame={self.frame_count} pid={pid} "
                                  f"gun_center=({gun_center[0]:.0f},{gun_center[1]:.0f}) "
                                  f"person_pos=({person_pos[0]:.0f},{person_pos[1]:.0f})")

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
                    # RobotPick 触发 → 退出 RobotReturn 阶段
                    self._in_robot_return_phase[pid] = False

            # --- 如果已在 RobotReturn 阶段（且本帧不是 RobotPick）：整帧算 RobotReturn ---
            if self._in_robot_return_phase.get(pid, False) and detected_step != "RobotPick":
                self.step_frame_counts[pid]["RobotReturn"] += 1
                result[pid] = "RobotReturn"
                continue

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
                    # 检查电枪框位置是否基本稳定（2帧移动距离 < 20像素）
                    history = self._gun_bbox_history.get(pid, [])
                    if len(history) >= 3:
                        cx1, cy1 = history[-3][0], history[-3][1]
                        cx2, cy2 = history[-2][0], history[-2][1]
                        cx3, cy3 = history[-1][0], history[-1][1]
                        d1 = np.sqrt((cx2-cx1)**2 + (cy2-cy1)**2)
                        d2 = np.sqrt((cx3-cx2)**2 + (cy3-cy2)**2)
                        if d1 < 8 and d2 < 8:
                            self._gun_size_stable[pid] = True
                            self._handtighten_frames[pid] += 1
                            if self._handtighten_frames[pid] >= self.HANDTIGHTEN_CONFIRM:
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
                        else:
                            self._gun_size_stable[pid] = False
                            self._handtighten_frames[pid] = 0
                            # HandTighten 阶段：人开始向右移动 + 枪还在画面 → 切换 ElectricGun
                            if person_moving_right:
                                self._electricgun_active[pid] = self.frame_count
                                self._electricgun_active_until[pid] = self.frame_count + self.ELECTRICGUN_DURATION
                                detected_step = "ElectricGun"
                                self._handtighten_frames[pid] = 0
                                self._has_seen_handtighten[pid] = True
                                self._handtighten_cooldown[pid] = self.HANDTIGHTEN_COOLDOWN
                    elif len(history) >= 2:
                        self._gun_size_stable[pid] = True

            # Step 5: ElectricGun - 电枪打螺母
            #   RobotReturn 阶段内枪出现 → 仍算 RobotReturn，不切换到 ElectricGun
            if detected_step is None and guns and persons and not self._in_robot_return_phase.get(pid, False):
                gun_near_person = False
                for gun in guns:
                    if self._is_near(person_pos, self._get_center(gun['bbox']), frame_shape):
                        gun_near_person = True
                        break

                if gun_near_person:
                    # 枪在人附近 → 检查枪面积是否缩小到 HandTighten 时的 70% 及以下
                    history = self._gun_bbox_history.get(pid, [])
                    ref_area = self._handtighten_gun_area.get(pid, 0)
                    current_area = history[-1][2] if history else 0
                    area_shrunk = (ref_area > 0 and current_area > 0 and
                                   current_area / ref_area <= 0.70)

                    if area_shrunk:
                        # 枪面积缩小到 70% 及以下 → ElectricGun
                        self._electricgun_active[pid] = self.frame_count
                        self._electricgun_active_until[pid] = self.frame_count + self.ELECTRICGUN_DURATION
                        detected_step = "ElectricGun"
                    elif self._electricgun_active.get(pid, 0) > 0 and self.frame_count <= self._electricgun_active_until.get(pid, 0):
                        # 仍在激活窗口内 → 延续 ElectricGun
                        detected_step = "ElectricGun"
                else:
                    # 枪不在人附近 → 检查激活窗口是否到期
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

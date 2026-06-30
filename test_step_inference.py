#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立测试 step_inference 的修复逻辑"""
import sys
sys.path.insert(0, '.')

from step_inference import StepInference

print("=" * 60)
print("测试 1: 视频从 ElectricGun 步骤开始（预热期内可独立触发）")
print("=" * 60)

# 模拟检测数据：person + E262C (车) + mechanical_arm + electric_gun 都出现
# 模拟电枪框大小变化（先稳定再突然变化）
def make_det(person_id, person_bbox, gun_bbox, arm_bbox=None, car_bbox=(0,0,200,200)):
    dets = [
        {'class_name': 'person', 'track_id': person_id, 'bbox': person_bbox},
        {'class_name': 'electric_gun', 'track_id': 99, 'bbox': gun_bbox},
        {'class_name': 'mechanical_arm', 'track_id': 88, 'bbox': arm_bbox or (200,200,300,300)},
        {'class_name': 'E262C', 'track_id': 77, 'bbox': car_bbox},
    ]
    return dets

inf = StepInference(proximity_threshold=0.30, warmup_frames=30)
h, w = 1080, 1920
frame_shape = (h, w, 3)

# 前 10 帧: 电枪在稳定位置（大小不变）→ 触发 HandTighten
for i in range(10):
    person = (400, 400, 500, 600)
    gun = (450, 450, 500, 500)  # 50x50 大小不变
    dets = make_det(1, person, gun)
    res = inf.infer_step(frame_shape, dets)
    if i in (0, 5, 9):
        print(f"  frame {i+1}: current={inf._current_step.get(1)} "
              f"handtighten_count={inf._handtighten_frames.get(1, 0)} "
              f"gun_stable={inf._gun_size_stable.get(1, False)}")

# 接下来: 电枪大小突变 → 触发 ElectricGun
print("\n  -- 电枪突然变大 --")
for i in range(15, 25):
    person = (400, 400, 500, 600)
    gun = (420, 420, 520, 540)  # 100x120 突然变大
    dets = make_det(1, person, gun)
    res = inf.infer_step(frame_shape, dets)
    if i in (15, 18, 20, 24):
        print(f"  frame {i+1}: current={inf._current_step.get(1)} "
              f"eg_triggered={inf._electricgun_triggered.get(1, False)}")

print("\n=== 最终结果 ===")
summary = inf.get_summary(fps=25)
print(f"summary: {summary}")
print(f"step_frame_counts: {dict(inf.step_frame_counts)}")

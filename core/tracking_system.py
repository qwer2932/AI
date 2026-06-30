#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSORT追踪系统
集成YOLOv8和DeepSORT进行人物追踪
"""
from core.step_inference import StepInference

import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import defaultdict
import json
from datetime import datetime

# DeepSORT相关导入
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    print("警告: 未安装deep_sort_realtime，将使用简单追踪")
    DEEPSORT_AVAILABLE = False

class TrackingSystem:
    def __init__(self, model_path, conf_threshold=0.5, iou_threshold=0.45):
        """
        初始化追踪系统
        
        Args:
            model_path: YOLO模型路径
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值
        """
        # 修复PyTorch兼容性问题
        os.environ['TORCH_WEIGHTS_ONLY'] = 'False'
        
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # 初始化模型和追踪器
        self._load_models()
    
    def _load_models(self):
        """加载YOLO模型和DeepSORT追踪器"""
        try:
            # 加载YOLO模型
            self.yolo_model = YOLO(self.model_path)
            self.class_names = {
                0: "person",              # 工人
                1: "310C",               # 车型1（车身）
                2: "E262C",              # 车型2（车身）
                3: "suspension_assembly", # 悬挂总成
                4: "mechanical_arm",     # 机械臂
                5: "electric_gun",       # 电枪
                6: "scanner"             # 扫码枪
            }
            
            # 初始化DeepSORT追踪器
            if DEEPSORT_AVAILABLE:
                self.tracker = DeepSort(
                    max_age=30,
                    n_init=2,
                    max_iou_distance=0.5,
                    max_cosine_distance=0.3,
                    nn_budget=50,
                    embedder="mobilenet",
                    half=True,
                    bgr=True
                )
            else:
                self.tracker = None
            
            # 追踪历史记录
            self.track_history = defaultdict(list)
            self.track_classes = {}
            self.track_smoothing = {}
            
            # 颜色映射
            self.colors = self._generate_colors(len(self.class_names))
            
            print(f"模型加载成功: {self.model_path}")
            print(f"类别数量: {len(self.class_names)}")
            print(f"DeepSORT可用: {DEEPSORT_AVAILABLE}")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
    def _generate_colors(self, num_classes):
        """为不同类别生成不同颜色"""
        colors = []
        for i in range(num_classes):
            hue = int(180 * i / num_classes)
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, color)))
        return colors
    
    def detect_and_track(self, frame, step_map=None):
        """
        对单帧图像进行检测和追踪

        Args:
            frame: 输入图像帧
            step_map: 当前帧各 person 的步骤名 {person_track_id: step_name}（可选，用于在画面上写步骤）

        Returns:
            tuple: (处理后的图像, 检测结果, 追踪结果)
        """
        try:
            # YOLOv8检测
            results = self.yolo_model(frame, conf=self.conf_threshold, iou=self.iou_threshold)[0]
            
            # 提取检测结果
            detections = []
            if results.boxes is not None:
                boxes = results.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confidences = results.boxes.conf.cpu().numpy()
                class_ids = results.boxes.cls.cpu().numpy().astype(int)
                
                for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                    if conf >= self.conf_threshold and cls_id in [0, 1, 2, 3, 4, 5, 6]:  # 追踪所有行为类别
                        # 转换为DeepSORT格式 (x1, y1, w, h)
                        x1, y1, x2, y2 = box
                        w, h = x2 - x1, y2 - y1
                        detections.append(([x1, y1, w, h], conf, cls_id))
            
            # 简化的追踪逻辑 - 直接使用检测结果进行简单追踪
            tracked_objects = []
            
            if detections:
                # 为每个检测分配或更新追踪ID - 处理所有行为类别
                for i, ((x1, y1, w, h), conf, cls_id) in enumerate(detections):
                    # 处理所有行为类别 (0, 1, 2, 4)
                    if cls_id not in [0, 1, 2, 3, 4, 5, 6]:
                        continue
                        
                    # 检查是否与现有追踪匹配
                    matched_track_id = None
                    best_iou = 0
                    
                    for track_id, track_info in self.track_classes.items():
                        if track_id in self.track_smoothing:
                            prev_bbox = self.track_smoothing[track_id]
                            iou = self._calculate_iou([x1, y1, x1+w, y1+h], 
                                                    [prev_bbox[0], prev_bbox[1], prev_bbox[2], prev_bbox[3]])
                            if iou > best_iou and iou > 0.3:
                                best_iou = iou
                                matched_track_id = track_id
                    
                    # 如果没有匹配的追踪，创建新的
                    if matched_track_id is None:
                        matched_track_id = len(self.track_classes) + 1
                    
                    # 位置平滑
                    smoothed_bbox = self._smooth_bbox(matched_track_id, [x1, y1, x1+w, y1+h])
                    
                    tracked_objects.append({
                        'track_id': matched_track_id,
                        'bbox': smoothed_bbox,
                        'class_id': cls_id,
                        'confidence': conf
                    })
                    
                    # 更新追踪信息
                    self.track_classes[matched_track_id] = cls_id
                    center = ((smoothed_bbox[0] + smoothed_bbox[2]) / 2, 
                             (smoothed_bbox[1] + smoothed_bbox[3]) / 2)
                    self.track_history[matched_track_id].append(center)
                    
                    # 限制历史长度
                    if len(self.track_history[matched_track_id]) > 20:
                        self.track_history[matched_track_id].pop(0)
                
                # 清理长时间未更新的追踪
                current_track_ids = {obj['track_id'] for obj in tracked_objects}
                for track_id in list(self.track_classes.keys()):
                    if track_id not in current_track_ids:
                        # 如果追踪超过5帧没有更新，删除它
                        if track_id not in self.track_smoothing:
                            del self.track_classes[track_id]
                            if track_id in self.track_history:
                                del self.track_history[track_id]
            
            # 如果没有追踪到目标，直接使用检测结果
            if not tracked_objects and detections:
                for i, ((x1, y1, w, h), conf, cls_id) in enumerate(detections):
                    tracked_objects.append({
                        'track_id': i,  # 使用检测索引作为临时ID
                        'bbox': [x1, y1, x1 + w, y1 + h],
                        'class_id': cls_id,
                        'confidence': conf
                    })
            
            # 绘制结果
            annotated_frame = self._draw_tracks(frame, tracked_objects, step_map=step_map)
            
            return annotated_frame, detections, tracked_objects
        
        except Exception as e:
            print(f"检测和追踪失败: {e}")
            return frame, [], []
    
    def _calculate_iou(self, box1, box2):
        """计算两个边界框的IoU值"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # 计算交集
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _smooth_bbox(self, track_id, bbox, alpha=0.7):
        """使用指数移动平均平滑边界框位置"""
        if track_id not in self.track_smoothing:
            self.track_smoothing[track_id] = bbox
            return bbox
        
        # 指数移动平均
        prev_bbox = self.track_smoothing[track_id]
        smoothed_bbox = [
            int(alpha * bbox[0] + (1 - alpha) * prev_bbox[0]),  # x1
            int(alpha * bbox[1] + (1 - alpha) * prev_bbox[1]),  # y1
            int(alpha * bbox[2] + (1 - alpha) * prev_bbox[2]),  # x2
            int(alpha * bbox[3] + (1 - alpha) * prev_bbox[3])   # y2
        ]
        
        self.track_smoothing[track_id] = smoothed_bbox
        return smoothed_bbox
    
    def _draw_tracks(self, frame, tracked_objects, step_map=None):
        """在图像上绘制追踪结果 - 绘制所有行为类别
        step_map: {person_track_id: step_name} 当前帧各 person 的步骤名（可选）"""
        annotated_frame = frame.copy()

        for obj in tracked_objects:
            track_id = obj['track_id']
            x1, y1, x2, y2 = map(int, obj['bbox'])
            class_id = obj['class_id']
            confidence = obj['confidence']
            
            # 绘制所有行为类别
            if class_id in [0, 1, 2, 3, 4, 5, 6]:
                # 获取类别名称和颜色
                class_name = self.class_names.get(class_id, f"Class_{class_id}")
                color = self.colors[class_id % len(self.colors)]
                
                # 绘制边界框
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                # 绘制标签
                label = f"ID:{track_id} {class_name} {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

                # 标签背景
                cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10),
                             (x1 + label_size[0], y1), color, -1)

                # 标签文字
                cv2.putText(annotated_frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # 如果是 person 且本帧有步骤名，在标签下方再写一行步骤
                if class_id == 0 and step_map and track_id in step_map:
                    step_name = step_map[track_id]
                    step_label = f"Step: {step_name}"
                    step_size = cv2.getTextSize(step_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    sy1 = y2 + step_size[1] + 6
                    sy2 = y2 + 6
                    cv2.rectangle(annotated_frame, (x1, sy1 - step_size[1] - 6),
                                 (x1 + step_size[0] + 6, sy2), (0, 140, 255), -1)
                    cv2.putText(annotated_frame, step_label, (x1 + 3, sy1 - 6),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 绘制追踪轨迹
                if track_id in self.track_history:
                    points = np.array(self.track_history[track_id], dtype=np.int32)
                    if len(points) > 1:
                        cv2.polylines(annotated_frame, [points], False, color, 2)
        
        return annotated_frame
    
    def analyze_video(self, video_path, analysis_id, progress_callback=None, step_inference=None):
        """分析视频文件

        Args:
            video_path: 视频文件路径
            analysis_id: 分析ID
            progress_callback: 进度回调函数
            step_inference: StepInference 实例（可选）。如果提供，每帧会同步跑步骤推理并把结果叠加到画面上

        Returns:
            result: 分析结果字典
        """
        # 如果外部没传 step_inference，内部创建一个（这样视频上才有 Step 标签）
        if step_inference is None:
            step_inference = StepInference(proximity_threshold=0.30, warmup_frames=30)
        try:
            # 重置追踪状态，确保每次分析都从ID 1开始
            self.track_history = defaultdict(list)
            self.track_classes = {}
            self.track_smoothing = {}
            
            print(f"开始分析视频: {video_path}, 分析ID: {analysis_id}")
            
            # 打开视频文件
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("无法打开视频文件")
            
            # 获取视频信息
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            video_info = {
                'fps': fps,
                'width': width,
                'height': height,
                'total_frames': total_frames,
                'duration': duration,
                'resolution': f"{width}x{height}"
            }
            
            print(f"视频信息: {video_info}")
            
            # 创建输出视频 - 保持原视频分辨率和帧率
            output_path = f"results/{analysis_id}_tracked.avi"
            output_width = width
            output_height = height
            # 使用 XVID 编码 AVI（兼容性更好）
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
            
            # 检查视频写入器是否成功初始化
            if not out.isOpened():
                print(f"警告: XVID 编码器失败，尝试 MJPEG")
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
                if not out.isOpened():
                    print(f"错误: 无法创建视频写入器")
                    cap.release()
                    return None
            
            # 追踪数据存储
            tracking_data = {}
            frame_data = []
            per_frame_detections = []  # 每帧所有检测结果，供后置推理用
            
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 更新进度
                if progress_callback:
                    progress_callback(frame_count, total_frames, f"处理第 {frame_count} 帧", analysis_id)
                
                # 同步跑步骤推理（如果传了 step_inference），得到当前帧各 person 的步骤名
                # 注意：必须先跑追踪拿到 track_id，再喂给 step_inference
                step_map = None
                if step_inference is not None:
                    try:
                        annotated_frame_pre, detections_pre, tracked_objects_pre = self.detect_and_track(frame)
                        # 用 tracked_objects 的 (track_id, bbox) 去给 detections 补 track_id
                        step_dets = []
                        for obj in tracked_objects_pre:
                            x1, y1, x2, y2 = map(float, obj['bbox'])
                            step_dets.append({
                                'class_name': self.class_names.get(int(obj['class_id']), f"class_{int(obj['class_id'])}"),
                                'track_id': int(obj['track_id']),
                                'bbox': [x1, y1, x2, y2],
                                'confidence': float(obj['confidence']),
                            })
                        step_map = step_inference.infer_step(frame.shape, step_dets) or {}
                        # 用 step_map 重新画图
                        annotated_frame = self._draw_tracks(frame, tracked_objects_pre, step_map=step_map)
                        # 把值传回外层（避免再跑一次 detect_and_track）
                        detections = detections_pre
                        tracked_objects = tracked_objects_pre
                    except Exception as _se:
                        if frame_count == 1:
                            print(f"[step_inference] 同步推理异常: {_se}")
                        step_map = None

                if step_map is None:
                    # 正常路径（没传 step_inference 或异常）
                    annotated_frame, detections, tracked_objects = self.detect_and_track(frame, step_map=step_map)
                
                # 存储每帧所有检测结果（供后置推理用）
                frame_all_detections = []
                for obj in tracked_objects:
                    frame_all_detections.append({
                        'track_id': int(obj['track_id']),
                        'bbox': [float(x) for x in obj['bbox']],
                        'class_id': int(obj['class_id']),
                        'class_name': self.class_names.get(obj['class_id'], f"class_{obj['class_id']}"),
                        'confidence': float(obj['confidence'])
                    })
                per_frame_detections.append({
                    'frame': frame_count,
                    'detections': frame_all_detections
                })
                
                # 处理追踪结果（只记录person的追踪）
                frame_tracks = []
                for obj in tracked_objects:
                    track_id = obj['track_id']
                    bbox = obj['bbox']
                    class_id = obj['class_id']
                    confidence = obj['confidence']
                    
                    # 存储追踪数据
                    if track_id not in tracking_data:
                        tracking_data[track_id] = {
                            'frames': [],
                            'bboxes': [],
                            'class_ids': [],
                            'confidences': [],
                            'first_frame': frame_count,
                            'last_frame': frame_count
                        }
                    
                    tracking_data[track_id]['frames'].append(frame_count)
                    # 确保所有数据都是Python原生类型，避免JSON序列化错误
                    tracking_data[track_id]['bboxes'].append([float(x) for x in bbox])
                    tracking_data[track_id]['class_ids'].append(int(class_id))
                    tracking_data[track_id]['confidences'].append(float(confidence))
                    tracking_data[track_id]['last_frame'] = frame_count
                    
                    frame_tracks.append({
                        'track_id': int(track_id),
                        'bbox': [float(x) for x in bbox],
                        'class_id': int(class_id),
                        'confidence': float(confidence),
                        'frame': frame_count
                    })
                
                frame_data.append({
                    'frame': frame_count,
                    'tracks': frame_tracks
                })
                
                # 调整帧大小并写入输出视频
                if annotated_frame.shape[1] != output_width or annotated_frame.shape[0] != output_height:
                    annotated_frame = cv2.resize(annotated_frame, (output_width, output_height))
                out.write(annotated_frame)
            
            # 释放资源
            cap.release()
            out.release()
            
            # 直接使用 AVI 文件
            print(f"视频生成完成: {output_path}")
            
            # 验证生成的视频文件
            if not self.validate_video_file(output_path):
                print(f"警告: 生成的视频文件可能有问题: {output_path}")
                # 尝试重新生成一个简单的视频文件
                self.create_fallback_video(output_path, width, height, fps)
            
            # 计算追踪统计
            tracked_ids = [int(track_id) for track_id in tracking_data.keys()]
            total_tracks = len(tracked_ids)
            
            # 按追踪时长排序，获取前三个
            track_durations = []
            for track_id, data in tracking_data.items():
                duration = data['last_frame'] - data['first_frame'] + 1
                track_durations.append((int(track_id), int(duration)))
            
            track_durations.sort(key=lambda x: x[1], reverse=True)
            top_tracks = track_durations[:3]
            
            result = {
                'analysis_id': analysis_id,
                'video_info': video_info,
                'tracking_data': tracking_data,
                'frame_data': frame_data,
                'per_frame_detections': per_frame_detections,  # 每帧所有检测，供后置推理
                'total_tracks': total_tracks,
                'tracked_ids': tracked_ids,
                'top_tracks': top_tracks,
                'video_path': output_path,
                'result_video_path': f"results/{analysis_id}_tracked.avi",  # 使用实际的 avi 文件
                'result': '分析完成'
            }
            
            print(f"视频分析完成: {total_tracks} 个追踪目标")
            return result
        
        except Exception as e:
            print(f"视频分析失败: {e}")
            raise
    
    def validate_video_file(self, video_path):
        """验证视频文件是否有效"""
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False
            
            # 尝试读取第一帧
            ret, frame = cap.read()
            cap.release()
            return ret and frame is not None
        except Exception as e:
            print(f"视频验证失败: {e}")
            return False
    
    def create_fallback_video(self, output_path, width, height, fps):
        """创建备用视频文件"""
        try:
            import cv2
            import numpy as np
            
            print(f"创建备用视频文件: {output_path}")
            
            # 使用更兼容的编码器
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                print(f"无法创建备用视频文件")
                return False
            
            # 创建10帧的简单视频
            for i in range(10):
                # 创建黑色背景
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                
                # 添加一些文字
                text = f"Video Processing Error - Frame {i+1}"
                cv2.putText(frame, text, (50, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                out.write(frame)
            
            out.release()
            print(f"备用视频文件创建完成: {output_path}")
            return True
            
        except Exception as e:
            print(f"创建备用视频失败: {e}")
            return False
    

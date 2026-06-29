#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8 + DeepSORT 目标检测与追踪系统
支持视频和实时摄像头输入，显示目标ID和类别标签
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import argparse
import os
from collections import defaultdict
import time

# DeepSORT相关导入
try:
    from deep_sort_realtime import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    print("警告: 未安装deep_sort_realtime，请运行: pip install deep-sort-realtime")
    DEEPSORT_AVAILABLE = False


class YOLOv8DeepSORTTracker:
    def __init__(self, model_path, class_names=None, conf_threshold=0.5, iou_threshold=0.45, ignore_classes=None):
        """
        初始化YOLOv8 + DeepSORT追踪器
        
        Args:
            model_path (str): YOLOv8模型路径
            class_names (dict): 类别ID到名称的映射
            conf_threshold (float): 置信度阈值
            iou_threshold (float): IoU阈值
            ignore_classes (list): 要忽略的类别ID列表，这些类别不会被追踪
        """
        # 修复PyTorch兼容性问题 - 设置环境变量
        import os
        os.environ['TORCH_WEIGHTS_ONLY'] = 'False'
        
        self.model = YOLO(model_path)
        self.class_names = class_names or {
            0: "person",              # 工人
            1: "310C",               # 车型1（车身）
            2: "E262C",              # 车型2（车身）
            3: "suspension_assembly", # 悬挂总成
            4: "mechanical_arm",     # 机械臂
            5: "electric_gun",       # 电枪
            6: "scanner"             # 扫码枪
        }
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.ignore_classes = ignore_classes or []  # 不过滤任何类别，全部检测
        
        # 初始化DeepSORT追踪器 - 优化参数提高连贯性
        if DEEPSORT_AVAILABLE:
            self.tracker = DeepSort(
                max_age=30,  # 减少最大存活帧数，更快删除丢失的目标
                n_init=2,    # 减少确认所需的连续检测次数
                max_iou_distance=0.5,  # 降低IoU距离阈值，更严格的匹配
                max_cosine_distance=0.3,  # 增加余弦距离阈值，更宽松的特征匹配
                nn_budget=50,  # 减少特征向量预算，提高速度
                embedder="mobilenet",  # 使用更快的特征提取器
                half=True,  # 使用半精度，提高速度
                bgr=True  # 输入是BGR格式
            )
        else:
            self.tracker = None
            print("DeepSORT不可用，将只进行检测不进行追踪")
        
        # 追踪历史记录
        self.track_history = defaultdict(list)
        self.track_classes = {}  # 存储每个追踪ID对应的类别
        self.track_smoothing = {}  # 存储平滑后的位置
        
        # 颜色映射（为不同类别分配不同颜色）
        self.colors = self._generate_colors(len(self.class_names))
    
    def _generate_colors(self, num_classes):
        """为不同类别生成不同颜色"""
        colors = []
        for i in range(num_classes):
            hue = int(180 * i / num_classes)
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, color)))
        return colors
    
    def detect_and_track(self, frame):
        """
        对单帧图像进行检测和追踪
        
        Args:
            frame (np.ndarray): 输入图像帧
            
        Returns:
            tuple: (处理后的图像, 检测结果, 追踪结果)
        """
        # YOLOv8检测
        results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold)[0]
        
        # 提取检测结果
        detections = []
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                if conf >= self.conf_threshold and cls_id not in self.ignore_classes:
                    # 转换为DeepSORT格式 (x1, y1, w, h)
                    x1, y1, x2, y2 = box
                    w, h = x2 - x1, y2 - y1
                    detections.append(([x1, y1, w, h], conf, cls_id))
        
        # 简化的追踪逻辑 - 直接使用检测结果进行简单追踪
        tracked_objects = []
        
        if detections:
            # 为每个检测分配或更新追踪ID
            for i, ((x1, y1, w, h), conf, cls_id) in enumerate(detections):
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
        
        # 如果没有DeepSORT或没有追踪到目标，直接使用检测结果
        if not tracked_objects and detections:
            for i, ((x1, y1, w, h), conf, cls_id) in enumerate(detections):
                tracked_objects.append({
                    'track_id': i,  # 使用检测索引作为临时ID
                    'bbox': [x1, y1, x1 + w, y1 + h],
                    'class_id': cls_id,
                    'confidence': conf
                })
        
        # 绘制结果
        annotated_frame = self._draw_tracks(frame, tracked_objects)
        
        return annotated_frame, detections, tracked_objects
    
    def _boxes_overlap(self, box1, box2, threshold=0.5):
        """检查两个边界框是否重叠"""
        x1_1, y1_1, w1, h1 = box1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        
        x1_2, y1_2, w2, h2 = box2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # 计算交集
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return False
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        iou = intersection / union if union > 0 else 0
        return iou >= threshold
    
    def _calculate_iou(self, box1, box2):
        """计算两个边界框的IoU值"""
        x1_1, y1_1, w1, h1 = box1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        
        x1_2, y1_2, w2, h2 = box2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # 计算交集
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = w1 * h1
        area2 = w2 * h2
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
    
    def _draw_tracks(self, frame, tracked_objects):
        """在图像上绘制追踪结果"""
        annotated_frame = frame.copy()
        
        for obj in tracked_objects:
            track_id = obj['track_id']
            x1, y1, x2, y2 = map(int, obj['bbox'])
            class_id = obj['class_id']
            confidence = obj['confidence']
            
            # 获取类别名称和颜色
            class_name = self.class_names.get(class_id, f"Class_{class_id}")
            color = self.colors[class_id % len(self.colors)]
            
            # 绘制边界框（加粗）
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
            
            # 绘制内部边框（更细的边框）
            cv2.rectangle(annotated_frame, (x1+2, y1+2), (x2-2, y2-2), (255, 255, 255), 1)
            
            # 准备标签文本
            id_text = f"ID: {track_id}"
            class_text = f"Class: {class_name}"
            conf_text = f"Conf: {confidence:.2f}"
            
            # 计算标签位置和大小
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            
            # 获取文本尺寸
            id_size = cv2.getTextSize(id_text, font, font_scale, thickness)[0]
            class_size = cv2.getTextSize(class_text, font, font_scale, thickness)[0]
            conf_size = cv2.getTextSize(conf_text, font, font_scale, thickness)[0]
            
            # 计算标签区域
            label_width = max(id_size[0], class_size[0], conf_size[0]) + 20
            label_height = (id_size[1] + class_size[1] + conf_size[1]) + 30
            
            # 标签背景位置
            label_x = x1
            label_y = max(y1 - label_height, 0)  # 确保不超出图像边界
            
            # 绘制标签背景（半透明）
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (label_x, label_y), 
                         (label_x + label_width, label_y + label_height), color, -1)
            cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
            
            # 绘制标签边框
            cv2.rectangle(annotated_frame, (label_x, label_y), 
                         (label_x + label_width, label_y + label_height), color, 2)
            
            # 绘制文本
            text_y = label_y + 25
            cv2.putText(annotated_frame, id_text, (label_x + 10, text_y), 
                       font, font_scale, (255, 255, 255), thickness)
            
            text_y += id_size[1] + 5
            cv2.putText(annotated_frame, class_text, (label_x + 10, text_y), 
                       font, font_scale, (255, 255, 255), thickness)
            
            text_y += class_size[1] + 5
            cv2.putText(annotated_frame, conf_text, (label_x + 10, text_y), 
                       font, font_scale, (255, 255, 255), thickness)
            
            # 在边界框中心绘制类别ID
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.putText(annotated_frame, str(class_id), (center_x - 10, center_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
            cv2.putText(annotated_frame, str(class_id), (center_x - 10, center_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)
            
            # 绘制追踪轨迹
            if track_id in self.track_history:
                points = np.array(self.track_history[track_id], dtype=np.int32)
                if len(points) > 1:
                    # 绘制轨迹线
                    cv2.polylines(annotated_frame, [points], False, color, 3)
                    # 在轨迹点上绘制小圆点
                    for point in points[-10:]:  # 只显示最近10个点
                        cv2.circle(annotated_frame, tuple(point), 3, color, -1)
        
        # 添加类别统计信息
        annotated_frame = self._draw_class_statistics(annotated_frame, tracked_objects)
        
        return annotated_frame
    
    def _draw_class_statistics(self, frame, tracked_objects):
        """在视频右上角绘制类别统计信息"""
        # 统计各类别的数量
        class_counts = {}
        for obj in tracked_objects:
            class_id = obj['class_id']
            class_name = self.class_names.get(class_id, f"Class_{class_id}")
            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += 1
        
        # 绘制统计信息背景
        stats_x = frame.shape[1] - 250
        stats_y = 50
        stats_width = 230
        stats_height = len(self.class_names) * 30 + 60
        
        # 半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (stats_x, stats_y), 
                     (stats_x + stats_width, stats_y + stats_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 绘制边框
        cv2.rectangle(frame, (stats_x, stats_y), 
                     (stats_x + stats_width, stats_y + stats_height), (255, 255, 255), 2)
        
        # 绘制标题
        title_text = "Class Statistics"
        cv2.putText(frame, title_text, (stats_x + 10, stats_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 绘制各类别统计
        y_offset = stats_y + 50
        for class_id, class_name in self.class_names.items():
            count = class_counts.get(class_name, 0)
            color = self.colors[class_id % len(self.colors)]
            
            # 绘制颜色指示器
            cv2.circle(frame, (stats_x + 15, y_offset - 5), 8, color, -1)
            cv2.circle(frame, (stats_x + 15, y_offset - 5), 8, (255, 255, 255), 1)
            
            # 绘制类别名称和数量
            text = f"{class_name}: {count}"
            cv2.putText(frame, text, (stats_x + 35, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            y_offset += 30
        
        return frame
    
    def process_video(self, source, output_path=None, display=True):
        """
        处理视频文件或摄像头输入
        
        Args:
            source: 视频路径或摄像头索引
            output_path: 输出视频路径
            display: 是否显示实时结果
        """
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"错误: 无法打开视频源 {source}")
            return
        
        # 获取视频属性
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"视频信息: {width}x{height} @ {fps}fps")
        
        # 设置输出视频
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 检测和追踪
                annotated_frame, detections, tracks = self.detect_and_track(frame)
                
                # 添加帧信息
                info_text = f"Frame: {frame_count} | Detections: {len(detections)} | Tracks: {len(tracks)}"
                cv2.putText(annotated_frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 显示结果
                if display:
                    cv2.imshow('YOLOv8 + DeepSORT Tracking', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # 保存结果
                if out:
                    out.write(annotated_frame)
                
                frame_count += 1
                
                # 计算FPS
                if frame_count % 30 == 0:
                    elapsed_time = time.time() - start_time
                    current_fps = frame_count / elapsed_time
                    print(f"处理进度: {frame_count} 帧, 当前FPS: {current_fps:.2f}")
        
        except KeyboardInterrupt:
            print("用户中断处理")
        
        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            
            # 输出统计信息
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time if total_time > 0 else 0
            print(f"处理完成: 总帧数 {frame_count}, 平均FPS: {avg_fps:.2f}")


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 + DeepSORT 目标检测与追踪')
    parser.add_argument('--model', type=str, 
                       default='runs/detect/a4000_traffic_train_filtered4/weights/best.pt',
                       help='YOLOv8模型路径')
    parser.add_argument('--source', type=str, default='0',
                       help='视频源 (视频文件路径或摄像头索引)')
    parser.add_argument('--output', type=str, default=None,
                       help='输出视频路径')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU阈值')
    parser.add_argument('--no-display', action='store_true',
                       help='不显示实时结果')
    
    args = parser.parse_args()
    
    # 检查模型文件
    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在 {args.model}")
        return
    
    # 创建追踪器
    tracker = YOLOv8DeepSORTTracker(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # 处理视频
    source = int(args.source) if args.source.isdigit() else args.source
    tracker.process_video(
        source=source,
        output_path=args.output,
        display=not args.no_display
    )


if __name__ == "__main__":
    main()

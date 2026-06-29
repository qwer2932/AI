#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库模型和操作
"""

import pymysql
import json
import os
from datetime import datetime
from typing import List, Dict, Optional

class DatabaseManager:
    def __init__(self, host: str = "localhost", port: int = 3306, user: str = "root", 
                 password: str = "111111", database: str = "ai_track_analysis"):
        """
        初始化数据库管理器
        
        Args:
            host: MySQL主机地址
            port: MySQL端口
            user: MySQL用户名
            password: MySQL密码
            database: 数据库名称
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.init_database()
    
    def get_connection(self):
        """获取MySQL数据库连接"""
        conn = pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
            charset='utf8mb4',
            autocommit=False
        )
        # 设置时区
        with conn.cursor() as cursor:
            cursor.execute("SET time_zone = '+08:00'")
        return conn
    
    def init_database(self):
        """初始化数据库表结构"""
        # 首先创建数据库（如果不存在）
        conn = pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            charset='utf8mb4'
        )
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        conn.close()
        
        # 连接到指定数据库
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # 创建分析历史表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                analysis_id VARCHAR(255) UNIQUE NOT NULL,
                filename VARCHAR(500) NOT NULL,
                original_filename VARCHAR(500) NOT NULL,
                video_path VARCHAR(500) NOT NULL,
                result_video_path VARCHAR(500),
                video_info JSON NOT NULL,
                tracking_data JSON NOT NULL,
                behavior_analysis JSON NOT NULL,
                total_tracks INT NOT NULL,
                tracked_ids JSON NOT NULL,
                top_tracks JSON NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        ''')
        
        # 创建追踪目标详情表（新：7个装配步骤）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS track_details (
                id INT AUTO_INCREMENT PRIMARY KEY,
                analysis_id VARCHAR(255) NOT NULL,
                track_id INT NOT NULL,
                total_time DECIMAL(10,2) NOT NULL,
                robotpick_time DECIMAL(10,2) NOT NULL DEFAULT 0,
                scan_time DECIMAL(10,2) NOT NULL DEFAULT 0,
                robotfix_time DECIMAL(10,2) NOT NULL DEFAULT 0,
                handtighten_time DECIMAL(10,2) NOT NULL DEFAULT 0,
                electricgun_time DECIMAL(10,2) NOT NULL DEFAULT 0,
                robotreturn_time DECIMAL(10,2) NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (analysis_id) REFERENCES analysis_history (analysis_id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        ''')
        
        # 创建索引
        try:
            cursor.execute('CREATE INDEX idx_analysis_id ON analysis_history (analysis_id)')
        except:
            pass  # 索引已存在
        try:
            cursor.execute('CREATE INDEX idx_created_at ON analysis_history (created_at)')
        except:
            pass  # 索引已存在
        try:
            cursor.execute('CREATE INDEX idx_track_analysis_id ON track_details (analysis_id)')
        except:
            pass  # 索引已存在
        
        conn.close()
    
    def save_analysis_result(self, analysis_data: Dict) -> bool:
        """
        保存分析结果到数据库
        
        Args:
            analysis_data: 分析结果数据
            
        Returns:
            bool: 保存是否成功
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # 准备数据
            analysis_id = analysis_data.get('analysis_id')
            video_info = analysis_data.get('video_info', {})
            tracking_data = analysis_data.get('tracking_data', {})
            behavior_analysis = analysis_data.get('behavior_analysis', {})
            
            # 插入主记录（使用ON DUPLICATE KEY UPDATE）
            cursor.execute('''
                INSERT INTO analysis_history (
                    analysis_id, filename, original_filename, video_path, result_video_path,
                    video_info, tracking_data, behavior_analysis, total_tracks, tracked_ids, top_tracks
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    filename = VALUES(filename),
                    original_filename = VALUES(original_filename),
                    video_path = VALUES(video_path),
                    result_video_path = VALUES(result_video_path),
                    video_info = VALUES(video_info),
                    tracking_data = VALUES(tracking_data),
                    behavior_analysis = VALUES(behavior_analysis),
                    total_tracks = VALUES(total_tracks),
                    tracked_ids = VALUES(tracked_ids),
                    top_tracks = VALUES(top_tracks),
                    updated_at = CURRENT_TIMESTAMP
            ''', (
                analysis_id,
                analysis_data.get('filename', ''),
                analysis_data.get('original_filename', ''),
                analysis_data.get('video_path', ''),
                analysis_data.get('result_video_path', ''),
                json.dumps(video_info, ensure_ascii=False),
                json.dumps(tracking_data, ensure_ascii=False),
                json.dumps(behavior_analysis, ensure_ascii=False),
                analysis_data.get('total_tracks', 0),
                json.dumps(analysis_data.get('tracked_ids', []), ensure_ascii=False),
                json.dumps(analysis_data.get('top_tracks', []), ensure_ascii=False)
            ))
            
            # 删除旧的追踪详情记录
            cursor.execute('DELETE FROM track_details WHERE analysis_id = %s', (analysis_id,))

            # 插入追踪详情记录（新：7个装配步骤）
            track_behaviors = behavior_analysis.get('track_behaviors', {})
            for track_id, track_data in track_behaviors.items():
                cursor.execute('''
                    INSERT INTO track_details (
                        analysis_id, track_id, total_time,
                        gettool_time, robotpick_time, robotfix_time,
                        handtighten_time, scan_time, electricgun_time, robotreturn_time
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ''', (
                    analysis_id,
                    int(track_id),
                    track_data.get('total_time', 0),
                    track_data.get('RobotPick', 0),
                    track_data.get('Scan', 0),
                    track_data.get('RobotFix', 0),
                    track_data.get('HandTighten', 0),
                    track_data.get('ElectricGun', 0),
                    track_data.get('RobotReturn', 0)
                ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"保存分析结果失败: {e}")
            return False
    
    def get_analysis_history_all(self) -> List[Dict]:
        """
        获取所有分析历史记录
        
        Returns:
            List[Dict]: 历史记录列表
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    analysis_id, filename, original_filename, video_path, result_video_path,
                    video_info, total_tracks, tracked_ids, top_tracks, created_at
                FROM analysis_history 
                ORDER BY created_at DESC
            ''')
            
            results = []
            for row in cursor.fetchall():
                print(f"数据库原始时间: {row[9]} (类型: {type(row[9])})")
                # 将datetime对象转换为字符串，保持原始格式
                created_at_str = row[9].strftime('%Y-%m-%d %H:%M:%S') if hasattr(row[9], 'strftime') else str(row[9])
                print(f"转换后的时间字符串: {created_at_str}")
                results.append({
                    'analysis_id': row[0],
                    'filename': row[1],
                    'original_filename': row[2],
                    'video_path': row[3],
                    'result_video_path': row[4] if row[4] else f"http://10.3.11.55:5000/api/video/results/{row[0]}_tracked.mp4",
                    'video_info': json.loads(row[5]),
                    'total_tracks': row[6],
                    'tracked_ids': json.loads(row[7]),
                    'top_tracks': json.loads(row[8]),
                    'created_at': created_at_str  # 使用转换后的字符串
                })
            
            conn.close()
            print(f"返回给API的时间数据: {[r['created_at'] for r in results]}")
            return results
            
        except Exception as e:
            print(f"获取所有分析历史失败: {e}")
            return []
    
    def get_analysis_history_by_days(self, days: int) -> List[Dict]:
        """
        获取最近N天的分析历史记录
        
        Args:
            days: 天数
            
        Returns:
            List[Dict]: 历史记录列表
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # 计算N天前的日期
            cursor.execute('''
                SELECT 
                    analysis_id, filename, original_filename, video_path, result_video_path,
                    video_info, total_tracks, tracked_ids, top_tracks, created_at
                FROM analysis_history 
                WHERE created_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
                ORDER BY created_at DESC
            ''', (days,))
            
            results = []
            for row in cursor.fetchall():
                # 将datetime对象转换为字符串，保持原始格式
                created_at_str = row[9].strftime('%Y-%m-%d %H:%M:%S') if hasattr(row[9], 'strftime') else str(row[9])
                results.append({
                    'analysis_id': row[0],
                    'filename': row[1],
                    'original_filename': row[2],
                    'video_path': row[3],
                    'result_video_path': row[4] if row[4] else f"http://10.3.11.55:5000/api/video/results/{row[0]}_tracked.mp4",
                    'video_info': json.loads(row[5]),
                    'total_tracks': row[6],
                    'tracked_ids': json.loads(row[7]),
                    'top_tracks': json.loads(row[8]),
                    'created_at': created_at_str  # 使用转换后的字符串
                })
            
            conn.close()
            return results
            
        except Exception as e:
            print(f"获取最近{days}天分析历史失败: {e}")
            return []
    
    def get_analysis_history_all_paginated(self, page: int = 1, per_page: int = 10) -> tuple[List[Dict], int]:
        """
        获取所有分析历史记录（分页）
        
        Args:
            page: 页码（从1开始）
            per_page: 每页条数
            
        Returns:
            tuple: (历史记录列表, 总记录数)
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # 获取总记录数
            cursor.execute('SELECT COUNT(*) FROM analysis_history')
            total = cursor.fetchone()[0]
            
            # 计算偏移量
            offset = (page - 1) * per_page
            
            # 获取分页数据
            cursor.execute('''
                SELECT 
                    analysis_id, filename, original_filename, video_path, result_video_path,
                    video_info, total_tracks, tracked_ids, top_tracks, created_at
                FROM analysis_history 
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
            ''', (per_page, offset))
            
            results = []
            for row in cursor.fetchall():
                # 将datetime对象转换为字符串，保持原始格式
                created_at_str = row[9].strftime('%Y-%m-%d %H:%M:%S') if hasattr(row[9], 'strftime') else str(row[9])
                results.append({
                    'analysis_id': row[0],
                    'filename': row[1],
                    'original_filename': row[2],
                    'video_path': row[3],
                    'result_video_path': row[4] if row[4] else f"http://10.3.11.55:5000/api/video/results/{row[0]}_tracked.mp4",
                    'video_info': json.loads(row[5]),
                    'total_tracks': row[6],
                    'tracked_ids': json.loads(row[7]),
                    'top_tracks': json.loads(row[8]),
                    'created_at': created_at_str  # 使用转换后的字符串
                })
            
            conn.close()
            return results, total
            
        except Exception as e:
            print(f"获取分页分析历史失败: {e}")
            return [], 0
    
    def get_analysis_history_by_days_paginated(self, days: int, page: int = 1, per_page: int = 10) -> tuple[List[Dict], int]:
        """
        获取最近N天的分析历史记录（分页）
        
        Args:
            days: 天数
            page: 页码（从1开始）
            per_page: 每页条数
            
        Returns:
            tuple: (历史记录列表, 总记录数)
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # 获取总记录数
            cursor.execute('''
                SELECT COUNT(*) FROM analysis_history 
                WHERE created_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
            ''', (days,))
            total = cursor.fetchone()[0]
            
            # 计算偏移量
            offset = (page - 1) * per_page
            
            # 获取分页数据
            cursor.execute('''
                SELECT 
                    analysis_id, filename, original_filename, video_path, result_video_path,
                    video_info, total_tracks, tracked_ids, top_tracks, created_at
                FROM analysis_history 
                WHERE created_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
            ''', (days, per_page, offset))
            
            results = []
            for row in cursor.fetchall():
                # 将datetime对象转换为字符串，保持原始格式
                created_at_str = row[9].strftime('%Y-%m-%d %H:%M:%S') if hasattr(row[9], 'strftime') else str(row[9])
                results.append({
                    'analysis_id': row[0],
                    'filename': row[1],
                    'original_filename': row[2],
                    'video_path': row[3],
                    'result_video_path': row[4] if row[4] else f"http://10.3.11.55:5000/api/video/results/{row[0]}_tracked.mp4",
                    'video_info': json.loads(row[5]),
                    'total_tracks': row[6],
                    'tracked_ids': json.loads(row[7]),
                    'top_tracks': json.loads(row[8]),
                    'created_at': created_at_str  # 使用转换后的字符串
                })
            
            conn.close()
            return results, total
            
        except Exception as e:
            print(f"获取最近{days}天分页分析历史失败: {e}")
            return [], 0
    
    def get_analysis_by_id(self, analysis_id: str) -> Optional[Dict]:
        """
        根据分析ID获取完整分析结果
        
        Args:
            analysis_id: 分析ID
            
        Returns:
            Optional[Dict]: 分析结果，如果不存在返回None
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    analysis_id, filename, original_filename, video_path, result_video_path,
                    video_info, tracking_data, behavior_analysis, total_tracks, 
                    tracked_ids, top_tracks, created_at
                FROM analysis_history 
                WHERE analysis_id = %s
            ''', (analysis_id,))
            
            row = cursor.fetchone()
            if row:
                result = {
                    'analysis_id': row[0],
                    'filename': row[1],
                    'original_filename': row[2],
                    'video_path': row[3],
                    'result_video_path': row[4] if row[4] else f"http://10.3.11.55:5000/api/video/results/{row[0]}_tracked.mp4",
                    'video_info': json.loads(row[5]),
                    'tracking_data': json.loads(row[6]),
                    'behavior_analysis': json.loads(row[7]),
                    'total_tracks': row[8],
                    'tracked_ids': json.loads(row[9]),
                    'top_tracks': json.loads(row[10]),
                    'created_at': row[11]
                }
                
                # 获取追踪详情（新：7个装配步骤）
                cursor.execute('''
                    SELECT track_id, total_time,
                           gettool_time, robotpick_time, robotfix_time,
                           handtighten_time, scan_time, electricgun_time, robotreturn_time
                    FROM track_details
                    WHERE analysis_id = %s
                    ORDER BY track_id
                ''', (analysis_id,))

                track_details = []
                for track_row in cursor.fetchall():
                    track_details.append({
                        'track_id': track_row[0],
                        'total_time': track_row[1],
                        'RobotPick': track_row[2],
                        'Scan': track_row[3],
                        'RobotFix': track_row[4],
                        'HandTighten': track_row[5],
                        'ElectricGun': track_row[6],
                        'RobotReturn': track_row[7]
                    })

                result['track_details'] = track_details
                conn.close()
                return result
            
            conn.close()
            return None
            
        except Exception as e:
            print(f"获取分析结果失败: {e}")
            return None
    
    def delete_analysis(self, analysis_id: str) -> bool:
        """
        删除分析记录
        
        Args:
            analysis_id: 分析ID
            
        Returns:
            bool: 删除是否成功
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # 删除追踪详情
            cursor.execute('DELETE FROM track_details WHERE analysis_id = %s', (analysis_id,))
            
            # 删除主记录
            cursor.execute('DELETE FROM analysis_history WHERE analysis_id = %s', (analysis_id,))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"删除分析记录失败: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """
        获取数据库统计信息
        
        Returns:
            Dict: 统计信息
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # 总分析次数
            cursor.execute('SELECT COUNT(*) FROM analysis_history')
            total_analyses = cursor.fetchone()[0]
            
            # 总追踪目标数
            cursor.execute('SELECT SUM(total_tracks) FROM analysis_history')
            total_tracks = cursor.fetchone()[0] or 0
            
            # 最近分析时间
            cursor.execute('SELECT MAX(created_at) FROM analysis_history')
            last_analysis = cursor.fetchone()[0]
            
            # 按日期统计
            cursor.execute('''
                SELECT DATE(created_at) as date, COUNT(*) as count
                FROM analysis_history 
                GROUP BY DATE(created_at)
                ORDER BY date DESC
                LIMIT 7
            ''')
            daily_stats = [{'date': row[0], 'count': row[1]} for row in cursor.fetchall()]
            
            conn.close()
            
            return {
                'total_analyses': total_analyses,
                'total_tracks': total_tracks,
                'last_analysis': last_analysis,
                'daily_stats': daily_stats
            }
            
        except Exception as e:
            print(f"获取统计信息失败: {e}")
            return {
                'total_analyses': 0,
                'total_tracks': 0,
                'last_analysis': None,
                'daily_stats': []
            }

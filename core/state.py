tracking_system = None
db_manager = None

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
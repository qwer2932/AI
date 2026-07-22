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
    'step_history': []
}
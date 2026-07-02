def calculate_line_balance(analysis_results):
    """
    基于行为分析数据计算线平衡率
    每个工位的价值时间优先使用第一个完整循环的 total_time
    """
    workstations = []
    video_details = []

    for i, analysis in enumerate(analysis_results):
        behavior = analysis.get('behavior_analysis', {})
        cycles = behavior.get('cycles', [])
        track_behaviors = behavior.get('track_behaviors', {})
        top_tracks = behavior.get('top_tracks', [])

        if cycles:
            # 取第一个完整循环（如果存在），否则取第一个循环
            cycle = None
            for c in cycles:
                if c.get('complete', False):
                    cycle = c
                    break
            if cycle is None and cycles:
                cycle = cycles[0]
            if cycle:
                value_time = cycle['total_time']
            else:
                value_time = 0
        else:
            # 回退：取总时间最长的 track
            if top_tracks:
                main_track = top_tracks[0]
                track_id = main_track['track_id']
                beh = track_behaviors.get(str(track_id), {})
                value_time = beh.get('total_time', 0)
            else:
                value_time = 0

        video_info = analysis.get('video_info', {})
        cycle_time = video_info.get('duration', 0)
        workstation_id = i + 1

        workstations.append({
            'track_id': workstation_id,
            'cycle_time': cycle_time,
            'value_time': value_time,
            'walking_time': 0,
            'waiting_time': 0,
            'non_value_time': 0,
            'filename': analysis.get('filename', ''),
            'analysis_id': analysis['analysis_id'],
            'total_frames': video_info.get('total_frames', 0),
            'value_frames': int(value_time * video_info.get('fps', 25)) if video_info.get('fps', 0) > 0 else 0
        })

        video_details.append({
            'filename': analysis.get('filename', ''),
            'analysis_id': analysis['analysis_id'],
            'workstation_id': workstation_id,
            'cycle_time': cycle_time,
            'value_time': value_time,
            'walking_time': 0,
            'waiting_time': 0,
            'non_value_time': 0,
            'efficiency': (value_time / cycle_time * 100) if cycle_time > 0 else 0,
            'video_duration': cycle_time,
            'total_frames': video_info.get('total_frames', 0),
            'value_frames': int(value_time * video_info.get('fps', 25)) if video_info.get('fps', 0) > 0 else 0
        })

    if not workstations:
        return {
            'global_result': {
                'balance_rate': 0,
                'total_value_time': 0,
                'workstation_count': 0,
                'bottleneck_value_time': 0,
                'bottleneck_id': None,
                'total_cycle_time': 0,
                'video_count': len(analysis_results)
            },
            'workstations': [],
            'video_details': []
        }

    max_value_time = max(w['value_time'] for w in workstations)
    bottleneck_id = next((w['track_id'] for w in workstations if w['value_time'] == max_value_time), None)
    total_value_time = sum(w['value_time'] for w in workstations)
    workstation_count = len(workstations)
    balance_rate = (total_value_time / (max_value_time * workstation_count) * 100) if (workstation_count > 0 and max_value_time > 0) else 0

    return {
        'global_result': {
            'balance_rate': round(balance_rate, 2),
            'total_value_time': round(total_value_time, 2),
            'workstation_count': workstation_count,
            'bottleneck_value_time': round(max_value_time, 2),
            'bottleneck_id': bottleneck_id,
            'total_cycle_time': round(sum(w['cycle_time'] for w in workstations), 2),
            'video_count': len(analysis_results)
        },
        'workstations': workstations,
        'video_details': video_details
    }
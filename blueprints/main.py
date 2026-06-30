import os
from flask import Blueprint, request, jsonify, send_from_directory, send_file, current_app
from core.utils import allowed_file

bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    return send_from_directory('static', 'index.html')

@bp.route('/test')
def test_page():
    return send_from_directory('static', 'test.html')

@bp.route('/video-test')
def video_test_page():
    return send_from_directory('static', 'video_test.html')

@bp.route('/browser-test')
def browser_test_page():
    return send_from_directory('static', 'browser_test.html')

@bp.route('/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

@bp.route('/video/<filename>')
def serve_video_universal(filename):
    try:
        file_path = os.path.join(current_app.config['RESULTS_FOLDER'], filename)
        if not os.path.exists(file_path):
            return jsonify({'error': '视频文件不存在'}), 404
        file_size = os.path.getsize(file_path)
        range_header = request.headers.get('Range')
        if range_header:
            byte_start = 0
            byte_end = file_size - 1
            if range_header.startswith('bytes='):
                parts = range_header[6:].split('-')
                if parts[0]:
                    byte_start = int(parts[0])
                if parts[1]:
                    byte_end = int(parts[1])
            byte_start = max(0, byte_start)
            byte_end = min(file_size - 1, byte_end)
            if byte_start > byte_end:
                return jsonify({'error': 'Invalid range'}), 416
            content_length = byte_end - byte_start + 1
            def generate():
                with open(file_path, 'rb') as f:
                    f.seek(byte_start)
                    remaining = content_length
                    while remaining > 0:
                        chunk = f.read(min(1024*1024, remaining))
                        if not chunk:
                            break
                        remaining -= len(chunk)
                        yield chunk
            response = current_app.response_class(
                generate(), 206,
                {
                    'Content-Type': 'video/mp4',
                    'Content-Length': str(content_length),
                    'Content-Range': f'bytes {byte_start}-{byte_end}/{file_size}',
                    'Accept-Ranges': 'bytes',
                    'Cache-Control': 'public, max-age=3600',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
                    'Access-Control-Allow-Headers': 'Range, Content-Range',
                    'X-Content-Type-Options': 'nosniff',
                }
            )
            return response
        else:
            def generate():
                with open(file_path, 'rb') as f:
                    while True:
                        chunk = f.read(1024*1024)
                        if not chunk:
                            break
                        yield chunk
            response = current_app.response_class(
                generate(), 200,
                {
                    'Content-Type': 'video/mp4',
                    'Content-Length': str(file_size),
                    'Accept-Ranges': 'bytes',
                    'Cache-Control': 'public, max-age=3600',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
                    'Access-Control-Allow-Headers': 'Range, Content-Range',
                    'X-Content-Type-Options': 'nosniff',
                }
            )
            return response
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/videos/<path:filename>')
def serve_video(filename):
    return serve_video_universal(filename)

@bp.route('/results/<filename>', methods=['GET', 'HEAD', 'OPTIONS'])
def serve_result_video_direct(filename):
    if request.method == 'OPTIONS':
        response = current_app.response_class()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, HEAD, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Range, Content-Range'
        return response
    return serve_video_universal(filename)

@bp.route('/api/video/results/<filename>')
def get_result_video(filename):
    file_path = os.path.join(current_app.config['RESULTS_FOLDER'], filename)
    if not os.path.exists(file_path):
        return jsonify({'error': '视频文件不存在'}), 404
    mimetype = 'video/x-msvideo' if filename.endswith('.avi') else 'video/mp4'
    return send_file(file_path, mimetype=mimetype, as_attachment=False, download_name=filename, conditional=False)
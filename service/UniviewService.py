# coding=utf-8
"""
宇视摄像头取流服务（基于独立测试脚本）
"""
import ctypes
import threading
import time
import logging
import os
import cv2
import numpy as np
from ctypes import (
    c_int, c_void_p, c_char, c_ushort, c_byte, c_bool,
    POINTER, Structure, byref, CFUNCTYPE, c_uint
)

logger = logging.getLogger(__name__)

# ======================== 摄像头配置 ========================
UNIVIEW_CAMERA_CONFIG = {
    "name": "TestCamera",
    "ip": "10.32.96.36",
    "username": "admin",
    "password": "Sgmw@5050",
    "port": 80,
}
# ===========================================================

# ---------------------------- 加载SDK ----------------------------
DLL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib', 'dll', 'NetDEVSDK.dll')
UNIVIEW_DLL = None

def load_uniview_sdk(dll_path):
    global UNIVIEW_DLL
    if not os.path.exists(dll_path):
        raise FileNotFoundError(f"DLL 不存在: {dll_path}")
    dll_dir = os.path.dirname(dll_path)
    os.environ["PATH"] = dll_dir + ";" + os.environ.get("PATH", "")
    UNIVIEW_DLL = ctypes.CDLL(dll_path, winmode=0)
    logger.info("宇视SDK动态库加载成功")
    return UNIVIEW_DLL

try:
    UNIVIEW_DLL = load_uniview_sdk(DLL_PATH)
except Exception as e:
    logger.warning(f"宇视SDK加载失败: {e}")

# ---------------------------- 常量 ----------------------------
NETDEV_LOGIN_PROTO_PRIVATE = 1
NETDEV_DTYPE_IPC = 1
DEFAULT_HTTP_PORT = 80
NETDEV_MEDIADIR_VIDEO = 0x01
NETDEV_STREAM_MAIN = 0x01
NETDEV_TRANSPROTOCOL_RTPTCP = 1

# ---------------------------- 结构体 ----------------------------
class NETDEV_DEVICE_LOGIN_INFO_S(Structure):
    _fields_ = [
        ("szIPAddr", c_char * 260),
        ("dwPort", c_int),
        ("szUserName", c_char * 132),
        ("szPassword", c_char * 128),
        ("dwLoginProto", c_int),
        ("dwDeviceType", c_int),
        ("byRes", c_byte * 256),
    ]

class NETDEV_SELOG_INFO_S(Structure):
    _fields_ = [
        ("dwSELogCount", c_int),
        ("dwSELogTime", c_int),
        ("byRes", c_byte * 64),
    ]

class NETDEV_PREVIEWINFO_S(Structure):
    _fields_ = [
        ("dwChannelID", c_int),
        ("dwStreamType", c_int),
        ("dwLinkMode", c_int),
        ("hPlayWnd", c_void_p),
        ("dwFluency", c_int),
        ("dwStreamMode", c_int),
        ("dwLiveMode", c_int),
        ("dwDisTributeCloud", c_int),
        ("dwallowDistribution", c_bool),
        ("dwTransType", c_int),
        ("dwStreamProtocol", c_int),
        ("bLoginDataByOpenAPI", c_bool),
        ("byRes", c_byte * 232),
    ]

class NETDEV_PICTURE_DATA_S(Structure):
    _fields_ = [
        ("pucData", c_void_p * 4),
        ("dwLineSize", c_int * 4),
        ("dwPicHeight", c_int),
        ("dwPicWidth", c_int),
        ("dwRenderTimeType", c_int),
        ("tRenderTime", c_int),
    ]

DecodeVideoCallBack = CFUNCTYPE(None, c_void_p, POINTER(NETDEV_PICTURE_DATA_S), c_void_p)

# ---------------------------- SDK函数签名 ----------------------------
if UNIVIEW_DLL is not None:
    UNIVIEW_DLL.NETDEV_Init.restype = c_bool
    UNIVIEW_DLL.NETDEV_Cleanup.restype = c_bool
    UNIVIEW_DLL.NETDEV_GetLastError.argtypes = []
    UNIVIEW_DLL.NETDEV_GetLastError.restype = c_int

    UNIVIEW_DLL.NETDEV_Login_V30.argtypes = [POINTER(NETDEV_DEVICE_LOGIN_INFO_S),
                                             POINTER(NETDEV_SELOG_INFO_S)]
    UNIVIEW_DLL.NETDEV_Login_V30.restype = c_void_p
    UNIVIEW_DLL.NETDEV_Logout.argtypes = [c_void_p]
    UNIVIEW_DLL.NETDEV_Logout.restype = c_bool

    UNIVIEW_DLL.NETDEV_RealPlay.argtypes = [c_void_p, POINTER(NETDEV_PREVIEWINFO_S),
                                            c_void_p, c_void_p]
    UNIVIEW_DLL.NETDEV_RealPlay.restype = c_void_p
    UNIVIEW_DLL.NETDEV_StopRealPlay.argtypes = [c_void_p]
    UNIVIEW_DLL.NETDEV_StopRealPlay.restype = c_bool

    UNIVIEW_DLL.NETDEV_SetPlayDecodeVideoCB.argtypes = [c_void_p, DecodeVideoCallBack, c_void_p]
    UNIVIEW_DLL.NETDEV_SetPlayDecodeVideoCB.restype = c_bool

    # 初始化 SDK
    if not UNIVIEW_DLL.NETDEV_Init():
        err = UNIVIEW_DLL.NETDEV_GetLastError()
        logger.error(f"SDK初始化失败，错误码: {err}")
        UNIVIEW_DLL = None
    else:
        logger.info("宇视SDK初始化成功")

# ---------------------------- 取流类（独立脚本稳定版本）----------------------------
class UniviewStreamCapture:
    def __init__(self):
        self.camera_name = UNIVIEW_CAMERA_CONFIG["name"]
        self.login_id = None
        self.play_id = None
        self._frame = None
        self._frame_lock = threading.Lock()
        self._running = False
        self._stop_event = threading.Event()
        self._decode_cb_ref = None
        self._frame_count = 0

        if UNIVIEW_DLL is None:
            raise RuntimeError("宇视SDK未加载")

        self._login_camera()
        self._start_preview()

    def _login_camera(self):
        config = UNIVIEW_CAMERA_CONFIG
        login_info = NETDEV_DEVICE_LOGIN_INFO_S()
        login_info.szIPAddr = config["ip"].encode('utf-8')
        login_info.dwPort = config["port"]
        login_info.szUserName = config["username"].encode('utf-8')
        login_info.szPassword = config["password"].encode('utf-8')
        login_info.dwLoginProto = NETDEV_LOGIN_PROTO_PRIVATE
        login_info.dwDeviceType = NETDEV_DTYPE_IPC
        selog_info = NETDEV_SELOG_INFO_S()

        login_id = UNIVIEW_DLL.NETDEV_Login_V30(byref(login_info), byref(selog_info))
        if login_id is None or login_id == 0:
            err = UNIVIEW_DLL.NETDEV_GetLastError()
            raise RuntimeError(f"登录失败 [{config['ip']}:{config['port']}] 错误码: {err}")
        self.login_id = login_id
        logger.info(f"宇视摄像头登录成功: {self.camera_name}")

    def _decode_callback(self, play_handle, pstPictureData, lpUser):
        print(">>> 解码回调触发 <<<")
        if not pstPictureData:
            return
        try:
            data = pstPictureData.contents
            width = data.dwPicWidth
            height = data.dwPicHeight
            if width <= 0 or height <= 0:
                return

            self._frame_count += 1
            if self._frame_count % 30 == 1:
                logger.info(f"宇视回调帧 #{self._frame_count}: {width}x{height}")

            # 获取 Y、U、V 平面指针
            y_ptr = data.pucData[0]
            u_ptr = data.pucData[1]
            v_ptr = data.pucData[2]
            if not y_ptr or not u_ptr or not v_ptr:
                return

            # 使用行跨距计算实际数据长度
            y_size = data.dwLineSize[0] * height
            u_size = data.dwLineSize[1] * (height // 2)
            v_size = data.dwLineSize[2] * (height // 2)

            y_data = ctypes.string_at(y_ptr, y_size)
            u_data = ctypes.string_at(u_ptr, u_size)
            v_data = ctypes.string_at(v_ptr, v_size)

            y_plane = np.frombuffer(y_data, dtype=np.uint8).reshape((height, data.dwLineSize[0]))
            u_plane = np.frombuffer(u_data, dtype=np.uint8).reshape((height // 2, data.dwLineSize[1]))
            v_plane = np.frombuffer(v_data, dtype=np.uint8).reshape((height // 2, data.dwLineSize[2]))

            # 裁剪到实际宽
            y_plane = y_plane[:, :width]
            u_plane = u_plane[:, :width//2]
            v_plane = v_plane[:, :width//2]

            yuv = np.concatenate([y_plane.flatten(), u_plane.flatten(), v_plane.flatten()])
            yuv = yuv.reshape((height * 3 // 2, width))
            bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

            with self._frame_lock:
                self._frame = bgr
        except Exception as e:
            logger.error(f"解码回调异常: {e}")

    def _start_preview(self):
        if self.login_id is None:
            raise RuntimeError("未登录")

        preview_info = NETDEV_PREVIEWINFO_S()
        preview_info.dwChannelID = 0
        preview_info.dwStreamType = NETDEV_STREAM_MAIN
        preview_info.dwLinkMode = NETDEV_TRANSPROTOCOL_RTPTCP
        preview_info.hPlayWnd = None
        preview_info.dwFluency = 0
        preview_info.dwStreamMode = 0
        preview_info.dwLiveMode = 0
        preview_info.dwDisTributeCloud = 0
        preview_info.dwallowDistribution = False
        preview_info.dwTransType = 0
        preview_info.dwStreamProtocol = 0
        preview_info.bLoginDataByOpenAPI = False

        play_id = UNIVIEW_DLL.NETDEV_RealPlay(self.login_id, byref(preview_info), None, None)
        if play_id is None or play_id == 0:
            err = UNIVIEW_DLL.NETDEV_GetLastError()
            raise RuntimeError(f"启动预览失败，错误码: {err}")
        self.play_id = play_id
        logger.info(f"预览启动成功，play_id={play_id}")

        cb = DecodeVideoCallBack(self._decode_callback)
        self._decode_cb_ref = cb
        if not UNIVIEW_DLL.NETDEV_SetPlayDecodeVideoCB(self.play_id, cb, None):
            err = UNIVIEW_DLL.NETDEV_GetLastError()
            UNIVIEW_DLL.NETDEV_StopRealPlay(self.play_id)
            self.play_id = None
            raise RuntimeError(f"注册解码回调失败，错误码: {err}")

        self._running = True
        logger.info("宇视解码回调注册成功")

    def get_frame(self):
        with self._frame_lock:
            return self._frame.copy() if self._frame is not None else None

    def is_running(self):
        return self._running

    def stop(self):
        self._running = False
        self._stop_event.set()
        if self.play_id:
            UNIVIEW_DLL.NETDEV_StopRealPlay(self.play_id)
            self.play_id = None
        if self.login_id:
            UNIVIEW_DLL.NETDEV_Logout(self.login_id)
            self.login_id = None
        UNIVIEW_DLL.NETDEV_Cleanup()
        logger.info(f"宇视流已停止: {self.camera_name}")

    def __del__(self):
        self.stop()

# 全局单例（供 api.py 调用）
_stream_capture = None
_stream_capture_lock = threading.Lock()

def get_uniview_stream():
    global _stream_capture
    if UNIVIEW_DLL is None:
        return None
    with _stream_capture_lock:
        if _stream_capture is None:
            try:
                _stream_capture = UniviewStreamCapture()
            except Exception as e:
                logger.error(f"创建宇视流捕获失败: {e}")
                _stream_capture = None
        return _stream_capture

def release_uniview_stream():
    global _stream_capture
    with _stream_capture_lock:
        if _stream_capture:
            _stream_capture.stop()
            _stream_capture = None
            logger.info("宇视流捕获已释放")
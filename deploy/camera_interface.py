"""
deploy/camera_interface.py
==========================
Ultra-low-latency Universal Camera Interface for Jetson Orin Nano:
- Hardware-accelerated CSI interface via GStreamer (nvarguscamerasrc)
- Low-latency USB camera interface via V4L2 with zero-buffering policy
- RTSP / Video file test stream support
- Dedicated daemon reader thread guaranteeing newest-frame access in < 1ms
- Automatic watchdog with auto-reconnection on camera disconnect
"""

from __future__ import annotations

import logging
import platform
import subprocess
import threading
import time
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np

log = logging.getLogger("apex_camera")


def build_csi_pipeline(
    sensor_id: int = 0,
    capture_width: int = 1280,
    capture_height: int = 720,
    display_width: int = 640,
    display_height: int = 360,
    framerate: int = 30,
    flip_method: int = 0,
) -> str:
    """
    Construct hardware-accelerated GStreamer pipeline string for Jetson CSI cameras
    (e.g., Sony IMX219, IMX477) using NVIDIA Jetson ISP nvarguscamerasrc.
    """
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )


class UniversalCamera:
    """
    Threaded, non-blocking camera reader that continuously drains frames
    to enforce a strict frame-drop policy and zero buffer latency.
    """

    def __init__(
        self,
        camera_type: str = "auto",
        device_id: int = 0,
        resolution: Tuple[int, int] = (640, 360),
        fps: int = 30,
        flip_method: int = 0,
        csi_sensor_id: int = 0,
        rtsp_url: str = "",
        reconnect_interval_sec: float = 5.0,
    ) -> None:
        self.camera_type = camera_type.lower()
        self.device_id = device_id
        self.width, self.height = resolution
        self.fps = fps
        self.flip_method = flip_method
        self.csi_sensor_id = csi_sensor_id
        self.rtsp_url = rtsp_url
        self.reconnect_interval = reconnect_interval_sec

        self.cap: Optional[cv2.VideoCapture] = None
        self.actual_source_type: str = "unknown"

        self._latest_frame: Optional[np.ndarray] = None
        self._frame_timestamp: float = 0.0
        self._frame_lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._connected = False
        self._consecutive_grab_failures = 0

        self.open()

    def open(self) -> bool:
        """Attempt to open camera connection based on configured type."""
        log.info("Opening camera interface (requested: %s)...", self.camera_type)

        if self.camera_type == "auto":
            success = self._try_open_csi() or self._try_open_usb() or self._try_open_rtsp()
        elif self.camera_type == "csi":
            success = self._try_open_csi()
        elif self.camera_type == "usb":
            success = self._try_open_usb()
        elif self.camera_type == "rtsp":
            success = self._try_open_rtsp()
        else:
            success = self._try_open_usb()

        if success and self.cap is not None and self.cap.isOpened():
            self._connected = True
            self._consecutive_grab_failures = 0
            log.info("Camera successfully initialized: %s (%dx%d @ %d FPS)", self.actual_source_type, self.width, self.height, self.fps)
            self._start_reader_thread()
            return True
        else:
            self._connected = False
            log.warning("Could not establish camera stream. Auto-reconnect watchdog will monitor.")
            self._start_reader_thread()
            return False

    def _try_open_csi(self) -> bool:
        """Attempt opening CSI camera via nvarguscamerasrc."""
        pipeline = build_csi_pipeline(
            sensor_id=self.csi_sensor_id,
            capture_width=1280,
            capture_height=720,
            display_width=self.width,
            display_height=self.height,
            framerate=self.fps,
            flip_method=self.flip_method,
        )
        try:
            log.info("Attempting CSI pipeline: %s", pipeline)
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    self.cap = cap
                    self.actual_source_type = f"CSI (sensor {self.csi_sensor_id})"
                    return True
                cap.release()
        except Exception as e:
            log.debug("CSI camera open failed: %s", e)
        return False

    def _try_open_usb(self) -> bool:
        """Attempt opening USB / V4L2 camera."""
        backend = cv2.CAP_V4L2 if platform.system() == "Linux" else cv2.CAP_ANY
        try:
            log.info("Attempting USB camera (device_id=%d)...", self.device_id)
            cap = cv2.VideoCapture(self.device_id, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, self.fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Zero buffering policy

                ret, frame = cap.read()
                if ret and frame is not None:
                    self.cap = cap
                    self.actual_source_type = f"USB (index {self.device_id})"
                    return True
                cap.release()
        except Exception as e:
            log.debug("USB camera open failed: %s", e)
        return False

    def _try_open_rtsp(self) -> bool:
        """Attempt opening RTSP or video file stream."""
        if not self.rtsp_url:
            return False
        try:
            log.info("Attempting RTSP stream: %s", self.rtsp_url)
            cap = cv2.VideoCapture(self.rtsp_url)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    self.cap = cap
                    self.actual_source_type = "RTSP / Network Stream"
                    return True
                cap.release()
        except Exception as e:
            log.debug("RTSP open failed: %s", e)
        return False

    def _start_reader_thread(self) -> None:
        """Start the background frame draining daemon."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, name="CameraReaderThread", daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        """Continuous grab loop to discard old frames and maintain zero-latency newest frame."""
        last_reconnect_attempt = 0.0

        while self._running:
            if not self._connected or self.cap is None or not self.cap.isOpened():
                now = time.time()
                if now - last_reconnect_attempt >= self.reconnect_interval:
                    last_reconnect_attempt = now
                    log.info("Watchdog: Attempting camera reconnection...")
                    if self.cap:
                        try:
                            self.cap.release()
                        except Exception:
                            pass
                    if self.camera_type == "auto":
                        self._connected = self._try_open_csi() or self._try_open_usb() or self._try_open_rtsp()
                    elif self.camera_type == "csi":
                        self._connected = self._try_open_csi()
                    else:
                        self._connected = self._try_open_usb()
                time.sleep(0.1)
                continue

            try:
                # Direct grab/retrieve to prevent OpenCV internal buffer buildup
                grabbed = self.cap.grab()
                if not grabbed:
                    self._consecutive_grab_failures += 1
                    if self._consecutive_grab_failures > 30:
                        log.warning("Lost camera feed (>30 failed grabs). Triggering reconnect.")
                        self._connected = False
                    time.sleep(0.01)
                    continue

                ret, frame = self.cap.retrieve()
                if ret and frame is not None:
                    self._consecutive_grab_failures = 0
                    if frame.shape[:2] != (self.height, self.width):
                        frame = cv2.resize(frame, (self.width, self.height))

                    with self._frame_lock:
                        self._latest_frame = frame
                        self._frame_timestamp = time.perf_counter()
                else:
                    self._consecutive_grab_failures += 1
            except Exception as e:
                log.error("Camera capture exception: %s", e)
                self._connected = False
                time.sleep(0.1)

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Retrieve the latest captured frame without blocking.
        Returns:
            (success: bool, frame: np.ndarray, timestamp: float)
        """
        with self._frame_lock:
            if self._latest_frame is None:
                return False, None, 0.0
            return True, self._latest_frame.copy(), self._frame_timestamp

    @property
    def is_connected(self) -> bool:
        return self._connected

    def release(self) -> None:
        """Stop background reader thread and release hardware handle."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        self._connected = False
        log.info("Camera interface released.")

    def __del__(self) -> None:
        self.release()

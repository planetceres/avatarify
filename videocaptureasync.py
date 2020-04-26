# https://github.com/gilbertfrancois/video-capture-async

import threading
import cv2
import time
import numpy as np
import pyrealsense2 as rs


WARMUP_TIMEOUT = 10.0


class VideoCaptureAsync:
    def __init__(self, src=0, width=640, height=480):
        self.src = src

        # Create a pipeline
        self.pipeline = rs.pipeline()

        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, 30)

        # Start streaming
        self.pipeline.start(config)

        frame = self.pipeline.wait_for_frames()
        color_frame = frame.get_color_frame()
        self.grabbed = True

        self.frame = np.asanyarray(color_frame.get_data())[:, :, ::-1]
        self.started = False
        self.read_lock = threading.Lock()

    def set(self, var1, var2):
        self.cap.set(var1, var2)

    def isOpened(self):
        return self.cap.isOpened()

    def start(self):
        if self.started:
            print('[!] Asynchronous video capturing has already been started.')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()

        # (warmup) wait for the first successfully grabbed frame 
        warmup_start_time = time.time()
        while not self.grabbed:
            warmup_elapsed_time = (time.time() - warmup_start_time)
            if warmup_elapsed_time > WARMUP_TIMEOUT:
                raise RuntimeError(f"Failed to succesfully grab frame from the camera (timeout={WARMUP_TIMEOUT}s). Try to restart.")

            time.sleep(0.5)
    
        return self

    def update(self):
        while self.started:
            grabbed = False
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())[:, :, ::-1]
            grabbed = True
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.pipeline.stop()

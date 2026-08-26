import cv2
import time
import queue
import threading
import os
import logging

class RTSPInputStreamer(object):
    '''
    RTSP Stream Input
    read from RTSP stream
    '''
    def __init__(self, url):
        self.url = url
        self._cap = cv2.VideoCapture(self.url)
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self._cap.get(cv2.CAP_PROP_FPS))


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._cap.release()

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self._cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        else:
            raise StopIteration

class FileInputStreamer(object):
    '''
    File Stream Input
    read from video file
    '''
    def __init__(self, path, keep_rate=False):
        self.path = path
        self._cap = cv2.VideoCapture(self.path)
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self._cap.get(cv2.CAP_PROP_FPS))
        self.frame_read = 0
        self.keep_rate = keep_rate

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._cap.release()

    def __iter__(self):
        return self
    
    def __next__(self):
        current_ts = time.time()
        if self.keep_rate:
            if self.frame_read == 0:
                self.start_ts = current_ts
            ts_gap = self.frame_read/self.fps - (current_ts - self.start_ts)
            if ts_gap > 0:
                time.sleep(ts_gap)
            self.frame_read += 1
        ret, frame = self._cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        else:
            raise StopIteration
        




class InputStreamer(object):
    """
    The return value of __next__ is a np.ndarray with shape (H, W, C) and dtype np.uint8 in RGB format
    """
    def __init__(self, source, keep_rate=False) -> None: 
        self.source = source
        if source.endswith(".mp4") and os.path.exists(source[7:]):
            input_streamer = FileInputStreamer(source, keep_rate)
        elif source.startswith("rtsp"):
            input_streamer = RTSPInputStreamer(source)
        else:
            raise NotImplementedError
        self.width = input_streamer.width
        self.height = input_streamer.height
        self.fps = input_streamer.fps
        self._input_queue = queue.Queue(maxsize=60)   
        self._terminate = False
        self._input_thread = threading.Thread(target=self._worker_func, args=(input_streamer, self._input_queue))
        self._input_thread.daemon = True
        self._input_thread.start()

    def _worker_func(self, input_streamer, input_queue):
        """
        input_streamer: cv2.VideoCapture
        input_queue: queue.Queue
        """
        for frame in input_streamer:
            if self._terminate:
                break
            logging.debug(f"InputQueueSize: {input_queue.qsize()}")
            input_queue.put(frame)
        input_queue.put(None)
    
    def terminate(self):
        logging.info("terminate input streamer")
        self._terminate = True

    def __iter__(self):
        return self

    def __next__(self):
        while self._input_queue.empty():
            if self._terminate:
                raise StopIteration
            time.sleep(0.01)
        frame = self._input_queue.get()
        if frame is None:
            raise StopIteration
        return frame




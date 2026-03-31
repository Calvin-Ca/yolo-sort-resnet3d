import os
import subprocess
import numpy as np
import cv2
import logging, queue, threading

class BlankFrame(object):
    pass

class BaseOutputStreamerBase:
    def __init__(self, width, height, fps):
        self.width = width
        self.height = height
        self.fps = fps

    def write(self, frame):
        """
        frame: np.ndarray, shape=(height, width, 3), dtype=np.uint8, RGB
        """
        raise NotImplementedError

class RTSPOutputStreamer(BaseOutputStreamerBase):
    def __init__(self, width, height, fps, push_url):
        super().__init__(width, height, fps)
        self.push_url = push_url
        if os.name == "nt":
            path ='D:/ffmpeg-7.0.2-essentials_build/bin/ffmpeg.exe'
        else:
            path ='/usr/bin/ffmpeg'

        self.command = [
            path,
            '-y', '-an',
            '-f', 'rawvideo',
            '-vcodec','rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', "{}x{}".format(width, height),
            '-r', str(fps),
            '-i', '-',
            '-preset', 'veryfast',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-x264-params', 'keyint=10:bframes=0',
            '-f', 'rtsp',
            '-rtsp_transport', 'tcp',  # 使用TCP推流，linux中一定要有这行
            push_url]

        self.pipe = subprocess.Popen(self.command, shell=False, stdin=subprocess.PIPE)
        self.blank_frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    def write(self, frame):
        try:
            if isinstance(frame, BlankFrame):
                frame = self.blank_frame
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.pipe.stdin.write(frame.astype(np.uint8).tobytes())
        except BrokenPipeError:
            print("BrokenPipeError")
            self.pipe = subprocess.Popen(self.command, shell=False, stdin=subprocess.PIPE)
            self.pipe.stdin.write(frame.astype(np.uint8).tobytes())

class FileOutputStreamer(BaseOutputStreamerBase):
    def __init__(self, width, height, fps, output_path):
        super().__init__(width, height, fps)
        self.output_path = output_path
        if os.name == "nt":
            path = 'D:/ffmpeg-7.0.2-essentials_build/bin/ffmpeg.exe'
        else:
            path = '/usr/bin/ffmpeg'         

        self.command = [
            path,
            '-y', '-an',
            '-f', 'rawvideo',
            '-vcodec','rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', "{}x{}".format(width, height),
            '-r', str(fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'veryfast',
            '-x264-params', 'keyint=10:bframes=0',
            '-movflags', '+faststart',  # 强制moov atom前置
            output_path]
        
        self.pipe = subprocess.Popen(self.command, shell=False, stdin=subprocess.PIPE, bufsize=0)
        self.blank_frame = np.zeros((height, width, 3), dtype=np.uint8)

    def write(self, frame):
        if isinstance(frame, BlankFrame):
            frame = self.blank_frame
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.pipe.stdin.write(frame.astype(np.uint8).tobytes())


class DummyOutputStreamer(BaseOutputStreamerBase):
    def __init__(self, width, height, fps):
        super().__init__(width, height, fps)
    
    def write(self, frame):
        pass


class OutputStreamer(object):
    def __init__(self, target, width, height, fps) -> None:
        self._target = target
        if target.startswith("file://"):
            target = target[7:]
            output_streamer = FileOutputStreamer(width, height, fps, target)
            self.streaming = False
        elif target.startswith("rtsp://"):
            output_streamer = RTSPOutputStreamer(width, height, fps, target)
            self.streaming = True
        elif target == "dummy":
            output_streamer = DummyOutputStreamer(width, height, fps)
            self.streaming = False
        else:
            raise NotImplementedError
        self._output_queue = queue.Queue(maxsize=30)  
        self._terminate = False
        self._output_thread = threading.Thread(target=self._worker_func, args=(output_streamer, self._output_queue))
        self._output_thread.start()
        self._output_streamer = output_streamer  # 保存output_streamer引用，用于后续关闭

    def _worker_func(self, output_streamer, output_queue):
        while True:
            try:
                # 先获取帧，不立即判断None，确保处理完所有帧
                frame = output_queue.get(block=True, timeout=1)
                logging.debug(f"OutputQueueSize: {output_queue.qsize()}")
            except queue.Empty:
                if self._terminate:
                    break
                else:
                    continue

            # 处理帧（即使收到None，也先处理完已有帧）
            if frame is None or self._terminate:
                # 处理完当前帧（若有）后，退出循环
                break
            output_streamer.write(frame)

    def terminate(self):
        logging.info("开始终止输出流，等待剩余帧处理...")
        self._terminate = True
        # 步骤1：等待工作线程处理完队列中所有帧（最多等待5秒，避免无限阻塞）
        self._output_thread.join(timeout=5)
        # 步骤2：关闭ffmpeg的输入管道（告知ffmpeg已无数据输入，触发收尾）
        if hasattr(self._output_streamer, 'pipe') and self._output_streamer.pipe.stdin:
            try:
                self._output_streamer.pipe.stdin.close()
                logging.info("已关闭ffmpeg输入管道")
            except Exception as e:
                logging.warning(f"关闭输入管道失败: {e}")
        # 步骤3：等待ffmpeg进程完成编码（最多等待10秒，确保写入moov atom）
        if hasattr(self._output_streamer, 'pipe'):
            try:
                # wait()会阻塞直到进程结束，返回退出码
                exit_code = self._output_streamer.pipe.wait(timeout=10)
                logging.info(f"ffmpeg进程已结束，退出码: {exit_code}")
            except subprocess.TimeoutExpired:
                # 超时未结束，强制杀死进程（万不得已）
                self._output_streamer.pipe.kill()
                logging.warning("ffmpeg进程超时，已强制终止")
        logging.info("输出流终止完成")

    def write(self, frame):
        if not self._terminate:
            self._output_queue.put(frame)


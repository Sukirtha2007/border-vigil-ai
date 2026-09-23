import cv2

class VideoLoader:
    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def read_first_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Could not read first frame")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return frame

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise StopIteration
        return frame

    def release(self):
        self.cap.release()

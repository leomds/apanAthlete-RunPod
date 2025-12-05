# app/video_utils.py
from typing import Generator, Tuple
import cv2
import numpy as np


def read_video_frames(path: str) -> Tuple[Generator, float, int, Tuple[int, int]]:
    """
    Lê o vídeo e retorna:
      - generator de frames (BGR, np.ndarray)
      - fps
      - número total de frames
      - (largura, altura)
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    def _frames() -> Generator[np.ndarray, None, None]:
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
        finally:
            cap.release()

    return _frames(), fps, frame_count, (width, height)

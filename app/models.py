# app/models.py
from functools import lru_cache

from ultralytics import YOLO
from rtmlib import RTMPose

from .config import MODEL_CFG


@lru_cache(maxsize=1)
def get_yolo_detector() -> YOLO:
    """
    Carrega YOLOv8-L detector pré-treinado em COCO (pessoas).
    O Ultralytics baixa automaticamente 'yolov8l.pt' na primeira vez que usar.
    """
    model = YOLO("yolo11x.pt")  # detector (não pose)
    model.to(MODEL_CFG.device)
    model.fuse()  # pequena otimização
    return model


@lru_cache(maxsize=1)
def get_rtmpose_model() -> RTMPose:
    """
    Carrega RTMPose-X (body7, 384x288) via rtmlib + ONNX Runtime GPU.

    Usamos o zip oficial da OpenMMLab (mesmo que a rtmlib usa internamente).
    """
    pose_url = (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
        "onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip"
    )
    model = RTMPose(
        onnx_model=pose_url,
        backend=MODEL_CFG.backend,
        device=MODEL_CFG.device,
    )
    return model

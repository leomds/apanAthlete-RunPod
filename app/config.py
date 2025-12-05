# app/config.py
from dataclasses import dataclass


@dataclass
class PoseKeypointIndices:
    # Índices COCO-17
    NOSE: int = 0
    LEFT_EYE: int = 1
    RIGHT_EYE: int = 2
    LEFT_EAR: int = 3
    RIGHT_EAR: int = 4
    LEFT_SHOULDER: int = 5
    RIGHT_SHOULDER: int = 6
    LEFT_ELBOW: int = 7
    RIGHT_ELBOW: int = 8
    LEFT_WRIST: int = 9
    RIGHT_WRIST: int = 10
    LEFT_HIP: int = 11
    RIGHT_HIP: int = 12
    LEFT_KNEE: int = 13
    RIGHT_KNEE: int = 14
    LEFT_ANKLE: int = 15
    RIGHT_ANKLE: int = 16


@dataclass
class MetricsConfig:
    # score mínimo do keypoint pra considerar válido
    kpt_score_thr: float = 0.4
    
    # parâmetros do Savitzky-Golay
    smoothing_window: int = 9
    smoothing_polyorder: int = 2
    
    # =========================================================
    # AJUSTE AQUI: Aumentando a altura mínima do salto
    # =========================================================
    # De 0.05 (5cm) para 0.10 (10cm) para evitar falsos positivos ao caminhar
    min_jump_height_m: float = 0.15  
    
    jump_hip_delta_m: float = 0.03


@dataclass
class ModelConfig:
    device: str = "cuda"
    backend: str = "onnxruntime"


POSE_IDXS = PoseKeypointIndices()
METRICS_CFG = MetricsConfig()
MODEL_CFG = ModelConfig()

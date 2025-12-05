# app/reid.py
"""
Módulo de Re-Identification (Re-ID) do atleta.

Ideia principal:
- Extrair um embedding visual (assinatura) do atleta a partir do crop da bbox.
- Manter um histórico (janela deslizante) dos últimos N embeddings.
- Usar a MÉDIA desses embeddings como "assinatura de referência" (centroide).
- Na seleção de alvo (no pipeline), comparar os candidatos com esse centroide
  usando similaridade cosseno, pra não trocar de atleta depois de oclusão.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Deque, List

from collections import deque

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

from .config import MODEL_CFG


# ==============================
#   CONSTANTES DE ReID
# ==============================

# tamanho da janela de histórico de embeddings
DEFAULT_HISTORY = 10

# thresholds de similaridade cosseno
# (0.0 = nada parecido, 1.0 = idêntico)
REID_STRONG_SIM = 0.75   # acima disso consideramos "mesma pessoa" com segurança
REID_WEAK_SIM   = 0.60   # abaixo disso, combinado com IOU fraco, preferimos NÃO trocar de alvo


# ==============================
#   PREPROCESSAMENTO
# ==============================

_REID_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((256, 128)),  # formato típico de ReID (H, W)
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


@lru_cache(maxsize=1)
def get_reid_encoder() -> nn.Module:
    """
    Usa ResNet-18 pré-treinado em ImageNet como extrator de features.
    Não é um modelo de ReID dedicado, mas funciona bem como embedding
    de aparência (cor, textura, forma).
    """
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Remove a FC final -> deixa só o extractor [B, 512, 1, 1]
    modules = list(backbone.children())[:-1]
    encoder = nn.Sequential(*modules)
    encoder.eval()
    encoder.to(MODEL_CFG.device)
    return encoder


def compute_reid_embedding(crop_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Recebe um crop BGR (np.ndarray) do frame original (OpenCV / YOLO).
    Retorna um vetor L2-normalizado (embedding) ou None se algo falhar.
    """
    if crop_bgr is None:
        return None
    if not isinstance(crop_bgr, np.ndarray) or crop_bgr.size == 0:
        return None

    # BGR -> RGB
    crop_rgb = crop_bgr[..., ::-1]

    try:
        img = Image.fromarray(crop_rgb.astype("uint8"))
    except Exception:
        return None

    tensor = _REID_TRANSFORM(img).unsqueeze(0)  # [1, 3, H, W]

    device = MODEL_CFG.device
    try:
        encoder = get_reid_encoder()
        tensor = tensor.to(device)

        with torch.no_grad():
            feat = encoder(tensor)  # [1, 512, 1, 1]
        feat = feat.view(-1).cpu().numpy().astype("float32")  # [512]

        # L2 normalize
        norm = np.linalg.norm(feat) + 1e-12
        feat /= norm
        return feat
    except Exception:
        return None


def cosine_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """
    Similaridade cosseno entre dois vetores.
    Retorna -1.0 se algo der errado (pra ser fácil de ignorar).
    """
    if a is None or b is None:
        return -1.0
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    if denom <= 0:
        return -1.0
    return float(np.dot(a, b) / denom)


# ==============================
#   CLASSE DE ESTADO ReID
# ==============================

class ReIDTracker:
    """
    Mantém um histórico dos últimos N embeddings do atleta alvo
    e expõe um "embedding de referência" (centroide da janela).

    Uso típico no pipeline:
      - Antes do loop: tracker = ReIDTracker(history_size=10)
      - A cada frame:
          * após escolher o índice final do atleta, recortar a bbox
            e chamar tracker.update(crop_bgr)
          * para decidir o alvo em um novo frame, usar
            tracker.reference_embedding como assinatura
    """

    def __init__(self, history_size: int = DEFAULT_HISTORY):
        self.history: Deque[np.ndarray] = deque(maxlen=history_size)

    # ---------------------------
    #   PROPRIEDADES
    # ---------------------------
    @property
    def reference_embedding(self) -> Optional[np.ndarray]:
        """
        Centroide (média) dos embeddings da janela.
        """
        if not self.history:
            return None
        stack = np.stack(list(self.history), axis=0)  # [N, D]
        ref = stack.mean(axis=0)
        # normaliza de novo, só por segurança
        norm = np.linalg.norm(ref) + 1e-12
        ref /= norm
        return ref.astype("float32")

    @property
    def has_history(self) -> bool:
        return len(self.history) > 0

    # ---------------------------
    #   MÉTODOS PÚBLICOS
    # ---------------------------

    def reset(self) -> None:
        """Limpa o histórico (ex: se for começar um novo vídeo)."""
        self.history.clear()

    def update(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Calcula embedding do crop atual e adiciona ao histórico.
        Retorna o embedding de referência atualizado (centroide).
        """
        emb = compute_reid_embedding(crop_bgr)
        if emb is not None:
            self.history.append(emb)
        return self.reference_embedding

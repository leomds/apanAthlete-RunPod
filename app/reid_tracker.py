# app/reid_tracker.py
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

from .reid import compute_reid_embedding


BBox = Tuple[float, float, float, float]  # x1, y1, x2, y2


def _bbox_center(b: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _bbox_area(b: BBox) -> float:
    x1, y1, x2, y2 = b
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    union = _bbox_area(a) + _bbox_area(b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    if denom <= 0:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


@dataclass
class ReIDState:
    """Estado do atleta para ReID com memória curta + longa."""
    last_bbox: Optional[BBox] = None
    last_center: Optional[Tuple[float, float]] = None

    # embedding "curto": do último frame confiável
    short_embed: Optional[np.ndarray] = None
    short_frame_idx: int = -1

    # embedding "longo": EMA dos últimos bons frames
    long_embed: Optional[np.ndarray] = None

    # contador de frames seguidos sem atleta
    miss_count: int = 0


class AthleteReIDTracker:
    """
    Tracker de atleta único com ReID + coerência espacial.

    Estratégia:
    - A cada frame, avalia TODOS os candidatos.
    - Score = combinação de:
        * sim_curta (embedding do último frame bom)
        * sim_longa (embedding EMA)
        * coerência espacial (distância do centro)
        * IoU com bbox anterior
    - Usa âncora curta para ser rígido em oclusão curta,
      e âncora longa para não “derrapar” se o atleta girar / mudar pose.
    """

    def __init__(
        self,
        max_misses: int = 20,
        short_weight: float = 0.55,
        long_weight: float = 0.25,
        motion_weight: float = 0.10,
        iou_weight: float = 0.10,
        # thresholds mais permissivos pra não perder o atleta fácil
        sim_threshold: float = 0.28,
        margin_best_second: float = 0.05,
        ema_momentum: float = 0.85,
    ):
        self.state = ReIDState()
        self.max_misses = max_misses
        self.short_weight = short_weight
        self.long_weight = long_weight
        self.motion_weight = motion_weight
        self.iou_weight = iou_weight
        self.sim_threshold = sim_threshold
        self.margin_best_second = margin_best_second
        self.ema_momentum = ema_momentum

    # --------------------------------------------------------
    # API principal
    # --------------------------------------------------------
    def reset(self):
        self.state = ReIDState()

    def update(
        self,
        frame_idx: int,
        frame_bgr: np.ndarray,
        candidates: List[Dict[str, Any]],
    ) -> Optional[int]:
        """
        Atualiza o tracking neste frame.

        Parameters
        ----------
        frame_idx : índice do frame
        frame_bgr : frame completo em BGR (np.ndarray)
        candidates : lista de dicts, cada um com:
            {
              "bbox": (x1, y1, x2, y2),
              "score": float,
              "crop": np.ndarray  # opcional; se não tiver, recortamos do frame
            }

        Returns
        -------
        idx_escolhido (int) ou None se não encontrou ninguém coerente.
        """
        st = self.state

        if not candidates:
            st.miss_count += 1
            if st.miss_count > self.max_misses:
                self.reset()
            return None

        # Se ainda não temos embedding, escolhe o melhor candidato só por lógica espacial (ou score)
        if st.short_embed is None and st.long_embed is None:
            best_idx = self._pick_initial_candidate(candidates, st)
            if best_idx is None:
                st.miss_count += 1
                return None
            self._update_state_with_candidate(
                frame_idx, frame_bgr, candidates[best_idx]
            )
            return best_idx

        # Caso já tenhamos embedding, calculamos score pra todos
        scores = []
        for i, cand in enumerate(candidates):
            s = self._score_candidate(frame_idx, frame_bgr, cand)
            scores.append((s, i))

        scores.sort(reverse=True, key=lambda x: x[0])
        best_score, best_idx = scores[0]
        second_score = scores[1][0] if len(scores) > 1 else -1.0

        # Critérios de aceitação:
        # 1) score precisa passar um threshold mínimo
        # 2) precisa ficar razoavelmente acima do segundo colocado
        if best_score < self.sim_threshold or (best_score - second_score) < self.margin_best_second:
            # Não confia -> considera o atleta perdido neste frame
            st.miss_count += 1
            if st.miss_count > self.max_misses:
                self.reset()
            return None

        # Aceita este candidato
        st.miss_count = 0
        self._update_state_with_candidate(frame_idx, frame_bgr, candidates[best_idx])
        return best_idx

    # --------------------------------------------------------
    # Internos
    # --------------------------------------------------------
    def _pick_initial_candidate(
        self,
        candidates: List[Dict[str, Any]],
        st: ReIDState,
    ) -> Optional[int]:
        """
        Escolha inicial do atleta (primeiro frame):
        - se não tiver last_center: pega maior bbox
        - se tiver last_center (por conta de ref_point), pega bbox mais perto do centro.
        """
        if st.last_center is None:
            # maior bbox
            best_idx = None
            best_area = -1.0
            for i, c in enumerate(candidates):
                area = _bbox_area(c["bbox"])
                if area > best_area:
                    best_area = area
                    best_idx = i
            return best_idx

        cx_target, cy_target = st.last_center
        best_idx = None
        best_dist = 1e12
        for i, c in enumerate(candidates):
            cx, cy = _bbox_center(c["bbox"])
            d = (cx - cx_target) ** 2 + (cy - cy_target) ** 2
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx

    def _score_candidate(
        self,
        frame_idx: int,
        frame_bgr: np.ndarray,
        cand: Dict[str, Any],
    ) -> float:
        st = self.state
        bbox = cand["bbox"]

        # --- embedding ---
        crop = cand.get("crop")
        if crop is None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            crop = frame_bgr[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        emb = compute_reid_embedding(crop)
        if emb is None:
            emb = np.zeros(512, dtype="float32")

        # similaridade curta
        short_sim = _cosine_similarity(emb, st.short_embed) if st.short_embed is not None else 0.0
        # similaridade longa
        long_sim = _cosine_similarity(emb, st.long_embed) if st.long_embed is not None else 0.0

        # movimento / coerência espacial
        motion_score = 0.0
        iou_score = 0.0
        if st.last_bbox is not None:
            iou_score = _bbox_iou(bbox, st.last_bbox)
            cx_prev, cy_prev = _bbox_center(st.last_bbox)
            cx, cy = _bbox_center(bbox)
            dist = np.sqrt((cx - cx_prev) ** 2 + (cy - cy_prev) ** 2)
            # quanto menor a distância, maior o score (normalizado de forma bem simples)
            motion_score = float(np.exp(-dist / 80.0))  # 80px = ~decai pra 0.37

        score = (
            self.short_weight * short_sim
            + self.long_weight * long_sim
            + self.motion_weight * motion_score
            + self.iou_weight * iou_score
        )
        return score

    def _update_state_with_candidate(
        self,
        frame_idx: int,
        frame_bgr: np.ndarray,
        cand: Dict[str, Any],
    ):
        st = self.state
        bbox = cand["bbox"]

        # crop
        crop = cand.get("crop")
        if crop is None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            crop = frame_bgr[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        emb = compute_reid_embedding(crop)
        if emb is None:
            # se deu ruim, não mexe na âncora de embedding, só posição
            st.last_bbox = bbox
            st.last_center = _bbox_center(bbox)
            return

        st.last_bbox = bbox
        st.last_center = _bbox_center(bbox)
        st.short_embed = emb
        st.short_frame_idx = frame_idx

        # atualiza embedding longo com EMA
        if st.long_embed is None:
            st.long_embed = emb.copy()
        else:
            m = self.ema_momentum
            st.long_embed = (m * st.long_embed + (1.0 - m) * emb).astype("float32")

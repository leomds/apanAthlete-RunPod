# app/tracker.py
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .reid import compute_reid_embedding


def _xyxy_to_cxcywh(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    return np.array([cx, cy, w, h], dtype=float)


def _cxcywh_to_xyxy(state: np.ndarray) -> np.ndarray:
    cx, cy, w, h = state
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return np.array([x1, y1, x2, y2], dtype=float)


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0

    area_a = max(0.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(0.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def _cosine_sim(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return -1.0
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    if denom <= 0:
        return -1.0
    return float(np.dot(a, b) / denom)


@dataclass
class TrackState:
    track_id: int
    bbox_xyxy: np.ndarray  # [x1,y1,x2,y2]
    score: float
    embedding: Optional[np.ndarray]
    x: np.ndarray  # Kalman state [cx, cy, vx, vy]
    P: np.ndarray  # Covariance
    age: int = 0
    hits: int = 0
    misses: int = 0


class SingleTargetTracker:
    """
    Tracker de alvo único para o atleta:
      - Kalman em (cx, cy, vx, vy)
      - Custo = IOU + similaridade de ReID
      - Hungarian (1xN) para escolha do melhor match
    """

    def __init__(
        self,
        dt: float,
        max_misses: int = 20,
        iou_threshold: float = 0.20,
        sim_threshold: float = 0.45,
        w_iou: float = 0.6,
        w_sim: float = 0.4,
    ) -> None:
        self.dt = dt if dt > 0 else 1 / 30.0
        self.max_misses = max_misses
        self.iou_threshold = iou_threshold
        self.sim_threshold = sim_threshold
        self.w_iou = w_iou
        self.w_sim = w_sim

        self.track: Optional[TrackState] = None
        self._next_id = 1

        # Matriz de transição (4D estado)
        dt_ = self.dt
        self.F = np.array(
            [
                [1.0, 0.0, dt_, 0.0],
                [0.0, 1.0, 0.0, dt_],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        # Medição = (cx, cy)
        self.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        # Ruído de processo e medição
        q_pos = 50.0
        q_vel = 5.0
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel])
        r_pos = 30.0
        self.R = np.diag([r_pos, r_pos])

    # ---------------- Kalman ----------------

    def _kalman_predict(self, x: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x_pred = self.F @ x
        P_pred = self.F @ P @ self.F.T + self.Q
        return x_pred, P_pred

    def _kalman_update(
        self, x_pred: np.ndarray, P_pred: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        y = z - (self.H @ x_pred)
        x_upd = x_pred + K @ y
        P_upd = (np.eye(4) - K @ self.H) @ P_pred
        return x_upd, P_upd

    # ---------------- Helpers ----------------

    def reset(self) -> None:
        self.track = None

    def _init_track(self, bbox_xyxy: np.ndarray, score: float, emb: Optional[np.ndarray]) -> None:
        cxcywh = _xyxy_to_cxcywh(bbox_xyxy)
        cx, cy, w, h = cxcywh
        x = np.array([cx, cy, 0.0, 0.0], dtype=float)
        P = np.diag([1000.0, 1000.0, 100.0, 100.0])

        self.track = TrackState(
            track_id=self._next_id,
            bbox_xyxy=bbox_xyxy.copy(),
            score=float(score),
            embedding=emb,
            x=x,
            P=P,
            age=1,
            hits=1,
            misses=0,
        )
        self._next_id += 1

    def _update_embedding(
        self, old_emb: Optional[np.ndarray], new_emb: Optional[np.ndarray], alpha: float = 0.9
    ) -> Optional[np.ndarray]:
        if new_emb is None:
            return old_emb
        if old_emb is None:
            return new_emb
        mix = alpha * old_emb + (1.0 - alpha) * new_emb
        norm = np.linalg.norm(mix) + 1e-12
        return (mix / norm).astype("float32")

    # ---------------- API principal ----------------

    def update(
        self,
        frame_bgr: np.ndarray,
        dets_xyxy: Optional[np.ndarray],
        det_scores: Optional[np.ndarray],
        ref_point: Optional[Tuple[float, float]] = None,
    ) -> Optional[np.ndarray]:
        """
        Atualiza o estado com as detecções do frame atual.
        Retorna sempre a bbox rastreada (xyxy) – prevista se não houve match –
        ou None se ainda não temos pista alguma.
        """
        # Normaliza detecções
        if dets_xyxy is None or len(dets_xyxy) == 0:
            dets = np.zeros((0, 4), dtype=float)
            scores = np.zeros((0,), dtype=float)
        else:
            dets = np.asarray(dets_xyxy, dtype=float)
            scores = (
                np.asarray(det_scores, dtype=float)
                if det_scores is not None
                else np.ones((len(dets),), dtype=float)
            )

        # ------------ ainda não temos track ------------

        if self.track is None:
            if len(dets) == 0:
                return None

            # escolhe det inicial
            if ref_point is not None:
                cx = 0.5 * (dets[:, 0] + dets[:, 2])
                cy = 0.5 * (dets[:, 1] + dets[:, 3])
                d = np.sqrt((cx - ref_point[0]) ** 2 + (cy - ref_point[1]) ** 2)
                idx = int(np.argmin(d))
            else:
                idx = int(np.argmax(scores))

            init_box = dets[idx]
            init_score = float(scores[idx])

            x1, y1, x2, y2 = init_box.astype(int)
            crop = frame_bgr[max(0, y1): max(0, y2), max(0, x1): max(0, x2)]
            emb = compute_reid_embedding(crop)

            self._init_track(init_box, init_score, emb)
            return self.track.bbox_xyxy.copy()

        # ------------ já existe track: predição ------------

        t = self.track
        x_pred, P_pred = self._kalman_predict(t.x, t.P)
        self.track.x, self.track.P = x_pred, P_pred

        # bbox prevista
        cx, cy, vx, vy = x_pred
        cxcywh_prev = _xyxy_to_cxcywh(t.bbox_xyxy)
        _, _, w_prev, h_prev = cxcywh_prev
        bbox_pred = _cxcywh_to_xyxy(np.array([cx, cy, w_prev, h_prev], dtype=float))

        # sem detecções -> só predição
        if len(dets) == 0:
            t.age += 1
            t.misses += 1
            if t.misses > self.max_misses:
                return None
            t.bbox_xyxy = bbox_pred
            return t.bbox_xyxy.copy()

        # ------------ custo IOU + ReID ------------

        ious = np.array([_iou(d, bbox_pred) for d in dets], dtype=float)

        cand_embs: List[Optional[np.ndarray]] = []
        for box in dets:
            x1, y1, x2, y2 = box.astype(int)
            crop = frame_bgr[max(0, y1): max(0, y2), max(0, x1): max(0, x2)]
            emb = compute_reid_embedding(crop)
            cand_embs.append(emb)

        sims = np.array(
            [_cosine_sim(e, t.embedding) for e in cand_embs],
            dtype=float,
        )

        costs = []
        for i in range(len(dets)):
            iou_i = ious[i]
            sim_i = sims[i]
            if not np.isfinite(sim_i) or sim_i < 0:
                c = 1.0 - iou_i
            else:
                c = self.w_iou * (1.0 - iou_i) + self.w_sim * (1.0 - sim_i)
            costs.append(c)
        costs = np.array(costs, dtype=float).reshape(1, -1)

        row_ind, col_ind = linear_sum_assignment(costs)
        j = int(col_ind[0])
        best_iou = float(ious[j])
        best_sim = float(sims[j])

        # gating: se IOU e SIM forem ruins, segue só com predição
        if (best_iou < self.iou_threshold) and (best_sim < self.sim_threshold):
            t.age += 1
            t.misses += 1
            if t.misses > self.max_misses:
                return None
            t.bbox_xyxy = bbox_pred
            return t.bbox_xyxy.copy()

        # ------------ atualização com detecção escolhida ------------

        chosen_box = dets[j]
        chosen_score = float(scores[j])
        chosen_emb = cand_embs[j]

        cxcywh = _xyxy_to_cxcywh(chosen_box)
        meas = np.array([cxcywh[0], cxcywh[1]], dtype=float)

        x_upd, P_upd = self._kalman_update(x_pred, P_pred, meas)
        self.track.x = x_upd
        self.track.P = P_upd

        cx_upd, cy_upd, vx_upd, vy_upd = x_upd
        _, _, w_det, h_det = cxcywh
        bbox_upd = _cxcywh_to_xyxy(np.array([cx_upd, cy_upd, w_det, h_det], dtype=float))

        self.track.bbox_xyxy = bbox_upd
        self.track.score = chosen_score
        self.track.embedding = self._update_embedding(self.track.embedding, chosen_emb)
        self.track.age += 1
        self.track.hits += 1
        self.track.misses = 0

        return self.track.bbox_xyxy.copy()

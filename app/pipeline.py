# app/pipeline.py
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from PIL import Image  # upscaling sem cv2

from .models import get_yolo_detector, get_rtmpose_model
from .video_utils import read_video_frames
from .metrics import (
    compute_scale_m_per_px,
    compute_speed_distance_from_hip,
    compute_stride_hybrid,
    detect_jump_from_hip,
)
from .config import POSE_IDXS, METRICS_CFG
from .filters import KalmanBBox
from .reid import compute_reid_embedding


# ============================================================
#                 CONSTANTES / AJUDANTES
# ============================================================

REID_IOU_STRONG = 0.5          # IOU "forte" pra manter continuidade
REID_SIM_THRESHOLD = 0.55      # similaridade mínima pra confiar no ReID
REID_SIM_UPDATE_MIN = 0.40     # mínimo de similaridade pra atualizar buffer
REID_BUFFER_SIZE = 10          # quantos embeddings recentes guardar


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Calcula IOU entre duas bboxes [x1,y1,x2,y2]."""
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
    """Similaridade cosseno entre dois vetores L2-normalizados. Retorna -1 se inválido."""
    if a is None or b is None:
        return -1.0
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    if denom <= 0:
        return -1.0
    return float(np.dot(a, b) / denom)


def update_embedding_buffer(
    buffer: List[np.ndarray],
    new_emb: Optional[np.ndarray],
) -> List[np.ndarray]:
    """Mantém um buffer FIFO de embeddings válidos."""
    if new_emb is None:
        return buffer
    if not isinstance(new_emb, np.ndarray) or new_emb.size == 0:
        return buffer

    if len(buffer) >= REID_BUFFER_SIZE:
        buffer.pop(0)
    buffer.append(new_emb)
    return buffer


def get_reference_embedding(buffer: List[np.ndarray]) -> Optional[np.ndarray]:
    """Retorna embedding médio L2-normalizado do buffer, ou None se vazio."""
    if len(buffer) == 0:
        return None
    mean_emb = np.mean(buffer, axis=0)
    norm = np.linalg.norm(mean_emb) + 1e-12
    return (mean_emb / norm).astype("float32")


def _expand_bbox(
    box: np.ndarray,
    img_w: int,
    img_h: int,
    scale: float = 1.6,
) -> Tuple[int, int, int, int]:
    """
    Expande bbox original por um fator de scale (zoom adaptativo).
    A ideia aqui é garantir pegar cabeça → pé do atleta.
    """
    x1, y1, x2, y2 = box
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = (x2 - x1) * scale
    h = (y2 - y1) * scale

    # levemente mais alto que largo (pra pegar o corpo)
    if h < w * 1.2:
        h = w * 1.2

    x1n = int(max(0, cx - w / 2))
    x2n = int(min(img_w - 1, cx + w / 2))
    y1n = int(max(0, cy - h / 2))
    y2n = int(min(img_h - 1, cy + h / 2))
    return x1n, y1n, x2n, y2n


def _upscale_for_pose(crop: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Aumenta a resolução do crop antes de mandar pro RTMPose.

    - Se o atleta estiver pequeno, sobe pra pelo menos ~320px de altura.
    - Retorna (crop_upscaled, scale_factor).
    """
    h, w = crop.shape[:2]

    target_min_h = 320
    max_scale = 3.0

    if h >= target_min_h:
        return crop, 1.0

    scale = min(max_scale, target_min_h / float(h))
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))

    pil_img = Image.fromarray(crop)
    pil_resized = pil_img.resize((new_w, new_h), Image.BICUBIC)
    crop_up = np.array(pil_resized)

    return crop_up, scale


def _pick_target_person(
    boxes: np.ndarray,
    scores: np.ndarray,
    ref_point: Optional[Tuple[float, float]],
    last_box: Optional[np.ndarray],
    embedding_ref: Optional[np.ndarray],
    candidates_embeddings: List[Optional[np.ndarray]],
) -> int:
    """
    Escolhe índice da pessoa-alvo usando, em ordem de prioridade:

      1) IOU forte com a bbox do frame anterior (continuidade espacial)
      2) Similaridade de embedding (Re-ID temporal)
      3) Ponto de referência (ref_point), se existir
      4) Maior score de detecção (fallback)
    """
    if boxes is None or len(boxes) == 0:
        return -1

    boxes = np.asarray(boxes, dtype=float)
    scores = np.asarray(scores, dtype=float)

    # -------------------------------------------------------
    # 1) Continuidade via IOU forte com a última bbox
    # -------------------------------------------------------
    if last_box is not None:
        ious = np.array([_iou(b, last_box) for b in boxes], dtype=float)
        best_iou_idx = int(np.argmax(ious))
        best_iou = float(ious[best_iou_idx])

        if best_iou >= REID_IOU_STRONG:
            return best_iou_idx

    # -------------------------------------------------------
    # 2) ReID temporal: comparar com embedding_ref médio
    # -------------------------------------------------------
    if embedding_ref is not None and candidates_embeddings:
        sims = np.array(
            [_cosine_sim(e, embedding_ref) for e in candidates_embeddings],
            dtype=float,
        )

        if np.isfinite(sims).any():
            best_reid_idx = int(np.nanargmax(sims))
            best_sim = float(sims[best_reid_idx])

            if best_sim >= REID_SIM_THRESHOLD:
                return best_reid_idx

    # -------------------------------------------------------
    # 3) Mais perto do ponto de referência (caso inicial)
    # -------------------------------------------------------
    if ref_point is not None:
        cx = 0.5 * (boxes[:, 0] + boxes[:, 2])
        cy = 0.5 * (boxes[:, 1] + boxes[:, 3])
        d = np.sqrt((cx - ref_point[0]) ** 2 + (cy - ref_point[1]) ** 2)
        return int(np.argmin(d))

    # -------------------------------------------------------
    # 4) Fallback: maior score de detecção
    # -------------------------------------------------------
    return int(np.argmax(scores))


# ============================================================
#                      PIPELINE PRINCIPAL
# ============================================================

def process_video(
    video_path: str,
    calib: Dict[str, Any],
    ref_point: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """
    Pipeline completo:
      - lê frames
      - YOLO detecta pessoas
      - ReID temporal + IOU + ref_point para manter o mesmo atleta
      - RTMPose extrai pose do crop (com upscaling)
      - Kalman suaviza trajetória do quadril (pra velocidade/distância)
      - constrói trajetória do quadril (hip) + tornozelos
      - calcula distância, velocidade, passada (stride) e salto

    Observação:
      - speed/dist usam quadril FILTRADO (Kalman).
      - stride/jump usam quadril CRU + tornozelos.
    """
    yolo = get_yolo_detector()
    rtmpose = get_rtmpose_model()

    frame_gen, fps, frame_count, (img_w, img_h) = read_video_frames(video_path)

    keypoints_series: List[Optional[np.ndarray]] = []

    # quadril cru (pra stride/jump)
    hip_raw_x_list: List[float] = []
    hip_raw_y_list: List[float] = []

    # quadril filtrado (pra velocidade/distância)
    hip_filt_x_list: List[float] = []
    hip_filt_y_list: List[float] = []

    LAx_list: List[float] = []
    LAy_list: List[float] = []
    RAx_list: List[float] = []
    RAy_list: List[float] = []

    bbox_series: List[Optional[Tuple[float, float, float, float]]] = []

    last_box: Optional[np.ndarray] = None
    last_hip_raw: Optional[Tuple[float, float]] = None

    # últimos tornozelos válidos
    last_LA: Optional[Tuple[float, float]] = None
    last_RA: Optional[Tuple[float, float]] = None

    # Kalman pro quadril
    kalman_hip: Optional[KalmanBBox] = None
    dt = 1.0 / float(fps) if fps and fps > 0 else 1.0 / 30.0

    # Buffer de embeddings ReID
    embedding_buffer: List[np.ndarray] = []

    # Loop de frames
    for frame_idx, frame in enumerate(frame_gen):
        frame_h, frame_w = frame.shape[:2]

        # -----------------------------------------
        # YOLO: detecção de pessoas (classe 0)
        # -----------------------------------------
        results = yolo.predict(
            frame,
            conf=0.25,
            classes=[0],
            verbose=False,
        )

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            # Sem detecção -> reaproveita última posição + Kalman predict

            if last_hip_raw is not None:
                hip_raw_x, hip_raw_y = last_hip_raw
            else:
                hip_raw_x, hip_raw_y = np.nan, np.nan

            if kalman_hip is not None and not np.isnan(hip_raw_x) and not np.isnan(hip_raw_y):
                hip_filt_x, hip_filt_y = kalman_hip.predict()
            else:
                hip_filt_x, hip_filt_y = hip_raw_x, hip_raw_y

            # tornozelos
            if last_LA is not None:
                la_x, la_y = last_LA
            else:
                la_x, la_y = np.nan, np.nan

            if last_RA is not None:
                ra_x, ra_y = last_RA
            else:
                ra_x, ra_y = np.nan, np.nan

            hip_raw_x_list.append(float(hip_raw_x))
            hip_raw_y_list.append(float(hip_raw_y))
            hip_filt_x_list.append(float(hip_filt_x))
            hip_filt_y_list.append(float(hip_filt_y))

            LAx_list.append(float(la_x))
            LAy_list.append(float(la_y))
            RAx_list.append(float(ra_x))
            RAy_list.append(float(ra_y))

            bbox_series.append(None)
            keypoints_series.append(None)
            continue

        # -----------------------------------------
        # Há detecções
        # -----------------------------------------
        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
        det_scores = results[0].boxes.conf.cpu().numpy()

        # Embeddings de cada candidato
        candidates_embeddings: List[Optional[np.ndarray]] = []
        for b in boxes_xyxy:
            x1, y1, x2, y2 = map(int, b)
            x1 = max(0, min(x1, frame_w - 1))
            x2 = max(0, min(x2, frame_w - 1))
            y1 = max(0, min(y1, frame_h - 1))
            y2 = max(0, min(y2, frame_h - 1))
            if x2 <= x1 or y2 <= y1:
                candidates_embeddings.append(None)
                continue
            crop_person = frame[y1:y2, x1:x2]
            emb = compute_reid_embedding(crop_person)
            candidates_embeddings.append(emb)

        # Embedding de referência (média)
        embedding_ref = get_reference_embedding(embedding_buffer)

        # Escolher atleta
        idx = _pick_target_person(
            boxes=boxes_xyxy,
            scores=det_scores,
            ref_point=ref_point,
            last_box=last_box,
            embedding_ref=embedding_ref,
            candidates_embeddings=candidates_embeddings,
        )

        if idx < 0:
            # Fallback similar ao "sem detecção"
            if last_hip_raw is not None:
                hip_raw_x, hip_raw_y = last_hip_raw
            else:
                hip_raw_x, hip_raw_y = np.nan, np.nan

            if kalman_hip is not None and not np.isnan(hip_raw_x) and not np.isnan(hip_raw_y):
                hip_filt_x, hip_filt_y = kalman_hip.predict()
            else:
                hip_filt_x, hip_filt_y = hip_raw_x, hip_raw_y

            if last_LA is not None:
                la_x, la_y = last_LA
            else:
                la_x, la_y = np.nan, np.nan

            if last_RA is not None:
                ra_x, ra_y = last_RA
            else:
                ra_x, ra_y = np.nan, np.nan

            hip_raw_x_list.append(float(hip_raw_x))
            hip_raw_y_list.append(float(hip_raw_y))
            hip_filt_x_list.append(float(hip_filt_x))
            hip_filt_y_list.append(float(hip_filt_y))

            LAx_list.append(float(la_x))
            LAy_list.append(float(la_y))
            RAx_list.append(float(ra_x))
            RAy_list.append(float(ra_y))

            bbox_series.append(None)
            keypoints_series.append(None)
            continue

        bbox = boxes_xyxy[idx]
        last_box = bbox.copy()
        bbox_series.append(
            (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        )

        # Atualiza buffer de embeddings com o atleta escolhido
        chosen_emb = candidates_embeddings[idx]
        if chosen_emb is not None:
            # Se já tem referência, só atualiza se for razoavelmente parecido
            if embedding_ref is not None:
                sim = _cosine_sim(chosen_emb, embedding_ref)
                if sim >= REID_SIM_UPDATE_MIN:
                    embedding_buffer = update_embedding_buffer(embedding_buffer, chosen_emb)
            else:
                embedding_buffer = update_embedding_buffer(embedding_buffer, chosen_emb)

        # -----------------------------------------
        # CROP + UPSCALING PARA RTMPOSE
        # -----------------------------------------
        x1, y1, x2, y2 = bbox
        bbox_h = y2 - y1

        # escala adaptativa baseada na altura
        if bbox_h < 80:
            scale = 2.8
        elif bbox_h < 140:
            scale = 2.2
        elif bbox_h < 220:
            scale = 1.8
        else:
            scale = 1.6

        xe1, ye1, xe2, ye2 = _expand_bbox(bbox, frame_w, frame_h, scale=scale)
        crop = frame[ye1:ye2, xe1:xe2]

        # Super-res simples (resize)
        crop_for_pose, scale_factor = _upscale_for_pose(crop)

        # RTMPose
        kpts = None
        scores = None
        try:
            k, s = rtmpose(crop_for_pose)  # esperado: [K,2], [K] ou batched
            kpts = np.asarray(k, dtype=float)
            scores = np.asarray(s, dtype=float)

            # compatibilidade com saída batched [1,K,2]
            if kpts.ndim == 3 and kpts.shape[0] == 1:
                kpts = kpts[0]
            if scores.ndim == 2 and scores.shape[0] == 1:
                scores = scores[0]
        except Exception:
            kpts = None
            scores = None

        # -----------------------------------------
        # POSE OK
        # -----------------------------------------
        if (
            kpts is not None
            and scores is not None
            and kpts.ndim == 2
            and kpts.shape[1] >= 2
        ):
            # desfaz o upscaling pra voltar ao tamanho do crop original
            if scale_factor != 1.0:
                kpts[:, 0] /= scale_factor
                kpts[:, 1] /= scale_factor

            # converter para coordenadas globais
            kpts_global = kpts.copy()
            kpts_global[:, 0] += xe1
            kpts_global[:, 1] += ye1

            keypoints_series.append(kpts_global)

            # quadril cru
            lh = POSE_IDXS.LEFT_HIP
            rh = POSE_IDXS.RIGHT_HIP
            max_idx_hip = max(lh, rh)

            if len(scores) > max_idx_hip and kpts_global.shape[0] > max_idx_hip:
                if (
                    scores[lh] >= METRICS_CFG.kpt_score_thr
                    and scores[rh] >= METRICS_CFG.kpt_score_thr
                ):
                    hip = 0.5 * (kpts_global[lh] + kpts_global[rh])
                    hip_raw_x = float(hip[0])
                    hip_raw_y = float(hip[1])
                else:
                    hip_raw_x = float(0.5 * (x1 + x2))
                    hip_raw_y = float(0.5 * (y1 + y2))
            else:
                hip_raw_x = float(0.5 * (x1 + x2))
                hip_raw_y = float(0.5 * (y1 + y2))

            # tornozelos
            la_idx = POSE_IDXS.LEFT_ANKLE
            ra_idx = POSE_IDXS.RIGHT_ANKLE
            max_idx_ank = max(la_idx, ra_idx)

            if len(scores) > max_idx_ank and kpts_global.shape[0] > max_idx_ank:
                # LEFT ANKLE
                if scores[la_idx] >= 0.2:
                    la_x = float(kpts_global[la_idx, 0])
                    la_y = float(kpts_global[la_idx, 1])
                    last_LA = (la_x, la_y)
                else:
                    if last_LA is not None:
                        la_x, la_y = last_LA
                    else:
                        la_x, la_y = np.nan, np.nan

                # RIGHT ANKLE
                if scores[ra_idx] >= 0.2:
                    ra_x = float(kpts_global[ra_idx, 0])
                    ra_y = float(kpts_global[ra_idx, 1])
                    last_RA = (ra_x, ra_y)
                else:
                    if last_RA is not None:
                        ra_x, ra_y = last_RA
                    else:
                        ra_x, ra_y = np.nan, np.nan
            else:
                if last_LA is not None:
                    la_x, la_y = last_LA
                else:
                    la_x, la_y = np.nan, np.nan

                if last_RA is not None:
                    ra_x, ra_y = last_RA
                else:
                    ra_x, ra_y = np.nan, np.nan

        # -----------------------------------------
        # FALHA NA POSE
        # -----------------------------------------
        else:
            keypoints_series.append(None)

            hip_raw_x = float(0.5 * (x1 + x2))
            hip_raw_y = float(0.5 * (y1 + y2))

            if last_LA is not None:
                la_x, la_y = last_LA
            else:
                la_x, la_y = np.nan, np.nan

            if last_RA is not None:
                ra_x, ra_y = last_RA
            else:
                ra_x, ra_y = np.nan, np.nan

        # -----------------------------------------
        # ATUALIZA KALMAN DO QUADRIL
        # -----------------------------------------
        if not np.isnan(hip_raw_x) and not np.isnan(hip_raw_y):
            if kalman_hip is None:
                kalman_hip = KalmanBBox(hip_raw_x, hip_raw_y, dt=dt)
                hip_filt_x, hip_filt_y = hip_raw_x, hip_raw_y
            else:
                hip_filt_x, hip_filt_y = kalman_hip.update((hip_raw_x, hip_raw_y))
        else:
            if kalman_hip is not None:
                hip_filt_x, hip_filt_y = kalman_hip.predict()
            else:
                hip_filt_x, hip_filt_y = hip_raw_x, hip_raw_y

        # -----------------------------------------
        # ACUMULA SÉRIES
        # -----------------------------------------
        hip_raw_x_list.append(float(hip_raw_x))
        hip_raw_y_list.append(float(hip_raw_y))
        hip_filt_x_list.append(float(hip_filt_x))
        hip_filt_y_list.append(float(hip_filt_y))

        LAx_list.append(float(la_x))
        LAy_list.append(float(la_y))
        RAx_list.append(float(ra_x))
        RAy_list.append(float(ra_y))

        last_hip_raw = (hip_raw_x, hip_raw_y)

    # =======================================================
    #                 MÉTRICAS FINAIS
    # =======================================================
    if len(hip_raw_x_list) == 0:
        raise RuntimeError("Nenhum atleta rastreado no vídeo.")

    # Converte listas para numpy arrays
    hip_raw_x_arr = np.array(hip_raw_x_list, dtype=float)
    hip_raw_y_arr = np.array(hip_raw_y_list, dtype=float)

    hip_filt_x_arr = np.array(hip_filt_x_list, dtype=float)
    hip_filt_y_arr = np.array(hip_filt_y_list, dtype=float)

    LAx_arr = np.array(LAx_list, dtype=float)
    LAy_arr = np.array(LAy_list, dtype=float)
    RAx_arr = np.array(RAx_list, dtype=float)
    RAy_arr = np.array(RAy_list, dtype=float)

    scale = compute_scale_m_per_px(calib)

    # 1. Calcula VELOCIDADE e DISTÂNCIA (Global)
    speed_data = compute_speed_distance_from_hip(
        hip_filt_x_arr, hip_filt_y_arr, scale, fps
    )
    
    # Recupera séries originais
    raw_speed_series = np.array(speed_data.get("speed_series_m_s", []), dtype=float)
    dist_cum_arr = np.array(speed_data.get("distance_series_cum_m", []), dtype=float)
    
    # Ajuste de segurança de tamanho (padding)
    if len(dist_cum_arr) != len(hip_raw_y_arr):
        diff = len(hip_raw_y_arr) - len(dist_cum_arr)
        if diff > 0:
            dist_cum_arr = np.pad(dist_cum_arr, (0, diff), mode='edge')
            raw_speed_series = np.pad(raw_speed_series, (0, diff), mode='edge')
        else:
            dist_cum_arr = dist_cum_arr[:len(hip_raw_y_arr)]
            raw_speed_series = raw_speed_series[:len(hip_raw_y_arr)]

    # 2. Calcula STRIDE (Passadas) - Usando distância relativa (Tesoura)
    stride = compute_stride_hybrid(
        hip_raw_x_arr,
        hip_raw_y_arr,
        LAx_arr,
        LAy_arr,
        RAx_arr,
        RAy_arr,
        scale,
        fps,
    )

    # 3. Calcula JUMP (Salto) - Passando a dist_cum_arr
    jump = detect_jump_from_hip(
        hip_raw_y_arr,
        scale,
        fps,
        hip_x=hip_raw_x_arr,
        distance_cum=dist_cum_arr,
    )

    # =======================================================
    #      LÓGICA DE SEPARAÇÃO: CORRIDA vs SALTO
    # =======================================================
    
    # Inicializa séries separadas
    run_speed_series = raw_speed_series.copy()
    jump_speed_series = np.zeros_like(raw_speed_series)
    
    # Filtra passos que ocorrem DURANTE o salto (para contagem parar)
    filtered_step_events = []
    step_events_raw = stride.get("step_events", [])

    if jump["has_jump"]:
        takeoff = jump["jump_takeoff_frame"]
        landing = jump["jump_landing_frame"]
        
        # 1. Separar Velocidades
        # Durante o salto (takeoff -> landing), a velocidade de corrida vira 0
        # e a velocidade de salto assume o valor real.
        if takeoff is not None and landing is not None:
            # Máscara para o intervalo do salto
            jump_mask = np.zeros_like(raw_speed_series, dtype=bool)
            jump_mask[takeoff:landing+1] = True
            
            jump_speed_series[jump_mask] = raw_speed_series[jump_mask]
            run_speed_series[jump_mask] = 0.0 # Zera corrida durante o salto
            
        # 2. Parar contagem de passos após o início do salto
        if takeoff is not None:
            for ev in step_events_raw:
                if ev <= takeoff:
                    filtered_step_events.append(ev)
        else:
            filtered_step_events = step_events_raw
    else:
        # Sem salto, tudo é corrida
        filtered_step_events = step_events_raw

    # Atualiza o objeto stride com os eventos filtrados
    stride["step_events"] = filtered_step_events
    stride["stride_count"] = len(filtered_step_events)

    # =======================================================
    # CRIAÇÃO DA SÉRIE TEMPORAL DE PASSOS (Step Count)
    # =======================================================
    n = len(hip_raw_x_arr)
    step_mask = np.zeros(n, dtype=int)
    
    # Marca 1 apenas nos frames onde o passo ocorre (usando lista filtrada)
    for idx in filtered_step_events:
        if 0 <= idx < n:
            step_mask[idx] += 1
            
    # Cria a série acumulada: [0, 0, 1, 1, 2, 2, 3...]
    step_count_series = np.cumsum(step_mask).tolist()

    # -------------------------------------------------------
    # BLOCO series NO JSON (compatível com overlay)
    # -------------------------------------------------------
    series = {
        "frames": list(range(n)),
        "hip_x": hip_filt_x_arr.tolist(),
        "hip_y": hip_filt_y_arr.tolist(),
        "hip_x_raw": hip_raw_x_arr.tolist(),
        "hip_y_raw": hip_raw_y_arr.tolist(),
        "LA_x": LAx_arr.tolist(),
        "LA_y": LAy_arr.tolist(),
        "RA_x": RAx_arr.tolist(),
        "RA_y": RAy_arr.tolist(),
        "bbox": bbox_series,
        
        # --- NOVAS SÉRIES DE VELOCIDADE ---
        "speed_m_s": run_speed_series.tolist(),       # Velocidade de Corrida (para no salto)
        "jump_speed_m_s": jump_speed_series.tolist(), # Velocidade do Salto (só existe no salto)
        
        "distance_cum_m": dist_cum_arr.tolist(),
        "distance_per_frame_m": speed_data.get("distance_per_frame_m", []),
        "step_count": step_count_series,
        "step_events": filtered_step_events,
        "skeleton": keypoints_series,
    }

    return {
        "fps": float(fps),
        "frame_count": int(frame_count),
        "scale_m_per_px": float(scale),
        "step_count_total": int(step_count_series[-1] if step_count_series else 0),
        "speed": speed_data,
        "stride": stride,
        "jump": jump,
        "series": series,
    }
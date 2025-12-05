# app/metrics.py

from typing import List, Dict, Any, Optional

import numpy as np
from scipy.signal import savgol_filter, find_peaks

from .config import METRICS_CFG


# ============================================================
#                  HELPERS BÁSICOS
# ============================================================

def _interp_nans(arr: np.ndarray) -> np.ndarray:
    """Interpolar NaNs em uma série 1D, se possível."""
    arr = np.asarray(arr, dtype=float)
    mask = np.isnan(arr)
    if not mask.any():
        return arr

    valid = ~mask
    if valid.sum() < 2:
        return np.nan_to_num(arr)

    arr[mask] = np.interp(
        np.flatnonzero(mask),
        np.flatnonzero(valid),
        arr[valid],
    )
    return arr


def _smooth_series(x: np.ndarray, window: Optional[int] = None) -> np.ndarray:
    """
    Suaviza uma série 1D usando Savitzky–Golay.
    """
    x = np.asarray(x, dtype=float)
    x = _interp_nans(x)

    n = len(x)
    if n < 3:
        return x

    if window is None:
        window = getattr(METRICS_CFG, "smoothing_window", 9)

    # janela ímpar
    if window % 2 == 0:
        window += 1
    if window >= n:
        window = n - 1 if (n - 1) % 2 == 1 else n - 2
    if window < 3:
        return x

    poly = getattr(METRICS_CFG, "smoothing_polyorder", 2)
    if poly >= window:
        poly = window - 1

    return savgol_filter(x, window_length=window, polyorder=poly)


def compute_scale_m_per_px(calib: Dict[str, Any]) -> float:
    p1 = np.array(calib["point1"], dtype=float)
    p2 = np.array(calib["point2"], dtype=float)
    real_d = float(calib["real_distance_m"])
    px_d = np.linalg.norm(p2 - p1)

    if px_d <= 0:
        raise ValueError("Distância em pixels inválida na calibração.")

    return real_d / px_d


# ============================================================
#           VELOCIDADE E DISTÂNCIA
# ============================================================

def _remove_outliers(speed: np.ndarray, threshold: float = 12.5) -> np.ndarray:
    speed = np.asarray(speed, dtype=float)
    speed[speed < 0] = np.nan
    speed[speed > threshold] = np.nan
    return speed


def _moving_average(arr: np.ndarray, window: int = 5) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(arr, kernel, mode="same")


def compute_speed_distance_from_hip(
    hip_x: np.ndarray,
    hip_y: np.ndarray,
    scale_m_per_px: float,
    fps: float,
) -> Dict[str, Any]:
    hip_x = np.asarray(hip_x, dtype=float)
    hip_y = np.asarray(hip_y, dtype=float)

    hip_x_s = _smooth_series(hip_x)
    hip_y_s = _smooth_series(hip_y)

    dx = np.diff(hip_x_s)
    dy = np.diff(hip_y_s)
    dist_px = np.sqrt(dx * dx + dy * dy)
    dist_m = dist_px * scale_m_per_px

    n = len(hip_x)
    if len(dist_m) == 0 or n == 0:
        return {
            "distance_m": 0.0,
            "velocity_mean_m_s": 0.0,
            "velocity_max_m_s": 0.0,
            "speed_series_m_s": [0.0] * n,
            "distance_series_cum_m": [0.0] * n,
            "distance_per_frame_m": [0.0] * n,
        }

    dt = 1.0 / float(fps)
    raw_speed = dist_m / dt

    clean_speed = _remove_outliers(raw_speed, threshold=12.5)
    clean_speed = _interp_nans(clean_speed)
    clean_speed = _moving_average(clean_speed, window=5)

    total_distance_m = float(np.sum(dist_m))
    mean_speed = float(np.nanmean(clean_speed))
    max_speed = float(np.nanmax(clean_speed))

    speed_series = np.concatenate(([0.0], clean_speed))
    distance_per_frame = np.concatenate(([0.0], dist_m))
    distance_cum = np.concatenate(([0.0], np.cumsum(dist_m)))

    if len(speed_series) != n:
        if len(speed_series) < n:
            speed_series = np.pad(speed_series, (0, n - len(speed_series)), mode="edge")
        else:
            speed_series = speed_series[:n]

    if len(distance_per_frame) != n:
        if len(distance_per_frame) < n:
            distance_per_frame = np.pad(distance_per_frame, (0, n - len(distance_per_frame)), mode="edge")
        else:
            distance_per_frame = distance_per_frame[:n]

    if len(distance_cum) != n:
        if len(distance_cum) < n:
            distance_cum = np.pad(distance_cum, (0, n - len(distance_cum)), mode="edge")
        else:
            distance_cum = distance_cum[:n]

    return {
        "distance_m": total_distance_m,
        "velocity_mean_m_s": mean_speed,
        "velocity_max_m_s": max_speed,
        "speed_series_m_s": speed_series.tolist(),
        "distance_series_cum_m": distance_cum.tolist(),
        "distance_per_frame_m": distance_per_frame.tolist(),
    }


# ============================================================
#            STRIDE (LÓGICA NOVA: DISTÂNCIA RELATIVA)
# ============================================================

def compute_stride_from_ankles_scissoring(
    LA_x, LA_y, RA_x, RA_y,
    scale_m_per_px: float,
    fps: float,
) -> Optional[Dict[str, Any]]:
    """
    Detecta passos calculando a distância relativa X (Left - Right).
    Quando essa distância é MÁXIMA (positiva ou negativa), temos um passo (Stance).
    """
    LA_x = _interp_nans(np.asarray(LA_x, dtype=float))
    RA_x = _interp_nans(np.asarray(RA_x, dtype=float))
    
    # Se não tiver dados suficientes dos tornozelos, aborta
    if np.all(np.isnan(LA_x)) or np.all(np.isnan(RA_x)) or len(LA_x) < 10:
        return None

    # Suaviza as trajetórias X dos tornozelos
    window_stride = min(11, getattr(METRICS_CFG, "smoothing_window", 9) + 2)
    LA_x_s = _smooth_series(LA_x, window=window_stride)
    RA_x_s = _smooth_series(RA_x, window=window_stride)

    # --- A LÓGICA MÁGICA: SINAL DA TESOURA ---
    # delta_x positivo = Pé Esquerdo na frente
    # delta_x negativo = Pé Direito na frente
    delta_x = LA_x_s - RA_x_s
    
    # Convertendo pra metros pra facilitar thresholds físicos
    delta_x_m = delta_x * scale_m_per_px

    # Parâmetros de detecção
    # Minima distância entre picos (0.25s = max 240 passos/min)
    min_dist_frames = int(max(1, fps * 0.25))
    
    # Amplitude mínima da passada (ex: pés devem se afastar pelo menos 15cm um do outro)
    min_stride_width_m = 0.15 

    # 1. Encontrar passos com Pé ESQUERDO na frente (Picos Positivos)
    # height=min_stride_width_m garante que só conta se a passada for larga
    peaks_left, _ = find_peaks(delta_x_m, distance=min_dist_frames, height=min_stride_width_m)

    # 2. Encontrar passos com Pé DIREITO na frente (Picos Negativos / Vales)
    # Invertemos o sinal para usar find_peaks
    peaks_right, _ = find_peaks(-delta_x_m, distance=min_dist_frames, height=min_stride_width_m)

    # Combina e ordena os eventos
    all_steps = np.concatenate([peaks_left, peaks_right])
    all_steps = np.sort(all_steps).astype(int)

    if len(all_steps) < 2:
        return None

    # Cálculo do comprimento da passada (Stride Length)
    # Stride Length = Distância entre o pé da frente e o de trás no momento do impacto
    stride_lengths = []
    step_times = []

    for i in range(len(all_steps)):
        idx = all_steps[i]
        # A "largura" do passo é exatamente o valor absoluto do delta_x naquele pico
        # stride_width = abs(x_left - x_right)
        s_len = abs(delta_x_m[idx])
        stride_lengths.append(s_len)
        
        if i > 0:
            dt = (all_steps[i] - all_steps[i-1]) / fps
            step_times.append(dt)

    if not stride_lengths:
        return None

    step_length_mean = float(np.mean(stride_lengths))
    # Stride (passada completa) é tecnicamente 2 passos (L->R + R->L)
    # Mas no esporte costuma-se usar stride length como o deslocamento de um passo
    # Se quiser 'stride' biomecânico (ciclo), multiplique por 2.
    # Vou manter a convenção de passo simples * 2 para consistência com o anterior.
    stride_length_mean_cycle = step_length_mean * 2.0

    mean_step_time = float(np.mean(step_times)) if step_times else 0.0
    cadence = 1.0 / mean_step_time if mean_step_time > 0 else None

    return {
        "stride_length_mean_m": stride_length_mean_cycle,
        "stride_cadence_hz": cadence,
        "stride_count": len(all_steps),
        "step_events": all_steps.tolist(), # Lista de frames exatos do impacto
        "peaks": all_steps.tolist() # Mantendo compatibilidade
    }


# ============================================================
#                STRIDE A PARTIR DO QUADRIL (FALLBACK)
# ============================================================

def compute_stride_from_hip(
    hip_x: np.ndarray,
    hip_y: np.ndarray,
    scale_m_per_px: float,
    fps: float,
) -> Dict[str, Any]:
    """
    Estima passada pelo quadril (método antigo, usado apenas se tornozelos falharem).
    """
    hip_x = np.asarray(hip_x, dtype=float)
    hip_y = np.asarray(hip_y, dtype=float)

    n = len(hip_y)
    if n < 10:
        return {
            "stride_length_mean_m": None,
            "stride_cadence_hz": None,
            "stride_count": 0,
            "peaks": [],
        }

    window_stride = min(7, getattr(METRICS_CFG, "smoothing_window", 9))
    hip_x_s = _smooth_series(hip_x, window=window_stride)
    hip_y_s = _smooth_series(hip_y, window=window_stride)

    # detectar "vales" em hip_y como picos de -hip_y
    signal = -hip_y_s
    
    # Fallback params
    prominence = (np.nanmax(signal) - np.nanmin(signal)) * 0.15
    peaks, _ = find_peaks(signal, distance=int(fps*0.25), prominence=prominence)
    peaks_list = [int(p) for p in peaks]

    # Lógica simplificada de fallback
    if len(peaks) < 2:
        return {
            "stride_length_mean_m": None,
            "stride_cadence_hz": None,
            "stride_count": len(peaks),
            "peaks": peaks_list,
        }
    
    # Calcula média simples
    x_m = hip_x_s * scale_m_per_px
    steps = []
    for i in range(len(peaks)-1):
        steps.append(abs(x_m[peaks[i+1]] - x_m[peaks[i]]))
    
    if not steps:
         return {
            "stride_length_mean_m": None,
            "stride_cadence_hz": None,
            "stride_count": len(peaks),
            "peaks": peaks_list,
        }
        
    return {
        "stride_length_mean_m": float(np.mean(steps)) * 2.0,
        "stride_cadence_hz": 1.0 / (float(np.mean(np.diff(peaks))) / fps),
        "stride_count": len(peaks),
        "peaks": peaks_list,
    }


# ============================================================
#                 STRIDE HÍBRIDO (ANKLE → HIP)
# ============================================================

def compute_stride_hybrid(
    hip_x, hip_y,
    LA_x, LA_y, RA_x, RA_y,
    scale_m_per_px: float,
    fps: float,
) -> Dict[str, Any]:
    """
    Usa a NOVA lógica de tesoura (ankle scissoring) como principal.
    """
    # Tenta o método preciso dos tornozelos (Scissoring)
    ankle_res = compute_stride_from_ankles_scissoring(
        LA_x, LA_y, RA_x, RA_y,
        scale_m_per_px, fps,
    )

    if ankle_res is not None:
        return {
            "stride_length_mean_m": ankle_res["stride_length_mean_m"],
            "stride_cadence_hz": ankle_res["stride_cadence_hz"],
            "stride_count": ankle_res["stride_count"],
            "step_events": ankle_res["step_events"], # Lista de frames
        }

    # Fallback para o quadril se os tornozelos estiverem ocultos
    hip_res = compute_stride_from_hip(
        hip_x, hip_y, scale_m_per_px, fps
    )

    return {
        "stride_length_mean_m": hip_res["stride_length_mean_m"],
        "stride_cadence_hz": hip_res["stride_cadence_hz"],
        "stride_count": hip_res["stride_count"],
        "step_events": hip_res.get("peaks", []),
    }


# ============================================================
#                      JUMP (SALTO PRO)
# ============================================================

def detect_jump_from_hip(
    hip_y: np.ndarray,
    scale_m_per_px: float,
    fps: float,
    hip_x: Optional[np.ndarray] = None,
    distance_cum: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Detecta salto a partir da trajetória vertical do quadril.
    """
    hip_y = np.asarray(hip_y, dtype=float)
    if len(hip_y) < 10:
        return {
            "has_jump": False,
            "jump_height_m": None,
            "jump_distance_m": None,
            "jump_start_distance_m": None,
            "jump_end_distance_m": None,
            "jump_apex_frame": None,
        }

    window_jump = min(9, getattr(METRICS_CFG, "smoothing_window", 9))
    hip_y_s = _smooth_series(hip_y, window=window_jump)

    apex_idx = int(np.argmin(hip_y_s))
    apex_y = float(hip_y_s[apex_idx])

    baseline_y = float(np.nanpercentile(hip_y_s, 70))
    height_px = baseline_y - apex_y

    if height_px <= 0:
        return {"has_jump": False, "jump_height_m": None}

    jump_height_m = float(height_px * scale_m_per_px)

    # Filtra saltos muito pequenos (lê do config, deve ser 0.10 para evitar andar)
    if jump_height_m < getattr(METRICS_CFG, "min_jump_height_m", 0.05):
        return {
            "has_jump": False,
            "jump_height_m": jump_height_m,
            "jump_distance_m": None,
            "jump_start_distance_m": None,
            "jump_end_distance_m": None,
            "jump_apex_frame": apex_idx,
        }

    tol = max(5.0, 0.25 * height_px)

    takeoff_idx = None
    for i in range(apex_idx - 1, -1, -1):
        if abs(hip_y_s[i] - baseline_y) < tol:
            takeoff_idx = i
            break

    landing_idx = None
    for i in range(apex_idx + 1, len(hip_y_s)):
        if abs(hip_y_s[i] - baseline_y) < tol:
            landing_idx = i
            break

    if takeoff_idx is None or landing_idx is None:
        return {
            "has_jump": False,
            "jump_height_m": jump_height_m,
            "jump_distance_m": None,
            "jump_start_distance_m": None,
            "jump_end_distance_m": None,
            "jump_apex_frame": apex_idx,
        }

    jump_distance_m = None
    jump_start_dist_m = None
    jump_end_dist_m = None

    if distance_cum is not None and len(distance_cum) == len(hip_y):
        start_m = float(distance_cum[takeoff_idx])
        end_m = float(distance_cum[landing_idx])
        jump_start_dist_m = start_m
        jump_end_dist_m = end_m
        jump_distance_m = end_m - start_m
    elif hip_x is not None:
        hip_x = np.asarray(hip_x, dtype=float)
        hip_x_s = _smooth_series(hip_x, window=window_jump)
        dx_px = hip_x_s[landing_idx] - hip_x_s[takeoff_idx]
        jump_distance_m = float(abs(dx_px) * scale_m_per_px)

    return {
        "has_jump": True,
        "jump_height_m": jump_height_m,
        "jump_distance_m": jump_distance_m,
        "jump_start_distance_m": jump_start_dist_m,
        "jump_end_distance_m": jump_end_dist_m,
        "jump_apex_frame": apex_idx,
        "jump_takeoff_frame": takeoff_idx,
        "jump_landing_frame": landing_idx,
        "jump_duration_s": (landing_idx - takeoff_idx) / fps,
    }
#app/filters.py
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class KalmanBBox:
    """
    Kalman simples 2D para rastrear posição (x, y) + velocidade (vx, vy).

    É suficiente pra suavizar a trajetória do quadril / centro da bbox
    entre frames, evitando "saltos" quando a detecção oscila.
    """
    x: float
    y: float
    dt: float = 1.0 / 30.0
    process_var: float = 5.0
    meas_var: float = 25.0

    def __post_init__(self) -> None:
        # estado: [x, y, vx, vy]^T
        self.state = np.array([self.x, self.y, 0.0, 0.0], dtype=float)

        dt = float(self.dt)
        self.F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        self.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=float,
        )

        q = float(self.process_var)
        self.Q = q * np.eye(4, dtype=float)

        r = float(self.meas_var)
        self.R = r * np.eye(2, dtype=float)

        self.P = np.eye(4, dtype=float) * 10.0

    # --------------------------------------------------------
    #                INTERFACE PÚBLICA
    # --------------------------------------------------------
    def predict(self) -> Tuple[float, float]:
        """Só faz a predição (sem nova medida)."""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

        x, y = self.state[0], self.state[1]
        return float(x), float(y)

    def update(self, meas: Tuple[float, float]) -> Tuple[float, float]:
        """
        Atualiza o filtro com uma nova medida (x_med, y_med).
        Retorna a posição filtrada (x_filt, y_filt).
        """
        z = np.array([[float(meas[0])], [float(meas[1])]], dtype=float)

        # predição
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

        # inovação
        y = z - (self.H @ self.state.reshape(-1, 1))
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # atualização
        self.state = (self.state.reshape(-1, 1) + K @ y).flatten()
        I = np.eye(4, dtype=float)
        self.P = (I - K @ self.H) @ self.P

        x, y_pos = self.state[0], self.state[1]
        return float(x), float(y_pos)

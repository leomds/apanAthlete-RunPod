# app/main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import os
import json
from typing import Optional, Tuple
import uvicorn

from .pipeline import process_video


app = FastAPI(title="Athlete AI Server", version="0.1.0")


# ---------------------------------------------------------
#  Conversão universal para JSON (NÃO QUEBRA NUNCA)
# ---------------------------------------------------------
def to_jsonable(x):
    """
    Converte recursivamente:
        - numpy.ndarray -> list
        - numpy.float32 -> float
        - numpy.int32 -> int
        - tuples -> lists
        - dicts internos -> JSON safe
    """
    import numpy as np

    # numpy scalar
    if isinstance(x, (np.float32, np.float64, np.int32, np.int64)):
        return x.item()

    # numpy array
    if isinstance(x, np.ndarray):
        return x.tolist()

    # lista ou tupla
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]

    # dict
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}

    return x


# ---------------------------------------------------------
#  HEALTHCHECK
# ---------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------
#  ENDPOINT PRINCIPAL
# ---------------------------------------------------------
@app.post("/analyze-video")
async def analyze_video(
    video: UploadFile = File(...),
    calib_json: str = Form(...),
    ref_point_json: Optional[str] = Form(None),
):
    """
    POST /analyze-video

    - video: vídeo enviado pelo cliente
    - calib_json: JSON com point1, point2 e real_distance_m
    - ref_point_json: ponto aproximado do atleta no frame inicial (opcional)
    """
    # -----------------------------------------------------
    # 1. Ler calibração
    # -----------------------------------------------------
    try:
        calib = json.loads(calib_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="calib_json inválido.")

    # -----------------------------------------------------
    # 2. Ler ponto de referência (opcional)
    # -----------------------------------------------------
    ref_point: Optional[Tuple[float, float]] = None
    if ref_point_json:
        try:
            rp = json.loads(ref_point_json)
            if isinstance(rp, (list, tuple)) and len(rp) == 2:
                ref_point = (float(rp[0]), float(rp[1]))
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="ref_point_json inválido.")

    # -----------------------------------------------------
    # 3. Salvar vídeo temporário
    # -----------------------------------------------------
    suffix = os.path.splitext(video.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await video.read())
        tmp_path = tmp.name

    # -----------------------------------------------------
    # 4. Processar vídeo
    # -----------------------------------------------------
    try:
        result = process_video(
            video_path=tmp_path,
            calib=calib,
            ref_point=ref_point,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # -----------------------------------------------------
    # 5. Converter para JSON safe (ESSENCIAL)
    # -----------------------------------------------------
    safe = to_jsonable(result)

    return JSONResponse(safe)


# ---------------------------------------------------------
#  RODAR DIRETO
# ---------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)

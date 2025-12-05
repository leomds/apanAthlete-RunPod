import runpod
import os
import tempfile
import base64
import json
import numpy as np
from app.pipeline import process_video
from app.models import get_yolo_detector, get_rtmpose_model

# ---------------------------------------------------------
# 1. Inicialização (Cold Start)
# ---------------------------------------------------------
# Carregamos os modelos fora da função handler.
# Isso garante que, se o worker estiver "quente", não precisa carregar de novo.
print("--> Inicializando modelos...")
yolo = get_yolo_detector()
rtmpose = get_rtmpose_model()
print("--> Modelos carregados na GPU!")

def decode_base64_video(base64_string, suffix=".mp4"):
    """Decodifica string base64 para um arquivo temporário."""
    video_data = base64.b64decode(base64_string)
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tfile.write(video_data)
    tfile.close()
    return tfile.name

def to_jsonable(x):
    """Helper para converter arrays numpy para listas Python."""
    if isinstance(x, (np.float32, np.float64, np.int32, np.int64)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    return x

# ---------------------------------------------------------
# 2. Função Handler (Executa a cada Request)
# ---------------------------------------------------------
def handler(job):
    """
    O 'job' é um dicionário contendo:
    job['input'] -> O JSON que você enviou.
    """
    job_input = job['input']

    # Validação básica
    if 'video_base64' not in job_input:
        return {"error": "Campo 'video_base64' é obrigatório."}
    
    if 'calib' not in job_input:
        return {"error": "Campo 'calib' (JSON object) é obrigatório."}

    video_path = None
    try:
        # 1. Salvar vídeo do Base64 para disco
        video_b64 = job_input['video_base64']
        video_path = decode_base64_video(video_b64)

        # 2. Ler dados de calibração e ref_point
        # Nota: No serverless, 'calib' já vem como dict, não precisa de json.loads
        calib = job_input['calib'] 
        ref_point = job_input.get('ref_point', None)

        # 3. Processar
        print(f"--> Processando vídeo: {video_path}")
        result = process_video(
            video_path=video_path,
            calib=calib,
            ref_point=ref_point
        )

        # 4. Retornar resultado limpo
        return to_jsonable(result)

    except Exception as e:
        return {"error": str(e)}
    
    finally:
        # Limpeza
        if video_path and os.path.exists(video_path):
            os.remove(video_path)

# ---------------------------------------------------------
# 3. Iniciar o Worker
# ---------------------------------------------------------
runpod.serverless.start({"handler": handler})

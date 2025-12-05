import runpod
import os
import tempfile
import base64
import json
import numpy as np
import requests
import shutil
import gc
import torch

from app.pipeline import process_video
from app.models import get_yolo_detector, get_rtmpose_model

# ---------------------------------------------------------
# 1. Inicialização (Cold Start)
# ---------------------------------------------------------
print("--> Inicializando modelos...")
yolo = get_yolo_detector()
rtmpose = get_rtmpose_model()
print("--> Modelos carregados na GPU!")

# ---------------------------------------------------------
# Helpers de Vídeo
# ---------------------------------------------------------
def decode_base64_video(base64_string, suffix=".mp4"):
    """Decodifica string base64 para um arquivo temporário."""
    try:
        video_data = base64.b64decode(base64_string)
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tfile.write(video_data)
        tfile.close()
        return tfile.name
    except Exception as e:
        raise ValueError(f"Erro ao decodificar Base64: {str(e)}")

def download_video_from_url(url, suffix=".mp4"):
    """Baixa vídeo de uma URL para um arquivo temporário (Stream)."""
    try:
        # Stream=True evita carregar o arquivo inteiro na RAM do worker
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            with open(tfile.name, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
            return tfile.name
    except Exception as e:
        raise ValueError(f"Erro ao baixar vídeo da URL: {str(e)}")

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
    video_path = None

    try:
        # Validação de Calibração
        if 'calib' not in job_input:
            return {"error": "Campo 'calib' (JSON object) é obrigatório."}
        
        calib = job_input['calib']
        ref_point = job_input.get('ref_point', None)

        # -----------------------------------------------------
        # 1. Obter o Vídeo (URL ou Base64)
        # -----------------------------------------------------
        if 'video_url' in job_input and job_input['video_url']:
            print(f"--> Baixando vídeo da URL: {job_input['video_url']}")
            video_path = download_video_from_url(job_input['video_url'])
            
        elif 'video_base64' in job_input and job_input['video_base64']:
            print("--> Decodificando vídeo via Base64")
            video_path = decode_base64_video(job_input['video_base64'])
            
        else:
            return {"error": "Nenhum vídeo fornecido. Envie 'video_url' ou 'video_base64'."}

        # -----------------------------------------------------
        # 2. Processar
        # -----------------------------------------------------
        print(f"--> Iniciando pipeline no arquivo: {video_path}")
        result = process_video(
            video_path=video_path,
            calib=calib,
            ref_point=ref_point
        )

        # -----------------------------------------------------
        # 3. Retornar resultado limpo
        # -----------------------------------------------------
        return to_jsonable(result)

    except Exception as e:
        print(f"❌ ERRO NO HANDLER: {str(e)}")
        return {"error": str(e), "status": "FAILED"}
    
    finally:
        # -----------------------------------------------------
        # 4. Limpeza Crítica (Arquivos e GPU)
        # -----------------------------------------------------
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except:
                pass
        
        # Limpa memória da GPU para não travar o próximo job
        gc.collect()
        torch.cuda.empty_cache()

# ---------------------------------------------------------
# 3. Iniciar o Worker
# ---------------------------------------------------------
runpod.serverless.start({"handler": handler})

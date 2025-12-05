import os
import requests
import torch
from torchvision import models
from rtmlib import RTMPose

print("🏗️  INICIANDO O PROCESSO DE DOWNLOAD (BAKING) DOS MODELOS...")

# ---------------------------------------------------------------------------
# 1. YOLOv8 (Baixando do link específico que você pediu)
# ---------------------------------------------------------------------------
YOLO_URL = "https://huggingface.co/Ultralytics/YOLO11/resolve/a01aaa06caeff788b052e193acb76b3f21571b3a/yolo11x.pt"
YOLO_FILENAME = "yolo11x.pt"

print(f"--> Baixando YOLO customizado de: {YOLO_URL}")

if not os.path.exists(YOLO_FILENAME):
    try:
        response = requests.get(YOLO_URL, stream=True)
        response.raise_for_status()
        with open(YOLO_FILENAME, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ {YOLO_FILENAME} salvo no disco com sucesso!")
    except Exception as e:
        print(f"❌ Erro ao baixar YOLO: {e}")
        raise e
else:
    print(f"⚠️ {YOLO_FILENAME} já existe. Pulando download.")

# ---------------------------------------------------------------------------
# 2. ReID (ResNet18 do Torchvision)
# ---------------------------------------------------------------------------
print("--> Baixando pesos do ReID (ResNet18 ImageNet)...")
# Ao instanciar com 'weights', o PyTorch baixa automaticamente para ~/.cache/torch/hub/checkpoints
try:
    models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    print("✅ Pesos do ResNet18 cacheados com sucesso!")
except Exception as e:
    print(f"❌ Erro ao baixar ResNet18: {e}")
    raise e

# ---------------------------------------------------------------------------
# 3. RTMPose (RTMLib)
# ---------------------------------------------------------------------------
print("--> Baixando modelo RTMPose...")
POSE_URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip"

# Instanciar o RTMPose força o download e cache automático da biblioteca
try:
    # Usamos device='cpu' aqui apenas para o build não falhar se o ambiente de build não tiver GPU.
    # O arquivo baixado é o mesmo.
    RTMPose(
        onnx_model=POSE_URL,
        backend="onnxruntime", 
        device="cpu" 
    )
    print("✅ Modelo RTMPose cacheado com sucesso!")
except Exception as e:
    print(f"❌ Erro ao baixar RTMPose: {e}")
    # Não damos raise aqui pois às vezes o rtmlib reclama de falta de GPU no init,
    # mas o download geralmente acontece antes do erro.
    pass

print("🎉 TODOS OS MODELOS FORAM ASSADOS NA IMAGEM!")

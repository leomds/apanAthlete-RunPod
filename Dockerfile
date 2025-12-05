# ======================================================================
# BASE IMAGE COM CUDA 12.2 + cuDNN 9 (necessário p/ onnxruntime-gpu)
# ======================================================================
FROM nvidia/cuda:12.2.0-cudnn9-runtime-ubuntu22.04


# ======================================================================
# CONFIGURAÇÕES DO SISTEMA
# ======================================================================
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive


# ======================================================================
# DEPENDÊNCIAS DO SISTEMA
# ======================================================================
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    libgl1-mesa-glx libglib2.0-0 \
    ffmpeg wget unzip git \
    && rm -rf /var/lib/apt/lists/*

# ======================================================================
# CONFIGURAR PYTHON
# ======================================================================
RUN python3 -m pip install --upgrade pip
WORKDIR /app

# ======================================================================
# INSTALAR DEPENDÊNCIAS PYTHON
# ======================================================================
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
# Fix do numpy + onnxruntime compatível
RUN pip install numpy<2
RUN pip install onnxruntime-gpu==1.17.0

# Instalar PyTorch com suporte CUDA 12.x
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ======================================================================
# COPIAR CÓDIGO
# ======================================================================
COPY . .

# ======================================================================
# PRÉ-DOWNLOAD DOS MODELOS
# ======================================================================
RUN python builder.py

# ======================================================================
# ENTRYPOINT
# ======================================================================
CMD ["python3", "-u", "handler.py"]

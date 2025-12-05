# ======================================================================
# BASE IMAGE COM PYTORCH + CUDA 12.1 + cuDNN 8
# ======================================================================
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

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
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# ======================================================================
# DIRETÓRIO DE TRABALHO
# ======================================================================
WORKDIR /app

# ======================================================================
# 1. INSTALAÇÃO DE DEPENDÊNCIAS PYTHON (CORREÇÃO DE VERSÕES)
# ======================================================================
COPY requirements.txt .

# AQUI ESTÁ O PULO DO GATO:
# 1. Instalamos requirements.txt
# 2. Forçamos o downgrade do NumPy para <2 (para salvar o PyTorch)
# 3. Desinstalamos onnxruntime padrão
# 4. Instalamos onnxruntime-gpu 1.17.x (A última compatível com cuDNN 8)
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install "numpy<2.0" && \
    pip uninstall -y onnxruntime onnxruntime-gpu && \
    pip install "onnxruntime-gpu>=1.17.0,<1.18.0" --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# ======================================================================
# 2. BAKING DOS MODELOS
# ======================================================================
COPY builder.py .
RUN python builder.py

# ======================================================================
# 3. CÓDIGO FONTE FINAL
# ======================================================================
COPY . .

# ======================================================================
# START
# ======================================================================
CMD ["python", "-u", "handler.py"]

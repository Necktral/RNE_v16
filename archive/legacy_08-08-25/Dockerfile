# Dockerfile para el proyecto AEON (Versión 3.0 - Instalación Estable)
# Base: Imagen oficial de NVIDIA con CUDA 12.1.1 y cuDNN 8 sobre Ubuntu 22.04
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Instala Python 3.11 y git
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y python3.11 python3.11-venv python3-pip git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3.11 -m pip install --upgrade pip && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

# Copia los archivos de requisitos a la imagen
COPY requirements.txt ./

# Instala las dependencias de Python, incluyendo las extensiones
# Usamos versiones específicas que tienen binarios precompilados para CUDA 12.1
RUN pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121 && \
    pip install causal-conv1d==1.1.0 --no-build-isolation && \
    pip install mamba-ssm==1.2.0.post1 --no-build-isolation

# Copia el resto del código de nuestra aplicación al contenedor
COPY . .

EXPOSE 8888
CMD ["/bin/bash"]
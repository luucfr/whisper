# Utiliser l'image NVIDIA avec CUDA 12.2 et Ubuntu 22.04 (devel)
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Installer PyTorch avec support CUDA (cu118 est compatible avec CUDA 11.8)
RUN pip install torch --index-url https://download.pytorch.org/whl/cu118

# Copier le fichier requirements.txt et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Installer Whisper depuis le dépôt Git pour garantir la compatibilité GPU
RUN pip install git+https://github.com/openai/whisper.git

RUN pip install python-multipart

# Copier le code de l'application dans le conteneur
COPY . .

# Lancer l'application FastAPI avec Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

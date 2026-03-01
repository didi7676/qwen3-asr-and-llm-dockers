FROM rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1

ENV HSA_OVERRIDE_GFX_VERSION=12.0.0
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_OFFLINE=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    libsndfile1 \
    ffmpeg \
    sox \
    libsox-fmt-all \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Клонируем свежий Qwen3-TTS (для актуального кода модели/токенизатора)
RUN git clone https://github.com/QwenLM/Qwen3-TTS.git /app/Qwen3-TTS \
    && cd /app/Qwen3-TTS \
    && python3 -m pip install --no-cache-dir -e . \
    && cd /app

# Ставим transformers строго 4.57.3 (или 4.57.6 — оба работают с Qwen3-TTS)
RUN python3 -m pip install --no-cache-dir "transformers==4.57.3"

# Остальное
RUN python3 -m pip install --no-cache-dir \
    "accelerate>=1.0.0" \
    "soundfile" \
    "librosa" \
    "protobuf" \
    "sentencepiece" \
    "einops" \
    "torchaudio"

COPY run_tts.py .

CMD ["python3", "run_tts.py"]
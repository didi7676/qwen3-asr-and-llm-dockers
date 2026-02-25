FROM rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1

ENV HSA_OVERRIDE_GFX_VERSION=12.0.0
WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# [Факт] Устанавливаем gptqmodel для поддержки AWQ в новых Transformers
RUN python3 -m pip install --no-cache-dir --no-build-isolation \
    "transformers[sentencepiece,tiktoken]>=4.51.0" \
    "accelerate>=1.1.0" \
    "autoawq>=0.2.8" \
    "optimum>=1.22.0" \
    "gptqmodel>=1.6.0" \
    "protobuf"

COPY check_llm.py .

CMD ["python3", "check_llm.py"]
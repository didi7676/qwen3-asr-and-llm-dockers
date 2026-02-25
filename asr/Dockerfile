FROM rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.7.1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl libsndfile1 ffmpeg && rm -rf /var/lib/apt/lists/*

# [Вероятность] Обновление pip иногда требует root, оставляем как есть
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir modelscope qwen-asr funasr websockets transformers peft diffusers

WORKDIR /app
RUN mkdir -p /models_mnt

# Настройка путей
ENV MODELSCOPE_CACHE=/models_mnt
ENV HF_HOME=/models_mnt

# Создаем скрипт запуска прямо внутри Dockerfile
RUN echo '#!/bin/bash\n\
echo "--- ПРОВЕРКА ЛОКАЛЬНЫХ ДАННЫХ ---"\n\
if [ ! -f "/models_mnt/Qwen3-ASR-0.6B/config.json" ]; then\n\
  echo "[ОШИБКА] Файлы не найдены в /models_mnt. Проверьте монтирование -v."\n\
  exit 1\n\
fi\n\
echo "[OK] Веса найдены. Права: \$(ls -l /models_mnt/Qwen3-ASR-0.6B/config.json | cut -d\" \" -f1)"\n\
exec "$@"' > /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["bash"]
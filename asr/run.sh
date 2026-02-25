docker run -it --rm \
    --device=/dev/kfd --device=/dev/dri \
    -v "$(pwd)/models:/models_mnt:ro" \
    -v "$(pwd)/harvard.wav:/app/harvard.wav:ro" \
    qwen-asr-local \
    /opt/venv/bin/python3 -c "import torch; from qwen_asr import Qwen3ASRModel; \
    print('--- ЗАГРУЗКА МОДЕЛИ ---'); \
    model = Qwen3ASRModel.from_pretrained('/models_mnt/Qwen3-ASR-0.6B', dtype=torch.bfloat16, device_map='cuda'); \
    print('--- РАСПОЗНАВАНИЕ ---'); \
    results = model.transcribe(audio='/app/harvard.wav'); \
    print(f'\nЯЗЫК: {results[0].language}'); \
    print(f'ТЕКСТ: {results[0].text}')"
import os
import sys
import torch
from unittest.mock import MagicMock

# Блокируем flash-attn
sys.modules["flash_attn"] = MagicMock()
import transformers
transformers.utils.is_flash_attn_2_available = lambda *a, **k: False

from qwen_tts import Qwen3TTSModel

MODEL_PATH = "/app/model"
OUTPUT_DIR = "/app/output"

print("[Факт] Попытка загрузки с принудительным SDPA...")

try:
    model = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map="cuda",
        dtype=torch.bfloat16,  # ← используй dtype вместо torch_dtype (как подсказало предупреждение)
        attn_implementation="sdpa",
        local_files_only=True,
        trust_remote_code=True
    )

    print("[Факт] Модель успешно загружена на SDPA (ROCm)!")

with torch.no_grad():
    wavs, sr = model.generate_custom_voice(
        text="Проверка... Я игнорирую флеш-аттеншн, и работаю на вашем железе.",
        language="Russian",
        speaker="Vivian",
        instruct="Спокойно, плавно, без спешки.",
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True,
    )    )

    import soundfile as sf
    output_path = os.path.join(OUTPUT_DIR, "output_final.wav")
    sf.write(output_path, wavs[0], sr)
    print(f"[Факт] Готово! Файл сохранён: {output_path}")

except Exception as e:
    print(f"[Ошибка] Всё ещё упало: {e}")
    import traceback
    traceback.print_exc()
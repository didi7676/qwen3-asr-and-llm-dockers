import torch
import os
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "/models/Qwen3-8B-AWQ"

print(f"--- [Проверка Qwen3 через AutoAWQ] ---")

try:
    print("--- [Загрузка токенайзера] ---")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print("--- [Загрузка модели через AutoAWQ] ---")
    # [Факт] AutoAWQ напрямую управляет ядрами деквантования
    model = AutoAWQForCausalLM.from_quantized(
        model_path, 
        fuse_layers=True, 
        trust_remote_code=True,
        device_map="auto"
    )
    
    print("🚀 ПОБЕДА! Модель успешно загружена через AutoAWQ.")

    # Тестовая генерация
    prompt = "Write a haiku about a red panda."
    # Форматирование для Qwen (ChatML-подобное)
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    
    print("--- [Генерация ответа] ---")
    outputs = model.generate(
        **inputs, 
        max_new_tokens=64,
        do_sample=True,
        temperature=0.7,
        top_p=0.95
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nОтвет:\n{response}")

except Exception as e:
    print(f"❌ Критическая ошибка: {e}")
    # Если и это не поможет, выведем диагностику импортов
    import subprocess
    print("\n--- Диагностика окружения ---")
    subprocess.run(["pip", "show", "autoawq", "gptqmodel"])
    
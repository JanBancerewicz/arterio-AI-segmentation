import os
import sys

# ==============================================================================
# 1. KONFIGURACJA ŚRODOWISKA (Wymuszenie trybu online)
# ==============================================================================
print("Konfigurowanie zmiennych środowiskowych...")
os.environ['HF_HUB_OFFLINE'] = '0'  # Wymuś połączenie z siecią
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1' # Wyłącz zbieranie danych (dla szybkości)

# Opcjonalnie: Zmienna pomocnicza, jeśli masz problemy z SSL
# os.environ['CURL_CA_BUNDLE'] = ''

from huggingface_hub import snapshot_download

# ==============================================================================
# 2. POBIERANIE
# ==============================================================================
model_id = "Qwen/Qwen2-VL-2B-Instruct"

print(f"\n--- START POBIERANIA: {model_id} ---")
print("To może potrwać kilka minut (ok. 4-5 GB).")
print("Jeśli pasek postępu stanie w miejscu, poczekaj chwilę - to może być weryfikacja plików.")

try:
    # Usunąłem resume_download, bo jest domyślne (i powodowało ostrzeżenie)
    local_dir = snapshot_download(
        repo_id=model_id,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"], # Pomijamy zbędne formaty (TensorFlow/Flax), pobieramy tylko PyTorch/Safetensors
    )

    print(f"\n✅ SUKCES! Model pobrany do:\n{local_dir}")

    # Weryfikacja plików
    files = os.listdir(local_dir)
    safetensors = [f for f in files if f.endswith('.safetensors')]

    if safetensors:
        print(f"Znaleziono pliki wag: {safetensors}")
        print("Możesz teraz uruchomić skrypt: run_inference_local.py")
    else:
        print("❌ UWAGA: Folder istnieje, ale nie widzę plików .safetensors. Coś jest nie tak.")

except Exception as e:
    print(f"\n❌ KRYTYCZNY BŁĄD POBIERANIA: {e}")
    print("\nSpróbuj wykonać następujące kroki w terminalu:")
    print("1. pip install --upgrade huggingface_hub")
    print("2. Sprawdź czy nie blokuje Cię firewall/antywirus")
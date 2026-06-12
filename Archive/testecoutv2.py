from faster_whisper import WhisperModel
import time
import os

# On donne le chemin absolu vers le dossier où CUDA a installé ses DLLs
cuda_bin_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"

if os.path.exists(cuda_bin_path):
    os.add_dll_directory(cuda_bin_path)
    print("Chemin CUDA injecté avec succès !")
else:
    print("Attention : Le dossier CUDA spécifié n'existe pas. Vérifie le numéro de version (v12.x).")

# 1. Configuration du modèle
# On utilise le modèle "base" (très bon compromis vitesse/précision)
model_size = "base"

print("Chargement du modèle...")
# device="cuda" force l'utilisation de ta GTX 1660 Ti
# compute_type="float16" optimise la VRAM pour ta carte graphique
model = WhisperModel(model_size, device="cuda", compute_type="float16")
print("Modèle chargé avec succès !")

# 2. Transcription
print("Début de la transcription...")
start_time = time.time()

# Remplace "audio_test.wav" par le chemin de ton fichier audio
# language="fr" force le français (gagne du temps en évitant la détection auto)
segments, info = model.transcribe("test_audio.wav", beam_size=5, language="fr")

# 3. Affichage des résultats
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

end_time = time.time()
print(f"\nTemps de traitement : {end_time - start_time:.2f} secondes")
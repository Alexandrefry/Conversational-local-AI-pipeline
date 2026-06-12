import pyaudio
import numpy as np
from faster_whisper import WhisperModel
import io
import wave
import os
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin")
# Configuration Audio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Whisper travaille nativement en 16kHz
CHUNK = 1024
SILENCE_THRESHOLD = 500  # Seuil de volume (à ajuster selon ton micro et le bruit)
SILENCE_DURATION = 1.5   # Temps de silence (en secondes) avant de lancer le STT

# Initialisation de Faster-Whisper
print("Chargement de Faster-Whisper...")
model = WhisperModel("base", device="cuda", compute_type="float16")
print("Robot prêt à écouter !")

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

def enregistrer_phrase():
    print("\n[ÉCOUTE] Parlez maintenant...")
    frames = []
    silence_chunks = 0
    parle_commence = False
    
    max_silence_chunks = int(RATE / CHUNK * SILENCE_DURATION)

    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        
        # Calcul du volume (RMS) pour détecter le silence
       # Conversion en float32 pour éviter l'overflow sur le calcul du carré
        audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        rms = np.sqrt(np.mean(audio_data**2)) if len(audio_data) > 0 else 0
        
        if rms > SILENCE_THRESHOLD:
            parle_commence = True
            silence_chunks = 0  # Réinitialise le compteur s'il y a du bruit
        elif parle_commence:
            silence_chunks += 1
            
        # Si l'utilisateur a parlé puis s'est tu pendant X secondes, on coupe
        if parle_commence and silence_chunks > max_silence_chunks:
            print("[STT] Traitement en cours...")
            break

    # Convertir les frames en format WAV en mémoire (sans écrire sur le disque)
    buffer = io.BytesIO()
    wf = wave.open(buffer, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    buffer.seek(0)
    
    return buffer

try:
    while True:
        audio_buffer = enregistrer_phrase()
        
        # Envoi direct du buffer mémoire à Faster-Whisper
        segments, _ = model.transcribe(audio_buffer, language="fr", beam_size=5)
        
        # Affichage du résultat
        for segment in segments:
            print(f"-> Passant : {segment.text}")
            
except KeyboardInterrupt:
    print("\nArrêt du script.")
    stream.stop_stream()
    stream.close()
    p.terminate()
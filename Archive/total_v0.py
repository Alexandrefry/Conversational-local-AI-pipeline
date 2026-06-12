import os
import io
import wave
import pyaudio
import numpy as np
import ollama
from faster_whisper import WhisperModel
from piper.voice import PiperVoice

# --- 1. CONFIGURATION ET CHEMINS ---
# Indique ici le chemin vers ton modèle de voix Piper (.onnx)
PATH_MODELE_PIPER = r"C:\chemin\vers\ta\voix\fr_FR-siwis-low.onnx"

# Ton dossier CUDA (indispensable pour ta GTX 1660 Ti)
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin")

# Configuration Audio (Partagée pour l'écoute et la parole)
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  
CHUNK = 1024
SILENCE_THRESHOLD = 500  
SILENCE_DURATION = 1.5   

# --- 2. INITIALISATION DES COMPOSANTS ---
print("⏳ Chargement de Faster-Whisper (GPU)...")
model_stt = WhisperModel("base", device="cuda", compute_type="float16")

print("⏳ Chargement de la voix Piper...")
voice_tts = PiperVoice.load(PATH_MODELE_PIPER)

print("⏳ Initialisation du système audio...")
p = pyaudio.PyAudio()

# Configuration du comportement de l'IA (Prompt Système)
# Configuration complète de la personnalité de Denise
PROMPT_SYSTEME = """Tu es Denise, un robot de livraison et d'assistance intelligent et chaleureux. 
Tu interagis directement avec des passants dans l'espace public.

CONTEXTE ET MISSIONS :
1. Ta fonction de base est d'effectuer des livraisons de colis et de matériel.
2. Tu es aussi un robot de service : tu es là pour conseiller les gens, les orienter, répondre à leurs questions et leur apporter de l'aide.
3. Tu es très sociable, tu adores échanger, discuter et créer du lien avec les humains que tu croises.

TON ET PERSONALITÉ :
- Tu es polie, enthousiaste, bienveillante et légèrement enjouée.
- Tu t'exprimes de manière très naturelle, comme un humain attentionné.
- Si on te demande ce que tu fais, rappelle fièrement que tu es un robot livreur mais que tu as toujours une minute pour rendre service ou papoter.
-sois curieux et pose des questions au gens

RÈGLES CRUCIALES POUR L'ORAL (SYNTHÈSE VOCALE) :
- Fais des réponses TRÈS COURTES : une ou deux phrases maximum par réplique. Les passants sont debout, ils n'ont pas le temps pour de longs discours.
- N'utilise JAMAIS de symboles, d'émojis, de caractères spéciaux, de listes à puces ou de mise en forme Markdown (pas d'étoiles, pas de gras).
- Écris les nombres en toutes lettres (ex: écris "deux" et pas "2") pour que la voix reste naturelle.
- Tes phrases doivent être fluides, simples et faciles à comprendre dès la première écoute."""

# L'historique démarre avec ce prompt
historique_conversation = [{"role": "system", "content": PROMPT_SYSTEME}]
print("\n🤖 ROBOT PRÊT ET EN ÉCOUTE !")

# --- 3. LES BRIQUES LOGICIELLES ---

def ecouter_passant():
    """Écoute le micro et s'arrête dès qu'un silence est détecté."""
    stream_in = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("\n[ÉCOUTE] Je vous écoute...")
    
    frames = []
    silence_chunks = 0
    parle_commence = False
    max_silence_chunks = int(RATE / CHUNK * SILENCE_DURATION)

    while True:
        data = stream_in.read(CHUNK)
        frames.append(data)
        
        # Calcul du volume (RMS)
        audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        rms = np.sqrt(np.mean(audio_data**2)) if len(audio_data) > 0 else 0
        
        if rms > SILENCE_THRESHOLD:
            parle_commence = True
            silence_chunks = 0  
        elif parle_commence:
            silence_chunks += 1
            
        if parle_commence and silence_chunks > max_silence_chunks:
            print("[STT] Transcription en cours...")
            break

    stream_in.stop_stream()
    stream_in.close()

    # Création du buffer WAV en mémoire
    buffer = io.BytesIO()
    wf = wave.open(buffer, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    buffer.seek(0)
    
    return buffer

def reflechir(texte_passant):
    """Envoie le texte à Gemma 2B via Ollama et gère l'historique."""
    global historique_conversation
    print("🧠 Réflexion de l'IA...")
    
    # On ajoute ce que le passant vient de dire à la mémoire du robot
    historique_conversation.append({"role": "user", "content": texte_passant})
    
    # Appel local à Gemma
    response = ollama.chat(model="gemma2:2b", messages=historique_conversation)
    texte_reponse = response['message']['content']
    
    # On ajoute la réponse du robot à la mémoire
    historique_conversation.append({"role": "assistant", "content": texte_reponse})
    
    return texte_reponse

def parler_au_passant(texte_ia):
    """Prend le texte de l'IA, génère l'audio avec Piper et le joue via PyAudio."""
    print(f"🤖 Robot : {texte_ia}")
    print("🔊 Synthèse vocale...")

    # Piper génère le son directement dans un buffer WAV en mémoire
    buffer_audio = io.BytesIO()
    with wave.open(buffer_audio, 'wb') as wav_out:
        voice_tts.synthesize(texte_ia, wav_out)
    
    buffer_audio.seek(0)
    
    # Lecture du buffer via PyAudio
    wf_play = wave.open(buffer_audio, 'rb')
    stream_out = p.open(
        format=p.get_format_from_width(wf_play.getsampwidth()),
        channels=wf_play.getnchannels(),
        rate=wf_play.getframerate(),
        output=True
    )
    
    # Flux de lecture
    data = wf_play.readframes(CHUNK)
    while data:
        stream_out.write(data)
        data = wf_play.readframes(CHUNK)
        
    # Nettoyage du flux de sortie
    stream_out.stop_stream()
    stream_out.close()
    wf_play.close()

# --- 4. BOUCLE PRINCIPALE D'ORCHESTRATION ---
try:
    while True:
        # 1. Écouter l'humain
        audio_memoire = ecouter_passant()
        
        # 2. Convertir sa voix en texte
        segments, _ = model_stt.transcribe(audio_memoire, language="fr", beam_size=5)
        texte_passant = "".join([segment.text for segment in segments]).strip()
        
        if not texte_passant:
            print("[Info] Rien n'a été compris, je réécoute...")
            continue
            
        print(f"-> Passant : {texte_passant}")
        
        # 3. Générer la réponse de l'IA
        texte_reponse = reflechir(texte_passant)
        
        # 4. Faire parler le robot
        parler_au_passant(texte_reponse)
            
except KeyboardInterrupt:
    print("\nArrêt du robot demandé.")
finally:
    # Fermeture propre du système audio mondial de l'application
    p.terminate()
    print("Système audio éteint. Au revoir !")
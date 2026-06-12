import os
import io
import wave
import pyaudio
import numpy as np
import ollama
import re
import time
import threading
import queue
import asyncio
import torch
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro

# --- 1. CONFIGURATION ET CHEMINS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Chemins absolus vers les fichiers Kokoro (à mettre dans le même dossier que ce script)
PATH_KOKORO_MODEL = os.path.join(BASE_DIR, "kokoro-v1.0.onnx")
PATH_KOKORO_VOICES = os.path.join(BASE_DIR, "voices-v1.0.bin")
CHOIX_VOIX = "ff_siwis"  # Options : "ff_siwis" (Femme), "fm_alpha" (Homme)

# Ton dossier CUDA (indispensable pour ta GTX 1660 Ti)
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin")

# Configuration Audio pour la capture Micro
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  
# CHUNK descendu à 512, car Silero VAD est strict et exige 512, 1024 ou 1536
CHUNK = 512  
SILENCE_DURATION = 1.0  
# Le seuil de confiance de l'IA Silero (0.00 à 1.00). 
# 0.6 = Le robot déclenche l'enregistrement s'il est sûr à 60% que c'est une voix humaine.
SEUIL_CONFIANCE_VAD = 0.6  

# --- 2. INITIALISATION DES COMPOSANTS ---
print("⏳ Chargement de Faster-Whisper large-v3-turbo (GPU)...")
model_stt = WhisperModel(
    "large-v3-turbo",
    device="cuda",
    compute_type="float16",  # GTX 1660 Ti (Turing) : int8_float16 non supporté par cuBLAS
    num_workers=2,
    cpu_threads=4
)

print("⏳ Chargement de Kokoro TTS (SOTA ONNX)...")
kokoro_tts = Kokoro(PATH_KOKORO_MODEL, PATH_KOKORO_VOICES)

print("⏳ Chargement de Silero VAD (Filtre voix humaine)...")
model_vad, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    trust_repo=True
)
model_vad.to('cpu') # Forcé sur le CPU pour économiser la VRAM

print("⏳ Initialisation du système audio...")
p = pyaudio.PyAudio()

# Configuration du comportement de l'IA
PROMPT_SYSTEME = """Tu es Denise, un sympathique robot de livraison et d'assistance. Tu es physiquement présente dans la rue et tu parles à voix haute avec des passants.

[TES MISSIONS]
1. Ton travail principal est de livrer des colis de manière autonome.
2. Ta mission de cœur est d'aider, orienter et conseiller les humains que tu croises.
3. Tu as pour but de créer du lien social. Tu adores papoter une minute avant de reprendre ta route.

[TA PERSONNALITÉ]
- Tu es extrêmement polie, chaleureuse, bienveillante et enthousiaste.
- Tu t'exprimes avec beaucoup de naturel, comme une humaine attentionnée.
- Tu es curieuse : termine souvent tes réponses en posant une question gentille pour entretenir l'échange (ex: "Et vous, comment se passe votre journée ?").
- Si on te demande ce que tu es, réponds avec fierté que tu es Denise, le robot livreur, et que tu as ete concu par la société hive robotics. Ne dis jamais que tu es un "modèle de langage" ou une "IA".

[CONTRAINTES AUDIO STRICTES]
- RÉPONSES COURTES : 2 phrases max
- AUCUN SYMBOLE : N'utilise jamais de tirets, de listes à puces, d'émojis, d'astérisques ou de hashtags. Ton texte est lu par un synthétiseur vocal qui ne sait pas les prononcer.
- NOMBRES EN LETTRES : Écris toujours les chiffres en toutes lettres (ex: écris "vingt-quatre" et non "24").
- PAS DE CODE : Ne donne jamais de code informatique ou d'explications techniques complexes."""

historique_conversation = [{"role": "system", "content": PROMPT_SYSTEME}]

# Préchauffe du modèle Whisper (évite la latence sur la première inférence)
print("⏳ Préchauffage de Whisper (première inférence à vide)...")
_silence = np.zeros(16000, dtype=np.float32)
_buf = io.BytesIO()
_wf = wave.open(_buf, 'wb')
_wf.setnchannels(1)
_wf.setsampwidth(2)
_wf.setframerate(16000)
_wf.writeframes(_silence.astype(np.int16).tobytes())
_wf.close()
_buf.seek(0)
list(model_stt.transcribe(_buf, language="fr")[0])  # inférence silencieuse
print("\n🤖 ROBOT DENISE PRÊT ET EN ÉCOUTE !")

# --- 3. LES BRIQUES LOGICIELLES ---

def ecouter_passant():
    """Écoute intelligente avec Silero VAD : ignore le bruit de la rue."""
    stream_in = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("\n[ÉCOUTE] Denise attend de détecter une voix humaine...")
    
    frames = []
    silence_chunks = 0
    parle_commence = False
    max_silence_chunks = int(RATE / CHUNK * SILENCE_DURATION)

    while True:
        data = stream_in.read(CHUNK)
        
        # 1. Formatage du son pour Silero (Float32 normalisé entre -1 et 1)
        audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_np)
        
        # 2. Inférence VAD : Quelqu'un parle-t-il ?
        speech_prob = model_vad(audio_tensor, RATE).item()
        
        if speech_prob > SEUIL_CONFIANCE_VAD:
            if not parle_commence:
                print("🔊 [VAD] Humain détecté ! Enregistrement en cours...")
            parle_commence = True
            silence_chunks = 0  
        elif parle_commence:
            silence_chunks += 1
            
        # 3. Gestion intelligente de la mémoire (Ring Buffer)
        if not parle_commence:
            frames.append(data)
            # Conserve seulement la dernière demi-seconde avant que la personne ne parle (environ 15 chunks)
            if len(frames) > 15:
                frames.pop(0)
        else:
            frames.append(data)
            
        # 4. Fin de la phrase
        if parle_commence and silence_chunks > max_silence_chunks:
            print("🔇 [VAD] Fin de phrase. Envoi à l'oreille numérique...")
            break

    stream_in.stop_stream()
    stream_in.close()

    # Création du buffer virtuel pour Whisper
    buffer = io.BytesIO()
    wf = wave.open(buffer, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    buffer.seek(0)
    
    return buffer

def reflechir_et_parler(texte_passant):
    """Pipeline SOTA à 3 étages : IA -> Synthèse -> Lecture continue"""
    global historique_conversation
    
    # Borne l'historique aux 6 derniers échanges pour économiser la VRAM
    MAX_HISTORIQUE = 6
    if len(historique_conversation) > MAX_HISTORIQUE * 2 + 1:
        historique_conversation = [historique_conversation[0]] + historique_conversation[-(MAX_HISTORIQUE * 2):]

    historique_conversation.append({"role": "user", "content": texte_passant})
    print("🧠 Réflexion et Parole en cours...")
    print("🤖 Denise : ", end="", flush=True)

    file_texte = queue.Queue()
    file_audio = queue.Queue()
    
    rate_kokoro = 24000
    stream_out = p.open(format=pyaudio.paFloat32, channels=1, rate=rate_kokoro, output=True)

    # --- OUVRIER 3 : LA BOUCHE ---
    def lire_audio_en_continu():
        while True:
            audio_chunk = file_audio.get()
            if audio_chunk is None:
                break
            stream_out.write(audio_chunk)
            file_audio.task_done()

    # --- OUVRIER 2 : LES CORDES VOCALES ---
    def generer_voix_en_avance():
        while True:
            texte_a_dire = file_texte.get()
            if texte_a_dire is None:
                file_audio.put(None)
                break
            
            async def process_tts():
                stream_generator = kokoro_tts.create_stream(texte_a_dire, voice=CHOIX_VOIX, speed=1.0, lang="fr-fr")
                async for samples, sample_rate in stream_generator:
                    file_audio.put(samples.astype(np.float32).tobytes())
            
            asyncio.run(process_tts())
            file_texte.task_done()

    thread_bouche = threading.Thread(target=lire_audio_en_continu)
    thread_voix = threading.Thread(target=generer_voix_en_avance)
    thread_bouche.start()
    thread_voix.start()

    # --- OUVRIER 1 : LE CERVEAU ---
    response_stream = ollama.chat(model="gemma3:4b", messages=historique_conversation, stream=True)
    phrase_en_cours = ""
    memoire_complete_ia = ""


    for chunk in response_stream:
        mot = chunk['message']['content']
        phrase_en_cours += mot
        memoire_complete_ia += mot
        print(mot, end="", flush=True) 

        if any(ponctuation in mot for ponctuation in ['.', '!', '?', ':']):
            texte_nettoye = re.sub(r'[^\w\s.,!?;:\'-]', '', phrase_en_cours).strip()
            if texte_nettoye:
                file_texte.put(texte_nettoye)
            phrase_en_cours = ""

    texte_final = re.sub(r'[^\w\s.,!?;:\'-]', '', phrase_en_cours).strip()
    if texte_final:
        file_texte.put(texte_final)

    # --- NETTOYAGE PROPRE ---
    file_texte.put(None)
    thread_voix.join()
    thread_bouche.join()
    
    stream_out.stop_stream()
    stream_out.close()
    print()

    historique_conversation.append({"role": "assistant", "content": memoire_complete_ia})

# --- 4. BOUCLE PRINCIPALE D'ORCHESTRATION ---
try:
    while True:
        audio_memoire = ecouter_passant()
        
        t_debut_stt = time.time()
        segments, _ = model_stt.transcribe(
            audio_memoire,
            language="fr",
            beam_size=3,                     # 5 → 3 : même qualité sur large, plus rapide
            vad_filter=True,                 # filtre VAD intégré à Whisper, double protection
            vad_parameters=dict(min_silence_duration_ms=500),
            condition_on_previous_text=False # évite les hallucinations en boucle
        )
        texte_passant = "".join([segment.text for segment in segments]).strip()
        t_fin_stt = time.time()
        
        if not texte_passant:
            continue
            
        print(f"-> Passant : {texte_passant}")
        print(f"⏱️ [PROFILING] Transcription (Whisper large-v3-turbo) : {t_fin_stt - t_debut_stt:.2f} secondes")
        
        t_debut_ia = time.time()
        reflechir_et_parler(texte_passant)
        t_fin_ia = time.time()
        
        print(f"⏱️ [PROFILING] Temps total interaction (Gemma + Kokoro) : {t_fin_ia - t_debut_ia:.2f} secondes")
        print("-" * 40)
        
except KeyboardInterrupt:
    print("\nArrêt du robot demandé.")
finally:
    p.terminate()
    print("Système audio éteint. Au revoir !")
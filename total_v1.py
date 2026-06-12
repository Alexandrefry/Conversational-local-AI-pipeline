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
import scipy.signal as signal
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

# Débruitage léger (passe-haut) : coupe les basses fréquences (bruit moteur, vent, vibrations)
# qui ne contiennent pas de voix humaine. Coût quasi nul (un seul filtre IIR).
DENOISE_ACTIF = True
DENOISE_FREQ_COUPURE = 100  # Hz : tout ce qui est sous cette fréquence est atténué
_b_denoise, _a_denoise = signal.butter(2, DENOISE_FREQ_COUPURE / (RATE / 2), btype='high')


def debruiter_chunk(audio_int16):
    """Applique un filtre passe-haut léger pour retirer le bruit grave (vent, moteur, vibrations).
    Coût CPU négligeable, ne dégrade pas la voix humaine (>100Hz)."""
    audio_np = audio_int16.astype(np.float32)
    audio_filtre = signal.lfilter(_b_denoise, _a_denoise, audio_np)
    return np.clip(audio_filtre, -32768, 32767).astype(np.int16)


# ============================================================
# UTILITAIRE DE LOG : horodatage précis pour chaque évènement
# ============================================================
def log(message, niveau="INFO"):
    """Affiche un message avec horodatage précis (heure:min:sec.millisecondes)."""
    t = time.strftime("%H:%M:%S", time.localtime()) + f".{int(time.time() * 1000) % 1000:03d}"
    print(f"[{t}] [{niveau}] {message}")


# --- 2. INITIALISATION DES COMPOSANTS ---
log("=" * 60)
log("DÉMARRAGE DU SYSTÈME DENISE")
log("=" * 60)

t0 = time.time()
log("Chargement de Faster-Whisper large-v3-turbo (GPU)...")
model_stt = WhisperModel(
    "large-v3-turbo",
    device="cuda",
    compute_type="float16",  # GTX 1660 Ti (Turing) : int8_float16 non supporté par cuBLAS
    num_workers=2,
    cpu_threads=4
)
log(f"Whisper chargé en {time.time() - t0:.2f} s", "TIMING")

t0 = time.time()
log("Chargement de Kokoro TTS (SOTA ONNX)...")
kokoro_tts = Kokoro(PATH_KOKORO_MODEL, PATH_KOKORO_VOICES)
log(f"Kokoro chargé en {time.time() - t0:.2f} s", "TIMING")

t0 = time.time()
log("Chargement de Silero VAD (Filtre voix humaine)...")
model_vad, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    trust_repo=True
)
model_vad.to('cpu')  # Forcé sur le CPU pour économiser la VRAM
log(f"Silero VAD chargé en {time.time() - t0:.2f} s", "TIMING")

t0 = time.time()
log("Initialisation du système audio (PyAudio)...")
p = pyaudio.PyAudio()
log(f"PyAudio initialisé en {time.time() - t0:.2f} s", "TIMING")

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
t0 = time.time()
log("Préchauffage de Whisper (première inférence à vide)...")
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
log(f"Whisper préchauffé en {time.time() - t0:.2f} s", "TIMING")

log("=" * 60)
log("🤖 ROBOT DENISE PRÊT ET EN ÉCOUTE !")
log(f"   Débruitage léger (passe-haut {DENOISE_FREQ_COUPURE} Hz) : {'ACTIF' if DENOISE_ACTIF else 'INACTIF'}", "AUDIO")
log(f"   Historique borné à {6} échanges max", "MEMOIRE")
log("=" * 60)


# --- 3. LES BRIQUES LOGICIELLES ---

def ecouter_passant():
    """Écoute intelligente avec Silero VAD : ignore le bruit de la rue."""
    stream_in = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    log("[ÉCOUTE] Denise attend de détecter une voix humaine... (en attente de signal micro)")

    frames = []
    silence_chunks = 0
    parle_commence = False
    max_silence_chunks = int(RATE / CHUNK * SILENCE_DURATION)

    t_debut_attente = time.time()
    t_debut_parole = None
    nb_chunks_total = 0
    nb_chunks_parole = 0
    proba_max = 0.0

    while True:
        data = stream_in.read(CHUNK)
        nb_chunks_total += 1

        # Débruitage léger (passe-haut) avant toute analyse
        if DENOISE_ACTIF:
            audio_int16 = np.frombuffer(data, dtype=np.int16)
            audio_int16 = debruiter_chunk(audio_int16)
            data = audio_int16.tobytes()

        # 1. Formatage du son pour Silero (Float32 normalisé entre -1 et 1)
        audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_np)

        # 2. Inférence VAD : Quelqu'un parle-t-il ?
        speech_prob = model_vad(audio_tensor, RATE).item()
        proba_max = max(proba_max, speech_prob)

        if speech_prob > SEUIL_CONFIANCE_VAD:
            if not parle_commence:
                t_attente = time.time() - t_debut_attente
                log(f"🔊 [VAD] Humain détecté (confiance={speech_prob:.2f}) après {t_attente:.2f} s d'attente. Enregistrement en cours...", "VAD")
                t_debut_parole = time.time()
            parle_commence = True
            nb_chunks_parole += 1
            silence_chunks = 0
        elif parle_commence:
            silence_chunks += 1
            nb_chunks_parole += 1

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
            duree_parole = time.time() - t_debut_parole
            log(f"🔇 [VAD] Fin de phrase détectée. Durée parole captée : {duree_parole:.2f} s "
                f"({nb_chunks_parole} chunks, confiance max={proba_max:.2f}). Envoi à Whisper...", "VAD")
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

    duree_audio_s = len(b''.join(frames)) / (RATE * p.get_sample_size(FORMAT))
    log(f"[ÉCOUTE] Buffer audio prêt : {duree_audio_s:.2f} s d'audio, {nb_chunks_total} chunks lus au total.", "AUDIO")

    return buffer


def reflechir_et_parler(texte_passant):
    """Pipeline SOTA à 3 étages : IA -> Synthèse -> Lecture continue"""
    global historique_conversation

    # Borne l'historique aux 6 derniers échanges pour économiser la VRAM
    MAX_HISTORIQUE = 6
    if len(historique_conversation) > MAX_HISTORIQUE * 2 + 1:
        avant = len(historique_conversation)
        historique_conversation = [historique_conversation[0]] + historique_conversation[-(MAX_HISTORIQUE * 2):]
        log(f"[HISTORIQUE] Élagage : {avant} -> {len(historique_conversation)} messages conservés.", "MEMOIRE")

    historique_conversation.append({"role": "user", "content": texte_passant})
    log(f"[HISTORIQUE] Taille actuelle : {len(historique_conversation)} messages.", "MEMOIRE")

    log("🧠 Début du pipeline LLM -> TTS -> Lecture audio (threads en parallèle)...", "PIPELINE")
    print("🤖 Denise : ", end="", flush=True)

    file_texte = queue.Queue()
    file_audio = queue.Queue()

    rate_kokoro = 24000
    stream_out = p.open(format=pyaudio.paFloat32, channels=1, rate=rate_kokoro, output=True)

    # Compteurs / horodatages partagés entre threads
    stats = {
        "t_debut_global": time.time(),
        "t_premier_token": None,
        "t_premiere_phrase_envoyee": None,
        "t_premier_audio_genere": None,
        "t_premier_audio_joue": None,
        "nb_phrases": 0,
        "nb_chunks_audio": 0,
        "attente_cumulee_bouche": 0.0,
        "attente_cumulee_voix": 0.0,
    }

    # --- OUVRIER 3 : LA BOUCHE ---
    def lire_audio_en_continu():
        log("  [THREAD-BOUCHE] Démarré, en attente du premier chunk audio...", "THREAD")
        while True:
            t_attente_debut = time.time()
            audio_chunk = file_audio.get()
            attente = time.time() - t_attente_debut
            stats["attente_cumulee_bouche"] += attente

            if audio_chunk is None:
                log(f"  [THREAD-BOUCHE] Signal d'arrêt reçu. Temps total passé à attendre : {stats['attente_cumulee_bouche']:.2f} s", "THREAD")
                break

            if stats["t_premier_audio_joue"] is None:
                stats["t_premier_audio_joue"] = time.time()
                delta = stats["t_premier_audio_joue"] - stats["t_debut_global"]
                log(f"  [THREAD-BOUCHE] 🔈 Premier son joué ! Latence depuis le début : {delta:.2f} s "
                    f"(a attendu {attente:.2f} s ce chunk)", "TIMING")

            stats["nb_chunks_audio"] += 1
            stream_out.write(audio_chunk)
            file_audio.task_done()

    # --- OUVRIER 2 : LES CORDES VOCALES ---
    def generer_voix_en_avance():
        log("  [THREAD-VOIX] Démarré, en attente de la première phrase texte...", "THREAD")
        while True:
            t_attente_debut = time.time()
            texte_a_dire = file_texte.get()
            attente = time.time() - t_attente_debut
            stats["attente_cumulee_voix"] += attente

            if texte_a_dire is None:
                file_audio.put(None)
                log(f"  [THREAD-VOIX] Signal d'arrêt reçu. Temps total passé à attendre du texte : {stats['attente_cumulee_voix']:.2f} s", "THREAD")
                break

            t_tts_debut = time.time()
            log(f"  [THREAD-VOIX] 🎤 Synthèse Kokoro pour : \"{texte_a_dire}\" (a attendu {attente:.2f} s)", "TTS")

            nb_chunks_phrase = 0

            async def process_tts():
                nonlocal nb_chunks_phrase
                stream_generator = kokoro_tts.create_stream(texte_a_dire, voice=CHOIX_VOIX, speed=1.0, lang="fr-fr")
                async for samples, sample_rate in stream_generator:
                    nb_chunks_phrase += 1
                    if stats["t_premier_audio_genere"] is None:
                        stats["t_premier_audio_genere"] = time.time()
                        delta = stats["t_premier_audio_genere"] - stats["t_debut_global"]
                        log(f"  [THREAD-VOIX] Premier sample audio généré par Kokoro après {delta:.2f} s (depuis début pipeline)", "TIMING")
                    file_audio.put(samples.astype(np.float32).tobytes())

            asyncio.run(process_tts())
            duree_tts = time.time() - t_tts_debut
            log(f"  [THREAD-VOIX] Synthèse terminée en {duree_tts:.2f} s ({nb_chunks_phrase} chunks audio générés)", "TIMING")
            file_texte.task_done()

    thread_bouche = threading.Thread(target=lire_audio_en_continu)
    thread_voix = threading.Thread(target=generer_voix_en_avance)
    thread_bouche.start()
    thread_voix.start()

    # --- OUVRIER 1 : LE CERVEAU ---
    log("  [THREAD-CERVEAU] Envoi de la requête à Ollama (gemma3:4b), démarrage du streaming...", "LLM")
    t_appel_llm = time.time()
    response_stream = ollama.chat(model="gemma3:4b", messages=historique_conversation, stream=True)

    phrase_en_cours = ""
    memoire_complete_ia = ""
    nb_tokens = 0

    for chunk in response_stream:
        mot = chunk['message']['content']
        nb_tokens += 1

        if stats["t_premier_token"] is None:
            stats["t_premier_token"] = time.time()
            delta = stats["t_premier_token"] - t_appel_llm
            log(f"  [THREAD-CERVEAU] ⚡ Premier token reçu après {delta:.2f} s (TTFT)", "TIMING")

        phrase_en_cours += mot
        memoire_complete_ia += mot
        print(mot, end="", flush=True)

        if any(ponctuation in mot for ponctuation in ['.', '!', '?', ':']):
            texte_nettoye = re.sub(r'[^\w\s.,!?;:\'-]', '', phrase_en_cours).strip()
            if texte_nettoye:
                stats["nb_phrases"] += 1
                if stats["t_premiere_phrase_envoyee"] is None:
                    stats["t_premiere_phrase_envoyee"] = time.time()
                    delta = stats["t_premiere_phrase_envoyee"] - t_appel_llm
                    log(f"  [THREAD-CERVEAU] 📤 Première phrase complète envoyée au TTS après {delta:.2f} s : \"{texte_nettoye}\"", "TIMING")
                file_texte.put(texte_nettoye)
            phrase_en_cours = ""

    texte_final = re.sub(r'[^\w\s.,!?;:\'-]', '', phrase_en_cours).strip()
    if texte_final:
        stats["nb_phrases"] += 1
        file_texte.put(texte_final)

    duree_llm = time.time() - t_appel_llm
    log(f"  [THREAD-CERVEAU] Génération LLM terminée en {duree_llm:.2f} s "
        f"({nb_tokens} tokens, {nb_tokens / duree_llm:.1f} tokens/s, {stats['nb_phrases']} phrases envoyées au TTS)", "TIMING")

    # --- NETTOYAGE PROPRE ---
    log("  [THREAD-CERVEAU] Envoi du signal d'arrêt aux threads voix/bouche...", "THREAD")
    file_texte.put(None)

    t_join_debut = time.time()
    thread_voix.join()
    log(f"  [SYNC] thread_voix terminé (attente jointure : {time.time() - t_join_debut:.2f} s)", "THREAD")

    t_join_debut = time.time()
    thread_bouche.join()
    log(f"  [SYNC] thread_bouche terminé (attente jointure : {time.time() - t_join_debut:.2f} s)", "THREAD")

    stream_out.stop_stream()
    stream_out.close()
    print()

    # --- RÉCAPITULATIF GLOBAL ---
    duree_totale = time.time() - stats["t_debut_global"]
    log("─" * 60)
    log("RÉCAPITULATIF DE L'INTERACTION", "RESUME")
    log(f"  • Temps total pipeline LLM+TTS+audio : {duree_totale:.2f} s", "RESUME")
    if stats["t_premier_token"]:
        log(f"  • Délai avant 1er token LLM (TTFT)   : {stats['t_premier_token'] - t_appel_llm:.2f} s", "RESUME")
    if stats["t_premiere_phrase_envoyee"]:
        log(f"  • Délai avant 1ère phrase -> TTS     : {stats['t_premiere_phrase_envoyee'] - t_appel_llm:.2f} s", "RESUME")
    if stats["t_premier_audio_genere"]:
        log(f"  • Délai avant 1er sample Kokoro      : {stats['t_premier_audio_genere'] - stats['t_debut_global']:.2f} s", "RESUME")
    if stats["t_premier_audio_joue"]:
        log(f"  • Délai avant 1er son joué (perçu)   : {stats['t_premier_audio_joue'] - stats['t_debut_global']:.2f} s", "RESUME")
    log(f"  • Nombre de phrases synthétisées     : {stats['nb_phrases']}", "RESUME")
    log(f"  • Nombre de chunks audio joués       : {stats['nb_chunks_audio']}", "RESUME")
    log("─" * 60)

    historique_conversation.append({"role": "assistant", "content": memoire_complete_ia})


# --- 4. BOUCLE PRINCIPALE D'ORCHESTRATION ---
try:
    while True:
        log("─" * 60)
        log("Nouvelle itération de la boucle principale : en attente d'un passant.", "BOUCLE")

        t_debut_ecoute = time.time()
        audio_memoire = ecouter_passant()
        t_fin_ecoute = time.time()
        log(f"Temps total de l'étape ÉCOUTE (attente + capture) : {t_fin_ecoute - t_debut_ecoute:.2f} s", "TIMING")

        t_debut_stt = time.time()
        log("Envoi du buffer audio à Whisper pour transcription...", "STT")
        segments, info = model_stt.transcribe(
            audio_memoire,
            language="fr",
            beam_size=3,                     # 5 → 3 : même qualité sur large, plus rapide
            vad_filter=True,                 # filtre VAD intégré à Whisper, double protection
            vad_parameters=dict(min_silence_duration_ms=500),
            condition_on_previous_text=False  # évite les hallucinations en boucle
        )
        liste_segments = list(segments)
        texte_passant = "".join([segment.text for segment in liste_segments]).strip()
        t_fin_stt = time.time()

        if not texte_passant:
            log("Transcription vide (silence/bruit filtré par Whisper). Retour à l'écoute.", "STT")
            continue

        log(f"Transcription terminée en {t_fin_stt - t_debut_stt:.2f} s "
            f"(langue détectée: {info.language}, confiance: {info.language_probability:.2f}, "
            f"{len(liste_segments)} segment(s))", "TIMING")
        log(f"-> Passant a dit : \"{texte_passant}\"", "STT")

        t_debut_ia = time.time()
        reflechir_et_parler(texte_passant)
        t_fin_ia = time.time()

        log(f"Temps total de l'étape RÉFLEXION+PAROLE : {t_fin_ia - t_debut_ia:.2f} s", "TIMING")
        log(f"Temps total de l'interaction complète (écoute->fin de parole) : "
            f"{t_fin_ia - t_debut_ecoute:.2f} s", "TIMING")
        log("=" * 60)

except KeyboardInterrupt:
    log("\nArrêt du robot demandé par l'utilisateur (Ctrl+C).", "ARRET")
finally:
    p.terminate()
    log("Système audio éteint. Au revoir !", "ARRET")
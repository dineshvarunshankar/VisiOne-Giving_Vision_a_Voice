'''import threading
import torch
import os
from TTS.api import TTS
from config import DEVICE, SPEAKER_WAV

class VoiceGuide:
    def __init__(self, model_name="tts_models/multilingual/multi-dataset/xtts_v2", language="en"):
        
        # --- CRITICAL FIX FOR MAC ---
        # XTTS crashes on Apple Silicon GPU (MPS) due to unsupported operations.
        # We must FORCE the audio model to use the CPU.
        # This does NOT affect YOLO or Depth (which stay on GPU).
        if DEVICE == 'mps':
            self.device = 'cpu'
            print(f"[TTS] Apple Silicon detected. Forcing XTTS to run on CPU (Stable).")
        else:
            self.device = DEVICE
            print(f"[TTS] Loading xTTS ({model_name}) on {self.device}...")

        # Load the model onto the specific device (CPU for Mac, CUDA for NVIDIA)
        self.tts = TTS(model_name=model_name, progress_bar=False).to(self.device)
        self.language = language
        self.lock = threading.Lock()
        
        # Verify reference audio
        if not os.path.exists(SPEAKER_WAV):
            print(f"[TTS WARNING] Reference audio '{SPEAKER_WAV}' not found! Audio will fail.")

    def speak(self, text, speaker_wav=None, output_path=None):
        
        target_wav = speaker_wav if speaker_wav else SPEAKER_WAV

        def _tts():
            with self.lock:
                try:
                    # Run inference (Safe on CPU now)
                    wav = self.tts.tts(
                        text=text,
                        speaker_wav=target_wav,
                        language=self.language,
                        split_sentences=True
                    )
                    
                    import numpy as np
                    import sounddevice as sd
                    wav_np = np.array(wav, dtype=np.float32)
                    sd.play(wav_np, samplerate=24000)
                    sd.wait()
                except Exception as e:
                    print(f"[TTS Error] {e}")

        threading.Thread(target=_tts, daemon=True).start()'''

import threading
import torch
import os
from TTS.api import TTS
# Assuming config.py defines DEVICE and SPEAKER_WAV
from config import DEVICE, SPEAKER_WAV 

# To handle audio playback in the separate thread
import numpy as np
import sounddevice as sd


class VoiceGuide:
    def __init__(self, model_name="tts_models/multilingual/multi-dataset/xtts_v2", language="en"):
        
                # Smart cross-platform device selection for xTTS
        if DEVICE == "mps":
            self.device = "cpu"
            print("[TTS] Apple Silicon (M1/M2/M3) detected → forcing xTTS to CPU for stability")
        else:
            self.device = DEVICE  # cuda / cpu / anything else → use it (CUDA works perfectly)
            print(f"[TTS] Running xTTS on {self.device.upper()} → optimal speed")

        # Load the model onto the specific device
        self.tts = TTS(model_name=model_name, progress_bar=False).to(self.device)
        self.language = language
        
        # Thread lock to prevent multiple threads from accessing the TTS model simultaneously
        self.lock = threading.Lock() 
        
        # Verify reference audio file exists
        if not os.path.exists(SPEAKER_WAV):
            print(f"[TTS WARNING] Reference audio '{SPEAKER_WAV}' not found! Audio will likely fail.")

    def speak(self, text, speaker_wav=None, output_path=None):
        """
        Synthesizes and plays audio in a separate thread to avoid blocking the main video loop.
        """
        target_wav = speaker_wav if speaker_wav else SPEAKER_WAV

        def _tts():
            """Function executed in the separate thread."""
            # Acquire the lock to ensure thread-safe access to the TTS model
            with self.lock:
                try:
                    # 1. Synthesize the audio (This is the slow part)
                    wav = self.tts.tts(
                        text=text,
                        speaker_wav=target_wav,
                        language=self.language,
                        split_sentences=True # Good for natural pauses
                    )
                    
                    # 2. Convert and Play the audio
                    wav_np = np.array(wav, dtype=np.float32)
                    # Samplerate for XTTS is typically 24000
                    sd.play(wav_np, samplerate=24000) 
                    sd.wait() # Wait for the audio playback to finish

                except Exception as e:
                    print(f"[TTS Error] Could not synthesize or play audio: {e}")

        # Start a new thread for the audio synthesis and playback
        # daemon=True ensures the thread doesn't prevent the main program from exiting
        threading.Thread(target=_tts, daemon=True).start()
import torch.multiprocessing as mp
import os
import queue
import time
import numpy as np
import threading
from config import TTS_ENGINE, TTS_MODEL, TTS_LANGUAGE, TTS_RATE, TTS_SPEAKER_WAV, TTS_BASE_RATE, DEVICE

class CoquiVoice:
    def __init__(self, config):

        # Import dependencies locally to manage dependencies for different engines
        from TTS.api import TTS
        
        self.config = config
        self.tts = None
        self.is_busy = False # Busy flag for AI engine

        # Smart cross-platform device selection
        if DEVICE == "mps":
            self.device = "cpu"
            print("[TTS] Apple Silicon (M1/M2/M3) detected - Running xTTS to CPU", flush=True)
        else:
            self.device = "cuda" # Force cuda for most stable MP/MT operation
            print(f"[TTS] Running xTTS on CUDA", flush=True)
        
        try:
            # Load the model in the thread/process where it will be used
            self.tts = TTS(model_name=self.config['model'], progress_bar=False).to(self.device)
            if self.tts is None or not hasattr(self.tts, 'tts'):
                 raise RuntimeError("TTS model failed to initialize fully.")
            print("[TTS] AI Voice Model Loaded.", flush=True)
        except Exception as e:
             print(f"[TTS Critical] AI Load Failed: {e}", flush=True)
             raise

    def speak(self, text):
        if self.is_busy: return # Drop request if busy
        threading.Thread(target=self._run_tts, args=(text,), daemon=True).start()

    def _run_tts(self, text):
        import sounddevice as sd
        self.is_busy = True # Lock the door
        
        try:
            mult = self.config['rate'] / self.config['base_rate']
            
            # 1. Synthesize
            wav = self.tts.tts(
                text=text,
                speaker_wav=self.config['speaker'],
                language=self.config['lang'],
                speed=mult
            )
            
            # 2. Play
            wav_np = np.array(wav, dtype=np.float32)
            sd.play(wav_np, samplerate=24000) 
            sd.wait() # Block this specific thread until audio finishes

        except Exception as e:
            print(f"[TTS Error] {e}", flush=True)
        
        finally:
            self.is_busy = False # Unlock the door

# System Voice (pyttsx3) Process
class SystemVoiceProcess(mp.Process):
    def __init__(self, input_queue, interrupt_event, ready_event, config):
        super().__init__()
        self.input_queue = input_queue
        self.interrupt_event = interrupt_event
        self.ready_event = ready_event
        self.config = config
        self.daemon = True 

    def run(self):
        import pyttsx3
        print("[TTS] System Voice Ready.", flush=True)
        self.ready_event.set()
        
        while True:
            try: text = self.input_queue.get(timeout=0.1)
            except queue.Empty: continue
            if text == "STOP": break
            
            self.interrupt_event.clear()
            try:
                # Re-initialize engine inside loop to prevent potential cross-process issues
                engine = pyttsx3.init() 
                engine.setProperty('rate', self.config['rate'])
                engine.say(text)
                engine.runAndWait()
                # Clean up to release resources
                del engine 
            except Exception as e: print(f"[TTS Error] {e}", flush=True)


#Main Voice Guide Class
class VoiceGuide:
    def __init__(self):
        # Configuration setup
        path = os.path.join(os.getcwd(), TTS_SPEAKER_WAV) if not os.path.exists(TTS_SPEAKER_WAV) else TTS_SPEAKER_WAV
        self.cfg = {'engine': TTS_ENGINE, 'device': DEVICE, 'model': TTS_MODEL, 'lang': TTS_LANGUAGE, 'rate': TTS_RATE, 'base_rate': TTS_BASE_RATE, 'speaker': path}
        
        self.engine = None
        self.queue = None
        self.interrupt_event = None
        self.process = None
        self.ready_event = mp.Event() 
        
        print(f"[TTS] Starting Voice Engine ({TTS_ENGINE.upper()})...")

        # Conditional Initialization
        if self.cfg['engine'] == "system":
            self.queue = mp.Queue()
            self.interrupt_event = mp.Event()
            self.process = SystemVoiceProcess(self.queue, self.interrupt_event, self.ready_event, self.cfg)
            self.process.start()
        
        elif self.cfg['engine'] == "ai":
            try:
                # Initialize the Coqui model directly in the main thread/process
                self.engine = CoquiVoice(self.cfg)
                self.ready_event.set() # Unblock immediately as initialization is done here
            except Exception as e:
                print(f"[TTS] Failed to initialize AI engine. System will not speak. Error: {e}", flush=True)
                self.engine = None # Ensure engine is None on failure

        else:
            print(f"[TTS] Unknown engine type '{TTS_ENGINE}'. Voice system disabled.")

        # BLOCKING WAIT (only blocks if using SystemVoiceProcess)
        self.ready_event.wait()
        print("[TTS] Engine Loaded. Starting Vision System.")


    def speak(self, text: str, interrupt: bool = False):
        if self.cfg['engine'] == "system":
            if self.queue:
                if interrupt:
                    self.interrupt_event.set()
                    while not self.queue.empty():
                        try: self.queue.get_nowait()
                        except queue.Empty: break
                self.queue.put(text)
        
        elif self.cfg['engine'] == "ai" and self.engine:
            # AI engine handles its own thread/busy logic internally (in CoquiVoice.speak)
            self.engine.speak(text)


    def stop(self):
        if self.cfg['engine'] == "system" and self.queue: 
            self.queue.put("STOP")
            self.process.join()
        
      
import whisper
import sounddevice as sd
import numpy as np
import threading
import time
from config import MIC_DEVICE_INDEX, WAKE_WORD, LISTENING_TIMEOUT, DEVICE

class QueryListener:
    def __init__(self, speaking_event=None, enabled: bool = True):
        self.enabled = enabled
        if not enabled: return
        
        print("Loading Whisper...")
        self.model = whisper.load_model("tiny.en", device=DEVICE)
        self.blocksize = 16000 * 3
        self.buffer = np.zeros(0, dtype=np.float32)
        
        self.is_waiting = False
        self.wait_start = 0
        self.query = None
        self.lock = threading.Lock()
        
        # Words to remove to isolate the object name
        self.stop_words = {
            "find", "search", "look", "see", "where", "wheres", "is", "are", 
            "the", "a", "an", "my", "me", "you", "can", "do", "did"
        }

    def start(self):
        if not self.enabled: return
        def callback(indata, frames, time, status):
            audio = indata[:, 0].astype(np.float32)
            if np.mean(np.abs(audio)) > 0.002:
                with self.lock:
                    self.buffer = np.append(self.buffer, audio)
                    if len(self.buffer) >= self.blocksize: self._process()
        try:
            self.stream = sd.InputStream(device=MIC_DEVICE_INDEX, samplerate=16000, channels=1, blocksize=self.blocksize, callback=callback)
            self.stream.start()
            print(f"Mic Active. Say '{WAKE_WORD}'.")
        except Exception as e: print(f"[Listener Error] {e}")

    def _process(self):
        try:
            res = self.model.transcribe(self.buffer, fp16=False)
            text = res['text'].strip().lower()
            clean = text.replace(",", "").replace(".", "").replace("!", "").replace("?", "")
            
            if len(clean) < 2: return

            target = WAKE_WORD.lower()
            now = time.time()

            # Timeout
            if self.is_waiting and (now - self.wait_start > LISTENING_TIMEOUT):
                print("[Listener] Timeout.")
                self.is_waiting = False

            command = ""
            if target in clean:
                parts = clean.split(target, 1)
                if len(parts) > 1 and len(parts[1].strip()) > 1:
                    command = parts[1].strip()
                else:
                    print(f"[Listener] Trigger heard. Waiting...")
                    self.is_waiting = True
                    self.wait_start = now
                    return
            elif self.is_waiting:
                command = clean

            if command:
                words = command.split()
                keywords = [w for w in words if w not in self.stop_words]
                final_q = " ".join(keywords)
                
                if len(final_q) > 1:
                    print(f"[Listener] Command: '{final_q}'")
                    self.query = final_q
                    self.is_waiting = False

        except Exception as e: print(f"[Listener Error] {e}")
        finally: self.buffer = np.zeros(0, dtype=np.float32)

    def get_query(self):
        if not self.enabled: return None
        with self.lock: q = self.query; self.query = None; return q

    def stop(self):
        if self.stream: self.stream.stop(); self.stream.close()
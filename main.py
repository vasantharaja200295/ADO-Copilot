import os
import sounddevice as sd
import numpy as np
import threading
import keyboard
import tempfile
from scipy.io.wavfile import write
from faster_whisper import WhisperModel

class RecordingApp:
    def __init__(self):
        self.model_size = "large-v3"
        self.model = WhisperModel(self.model_size, device="cuda", compute_type="float16")
        self.sample_rate = 16000
        self.channels = 1
        self.stream = None
        self.recording = np.array([], dtype='float32')
        self.is_recording = False
        self.stop_event = threading.Event()
        self.cwd = os.getcwd()

    def start_recording(self, key):
        if not self.is_recording:
            self.is_recording = True
            print("Recording started. Hold space bar to talk...")
    def stop_recording(self, key):
        if self.is_recording:
            self.is_recording = False
            print("Recording stopped. Saving audio...")
            self.save_audio()
            

    def record_audio(self, indata, frames, time, status):
        if status:
            print(status, flush=True)
        if self.is_recording:
            self.recording = np.concatenate((self.recording, indata.flatten()))

    def save_audio(self):
        temp_file = tempfile.NamedTemporaryFile(delete=False, dir=self.cwd,suffix='.wav')
        write(temp_file.name, self.sample_rate, self.recording)
        self.transcribe(temp_file.name)
        self.recording = np.array([], dtype='float32')  # Clear the recording buffer
        
    def transcribe(self, file_path):
        print("Transcription started...")
        segments, info = self.model.transcribe(file_path, beam_size=5)
        print(f"Detected language '{info.language}' with probability {info.language_probability}")
        full_transcription = ""
        for segment in segments:
            full_transcription += segment.text
        print("Full Transcription: ",full_transcription)

    def run(self):
        keyboard.on_press_key("space", self.start_recording)
        keyboard.on_release_key("space", self.stop_recording)

        try:
            print("Press space bar to start recording...")
            with sd.InputStream(channels=self.channels, samplerate=self.sample_rate, callback=self.record_audio):
                
                keyboard.wait()
        except KeyboardInterrupt:
            self.stop_event.set()
            print("\nStopping...")

if __name__ == "__main__":
    app = RecordingApp()
    app.run()
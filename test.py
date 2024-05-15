import threading
from faster_whisper import WhisperModel
import pyaudio
import wave
import keyboard

class RealTimeTranscription:
    def __init__(self):
        self.model_size = "large-v3"
        self.model = WhisperModel(self.model_size, device="cuda", compute_type="float16")
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.is_recording = False
        self.transcription_thread = None

    def start_recording(self, *args):
        if not self.is_recording:
            print("Hold space bar to talk")
            self.stream = self.audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
            self.frames = []
            self.is_recording = True
            self.transcription_thread = threading.Thread(target=self.transcribe_audio)
            self.transcription_thread.start()

    def stop_recording(self, *args):
        if self.is_recording:
            self.is_recording = False
            self.stream.stop_stream()
            self.stream.close()

    def transcribe_audio(self):
        while self.is_recording:
            data = self.stream.read(1024)
            self.frames.append(data)

        print("Transcription started...")
        wf = wave.open("temp.wav", "wb")
        wf.setnchannels(1)
        wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(self.frames))
        wf.close()

        segments, info = self.model.transcribe("temp.wav", beam_size=5)
        print(f"Detected language '{info.language}' with probability {info.language_probability}")
        for segment in segments:
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

    def run(self):
        try:
            keyboard.on_press_key("space", self.start_recording)
            keyboard.on_release_key("space", self.stop_recording)
            keyboard.wait()
        except KeyboardInterrupt:
            self.stop_recording()

if __name__ == "__main__":
    transcriber = RealTimeTranscription()
    transcriber.run()
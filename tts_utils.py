import pyttsx3
import threading
import queue

# TTS 엔진 초기화
tts_engine = pyttsx3.init()
tts_queue = queue.Queue()

def tts_worker():
    while True:
        text = tts_queue.get()
        if text is None:
            break
        tts_engine.say(text)
        tts_engine.runAndWait()
        tts_queue.task_done()

tts_thread = threading.Thread(target=tts_worker)
tts_thread.daemon = True
tts_thread.start()

def speak_text(text):
    tts_queue.put(text)
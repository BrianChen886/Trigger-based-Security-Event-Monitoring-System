import pyaudio
import numpy as np
import speech_recognition as sr
import multiprocessing
import time
from tt_2 import Fight_Detect

class AudioProcessor:
    def __init__(self):
        self.format = pyaudio.paInt16  # 數據格式
        self.channels = 2  # 通道數
        self.rate = 32000  # 採樣率
        self.chunk = 1000  # 每次讀取的樣本數
        self.p = pyaudio.PyAudio()
        self.sr = sr.Recognizer()
        self.mc = sr.Microphone()
        self.text_dictionary = ["hello", "world", "test", "example"]  # 假定的文字字典
        self.recent_matches = []  # 存儲最近的匹配狀態
        self.high_db_chunks = 0  # 高分貝音頻塊計數
        self.last_check_time = time.time()  # 上次檢查時間

    def check_text_match(self, recognized_text):
        """檢查識別的文字是否至少部分匹配字典中的任一項"""
        for word in self.text_dictionary:
            if word in recognized_text:
                return True
        return False

    def sperec(self):
        with self.mc as source:
            while True:
                audio = self.sr.listen(source, timeout=5)
                try:
                    recognized_text = self.sr.recognize_google(audio, language="zh-TW")
                    print(f"識別結果:{recognized_text}")
                    if self.check_text_match(recognized_text):
                        self.recent_matches.append(True)
                    else:
                        self.recent_matches.append(False)
                    # 保持最近四次匹配記錄
                    self.recent_matches = self.recent_matches[-4:]
                    # 檢查是否滿足特定條件
                    if len(self.recent_matches) >= 4 and all(self.recent_matches) and self.high_db_chunks >= 15:
                        Fight_Detect.send_notification('Quarrel alert!!!') # 發送網頁通知
                except sr.UnknownValueError:
                    print('無法識別語音')
                except sr.RequestError as e:
                    print(f"語音識別錯誤:{e}")

    def intensitygraph(self):
        stream = self.p.open(format=self.format, channels=self.channels, rate=self.rate, input=True, frames_per_buffer=self.chunk)
        start_time = time.time()
        while True:
            data = np.frombuffer(stream.read(self.chunk), dtype=np.int16)
            amplitude = np.abs(data)
            amplitude[amplitude < 1] = 1
            db = 20 * np.log10(amplitude)
            avg_db = np.mean(db)
            if avg_db > 70:
                self.high_db_chunks += 1
            # 每20秒重置高分貝塊計數
            if time.time() - start_time > 20:
                self.high_db_chunks = 0
                start_time = time.time()

    def run(self):
        self.audio_spec_process = multiprocessing.Process(target=self.sperec)
        self.audio_db_process = multiprocessing.Process(target=self.intensitygraph)
        
        self.audio_spec_process.start()
        self.audio_db_process.start()

    def stop(self):
        # 停止所有進程
        if self.audio_spec_process.is_alive():
            self.audio_spec_process.terminate()
            self.audio_spec_process.join()
        
        if self.audio_db_process.is_alive():
            self.audio_db_process.terminate()
            self.audio_db_process.join()

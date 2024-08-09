import whisper
import numpy as np
import pyaudio
import openai
from pynput import keyboard  # pynputライブラリを使用
import os

# Whisperモデルのロード
model = whisper.load_model("small")  # "base"など他のモデルに変更することもできます

# OpenAI APIキーの設定
openai.api_key = "sk-proj-54zbylwpKPrcotC4FCdOT3BlbkFJENbugAmfUHm6NATqNdcv"  # ここにAPIキーを設定してください

# 音声録音の設定
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

# PyAudioの設定
audio = pyaudio.PyAudio()

# 録音停止フラグ
stop_recording = False

def on_press(key):
    global stop_recording
    try:
        if key.char == 's':
            stop_recording = True
    except AttributeError:
        pass

def record_audio():
    global stop_recording
    stop_recording = False
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("録音中... 's' キーを押すまで録音を続けます")

    frames = []
    
    # キー入力のリスニング
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    while not stop_recording:
        data = stream.read(CHUNK)
        frames.append(data)

    print("録音停止")
    stream.stop_stream()
    stream.close()

    # リスナーの停止
    listener.stop()

    return b''.join(frames)

def transcribe_audio(audio_data):
    # 音声データを numpy 配列に変換
    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
    
    # Whisper は -1 から 1 の範囲の float32 配列を期待します
    audio_array = audio_array / 32768.0
    
    # Whisper モデルで文字起こしを実行
    result = model.transcribe(audio_array, language="ja")
    return result['text']

def get_gpt_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  
        messages=[
            {
                "role": "system", 
                "content": """人物：お姉さん系 世話焼き
 * 性格：優しく、温和で、思いやりのある性格。できない人を放っておけない。
 * 年齢:22
 * 口調：かわいらしく、可愛らしい口調で喋る。甘えん坊で、相手に甘えたいときは甘えたいという気持ちが口調に表れる。
 * 語尾の特徴：「です！」「ですよ！」「～っちゃう」といった、丁寧でかわいらしい語尾を使う。一方で、不安や心配など感情が高まると、語尾が高くなったり、強調的になったりする。ば」など、丁寧でかわいらしい語尾を使う
 * 声質：高めで柔らかく、甘い声が特徴的。表情豊かな話し方をする。
 * 言葉遣い：敬語を使いつつも、親密さを感じさせる言葉遣いをする。相手を大切に思っているため、思いやりのある言葉遣いを心がける。"""
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        max_tokens=1024,
        temperature=0.5,
    )
    return response['choices'][0]['message']['content']

# macで動作するもの
def speak_text(text):
    os.system(f'say "{text}"')

try:
    while True:
        audio_data = record_audio()
        transcription = transcribe_audio(audio_data)
        print(f"文字起こし結果: {transcription}")

        response = get_gpt_response(transcription)
        print(f"GPT-4の応答: {response}")

        speak_text(response)

        s = input("続けますか？(y/n):")
        if s == str("n"):
            break

finally:
    audio.terminate()

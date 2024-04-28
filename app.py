import os
import requests
import json
import base64
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# 環境変数のロード
load_dotenv(Path(__file__).parent / ".env")

# Google Cloud Text-to-Speech APIキーの取得
API_KEY = os.getenv('GOOGLE_CLOUD_API_KEY')

class GroqAPI:
    def __init__(self, model_name: str):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model_name = model_name

    def _response(self, message):
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=message,
            temperature=0,
            max_tokens=4096,
            stream=True,
            stop=None,
        )

    def response_stream(self, message):
        full_response = ""
        for chunk in self._response(message):
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
        return full_response

def synthesize_text(text):
    """Google Text-to-Speech APIを使用してテキストを音声に変換する"""
    url = "https://texttospeech.googleapis.com/v1/text:synthesize"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": API_KEY
    }
    body = {
        "input": {"text": text},
        "voice": {"languageCode": "ja-JP", "ssmlGender": "FEMALE"},
        "audioConfig": {"audioEncoding": "MP3"}
    }

    response = requests.post(url, headers=headers, json=body)
    if response.status_code == 200:
        audio_content = response.json()['audioContent']
        filename = "output.mp3"
        with open(filename, "wb") as out:
            out.write(base64.b64decode(audio_content))
        return filename
    else:
        st.error(f"Error: {response.status_code}, {response.text}")
        return None

class Message:  # インデントを修正
    system_prompt: str = (
        """あなたは愉快なAIです。ユーザの入力に全て日本語で返答を生成してください"""
    )

    def __init__(self):
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "system",
                    "content": self.system_prompt,
                }
            ]

    def add(self, role: str, content: str):
        st.session_state.messages.append({"role": role, "content": content})

    def display_chat_history(self):
        for message in st.session_state.messages:
            if message["role"] == "system":
                continue
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def display_stream(self, generater):
        with st.chat_message("assistant"):
            return st.write_stream(generater)

def main():
    st.title("AI Chatbot with Voicevox TTS")
    user_input = st.text_input("私に話しかけてみてください")

    if st.button("話しかける"):
        if user_input:
            # ユーザーの入力を表示
            st.write(f"ユーザー: {user_input}")

            # APIからの応答を取得
            ai_response = get_api_response(user_input, "llama3-70b-8192")

            # AIの応答を表示
            st.write(f"AI: {ai_response}")

            # テキストを音声に変換
            if ai_response:
                speaker_id = '1'  # 常に話者IDを "1" に設定
                audio_content = text_to_speech(ai_response, speaker_id)
                if audio_content:
                    st.audio(audio_content, format='audio/mp3')
                else:
                    st.error("音声合成に失敗しました。")
        else:
            st.error("テキストを入力してください.")

if __name__ == "__main__":
    main()
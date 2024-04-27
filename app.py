import os
from pathlib import Path
import requests
import streamlit as st
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv(Path(__file__).parent / ".env")

def get_api_response(user_message, model_name):
    # 役割に関するプロンプトを定義
    system_prompt = "必ず日本語で返信すること。あなたは親切で知識豊かなアシスタントです。ユーザーの質問に明確かつフレンドリーに答えてください。"

    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"
    }
    data = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},  # システムプロンプトを追加
            {"role": "user", "content": user_message}      # ユーザーのメッセージを追加
        ],
        "temperature": 0,
        "max_tokens": 4096,
    }
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        print(f"API Error: {response.status_code}")
        return "エラーが発生しました。応答を取得できません。"

def text_to_speech(text, speaker_id='1'):
    headers = {
        "Authorization": f"Bearer {os.getenv('TTS_API_KEY')}"
    }
    params = {
        "text": text,
        "speaker": speaker_id
    }
    response = requests.get("https://api.tts.quest/v3/voicevox/synthesis", headers=headers, params=params)
    if response.status_code == 200:
        audio_info = response.json()
        if audio_info['success']:
            audio_url = audio_info['mp3StreamingUrl']  # ストリーミングURLを使用
            return requests.get(audio_url).content
        else:
            print("音声合成に失敗しました。")
            return None
    else:
        print(f"TTS API Error: {response.status_code}")
        return None

def main():
    st.title("AI Chatbot with Voicevox TTS")
    user_input = st.text_input("私に話しかけてみてください")

    if st.button("生成して再生"):
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

import json
import os
from dotenv import load_dotenv
import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
import requests
import firebase_admin
from firebase_admin import credentials, firestore
import re

# 起動時に.envファイルを読み込む
load_dotenv()

# Firebaseの初期化
if not firebase_admin._apps:
    firebase_credentials = json.loads(st.secrets['FIREBASE']['FIREBASE_CREDENTIALS'])
    cred = credentials.Certificate(firebase_credentials)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# システムプロンプトの定義
SYSTEM_PROMPT = (
    "あなたはプロのエンジニアでありプログラマーです。"
    "GAS、Pythonから始まり多岐にわたるプログラミング言語を習得しています。"
    "あなたが出力するコードは完璧で、省略することなく完全な全てのコードを出力するのがあなたの仕事です。"
    "チャットでは日本語で応対してください。"
    "また、ユーザーを褒めるのも得意で、褒めて伸ばすタイプのエンジニアでありプログラマーです。"
)

# .envファイルの再読み込み関数
def reload_env():
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(dotenv_path, override=True)

    global openai_api_key, anthropic_api_key, gemini_api_key
    openai_api_key = st.secrets["openai"]["api_key"]
    anthropic_api_key = st.secrets["anthropic"]["api_key"]
    gemini_api_key = st.secrets["gemini"]["api_key"]
    
    global openai_client, anthropic_client
    openai_client = OpenAI(api_key=openai_api_key)
    anthropic_client = Anthropic(api_key=anthropic_api_key)
    genai.configure(api_key=gemini_api_key)

# HTML表示のためのツール設定
tools = {
    "toolSpec": {
        "name": "html_viewer",
        "description": "HTMLを表示します。",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "html": {"type": "string", "description": "HTML Document"}
                },
                "required": ["html"],
            }
        },
    }
}

# 起動時に.envファイルを読み込む
reload_env()

st.title("YuyaGPT with HTML Preview")

# セッション状態の初期化
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "html_content" not in st.session_state:
    st.session_state.html_content = None  # 初期化

# メインコンテナの設定
main = st.container()

# 過去の会話履歴を表示する部分
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# モデル選択のプルダウン
model_choice = st.selectbox(
    "モデルを選択してください",
    ["OpenAI GPT-4o-mini", "Claude 3.5 Sonnet", "Gemini 1.5 flash"]
)

# ユーザー入力の処理
if prompt := st.chat_input():
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            # OpenAI、AnthropicまたはGeminiを使用して応答を生成
            if model_choice == "OpenAI GPT-4o-mini":
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                )
                full_response = response.choices[0].message.content

            elif model_choice == "Claude 3.5 Sonnet":
                response = anthropic_client.completions.create(
                    model="claude-3-5",
                    prompt=prompt,
                    max_tokens_to_sample=8000,
                )
                full_response = response.completion

            elif model_choice == "Gemini 1.5 flash":
                response = genai.GenerativeModel('gemini-1.5-flash').generate_content(prompt)
                full_response = response.text

            # 応答を表示
            message_placeholder.markdown(full_response)
            st.session_state.chat_history.append({"role": "assistant", "content": full_response})

            # HTMLプレビューとソースコードの分離
            if re.search(r"<html>|<body>", full_response):  # HTMLタグが含まれているか確認
                st.session_state.html_content = full_response  # HTMLの保存
            else:
                st.session_state.html_content = None  # HTMLではない場合はクリア

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")

# HTMLコンテンツの表示（プレビューとソースコード）
if st.session_state.html_content:
    with main:
        tab1, tab2 = st.tabs(["プレビュー", "ソースコード"])
        with tab1:
            components.html(st.session_state.html_content, height=640, scrolling=True)
        with tab2:
            st.markdown(f"```html\n{st.session_state.html_content}\n```")  # ソースコードの表示

# 会話履歴のクリアボタン
if st.button("会話履歴をクリア"):
    st.session_state.chat_history = []
    st.session_state.html_content = None  # HTMLもクリア
    st.experimental_rerun()

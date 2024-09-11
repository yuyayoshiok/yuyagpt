import json
import os
import re
from dotenv import load_dotenv
import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
import firebase_admin
from firebase_admin import credentials, firestore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 起動時に.envファイルを読み込む
load_dotenv()

# Firebaseの初期化
if not firebase_admin._apps:
    try:
        firebase_credentials = json.loads(st.secrets['FIREBASE']['CREDENTIALS_JSON'])
        cred = credentials.Certificate(firebase_credentials)
        firebase_admin.initialize_app(cred)
    except Exception as e:
        st.error(f"Firebaseの初期化に失敗しました: {str(e)}")
        st.stop()

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

# URLを検出する関数
def detect_url(text):
    url_pattern = re.compile(r'https?://\S+')
    return url_pattern.search(text)

# スクレイピングと要約の関数
def scrape_and_summarize(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # タイトルの取得
        title = soup.title.string if soup.title else "No title found"
        
        # メタディスクリプションの取得
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        description = meta_desc['content'] if meta_desc else "No description found"
        
        # 本文の取得
        paragraphs = soup.find_all('p')
        content = ' '.join([p.text for p in paragraphs])
        
        # 要約（最初の500文字）
        summary = content[:500] + "..." if len(content) > 500 else content
        
        return f"Title: {title}\nDescription: {description}\nSummary: {summary}"
    except Exception as e:
        return f"エラーが発生しました: {str(e)}"

# コンテキストを取得する関数
def get_context(current_query, chat_history, max_tokens=1000):
    context = []
    total_tokens = 0
    
    relevant_history = select_relevant_conversations(current_query, chat_history)
    
    for conversation in relevant_history:
        summary = conversation['summary_title']
        messages = conversation['messages']
        
        if total_tokens + len(summary.split()) > max_tokens:
            break
        
        context.append(f"過去の関連会話: {summary}")
        for msg in messages:
            if total_tokens + len(msg['content'][0]['text'].split()) > max_tokens:
                break
            context.append(f"{msg['role']}: {msg['content'][0]['text']}")
            total_tokens += len(msg['content'][0]['text'].split())
    
    return '\n\n'.join(context)

# OpenAIの応答をパースする関数
def parse_openai_response(response):
    if response and hasattr(response, 'choices') and len(response.choices) > 0:
        message = response.choices[0].message
        content = message.content if message.content else ""
        tool_calls = getattr(message, 'tool_calls', [])
        
        parsed_content = [{"text": content}] if content else []
        for tool_call in tool_calls:
            if tool_call.function.name == "html_viewer":
                parsed_content.append({
                    "toolUse": {
                        "toolUseId": tool_call.id,
                        "name": tool_call.function.name,
                        "input": json.loads(tool_call.function.arguments)
                    }
                })
        
        return {"role": "assistant", "content": parsed_content}
    else:
        return {"role": "assistant", "content": [{"text": "No valid response received from the API."}]}

# 起動時に.envファイルを読み込む
reload_env()

st.title("YuyaGPT")

# APIキーが正しく取得できたか確認
if not openai_api_key or not anthropic_api_key or not gemini_api_key:
    st.error("APIキーが正しく設定されていません。.envファイルを確認してください。")
    st.stop()

# セッション状態の初期化
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": [{"text": SYSTEM_PROMPT}]}
    ]

# メインコンテナの設定
main = st.container()

# サイドバーの設定
sidebar = st.sidebar

# モデル選択のプルダウン
model_choice = sidebar.selectbox(
    "モデルを選択してください",
    ["OpenAI GPT-4o-mini", "Claude 3.5 Sonnet", "Gemini 1.5 flash"]
)

# ユーザー入力の処理
if prompt := st.chat_input():
    # URLの検出
    url_match = detect_url(prompt)
    if url_match:
        url = url_match.group()
        summary = scrape_and_summarize(url)
        prompt += f"\n\nURL content summary:\n{summary}"

    with sidebar:
        with st.chat_message("user"):
            st.markdown(prompt)

    user_message = {"role": "user", "content": [{"text": prompt}]}
    st.session_state.messages.append(user_message)

    # チェック: ユーザーのメッセージが連続していないか確認
    if len(st.session_state.messages) > 1 and st.session_state.messages[-2]["role"] == "user":
        st.error("ユーザーのメッセージが連続しています。AI応答を待ってください。")
    else:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            try:
                # AIモデルにプロンプトを送信し、応答を生成
                if model_choice == "OpenAI GPT-4o-mini":
                    response = openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": m["role"], "content": m["content"][0]["text"]} for m in st.session_state.messages],
                    )
                    ai_message = parse_openai_response(response)

                # OpenAIのレスポンスが無効な場合のエラーチェック
                if ai_message is None or not ai_message['content']:
                    ai_message = {"role": "assistant", "content": [{"text": "OpenAI APIの応答が無効です。"}]}

                st.session_state.messages.append(ai_message)

                for content in ai_message["content"]:
                    if "text" in content:
                        message_placeholder.markdown(content["text"])

            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")
                st.error("APIキーを確認し、再試行してください。")
                st.error(f"現在のモデル選択: {model_choice}")

# 会話履歴のクリアボタン
if sidebar.button("会話履歴をクリア"):
    st.session_state.messages = [
        {"role": "system", "content": [{"text": SYSTEM_PROMPT}]}
    ]
    st.rerun()

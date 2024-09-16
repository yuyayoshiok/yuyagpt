import os
from dotenv import load_dotenv
import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
import cohere
from groq import Groq
import requests
from bs4 import BeautifulSoup
import re
import firebase_admin
from firebase_admin import credentials, firestore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import time

# 起動時に.envファイルを読み込む
load_dotenv()

# Firebaseの初期化
if not firebase_admin._apps:
    firebase_credentials = json.loads(st.secrets['FIREBASE']['CREDENTIALS_JSON'])
    cred = credentials.Certificate(firebase_credentials)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# システムプロンプトの定義
SYSTEM_PROMPT = (
    "あなたはプロのエンジニアでありプログラマーです。"
    "GAS、Pythonから始まり多岐にわたるプログラミング言語を習得しています。"
    "あなたが出力するコードは完璧で、省略することなく完全な全てのコードを出力するのがあなたの仕事です。"
    "チャットでは日本語で応対してください。"
    "制約条件として、出力した文章とプログラムコード（コードブロック）は分けて出力してください。"
)

import os
from dotenv import load_dotenv
import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
import cohere
from groq import Groq
import requests
from bs4 import BeautifulSoup
import re
import firebase_admin
from firebase_admin import credentials, firestore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import time

# 起動時に.envファイルを読み込む
load_dotenv()

# Firebaseの初期化
if not firebase_admin._apps:
    firebase_credentials = json.loads(st.secrets['FIREBASE']['CREDENTIALS_JSON'])
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
    
    global openai_api_key, anthropic_api_key, gemini_api_key, cohere_api_key, groq_api_key
    openai_api_key = st.secrets["openai"]["api_key"]
    anthropic_api_key = st.secrets["anthropic"]["api_key"]
    gemini_api_key = st.secrets["gemini"]["api_key"]
    cohere_api_key = st.secrets["cohere"]["api_key"]
    groq_api_key = st.secrets["groq"]["api_key"]
    
    global openai_client, anthropic_client, co, groq_client
    openai_client = OpenAI(api_key=openai_api_key)
    anthropic_client = Anthropic(api_key=anthropic_api_key)
    genai.configure(api_key=gemini_api_key)
    co = cohere.Client(api_key=cohere_api_key)
    groq_client = Groq(api_key=groq_api_key)

# スクレイピングと要約の関数
def scrape_and_summarize(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([p.text for p in paragraphs])
        summary = content[:500] + "..." if len(content) > 500 else content
        return summary
    except Exception as e:
        return f"エラーが発生しました: {str(e)}"

# 関連する過去の会話を選択する関数
def select_relevant_conversations(query, chat_history, top_n=3):
    if not chat_history:
        return []
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([query] + [conv['summary_title'] for conv in chat_history])
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-top_n-1:-1]
    return [chat_history[i] for i in related_docs_indices]

# コンテキストを取得する関数
def get_context(current_query, chat_history, max_tokens=1000):
    context = []
    total_tokens = 0
    relevant_history = select_relevant_conversations(current_query, chat_history)
    for conversation in relevant_history:
        summary = conversation['summary_title']
        messages = conversation['messages']
        if isinstance(summary, str):
            if total_tokens + len(summary.split()) > max_tokens:
                break
            context.append(f"過去の関連会話: {summary}")
            total_tokens += len(summary.split())
        for msg in messages:
            content = msg.get('content', '')
            if isinstance(content, str):
                if total_tokens + len(content.split()) > max_tokens:
                    break
                context.append(f"{msg['role']}: {content}")
                total_tokens += len(content.split())
    return '\n\n'.join(context)

# Cohereを使用した会話機能（ストリーミング対応）
def cohere_chat_stream(prompt):
    response = co.chat(
        model="command-r-plus-08-2024",
        message=prompt,
        temperature=0.5,
        max_tokens=5000,
        stream=True
    )
    for event in response:
        if event.event_type == "text-generation":
            yield event.text

# Groqを使用した会話機能（ストリーミング対応）
def groq_chat_stream(prompt):
    chat_history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=chat_history,
        max_tokens=5000,
        temperature=0.5,
        stream=True
    )
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content

# Firebaseのドキュメントを全て削除する関数
def clear_firebase_documents():
    docs = db.collection('chat_history').get()
    for doc in docs:
        doc.reference.delete()

# 起動時に.envファイルを読み込む
reload_env()

st.title("YuyaGPT")

# APIキーが正しく取得できたか確認
if not openai_api_key or not anthropic_api_key or not gemini_api_key or not cohere_api_key or not groq_api_key:
    st.error("APIキーが正しく設定されていません。.envファイルを確認してください。")
    st.stop()

# セッション状態の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []
if "html_content" not in st.session_state:
    st.session_state.html_content = None

# メインコンテナの設定
main = st.container()

# モデル選択のプルダウン
model_choice = st.selectbox(
    "モデルを選択してください",
    ["OpenAI GPT-4o-mini", "Claude 3.5 Sonnet", "Gemini 1.5 flash", "Cohere Command-R Plus", "Groq"]
)

# 現在の会話を表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ユーザー入力の処理
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            # 過去の関連する会話のコンテキストを取得
            chat_history = db.collection('chat_history').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(20).get()
            context = get_context(prompt, [doc.to_dict() for doc in chat_history])
            
            # AIプロンプトの作成
            ai_prompt = f"{SYSTEM_PROMPT}\n\n過去の関連する会話:\n{context}\n\n現在の質問: {prompt}"
            
            # AIモデルにプロンプトを送信し、応答を生成
            if model_choice == "OpenAI GPT-4o-mini":
                for chunk in openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": ai_prompt}],
                    stream=True
                ):
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")

            elif model_choice == "Claude 3.5 Sonnet":
                with anthropic_client.messages.stream(
                    model="claude-3-5-sonnet-20240620",
                    max_tokens=8000,
                    messages=[{"role": "user", "content": ai_prompt}]
                ) as stream:
                    for text in stream.text_stream:
                        full_response += text
                        message_placeholder.markdown(full_response + "▌")

            elif model_choice == "Gemini 1.5 flash":
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(ai_prompt, stream=True)
                for chunk in response:
                    full_response += chunk.text
                    message_placeholder.markdown(full_response + "▌")

            elif model_choice == "Cohere Command-R Plus":
                for chunk in cohere_chat_stream(ai_prompt):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")

            else:  # Groq llama3-70b-8192
                for chunk in groq_chat_stream(ai_prompt):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

            # 会話履歴をFirebaseに保存（最新の会話のみ）
            latest_conversation = st.session_state.messages[-2:]
            
            # 会話の要約タイトルを生成
            summary_prompt = f"以下の会話を5単語以内で要約してタイトルを作成してください：\nユーザー: {latest_conversation[0]['content']}\nAI: {latest_conversation[1]['content']}"
            summary_response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=10
            )
            summary_title = summary_response.choices[0].message.content.strip()
            
            db.collection('chat_history').add({
                'messages': latest_conversation,
                'summary_title': summary_title,
                'timestamp': firestore.SERVER_TIMESTAMP
            })

            # HTMLコンテンツの抽出
            html_start = full_response.find("<html")
            if html_start != -1:
                html_end = full_response.rfind("</html>") + 7
                st.session_state.html_content = full_response[html_start:html_end]
                text_content = full_response[:html_start] + full_response[html_end:]
                message_placeholder.markdown(text_content)

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
            st.error("APIキーを確認し、再試行してください。")
            st.error(f"現在のモデル選択: {model_choice}")

# HTMLコンテンツの表示
if st.session_state.html_content:
    with main:
        tab1, tab2 = st.tabs(["プレビュー", "ソースコード"])
        with tab1:
            components.html(st.session_state.html_content, height=640, scrolling=True)
        with tab2:
            st.code(st.session_state.html_content, language="html")

# 会話履歴のクリアボタン
if st.button("会話履歴をクリア"):
    st.session_state.messages = []
    st.session_state.html_content = None
    clear_firebase_documents()  # Firebaseのドキュメントを削除
    st.success("会話履歴とFirebaseのドキュメントが削除されました。")
    st.rerun()  # ページを再読み込み
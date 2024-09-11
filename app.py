import json
import os
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
    "必ず、文章とコードブロックは分けて出力してください。"
)

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
        
        if total_tokens + len(summary.split()) > max_tokens:
            break
        
        context.append(f"過去の関連会話: {summary}")
        for msg in messages:
            if total_tokens + len(msg['content'].split()) > max_tokens:
                break
            context.append(f"{msg['role']}: {msg['content']}")
            total_tokens += len(msg['content'].split())
    
    return '\n\n'.join(context)

# Gemini用のメッセージ変換関数
def convert_messages_for_gemini(messages):
    converted = []
    for msg in messages:
        if msg['role'] == 'user':
            converted.append({"role": "user", "parts": [{"text": msg['content']}]})
        elif msg['role'] == 'assistant':
            converted.append({"role": "model", "parts": [{"text": msg['content']}]})
        elif msg['role'] == 'system':
            # システムメッセージは最初のユーザーメッセージに組み込む
            if converted and converted[0]['role'] == 'user':
                converted[0]['parts'][0]['text'] = msg['content'] + "\n" + converted[0]['parts'][0]['text']
            else:
                converted.insert(0, {"role": "user", "parts": [{"text": msg['content']}]})
    return converted

# Claude用のメッセージ変換関数
def convert_messages_for_claude(messages):
    system_message = next((msg['content'] for msg in messages if msg['role'] == 'system'), None)
    user_assistant_messages = [msg for msg in messages if msg['role'] != 'system']
    return system_message, user_assistant_messages

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
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

# メインコンテナの設定
main = st.container()

# モデル選択のプルダウン
model_choice = st.selectbox(
    "モデルを選択してください",
    ["OpenAI GPT-4o-mini", "Claude 3.5 Sonnet", "Gemini 1.5 flash"]
)

# 過去のメッセージを表示
for message in st.session_state.messages[1:]:  # システムメッセージをスキップ
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ユーザー入力の処理
if prompt := st.chat_input():
    # URLの検出
    url_match = detect_url(prompt)
    if url_match:
        url = url_match.group()
        summary = scrape_and_summarize(url)
        prompt += f"\n\nURL content summary:\n{summary}"

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
            
            # AIモデルにプロンプトを送信し、応答を生成
            if model_choice == "OpenAI GPT-4o-mini":
                for chunk in openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=st.session_state.messages,
                    stream=True
                ):
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")

            elif model_choice == "Claude 3.5 Sonnet":
                system_message, user_assistant_messages = convert_messages_for_claude(st.session_state.messages)
                with anthropic_client.messages.stream(
                    model="claude-3-5-sonnet-20240620",
                    max_tokens=8000,
                    system=system_message,
                    messages=user_assistant_messages
                ) as stream:
                    for text in stream.text_stream:
                        full_response += text
                        message_placeholder.markdown(full_response + "▌")

            else:  # Gemini 1.5 flash
                model = genai.GenerativeModel('gemini-1.5-flash')
                gemini_messages = convert_messages_for_gemini(st.session_state.messages[-5:])  # 直近5つのメッセージのみを使用
                response = model.generate_content(gemini_messages, stream=True)
                for chunk in response:
                    if hasattr(chunk, 'text'):
                        chunk_text = chunk.text
                        if chunk_text is not None:
                            full_response += chunk_text
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

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
            st.error("APIキーを確認し、再試行してください。")
            st.error(f"現在のモデル選択: {model_choice}")

# HTMLコンテンツの表示
if 'html_content' in st.session_state and st.session_state.html_content:
    with main:
        tab1, tab2 = st.tabs(["プレビュー", "ソースコード"])
        with tab1:
            components.html(st.session_state.html_content, height=640, scrolling=True)
        with tab2:
            st.code(st.session_state.html_content, language="html")

# 会話履歴のクリアボタン
if st.button("会話履歴をクリア"):
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    st.rerun()
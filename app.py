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

# HTML生成ツールの定義
html_viewer_tool = {
    "type": "function",
    "function": {
        "name": "html_viewer",
        "description": "HTMLを表示します。",
        "parameters": {
            "type": "object",
            "properties": {
                "html": {"type": "string", "description": "HTML Document"}
            },
            "required": ["html"],
        }
    }
}

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
            if total_tokens + len(msg['content'][0]['text'].split()) > max_tokens:
                break
            context.append(f"{msg['role']}: {msg['content'][0]['text']}")
            total_tokens += len(msg['content'][0]['text'].split())
    
    return '\n\n'.join(context)

# Gemini用のメッセージ変換関数
def convert_messages_for_gemini(messages):
    converted = []
    for msg in messages:
        if msg['role'] == 'user':
            converted.append({"role": "user", "parts": [{"text": msg['content'][0]['text']}]})
        elif msg['role'] == 'assistant':
            converted.append({"role": "model", "parts": [{"text": msg['content'][0]['text']}]})
        elif msg['role'] == 'system':
            # システムメッセージは最初のユーザーメッセージに組み込む
            if converted and converted[0]['role'] == 'user':
                converted[0]['parts'][0]['text'] = msg['content'][0]['text'] + "\n" + converted[0]['parts'][0]['text']
            else:
                converted.insert(0, {"role": "user", "parts": [{"text": msg['content'][0]['text']}]})
    return converted

# Claude用のメッセージ変換関数
def convert_messages_for_claude(messages):
    system_message = next((msg['content'][0]['text'] for msg in messages if msg['role'] == 'system'), None)
    user_assistant_messages = [{"role": msg['role'], "content": msg['content'][0]['text']} for msg in messages if msg['role'] != 'system']
    return system_message, user_assistant_messages

# OpenAIの応答をパースする関数
def parse_openai_response(response):
    message = response.choices[0].message
    content = message.content if message.content else ""
    tool_calls = message.tool_calls if hasattr(message, 'tool_calls') else []
    
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

# 過去のメッセージを表示
with sidebar:
    st.subheader("チャットエリア")
    for message in st.session_state.messages:
        if message["role"] != "system":
            for content in message["content"]:
                if "text" in content and not (content["text"] == "OK."):
                    with st.chat_message(message["role"]):
                        st.markdown(content["text"])

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

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            # AIモデルにプロンプトを送信し、応答を生成
            if model_choice == "OpenAI GPT-4o-mini":
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": m["role"], "content": m["content"][0]["text"]} for m in st.session_state.messages],
                    tools=[html_viewer_tool],
                    tool_choice="auto"
                )
                ai_message = parse_openai_response(response)
                
            elif model_choice == "Claude 3.5 Sonnet":
                system_message, user_assistant_messages = convert_messages_for_claude(st.session_state.messages)
                response = anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    max_tokens=8000,
                    system=system_message,
                    messages=user_assistant_messages,
                    tools=[html_viewer_tool]
                )
                ai_message = {"role": "assistant", "content": [{"text": response.content[0].text}] if response.content else []}

            else:  # Gemini 1.5 flash
                model = genai.GenerativeModel('gemini-1.5-flash')
                gemini_messages = convert_messages_for_gemini(st.session_state.messages[-5:])  # 直近5つのメッセージのみを使用
                response = model.generate_content(gemini_messages)
                ai_message = {"role": "assistant", "content": [{"text": response.text}] if response.text else []}

            st.session_state.messages.append(ai_message)

            for content in ai_message["content"]:
                if "text" in content:
                    message_placeholder.markdown(content["text"])
                if "toolUse" in content:
                    tool_use = content["toolUse"]
                    tool_use_id = tool_use["toolUseId"]
                    name = tool_use["name"]
                    html = tool_use["input"]["html"]
                    with main:
                        tab1, tab2 = st.tabs(["プレビュー", "ソースコード"])
                        with tab1:
                            components.html(
                                html,
                                height=640,
                                scrolling=True,
                            )
                        with tab2:
                            st.markdown(f"```html\n{html}\n```")
                    st.session_state.messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "toolResult": {
                                        "toolUseId": tool_use_id,
                                        "content": [{"text": "Done."}],
                                    }
                                }
                            ],
                        }
                    )
                    st.session_state.messages.append({"role": "assistant", "content": [{"text": "OK."}]})

            # 会話履歴をFirebaseに保存（最新の会話のみ）
            latest_conversation = st.session_state.messages[-2:]
            
            # 会話の要約タイトルを生成
            summary_prompt = f"以下の会話を5単語以内で要約してタイトルを作成してください：\nユーザー: {latest_conversation[0]['content'][0]['text']}\nAI: {latest_conversation[1]['content'][0]['text']}"
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

# 会話履歴のクリアボタン
if sidebar.button("会話履歴をクリア"):
    st.session_state.messages = [
        {"role": "system", "content": [{"text": SYSTEM_PROMPT}]}
    ]
    st.rerun()
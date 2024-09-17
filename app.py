import os
import re
import base64
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
import hashlib

# .envファイルを読み込む
load_dotenv()

# システムプロンプトの定義
SYSTEM_PROMPT = """
あなたはプロのエンジニアでありプログラマーです。
GAS、Pythonから始まり多岐にわたるプログラミング言語を習得しています。
あなたが出力するコードは完璧で、省略することなく完全な全てのコードを出力するのがあなたの仕事です。
チャットでは日本語で応対してください。
コードを出力する際は、必ず適切な言語名を指定してコードブロックを使用してください。
同様に、Python、JavaScript、CSSなどのコードも適切なコードブロックで囲んでください。
"""

# ユーザー認証情報（実際の使用時はデータベースなどを使用してください）
USERS = {
    "yuyayoshiok@gmail.com": hashlib.sha256("Yoshi0731".encode()).hexdigest(),
}

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

# URLの検出関数
def detect_url(text):
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    urls = url_pattern.findall(text)
    return urls[0] if urls else None

# スクレイピングと要約の関数
def scrape_and_summarize(url, model_choice):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # メタデータの取得
        title = soup.title.string if soup.title else "No title"
        description = soup.find('meta', attrs={'name': 'description'})
        description = description['content'] if description else "No description"
        
        # 本文の取得
        paragraphs = soup.find_all('p')
        content = ' '.join([p.text for p in paragraphs])
        
        # 長すぎる場合は切り詰める
        max_content_length = 5000  # 適宜調整してください
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."
        
        # AIモデルによる要約
        summary = summarize_with_ai(title, description, content, model_choice)
        
        return summary
    except Exception as e:
        return f"エラーが発生しました: {str(e)}"

# AIモデルによる要約関数
def summarize_with_ai(title, description, content, model_choice):
    prompt = f"""以下のウェブページの内容を要約してください。
タイトル: {title}
説明: {description}
本文:
{content}

要約:"""
    
    if model_choice == "OpenAI GPT-4o-mini":
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        return response.choices[0].message.content
    
    elif model_choice == "Claude 3.5 Sonnet":
        response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    elif model_choice == "Gemini 1.5 flash":
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    
    elif model_choice == "Cohere Command-R Plus":
        response = co.summarize(
            text=content,
            length='medium',
            format='paragraph',
            model='command-r-plus-08-2024',
            additional_command=f"Title: {title}\nDescription: {description}"
        )
        return response.summary
    
    else:  # Groq llama3-70b-8192
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        return response.choices[0].message.content

# メッセージの役割を適切に変換する関数
def convert_role_for_api(role):
    if role == 'human':
        return 'user'
    elif role == 'ai':
        return 'assistant'
    return role

# メッセージを適切な形式に整形する関数
def format_messages_for_claude(messages):
    formatted_messages = []
    for i, message in enumerate(messages):
        if i == 0 and message['role'] == 'assistant':
            # 最初のメッセージがアシスタントの場合、スキップ
            continue
        if i > 0 and message['role'] == formatted_messages[-1]['role']:
            # 同じ役割が連続する場合、内容を結合
            formatted_messages[-1]['content'] += "\n" + message['content']
        else:
            formatted_messages.append(message)
    return formatted_messages

# HTMLコンテンツを抽出する関数
def extract_html_content(text):
    html_blocks = re.findall(r'```html\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
    if html_blocks:
        return html_blocks[0].strip()
    return None

# HTMLをbase64エンコードしてdata URLを作成する関数
def get_html_data_url(html):
    b64 = base64.b64encode(html.encode()).decode()
    return f"data:text/html;base64,{b64}"

# HTMLプレビューを表示する関数
def display_html_preview(html_content):
    html_data_url = get_html_data_url(html_content)
    components.iframe(html_data_url, height=600, scrolling=True)

# Cohereを使用した会話機能（ストリーミング対応、新しいSDKバージョンに対応）
def cohere_chat_stream(prompt):
    chat_history = [
        {"role": "USER" if m.type == "human" else "CHATBOT", "message": m.content}
        for m in st.session_state.memory.chat_memory.messages
    ]
    response = co.chat_stream(
        model='command-r-plus-08-2024',
        message=prompt,
        temperature=0.5,
        chat_history=chat_history,
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

# AIモデルにプロンプトを送信し、応答を生成
def generate_response(prompt, model_choice, memory):
    url = detect_url(prompt)
    if url:
        summary = scrape_and_summarize(url, model_choice)
        full_response = f"ウェブページの要約:\n\n{summary}\n\n元のURL: {url}"
        yield full_response
        memory.chat_memory.add_ai_message(full_response)
        return

    full_response = ""
    chat_history = memory.chat_memory.messages
    
    try:
        if model_choice == "OpenAI GPT-4o-mini":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
                {"role": convert_role_for_api(m.type), "content": m.content} for m in chat_history
            ] + [{"role": "user", "content": prompt}]
            
            for chunk in openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                stream=True
            ):
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    yield full_response

        elif model_choice == "Claude 3.5 Sonnet":
            messages = [
                {"role": convert_role_for_api(m.type), "content": m.content} for m in chat_history
            ] + [{"role": "user", "content": prompt}]
            
            formatted_messages = format_messages_for_claude(messages)
            
            with anthropic_client.messages.stream(
                model="claude-3-5-sonnet-20240620",
                max_tokens=8000,
                system=SYSTEM_PROMPT,
                messages=formatted_messages
            ) as stream:
                for text in stream.text_stream:
                    full_response += text
                    yield full_response

        elif model_choice == "Gemini 1.5 flash":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
                {"role": convert_role_for_api(m.type), "content": m.content} for m in chat_history
            ] + [{"role": "user", "content": prompt}]
            
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content([m["content"] for m in messages], stream=True)
            for chunk in response:
                full_response += chunk.text
                yield full_response

        elif model_choice == "Cohere Command-R Plus":
            for chunk in cohere_chat_stream(prompt):
                full_response += chunk
                yield full_response

        else:  # Groq llama3-70b-8192
            for chunk in groq_chat_stream(prompt):
                full_response += chunk
                yield full_response

        # 会話履歴に応答を追加
        memory.chat_memory.add_ai_message(full_response)

    except Exception as e:
        st.error(f"エラーが発生しました: {str(e)}")
        yield "申し訳ありません。エラーが発生しました。もう一度お試しください。"

# ログイン状態の確認
def check_login_status():
    return st.session_state.get('logged_in', False)

# ログインページ
def login_page():
    st.title("ログイン")
    username = st.text_input("ユーザー名")
    password = st.text_input("パスワード", type="password")
    if st.button("ログイン"):
        if username in USERS and USERS[username] == hashlib.sha256(password.encode()).hexdigest():
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.success("ログインに成功しました。")
            st.rerun()
        else:
            st.error("ユーザー名またはパスワードが間違っています。")

# メイン機能の関数
def main_app():
    st.title("YuyaGPT")

    # ログアウトボタン
    if st.sidebar.button("ログアウト"):
        st.session_state['logged_in'] = False
        st.session_state.pop('username', None)
        st.success("ログアウトしました。")
        st.rerun()

    # セッション状態の初期化
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )

    if "html_content" not in st.session_state:
        st.session_state.html_content = ""

    # モデル選択のプルダウン
    model_choice = st.selectbox(
        "使用するモデルを選択してください",
        ["OpenAI GPT-4o-mini", "Claude 3.5 Sonnet", "Gemini 1.5 flash", "Cohere Command-R Plus", "Groq"]
    )

    # メインコンテンツエリアの作成
    main = st.container()

    # 会話履歴の表示
    for message in st.session_state.memory.chat_memory.messages:
        with st.chat_message(message.type):
            st.markdown(message.content)

    # ユーザー入力の処理
    if prompt := st.chat_input("質問を入力してください"):
        st.session_state.memory.chat_memory.add_user_message(prompt)
        with st.chat_message("human"):
            st.markdown(prompt)

        # AI応答の生成
        with st.chat_message("ai"):
            message_placeholder = st.empty()
            try:
                full_response = ""
                for response in generate_response(prompt, model_choice, st.session_state.memory):
                    full_response = response
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                
                # HTMLコンテンツの抽出
                html_content = extract_html_content(full_response)
                if html_content:
                    st.session_state.html_content = html_content
            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")
                message_placeholder.markdown("申し訳ありません。エラーが発生しました。もう一度お試しください。")

    # HTMLコンテンツの表示
    if st.session_state.html_content:
        with main:
            tab1, tab2 = st.tabs(["プレビュー", "ソースコード"])
            with tab1:
                st.subheader("HTMLプレビュー")
                display_html_preview(st.session_state.html_content)
            with tab2:
                st.subheader("HTMLソースコード")
                st.code(st.session_state.html_content, language="html")

    # 会話履歴のクリアボタン
    if st.sidebar.button("会話履歴をクリア"):
        st.session_state.memory.clear()
        st.session_state.html_content = ""
        st.rerun()

# メイン処理
def main():
    reload_env()
    if check_login_status():
        main_app()
    else:
        login_page()

if __name__ == "__main__":
    main()
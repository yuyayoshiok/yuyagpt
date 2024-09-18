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
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
import hashlib
import time
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import DuckDuckGoSearchException
import streamlit as st
import random
from concurrent.futures import ThreadPoolExecutor, TimeoutError



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

# DuckDuckGo検索機能
def duckduckgo_search(prompt):
    search_type = "text"
    keywords = prompt

    if "画像" in prompt and "調べて" in prompt:
        search_type = "images"
        keywords = prompt.replace("画像を調べて", "").strip()
    elif "動画" in prompt and "調べて" in prompt:
        search_type = "videos"
        keywords = prompt.replace("動画を調べて", "").strip()
    elif "ニュース" in prompt and "調べて" in prompt:
        search_type = "news"
        keywords = prompt.replace("最新のニュースを調べて", "").replace("ニュースを調べて", "").strip()
    elif "調べて" in prompt:
        keywords = prompt.replace("調べて", "").strip()

    try:
        if search_type == "text":
            results = DDGS().text(keywords, region="jp-jp", max_results=3)
        elif search_type == "images":
            results = DDGS().images(keywords, region="jp-jp", safesearch="moderate", max_results=3)
        elif search_type == "videos":
            results = DDGS().videos(keywords, region="jp-jp", safesearch="moderate", max_results=3)
        elif search_type == "news":
            results = DDGS().news(keywords, region="jp-jp", max_results=3)
        return list(results), search_type
    except Exception as e:
        st.error(f"検索中にエラーが発生しました: {str(e)}")
        return None, search_type

# AIモデルにプロンプトを送信し、応答を生成
def generate_response(prompt, model_choice, memory):
    search_results, search_type = duckduckgo_search(prompt)
    
    full_response = ""
    chat_history = memory.chat_memory.messages
    
    try:
        if search_results:
            full_response += f"DuckDuckGo検索結果 ({search_type}):\n\n"
            for result in search_results:
                if search_type == "text":
                    full_response += f"- {result['body']}\n  URL: {result['href']}\n\n"
                elif search_type == "images":
                    full_response += f"![画像]({result['image']})\n  URL: {result['url']}\n\n"
                elif search_type == "videos":
                    full_response += f"動画: {result['title']}\n  URL: {result['content']}\n\n"
                elif search_type == "news":
                    full_response += f"- {result['body']}\n  URL: {result['url']}\n  日付: {result['date']}\n\n"
            
            full_response += "\n検索結果の解釈：\n"

        if model_choice == "OpenAI GPT-4o-mini":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
                {"role": convert_role_for_api(m.type), "content": m.content} for m in chat_history
            ] + [{"role": "user", "content": prompt}]
            if search_results:
                messages.append({"role": "system", "content": f"以下の検索結果を解釈し、ユーザーの質問に答えてください：\n{full_response}"})
            
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
            if search_results:
                messages.append({"role": "assistant", "content": f"以下の検索結果を解釈し、ユーザーの質問に答えてください：\n{full_response}"})
            
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
            if search_results:
                messages.append({"role": "system", "content": f"以下の検索結果を解釈し、ユーザーの質問に答えてください：\n{full_response}"})
            
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content([m["content"] for m in messages], stream=True)
            for chunk in response:
                full_response += chunk.text
                yield full_response

        elif model_choice == "Cohere Command-R Plus":
            chat_history = [
                {"role": "USER" if m.type == "human" else "CHATBOT", "message": m.content}
                for m in st.session_state.memory.chat_memory.messages
            ]
            if search_results:
                chat_history.append({"role": "CHATBOT", "message": f"以下の検索結果を解釈し、ユーザーの質問に答えてください：\n{full_response}"})
            chat_history.append({"role": "USER", "message": prompt})
            
            response = co.chat_stream(
                model='command-r-plus-08-2024',
                message=prompt,
                temperature=0.5,
                chat_history=chat_history,
            )
            for event in response:
                if event.event_type == "text-generation":
                    full_response += event.text
                    yield full_response

        else:  # Groq llama-3.1-70b-versatile
            chat_history = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            if search_results:
                chat_history.insert(1, {"role": "assistant", "content": f"以下の検索結果を解釈し、ユーザーの質問に答えてください：\n{full_response}"})
            
            response = groq_client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=chat_history,
                max_tokens=5000,
                temperature=0.5,
                stream=True
            )
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
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

def generate_response_with_timeout(prompt, model_choice, memory, timeout=60):
    def generate():
        try:
            for response in generate_response(prompt, model_choice, memory):
                yield response
        except Exception as e:
            st.error(f"応答生成中にエラーが発生しました: {str(e)}")
            yield "エラーが発生しました。もう一度お試しください。"

    with ThreadPoolExecutor() as executor:
        future = executor.submit(lambda: list(generate()))
        try:
            result = future.result(timeout=timeout)
            return result[-1] if result else "タイムアウトにより応答を生成できませんでした。"
        except TimeoutError:
            return "応答生成がタイムアウトしました。もう一度お試しください。"

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

    # デバッグ情報の表示
    st.sidebar.subheader("デバッグ情報")
    st.sidebar.write(f"セッション状態: {st.session_state}")

    # ユーザー入力の処理
    if prompt := st.chat_input("質問を入力してください"):
        st.session_state.memory.chat_memory.add_user_message(prompt)
        with st.chat_message("human"):
            st.markdown(prompt)

        # AI応答の生成
        with st.chat_message("ai"):
            message_placeholder = st.empty()
            try:
                start_time = time.time()
                full_response = generate_response_with_timeout(prompt, model_choice, st.session_state.memory)
                end_time = time.time()
                
                message_placeholder.markdown(full_response)
                st.sidebar.write(f"応答生成時間: {end_time - start_time:.2f}秒")

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
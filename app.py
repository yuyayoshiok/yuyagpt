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

# .envファイルを読み込む
load_dotenv()

# システムプロンプトの定義
SYSTEM_PROMPT = """
あなたはプロのエンジニアでありプログラマーです。
GAS、Pythonから始まり多岐にわたるプログラミング言語を習得しています。
あなたが出力するコードは完璧で、省略することなく完全な全てのコードを出力するのがあなたの仕事です。
チャットでは日本語で応対してください。
コードを出力する際は、必ず適切な言語名を指定してコードブロックを使用してください。
例えば、HTMLコードを出力する場合は以下のようにしてください：

```html
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>例題</title>
</head>
<body>
    <h1>これは例です</h1>
</body>
</html>
```

同様に、Python、JavaScript、CSSなどのコードも適切なコードブロックで囲んでください。
"""

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
    response = co.chat_stream(
        model="command-r-plus-08-2024",
        message=prompt,
        temperature=0.5,
        max_tokens=4096
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

# AIモデルにプロンプトを送信し、応答を生成
def generate_response(ai_prompt, model_choice, memory):
    full_response = ""
    chat_history = memory.chat_memory.messages
    
    try:
        if model_choice == "OpenAI GPT-4o-mini":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
                {"role": convert_role_for_api(m.type), "content": m.content} for m in chat_history
            ] + [{"role": "user", "content": ai_prompt}]
            
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
            ] + [{"role": "user", "content": ai_prompt}]
            
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
            ] + [{"role": "user", "content": ai_prompt}]
            
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content([m["content"] for m in messages], stream=True)
            for chunk in response:
                full_response += chunk.text
                yield full_response

        elif model_choice == "Cohere Command-R Plus":
            for chunk in cohere_chat_stream(ai_prompt):
                full_response += chunk
                yield full_response

        else:  # Groq llama3-70b-8192
            for chunk in groq_chat_stream(ai_prompt):
                full_response += chunk
                yield full_response

        # 会話履歴に応答を追加
        memory.chat_memory.add_ai_message(full_response)

    except Exception as e:
        st.error(f"エラーが発生しました: {str(e)}")
        yield "申し訳ありません。エラーが発生しました。もう一度お試しください。"

# 起動時に.envファイルを読み込む
reload_env()

st.title("YuyaGPT")

# セッション状態の初期化
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

if "html_content" not in st.session_state:
    st.session_state.html_content = ""

# PDFファイルのアップロード
uploaded_file = st.file_uploader("PDFファイルをアップロードしてください", type="pdf")

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
if st.button("会話履歴をクリア"):
    st.session_state.memory.clear()
    st.session_state.html_content = ""
    st.rerun()  # 最新のStreamlit APIを使用してページを再読み込み
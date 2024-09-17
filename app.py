import os
import tempfile
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
import json
import time
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# .envファイルを読み込む
load_dotenv()

# システムプロンプトの定義
SYSTEM_PROMPT = (
    "あなたはプロのエンジニアでありプログラマーです。"
    "GAS、Pythonから始まり多岐にわたるプログラミング言語を習得しています。"
    "あなたが出力するコードは完璧で、省略することなく完全な全てのコードを出力するのがあなたの仕事です。"
    "チャットでは日本語で応対してください。"
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

# AIモデルにプロンプトを送信し、応答を生成
def generate_response(ai_prompt, model_choice, message_placeholder):
    full_response = ""

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

    return full_response

# 起動時に.envファイルを読み込む
reload_env()

st.title("YuyaGPT")

# PDFファイルのアップロード
uploaded_file = st.file_uploader("PDFファイルをアップロードしてください", type="pdf")

# セッション状態の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

# モデル選択のプルダウン
model_choice = st.selectbox(
    "使用するモデルを選択してください",
    ["OpenAI GPT-4o-mini", "Claude 3.5 Sonnet", "Gemini 1.5 flash", "Cohere Command-R Plus", "Groq"]
)

if uploaded_file:
    # 一時ファイルにPDFを書き込みバスを取得
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # PDFを読み込む
    loader = PyMuPDFLoader(file_path=tmp_file_path)
    documents = loader.load()

    # テキストをチャンク化
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)

    # 埋め込みを作成
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vector_store = Chroma(
        persist_directory="./.data",
        embedding_function=embeddings,
    )

    vector_store.add_documents(chunks)

    # モデルを使用した会話チェーンを作成
    retriever = vector_store.as_retriever()
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(),
        retriever=retriever,
        memory=st.session_state.memory,
    )

    # UI用の会話履歴を表示
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ユーザー入力の処理
    user_input = st.text_input("質問を入力してください", key="user_input")  # ここで st.text_input を使用

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # AI応答の生成
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                response = chain.run({"question": user_input})
                st.markdown(response)

        # 会話履歴に保存
        st.session_state.messages.append({"role": "assistant", "content": response})



# 会話履歴のクリアボタン
if st.button("会話履歴をクリア"):
    st.session_state.messages = []
    st.session_state.memory.clear()
    st.experimental_rerun()

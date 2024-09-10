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
import re
import firebase_admin
from firebase_admin import credentials, firestore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, PromptTemplate
from llama_index.llms.cohere import Cohere
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank
import gc

# 起動時に.envファイルを読み込む
load_dotenv()

# Firebaseの初期化
if not firebase_admin._apps:
    firebase_credentials = json.loads(st.secrets['FIREBASE']['CREDENTIALS_JSON'])
    cred = credentials.Certificate(firebase_credentials)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# .envファイルの再読み込み関数
def reload_env():
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(dotenv_path, override=True)
    
    global openai_api_key, anthropic_api_key, gemini_api_key, cohere_api_key
    openai_api_key = st.secrets["openai"]["api_key"]
    anthropic_api_key = st.secrets["anthropic"]["api_key"]
    gemini_api_key = st.secrets["gemini"]["api_key"]
    cohere_api_key = st.secrets["cohere"]["api_key"]
    
    global openai_client, anthropic_client
    openai_client = OpenAI(api_key=openai_api_key)
    anthropic_client = Anthropic(api_key=anthropic_api_key)
    genai.configure(api_key=gemini_api_key)

reload_env()

st.title("YuyaGPT")

# APIキーが正しく取得できたか確認
if not openai_api_key or not anthropic_api_key or not gemini_api_key or not cohere_api_key:
    st.error("APIキーが正しく設定されていません。.envファイルを確認してください。")
    st.stop()

# モデル選択のプルダウンにCommand-R+を追加
model_choice = st.selectbox(
    "モデルを選択してください",
    ["OpenAI GPT-4o-mini", "Claude 3.5 Sonnet", "Gemini 1.5 flash", "Command-R+"]
)

# チャット履歴の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []
if "html_content" not in st.session_state:
    st.session_state.html_content = None

# メインコンテナの設定
main = st.container()

# Command-R+が選択された場合のセットアップ
if model_choice == "Command-R+":
    st.header(f"Command-R+ Model Selected")
    
    # Cohere APIキーを設定
    API_KEY = cohere_api_key

    # アップロードされたファイルがある場合
    uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")
    if uploaded_file and API_KEY:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                file_key = f"{session_id}-{uploaded_file.name}"
                st.write("Indexing your document...")

                if file_key not in st.session_state.get('file_cache', {}):

                    loader = SimpleDirectoryReader(input_dir=temp_dir, required_exts=[".pdf"], recursive=True)
                    docs = loader.load_data()

                    llm = Cohere(api_key=API_KEY, model="command-r-plus")
                    embed_model = CohereEmbedding(cohere_api_key=API_KEY, model_name="embed-english-v3.0", input_type="search_query")
                    cohere_rerank = CohereRerank(model='rerank-english-v3.0', api_key=API_KEY)

                    Settings.embed_model = embed_model
                    index = VectorStoreIndex.from_documents(docs, show_progress=True)

                    Settings.llm = llm
                    query_engine = index.as_query_engine(streaming=True, node_postprocessors=[cohere_rerank])

                    qa_prompt_tmpl_str = (
                    "Context information is below.\n"
                    "---------------------\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "Given the context information above I want you to think step by step to answer the query in a crisp manner, incase case you don't know the answer say 'I don't know!'.\n"
                    "Query: {query_str}\n"
                    "Answer: "
                    )
                    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
                    query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})

                    st.session_state.file_cache[file_key] = query_engine
                else:
                    query_engine = st.session_state.file_cache[file_key]

                st.success("Ready to Chat!")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()

# 現在の会話を表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ユーザー入力の処理
if prompt := st.chat_input("What's up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            if model_choice == "OpenAI GPT-4o-mini":
                for chunk in openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    stream=True
                ):
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")

            elif model_choice == "Claude 3.5 Sonnet":
                with anthropic_client.messages.stream(
                    model="claude-3-5-sonnet-20240620",
                    max_tokens=8000,
                    messages=[{"role": "user", "content": prompt}]
                ) as stream:
                    for text in stream.text_stream:
                        full_response += text
                        message_placeholder.markdown(full_response + "▌")

            elif model_choice == "Gemini 1.5 flash":
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(prompt, stream=True)
                for chunk in response:
                    full_response += chunk.text
                    message_placeholder.markdown(full_response + "▌")

            elif model_choice == "Command-R+":
                streaming_response = query_engine.query(prompt)
                for chunk in streaming_response.response_gen:
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")

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
import json #追加

# 起動時に.envファイルを読み込む
load_dotenv()  # .envファイルの内容を環境変数としてロードする

# Firebaseの初期化
if not firebase_admin._apps:
    # Firebaseのシークレット情報をst.secretsから取得してJSON形式に変換
    firebase_credentials = json.loads(st.secrets['FIREBASE']['CREDENTIALS_JSON'])
    
    # Firebase認証情報を使ってアプリを初期化
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
    
    global openai_api_key, anthropic_api_key, gemini_api_key, dify_api_key, dify_api_url
    
    try:
        openai_api_key = st.secrets["openai"]["api_key"]
        anthropic_api_key = st.secrets["anthropic"]["api_key"]
        gemini_api_key = st.secrets["gemini"]["api_key"]
        
        # Difyの設定を取得し、URLとキーが正しいか確認
        dify_config = st.secrets["dify"]
        dify_api_key = dify_config["api_key"]
        dify_api_url = dify_config["api_url"]
        
        # URLとキーが逆になっている可能性をチェック
        if dify_api_key.startswith("http") and not dify_api_url.startswith("http"):
            dify_api_key, dify_api_url = dify_api_url, dify_api_key
        
        # URLの形式チェック
        if not dify_api_url.startswith("http"):
            raise ValueError("Dify API URLが正しくありません")
        
        # APIキーの簡易チェック（完全な検証は難しいですが、明らかに間違っているものを排除）
        if len(dify_api_key) < 10:  # 適切な最小長さに調整してください
            raise ValueError("Dify APIキーが短すぎます")
        
    except KeyError as e:
        st.error(f"環境変数の設定エラー: {e} が見つかりません。設定を確認してください。")
        return False
    except ValueError as e:
        st.error(f"環境変数の値エラー: {e}")
        return False
    
    global openai_client, anthropic_client
    openai_client = OpenAI(api_key=openai_api_key)
    anthropic_client = Anthropic(api_key=anthropic_api_key)
    genai.configure(api_key=gemini_api_key)
    
    return True

# Difyクライアントの初期化関数
def init_dify_client():
    return {
        "api_key": dify_api_key,
        "api_url": dify_api_url
    }

# スクレイピングと要約の関数
def scrape_and_summarize(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 本文の取得（単純化のため、pタグのテキストのみを取得）
        paragraphs = soup.find_all('p')
        content = ' '.join([p.text for p in paragraphs])
        
        # 内容の要約（ここでは簡単な要約として最初の500文字を使用）
        summary = content[:500] + "..." if len(content) > 500 else content
        
        return summary
    except Exception as e:
        return f"エラーが発生しました: {str(e)}"

# 関連する過去の会話を選択する関数
def select_relevant_conversations(query, chat_history, top_n=3):
    if not chat_history:  # チャット履歴が空の場合
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

# Difyを使用してメッセージを送信する関数
def send_message_to_dify(dify_client, message):
    headers = {
        "Authorization": f"Bearer {dify_client['api_key']}",
        "Content-Type": "application/json"
    }
    data = {
        "inputs": {},
        "query": message,
        "response_mode": "streaming",
        "conversation_id": None,
        "user": "user"
    }
    response = requests.post(f"{dify_client['api_url']}/chat-messages", headers=headers, json=data, stream=True)
    return response

# メイン処理の開始
if reload_env():
    st.title("YuyaGPT")

    # APIキーが正しく取得できたか確認
    if not openai_api_key or not anthropic_api_key or not gemini_api_key or not dify_api_key or not dify_api_url:
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
        ["OpenAI GPT-4o-mini", "Claude 3.5 Sonnet", "Gemini 1.5 flash", "モダン会議室"]
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

                elif model_choice == "モダン会議室":
                    dify_client = init_dify_client()
                    response = send_message_to_dify(dify_client, prompt)
                    for chunk in response.iter_lines():
                        if chunk:
                            chunk_data = json.loads(chunk.decode('utf-8'))
                            if 'answer' in chunk_data:
                                full_response += chunk_data['answer']
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
        st.session_state.reload_page = True  # ページの再読み込みフラグを設定

    # ページの再読み込み処理
    if 'reload_page' in st.session_state and st.session_state.reload_page:
        st.session_state.reload_page = False
        st.session_state.reload_page = True  # ページの再読み込みフラグを設定

else:
    st.error("環境変数の読み込みに失敗しました。アプリケーションを続行できません。")
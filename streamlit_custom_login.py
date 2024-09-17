import streamlit as st
import streamlit.components.v1 as components
import hashlib

# 認証されたユーザーの情報
AUTHORIZED_EMAIL = "yuyayoshiok@gmail.com"
HASHED_PASSWORD = hashlib.sha256("Yoshi0731-".encode()).hexdigest()  # パスワードをハッシュ化して保存

def login_page():
    st.title("ログイン")
    
    # HTMLとCSSを使用したカスタムログインフォーム
    login_html = """
    <style>
        .login-form {
            max-width: 300px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f0f0f0;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }
        .login-form input {
            width: 100%;
            padding: 10px;
            margin-bottom: 10px;
            border: 1px solid #ddd;
            border-radius: 3px;
        }
        .login-form button {
            width: 100%;
            padding: 10px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
        }
        .login-form button:hover {
            background-color: #45a049;
        }
    </style>
    <div class="login-form">
        <input type="email" id="email" placeholder="メールアドレス">
        <input type="password" id="password" placeholder="パスワード">
        <button onclick="login()">ログイン</button>
    </div>
    <script>
        function login() {
            var email = document.getElementById('email').value;
            var password = document.getElementById('password').value;
            if (email && password) {
                window.parent.postMessage({type: 'streamlit:setComponentValue', value: JSON.stringify({email: email, password: password})}, '*');
            } else {
                alert('メールアドレスとパスワードを入力してください。');
            }
        }
    </script>
    """
    
    login_data = components.html(login_html, height=200)
    if login_data:
        login_info = eval(login_data)
        if login_info['email'] == AUTHORIZED_EMAIL and hashlib.sha256(login_info['password'].encode()).hexdigest() == HASHED_PASSWORD:
            st.session_state.logged_in = True
            st.experimental_rerun()
        else:
            st.error("メールアドレスまたはパスワードが正しくありません。")

def logout():
    st.session_state.logged_in = False
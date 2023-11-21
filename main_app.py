import streamlit as st
import openai

# OpenAIのAPIキーを設定
openai.api_key = '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'

# セッション内で使用するモデルが指定されていない場合のデフォルト値
if "openai_model" not in st.session_state:
    st.session_state.openai_model = "text-davinci-003"  # GPT-3モデルを指定

# セッション内のメッセージが指定されていない場合のデフォルト値
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title('要約アプリ')
st.caption('愛媛新聞の要約されたものがみれます。')

# テキストボックス
input_text = st.text_input('要約したい文章')

# ボタンがクリックされたときの処理
if st.button('送信'):
    # ユーザーの入力をセッション内のメッセージに追加
    st.session_state.messages.append({"role": "user", "content": input_text})

    # 対話を生成
    response = openai.ChatCompletion.create(
        model=st.session_state.openai_model,
        messages=st.session_state.messages,
    )

    # 生成された要約をセッション内のメッセージに追加
    summary = response.choices[0].message['content']
    st.session_state.messages.append({"role": "assistant", "content": summary})

# チャット履歴を表示
for message in st.session_state.messages:
    if message["role"] == "user":
        st.text("User: " + message["content"])
    else:
        st.text("Assistant: " + message["content"])
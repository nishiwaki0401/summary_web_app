import streamlit as st
from PIL import Image


st.title('要約アプリ')
st.caption('愛媛新聞の要約されたものがみれます。')

#テキストボックス
input = st.text_input('要約したい文章')

#ボタン
submit_btn = st.button('送信')

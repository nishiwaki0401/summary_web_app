import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.combine_documents.base import Document
import os
from datetime import datetime  # 新しい行を追加
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

# # OpenAI APIキーを設定
# os.environ["OPENAI_API_KEY"] = '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'

def init_page():
    st.set_page_config(
        page_title="要約アプリ",
        page_icon="🧠"
    )
    st.header("要約アプリ 🧠")
    
    # サイドバーのタイトルを表示
    st.sidebar.title("モデル選択")
    st.session_state.costs = []

def init_messages():
    # サイドバーにボタンを設置
    clear_button = st.sidebar.button("履歴削除", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="デモ段階であるため、ただchatgptのapiを使用してwebappを作成しただけになっているが今後要約アプリとして工夫していく")
        ]
        st.session_state.costs = []

def select_model():
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
    if model == "GPT-3.5":
        model_name = "gpt-3.5-turbo"
    else:
        model_name = "gpt-4"

    return ChatOpenAI(temperature=0, model_name=model_name)

def get_text_input():
    text_input = st.text_area("テキストを入力してください:", key="input", height=200)
    text_splitter = CharacterTextSplitter(separator="。", chunk_size=200)
    texts = text_splitter.split_text(text_input)
    return texts

def summarize(llm, docs):
    prompt_template = """
    見出し: 女性美　表現の変遷　新居浜市美術館「描かれた女たち」　明治から現代　日本の洋画８１点　来月２６日まで
    本文: 　西洋美術との出合いは、日本人画家が描く人体像にどんな変化をもたらしたのか―。女性をモチーフにした明治時代から現代の洋画を通じその変遷に迫る特別展「描かれた女たち―女性像にみるフォルム／現実／夢」が、新居浜市坂井町２丁目の市美術館で開かれている。日動美術財団所蔵の７５点を中心に計８１点が並ぶ。６月２６日まで。 　明治期に西洋美術に接した日本画壇は、対象の科学的な捉え方や陰影法などの技法だけでなく、絵画とは何かといった概念も吸収した。特別展では東郷青児、竹久夢二、百武兼行、絹谷幸二、岸田劉生、藤島武二らが描いた「女性美」から、表現の多様性を浮かび上がらせている。展示作品の一部を紹介する。（所蔵は全て日動美術財団）
	人間が作成した要約文章:女性をモチーフにした明治時代から現代の洋画を通じその変遷に迫る特別展「描かれた女たち―女性像にみるフォルム／現実／夢」が、新居浜市坂井町２丁目の市美術館で開かれている。日動美術財団所蔵の７５点を中心に計８１点が並び、展示期間は6月26日までである。

    見出し: 大王製紙　おむつ原料　自社生産　戦略説明会　輸入から切り替え
    本文: 　大王製紙（四国中央市、若林賴房社長）は２６日、オンラインで戦略説明会を開き、２０２１～２３年度の第４次中期事業計画の進展や今後の方向性を明らかにした。家庭紙などのホーム＆パーソナルケア（Ｈ＆ＰＣ）事業推進や主力の三島工場（四国中央市）、岐阜県の新工場などを軸に構造改革を進めるとしている。Ｈ＆ＰＣ事業では、紙おむつなどの吸収体製品に使うフラッフ（綿状）パルプを北米からの輸入に頼ってきたが、予定を前倒しし２３年７月から三島工場で生産を始める予定。若林社長は「フラッフパルプの価格は今後も高止まりするとみている」とし、一部を自社で製造しコストを下げる。三菱自動車の子会社・パジェロ製造（岐阜県坂祝町）から２３年１月に購入する新工場予定地は４月に物流拠点として稼働させ、２４年１０月からはティッシュペーパーやトイレットペーパーを生産する。近くの可児工場（同県可児市）にも家庭紙用抄紙機を増設し、両工場で月産計３千トンを見込む。一連の設備投資額は約１７０億円。需要の多い首都圏への配送を強化する。海外関連では事業の複合化を進める。中国市場でフェミニンケア商品の現地生産 を始めており、衛生用紙のラインアップを拡充。ベビー用紙おむつ以外の売り上げ構成を２１年の１０％から２２年は２５％に増やす。コスト構造変化への対応と循環型社会への取り組みとして、板紙への難処理古紙配合率を３０年度に３０％とする目標を掲げた。使用済み紙おむつのリサイクル技術の確立にも取り組む。第４次計画最終の２３年度連結決算で売上高７２００億円、営業利益５１０億円を目標とする一方、原燃料価格高騰が収益を圧迫し、連結営業利益は２１年度の３７６億円から２２年度は２５０億円に減益となる見通しを示した。（菅亮輔）
	人間が作成した要約文章:大王製紙は26日、２０２１～２３年度の第４次中期事業計画の進展や今後の方向性として、家庭紙などのホーム＆パーソナルケア（Ｈ＆ＰＣ）事業推進や主力の三島工場（四国中央市）、岐阜県の新工場などを軸に構造改革を進めると明らかにした。Ｈ＆ＰＣ事業では、紙おむつなどの吸収体製品に使うフラッフ（綿状）パルプを２３年７月から三島工場で生産開始予定である。岐阜県の新工場は、２３年４月に物流拠点として稼働させ、２４年１０月からはティッシュペーパーやトイレットペーパーを生産する。海外関連では事業の複合化を進め、衛生用紙のラインアップを拡充する。コスト構造変化への対応と循環型社会への取り組みとして、板紙への難処理古紙配合率を３０年度に３０％とする目標を掲げた。第４次計画最終の２３年度連結決算で売上高７２００億円、営業利益５１０億円を目標とする一方、原燃料価格高騰が収益を圧迫し、連結営業利益は２１年度の３７６億円から２２年度は２５０億円に減益となる見通しを示した。


    以上での例の人間のように入力された文章を要約して下さい。
    #入力する文章
    {text:}
    #出力形式
    要約した文章:
"""

    prompt_refine = """
    見出し: 女性美　表現の変遷　新居浜市美術館「描かれた女たち」　明治から現代　日本の洋画８１点　来月２６日まで
    本文: 　西洋美術との出合いは、日本人画家が描く人体像にどんな変化をもたらしたのか―。女性をモチーフにした明治時代から現代の洋画を通じその変遷に迫る特別展「描かれた女たち―女性像にみるフォルム／現実／夢」が、新居浜市坂井町２丁目の市美術館で開かれている。日動美術財団所蔵の７５点を中心に計８１点が並ぶ。６月２６日まで。 　明治期に西洋美術に接した日本画壇は、対象の科学的な捉え方や陰影法などの技法だけでなく、絵画とは何かといった概念も吸収した。特別展では東郷青児、竹久夢二、百武兼行、絹谷幸二、岸田劉生、藤島武二らが描いた「女性美」から、表現の多様性を浮かび上がらせている。展示作品の一部を紹介する。（所蔵は全て日動美術財団）
	人間が作成した要約文章:女性をモチーフにした明治時代から現代の洋画を通じその変遷に迫る特別展「描かれた女たち―女性像にみるフォルム／現実／夢」が、新居浜市坂井町２丁目の市美術館で開かれている。日動美術財団所蔵の７５点を中心に計８１点が並び、展示期間は6月26日までである。

    見出し: 大王製紙　おむつ原料　自社生産　戦略説明会　輸入から切り替え
    本文: 　大王製紙（四国中央市、若林賴房社長）は２６日、オンラインで戦略説明会を開き、２０２１～２３年度の第４次中期事業計画の進展や今後の方向性を明らかにした。家庭紙などのホーム＆パーソナルケア（Ｈ＆ＰＣ）事業推進や主力の三島工場（四国中央市）、岐阜県の新工場などを軸に構造改革を進めるとしている。Ｈ＆ＰＣ事業では、紙おむつなどの吸収体製品に使うフラッフ（綿状）パルプを北米からの輸入に頼ってきたが、予定を前倒しし２３年７月から三島工場で生産を始める予定。若林社長は「フラッフパルプの価格は今後も高止まりするとみている」とし、一部を自社で製造しコストを下げる。三菱自動車の子会社・パジェロ製造（岐阜県坂祝町）から２３年１月に購入する新工場予定地は４月に物流拠点として稼働させ、２４年１０月からはティッシュペーパーやトイレットペーパーを生産する。近くの可児工場（同県可児市）にも家庭紙用抄紙機を増設し、両工場で月産計３千トンを見込む。一連の設備投資額は約１７０億円。需要の多い首都圏への配送を強化する。海外関連では事業の複合化を進める。中国市場でフェミニンケア商品の現地生産 を始めており、衛生用紙のラインアップを拡充。ベビー用紙おむつ以外の売り上げ構成を２１年の１０％から２２年は２５％に増やす。コスト構造変化への対応と循環型社会への取り組みとして、板紙への難処理古紙配合率を３０年度に３０％とする目標を掲げた。使用済み紙おむつのリサイクル技術の確立にも取り組む。第４次計画最終の２３年度連結決算で売上高７２００億円、営業利益５１０億円を目標とする一方、原燃料価格高騰が収益を圧迫し、連結営業利益は２１年度の３７６億円から２２年度は２５０億円に減益となる見通しを示した。（菅亮輔）
	人間が作成した要約文章:大王製紙は26日、２０２１～２３年度の第４次中期事業計画の進展や今後の方向性として、家庭紙などのホーム＆パーソナルケア（Ｈ＆ＰＣ）事業推進や主力の三島工場（四国中央市）、岐阜県の新工場などを軸に構造改革を進めると明らかにした。Ｈ＆ＰＣ事業では、紙おむつなどの吸収体製品に使うフラッフ（綿状）パルプを２３年７月から三島工場で生産開始予定である。岐阜県の新工場は、２３年４月に物流拠点として稼働させ、２４年１０月からはティッシュペーパーやトイレットペーパーを生産する。海外関連では事業の複合化を進め、衛生用紙のラインアップを拡充する。コスト構造変化への対応と循環型社会への取り組みとして、板紙への難処理古紙配合率を３０年度に３０％とする目標を掲げた。第４次計画最終の２３年度連結決算で売上高７２００億円、営業利益５１０億円を目標とする一方、原燃料価格高騰が収益を圧迫し、連結営業利益は２１年度の３７６億円から２２年度は２５０億円に減益となる見通しを示した。


    以上での例の人間のように入力された文章を要約して下さい。
    #入力する文章
    {existing_answer}
    {text:}
    #出力形式
    要約した文章:
    """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    REFINE_PROMPT = PromptTemplate(template=prompt_refine, input_variables=["existing_answer","text"])

    with get_openai_callback() as cb:
        chain = load_summarize_chain( 
            llm,
            chain_type="refine",
            verbose=True,
            question_prompt=PROMPT,
            refine_prompt = REFINE_PROMPT
        )

        # Create a Document with page_content set to content
        document = docs
        response = chain({"input_documents": docs}, return_only_outputs=True)
        
    return response['output_text'], cb.total_cost

def main():
    init_page()
    llm = select_model()
    init_messages()

    container = st.container()
    response_container = st.container()

    with container:
        text_input = get_text_input()
        run_button = st.button("実行")
    start_time = None  # 新しい行を追加
    if text_input and run_button:
        document = [Document(page_content=t) for t in text_input]

        with st.spinner("ChatGPT is typing ..."):
            start_time = datetime.now()  # 新しい行を追加
            output_text, cost = summarize(llm, document)
        st.session_state.costs.append(cost)
    else:
        output_text = None

    if output_text:
        with response_container:
            st.markdown("## 要約された文章")
            st.write(text_input)
            st.markdown("## 要約された文章")
            st.write(output_text)
    end_time = datetime.now()  # 新しい行を追加
    execution_time = (end_time - start_time).total_seconds() if start_time else None  # 新しい行を追加

    if execution_time:
        st.sidebar.markdown(f"**実行時間: {execution_time:.2f}秒**")
    costs = st.session_state.get('costs', [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")

if __name__ == '__main__':
    main()

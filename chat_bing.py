import os
from uuid import uuid4

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    ComplexField,
    CorsOptions,
    SearchIndex,
    ScoringProfile,
    SearchFieldDataType,
    SimpleField,
    SearchableField
)
from dotenv import load_dotenv
import streamlit as st

from langchain.agents import AgentType, initialize_agent, Tool
from langchain.chat_models import AzureChatOpenAI
from langchain.globals import get_llm_cache, set_llm_cache
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import AzureCognitiveSearchRetriever
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import BingSearchAPIWrapper
from streamlit_chat import message


# 環境変数の読み込み
load_dotenv()

unique_id = uuid4().hex[0:8]
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_PROJECT"] = f"Tracing Walkthrough - {unique_id}"
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
OPENAI_API_TYPE = os.getenv("OPENAI_API_TYPE")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
model: str = os.getenv("ADA_DEPLOY_NAME")
service_name = os.getenv("COGNITIVE_SEARCH_NAME")
admin_key = os.getenv("COGNITIVE_SEARCH_ADMIN_KEY")
index_name = os.getenv("INDEX_NAME")

# 以下ベクトル検索の際に使用？（現在コメントアウト）
# Azure OpenAIのエンドポイントとAPIキー
#AZURE_OPENAI_ENDPOINT = os.getenv("OPENAI_API_BASE")
#AZURE_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Embeddingモデルのデプロイ名
#AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME")
# Azure SearchのエンドポイントとAPIキー
#AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
#AZURE_SEARCH_SERVICE_NAME = os.getenv("AZURE_SEARCH_SERVICE_NAME")
#AZURE_SEARCH_API_KEY_ADMIN = os.getenv("AZURE_SEARCH_API_KEY_ADMIN")

# Create an SDK client
endpoint = "https://{}.search.windows.net/".format(service_name)
admin_client = SearchIndexClient(endpoint=endpoint,
                      index_name=index_name,
                      credential=AzureKeyCredential(admin_key))

search_client = SearchClient(endpoint=endpoint,
                      index_name=index_name,
                      credential=AzureKeyCredential(admin_key))

# Cognitive Search検索
def cognitive_search(search_text: str) -> str:
    results = search_client.search(search_text=search_text,select='content',top=3)
    output = ""
    for result in results:
        output += str(result) + "\n"
    return output

# オプション
def select_model():
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
    if model == "GPT-3.5":
        model_name = os.getenv("OPENAI_API_MODEL_DEPROY")
    else:
        model_name = os.getenv("OPENAI_API_MODEL_DEPROY_4")
        
    temperature = st.sidebar.slider("Temperature:", min_value=0.0, max_value=1.5, value=0.7, step=0.1)    

    return AzureChatOpenAI(
        openai_api_base = os.getenv("OPENAI_API_BASE"),
        openai_api_version = os.getenv("OPENAI_API_VERSION"),
        deployment_name = model_name,
        openai_api_key = os.getenv("OPENAI_API_KEY"),
        openai_api_type = os.getenv("OPENAI_API_TYPE"),
        temperature=temperature,
    )

# クリアボタン
def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button:
        chat_history = []
        if "memory" in st.session_state:
            del st.session_state["memory"]

# 特殊文字のエスケープ
def escape_markdown(text):
    # マークダウンでの特殊文字
    special_chars = ['\\', '`', '*', '_', '{', '}', '[', ']', '(', ')', '#', '+', '-', '.', '!']
    for char in special_chars:
        text = text.replace(char, '\\' + char)
    return text

# アプリ全体        
def main():  
    # クリアボタンの追加
    init_messages()
     
    # ChatGPT-3.5のモデルのインスタンスの作成
    chat = select_model()

    # bingエンジンを使えるようにtoolsを定義（有料のため現在コメントアウト）
    #BING_SUBSCRIPTION_KEY = os.getenv("BING_SUBSCRIPTION_KEY")
    #BING_SEARCH_URL = os.getenv("BING_SEARCH_URL")
    #BING_CUSTOM_SEARCH_ENDPOINT = os.getenv("BING_CUSTOM_SEARCH_ENDPOINT")
    #BING_CUSTOM_CONFIG = os.getenv("BING_CUSTOM_CONFIG")

    #search = BingSearchAPIWrapper()

    search = DuckDuckGoSearchRun()

    tools = [
        Tool(
            name = "CognitiveSearch",
            func = cognitive_search,
            description= "Informatica、インフォマティカに関する情報を取得するときはこのツールを使用してより適切な情報を探します。Azureに格納されたデータの中から情報を検索します",
        ),         
        Tool(
            name = "WebSearch",
            func = search.run,
            description= "Informatica、インフォマティカに関する情報を取得するときはこのツールを使用してより適切な情報を探します。ウェブで最新の情報を検索します",
        ) 
    ]   

    # セッション内に保存されたチャット履歴のメモリの取得
    try:
        memory = st.session_state["memory"]
    except:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )

    system_message = """
    You are an Informatica expert and engineer.
    
    If the value of "Observation" is "Invalid or incomplete response" more than 4 times, output "回答を生成できませんでした".
    
    Answer the following questions as best you can, but speaking Japanese.
    """

    # チャット用のチェーンのインスタンスの作成
    agent = initialize_agent(
        tools=tools,
        llm=chat,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        agent_kwargs={"system_message": system_message},
        handle_parsing_errors=True,    
    )

    # Streamlitによって、タイトル部分のUIをの作成
    st.title("Chatbot with OpenAI")
    st.caption("testのチャットです")
        
    # チャット履歴（HumanMessageやAIMessageなど）を格納する配列の初期化
    chat_history = []
    
    # 入力フォームと送信ボタンのUIの作成
    if text_input := st.chat_input("Enter your message"):

        # ChatGPTの実行
        agent.run(text_input)

        # セッションへのチャット履歴の保存
        st.session_state["memory"] = memory

        # チャット履歴（HumanMessageやAIMessageなど）の読み込み
        try:
            chat_history = memory.load_memory_variables({})["chat_history"]
        except Exception as e:
            st.error(e)

    # チャット履歴の表示
    for chat_message in chat_history:
        if type(chat_message) == AIMessage:
            with st.chat_message('assistant'):
                st.markdown(escape_markdown(chat_message.content))
        elif type(chat_message) == HumanMessage:
            with st.chat_message('user'):
                st.markdown(escape_markdown(chat_message.content))
                 
if __name__ == '__main__':
    main()                                     
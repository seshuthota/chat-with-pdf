import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit_chat import message

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = LS_API_KEY = os.getenv("LS_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "streamlit-chatbot"


def initialize_state():
    if "history" not in st.session_state:
        st.session_state.history = []

    if "generated" not in st.session_state:
        st.session_state.generated = ["Hello! Ask me a question!"]

    if "past" not in st.session_state:
        st.session_state.past = ["Hey ! "]


def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]


def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key="query_form", clear_on_submit=True):
            query_input = st.text_input("Question: ", placeholder="Ask anything about the pdfs")
            submit_button = st.form_submit_button("Submit")

        if submit_button and query_input:
            with st.spinner("Thinking..."):
                output = conversation_chat(query_input, chain, st.session_state.history)

            st.session_state.past.append(query_input)
            st.session_state.generated.append(output)
    if st.session_state.generated:
        with reply_container:
            for i in range(len(st.session_state.generated)):
                message(st.session_state.past[i], is_user=True, key=str(i) + "_user", avatar_style="thumbs")
                message(st.session_state.generated[i], key=str(i) + "_bot", avatar_style="fun-emoji")


def create_conversational_chain(vector_store, api_key=None):
    llm = ChatOpenAI(model="gpt-3.5-turbo-1106", streaming=True, verbose=True, openai_api_key=api_key)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                  retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
                                                  memory=memory)
    return chain


def main():
    initialize_state()

    st.title("Multi-Document Question Answering")

    st.sidebar.title("Multi-PDF chatbot with OpenAI :books:")

    st.sidebar.title("Document processing")
    api_key = st.sidebar.text_input("OPENAI API Key")
    uploaded_files = st.sidebar.file_uploader("Upload PDFs", accept_multiple_files=True)

    if uploaded_files:
        text = []
        for file in uploaded_files:
            file_extension = file.name.split(".")[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name
            loader = None
            if file_extension == "pdf":
                loader = PyPDFLoader(temp_file_path)
            if loader:
                text.extend(loader.load())
                os.remove(temp_file_path)

        test_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

        text_chunks = test_splitter.split_documents(text)

        print("Number of chunks", len(text_chunks))

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
                                           model_kwargs={"device": "cpu"}, show_progress=True)

        vector_store = FAISS.from_documents(text_chunks, embeddings)

        chain = create_conversational_chain(vector_store, api_key=api_key)

        display_chat_history(chain)


if __name__ == "__main__":
    main()

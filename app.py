from llama_index.llms.ollama import Ollama
from llama_index.core.chat_engine.types import ChatMode
import chromadb
import streamlit as st
from pathlib import Path
from rag_functions import (
    Create_Vector,
    init_llm_ollama,
    init_llm_hf,
    init_index,
)
import shutil
import os


def chatbot(index) -> None:
    if "chat_engine" not in st.session_state.keys():
            st.session_state.chat_engine = index.as_chat_engine(
                chat_mode="condense_question", streaming=True, verbose=True
                )
    if "messages" not in st.session_state.keys():
            st.session_state.messages = [
                {
                "role": "assistant",
                "content": "Ask me anything",
                }
            ]
    if prompt := st.chat_input("Ask me a question"):
            st.session_state.messages.append({"role": "user", "content": prompt})
    for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.empty() #this is unneeded however this is added to fix the ghosting text (double text)
                st.write(message["content"])

    if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.chat_engine.chat(message=prompt)
                    st.write(response.response)
                    message = {"role": "assistant", "content": response.response}
                    st.session_state.messages.append(message)
    
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="./images/llama.png",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items=None,
)
    
def Clear_Chat() -> None:
    keys = list(st.session_state.keys())
    for key in keys:
        st.session_state.pop(key)


def main() -> None:
    menu = ["Document Ingestion", "Chatbot"]
    # st.sidebar.image("./images/llama.png", use_column_width=True)
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Chatbot":
        st.title("RAG Chat Bot")
        llmtype = st.radio(
            "Choose your LLM",
            [
                "Ollama llama3 (local)",
                "Huggingface Mistral",
            ],
            horizontal=True,
        )
        if llmtype == "Ollama llama3 (local)":
            st.sidebar.write(
                "*Ensure that you're running Ollama locally and also have downloaded the llama3 model*"
            )
            init_llm_ollama()
            try:
                index = init_index()
                chatbot(index)
            except:
                st.info("Please upload documents first using Document Ingestion")
                st.stop()
        elif llmtype == "Huggingface Mistral":
            HF_Token = st.sidebar.text_input("Huggingface Token", type="password")
            st.sidebar.write(
                "*Sigup for a free account at https://huggingface.co/. Create a free API key and input above*"
            )
            init_llm_hf("mistralai/Mistral-7B-Instruct-v0.3", HF_Token)
            try:
                index = init_index()
                chatbot(index)
            except:
                st.info("Please upload documents first using Document Ingestion")
                st.stop()
        

    elif choice == "Document Ingestion":
        st.subheader("Document Ingestion")
        with st.form("Cleanup"):
            clean = st.form_submit_button("Reinitialize")
            st.write("This helps remove all previous data & chat history")
            if clean:
                try:
                    shutil.rmtree("./data")
                    os.makedirs("data")
                    Clear_Chat()
                    st.success("Successfully Reinitialized")
                except:
                    #os.makedirs("data")
                    st.error("Nothing to initialize")
        with st.form(key="FileUpload", clear_on_submit=True):
            uploaded_file = st.file_uploader(
                "Drop your files here", type=["txt", "csv", "pdf", "py"]
            )
            Submit = st.form_submit_button(label="Save")
            if Submit:
                st.markdown("**The file is sucessfully Saved.**")
                # Save uploaded file to './data' folder.
                save_folder = "./data"
                save_path = Path(save_folder, uploaded_file.name)
                with open(save_path, mode="wb") as f:
                    f.write(uploaded_file.getvalue())
        if uploaded_file is not None:
            select_rag = st.radio(
                "Select RAG Type",
                ["Vector Database", "Knowledge Graph"],
                horizontal=True,
            )
            if select_rag == "Knowledge Graph":
                st.info("Not Implemented Yet")
            vectorized = st.button("Embed & Vectorize")
            if vectorized:
                Clear_Chat()
                if select_rag == "Vector Database":
                    with st.spinner("Please wait while I vectorize..."):
                        Create_Vector()
                        st.success("Done...You can launch the Chatbot now")
                elif select_rag == "Graph Database":
                    st.info ("Not Implemented Yet")


if __name__ == "__main__":
    main()

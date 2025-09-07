"""Streamlit ui for demostrations
"""

import streamlit as st
from kb_articles_rag_prd.common.config import KnowledgeBaseGenConfig, KnowledgeBaseIndexerConfig
from kb_articles_rag_prd.provider.generator import KBArticlesTextGenerator
from kb_articles_rag_prd.provider.indexer import KBArticleTextIndexer

st.title("Knowledge Base RAG")

with st.sidebar:
    tab1, tab2 = st.tabs(["Generation", "Indexing"])
    with tab1:
        st.header("Generation config")
        vector_store_engine = st.radio(
            "What is your vector database",
            ["chroma", "faiss"],
        )
        vector_store_persist_path = st.text_input("vector store persist directory", "vectordb/" )
        vector_store_collection_name = st.text_input("vector store collection", "kbarticles" )
        embedding_engine = st.radio(
            "What is your embedding engine",
            ["vertex","openai", "huggingface"],
        )
        embedding_engine_model = st.text_input("Embedding model", "text-embedding-005" )
        model_provier = st.radio(
            "What is your model provider",
            ["google_vertexai", "openai"],
        )
        model = st.radio(
            "What is your model",
            ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"],
        )
    with tab2:
        if st.button("Generate indexes"):
            with st.spinner("Assistant is Generating index 🤔"):
                config = KnowledgeBaseIndexerConfig()
                indexer = KBArticleTextIndexer(
                    config=config
                )
                result = indexer.index()
                st.write("Indexing is complete, go ahead and generate")


        




st.header("Knowlede Base chat")
if "kbconfig" not in st.session_state:
    st.session_state.kbconfig = KnowledgeBaseGenConfig(
        vector_store_engine=vector_store_engine,
        vector_store_persist_path=vector_store_persist_path,
        embedding_engine=embedding_engine,
        embedding_engine_model=embedding_engine_model,
        model_provier=model_provier,
        model=model
    )
if "messages" not in st.session_state:
    st.session_state.messages = []

if "generator" not in st.session_state:
    st.session_state.generator = KBArticlesTextGenerator(st.session_state.kbconfig)

def gen_ai_response(question: str) -> str:
    """simulates ai response

    Args:
        question (str): _description_

    Returns:
        str: response
    """
    reply = st.session_state.generator.respond(question)
    return reply.content

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input(" What are you working on ? "):
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_prompt
        }
    )
    with st.chat_message("user"):
        st.markdown(user_prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Assistant is thinking 🤔"):
            response = gen_ai_response(user_prompt)
            st.markdown(response)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response
                }
            )

    

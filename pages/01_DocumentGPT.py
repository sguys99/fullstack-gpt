from typing import Any, Dict, List
from uuid import UUID
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.output import ChatGenerationChunk, GenerationChunk
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st


st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

# if "messages" not in st.session_state:
#     st.session_state["messages"] = []


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""
    
    def on_llm_start(self, *args, **kwargs):
        # with st.sidebar:
        #     st.write("llm started!")
        self.message_box = st.empty()
        
    def on_llm_end(self, *args, **kwargs):
        # with st.sidebar:
        #     st.write("llm ended!")
        save_message(self.message, "ai")
    
    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)
        

llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler()
    ]
)


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    
    with open(file_path, "wb") as f:
        f.write(file_content)
        
    cache_dir = LocalFileStore("./.cache/embeddings/{file.name}")
    
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader("./files/chapter_one.txt")
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    docs = retriever.invoke("ministry of truth")
    return retriever

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def pain_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """
     Answer the question using ONLY the following context. If you don't know the answer
     just say you don't know. DON'T make anything up.
     
     Context: {context}
     """
     ),
    ("human", "{question}")
])


st.title("Document GPT")
st.markdown("""
Welcome!

Use this chatbot to ask questions to an AI about your files!

Upload your files on the sidebar.
""")

with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file", 
        type=["pdf", "txt", "docx"]
        )

if file:
    retriever = embed_file(file)
    
    send_message("I'm ready! Ask away!", "ai", save=False)
    pain_history()
    message = st.chat_input("Ask anything about your file....")
    
    if message:
        send_message(message, "human")
        # docs = retriever.invoke(message)
        # docs = "\n\n".join(document.page_content for document in docs)
        # prompt = prompt.format_messages(context=docs, question=message)
        
        chain = {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        } | prompt | llm
        
        with st.chat_message("ai"):
            chain.invoke(message)
        
else:
    st.session_state["messages"] = []














# if "messages" not in st.session_state:
#     st.session_state["messages"] = []
    

# def send_message(message, role, save=True):
#     with st.chat_message(role):
#         st.write(message)
#     if save:
#         st.session_state["messages"].append({"message":message, "role":role})


# for message in st.session_state["messages"]:
#     send_message(message["message"], message["role"], save=False)


# message = st.chat_input("Send a message to the ai")

# if message:
#     send_message(message, "human")
#     time.sleep(2)
#     send_message(f"You said: {message}", "ai")



# with st.chat_message("human", avatar="lfc-icon.png"):
#     st.write("Hello..")

# with st.chat_message("ai", avatar="man-icon.png"):
#     st.write("How are you")
    


# if "messages" not in st.session_state:
#     st.session_state["messages"] = []
    
    
# with st.chat_message("human"):
#     st.write("Hello..")
    
# with st.chat_message("ai"):
#     st.write("How are you")
    

# with st.status("Embedding file...", expanded=True) as status:
#     time.sleep(2)
#     st.write("Getting the file")
#     time.sleep(2)
#     st.write("Embedding the file")
#     time.sleep(2)
#     st.write("Caching the file")
    
#     status.update(label="Error", state="error")
            
# st.chat_input("Send a message to the AI")
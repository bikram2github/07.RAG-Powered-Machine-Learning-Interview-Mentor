import streamlit as st
import os
import tempfile


from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage



st.title("RAG-Powered Machine Learning Interview Mentor")
st.write("Your personal AI mentor for mastering Machine Learning — upload your study materials and start asking interview-style questions instantly!")



with st.sidebar:
    st.title("Configuration")
    api_key = st.text_input("Enter your Groq API Key", type="password")
    if st.sidebar.button("🔄 Reset Session"):
        st.cache_resource.clear() 
        st.session_state.clear()   
        st.rerun() 

file = st.file_uploader("Upload a PDF book on Machine Learning", type=["pdf"])




@st.cache_resource(show_spinner=False)
def get_retriever(file,original_name):
    loader = PyPDFLoader(file)
    docs= loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    INDEX_DIR = "faiss_index"
    os.makedirs(INDEX_DIR, exist_ok=True)
    base_name = os.path.splitext(original_name)[0]
    base_name = base_name.replace(" ", "_").replace(".", "_").replace("/", "_")
    INDEX_PATH = os.path.join(INDEX_DIR, f"{base_name}_index")

    if os.path.exists(INDEX_PATH):
        db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        db = FAISS.from_documents(texts, embeddings)
        db.save_local(INDEX_PATH)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    return retriever



@st.cache_resource(show_spinner=False)
def model(api_key):
    return ChatGroq(api_key=api_key, model="openai/gpt-oss-20b", temperature=0)




def chain(retriever,llm):

    system_prompt = (
    "You are a helpful and precise AI assistant. "
    "Answer the user's question strictly based on the provided context. "
    "If the context does not contain enough information to answer, respond with: 'I don't know.' "
    "Do not use any external or prior knowledge. "
    "If the user makes grammar mistakes, fix them in your answer. "
    "Keep your answer clear, concise, and directly relevant to the context below.\n\n"
    "Context:\n{context}"
    )


    contextualize_q_system_prompt = (
    "You are helping to improve question retrieval in a study assistant system. "
    "Given the chat history and the user's latest question, "
    "rewrite the question so that it is clear, complete, and self-contained. "
    "If the question already makes sense on its own, keep it as is. "
    "Do not attempt to answer it — your only task is to reformulate it for better context understanding."
    )



    contextualize_q_prompt=ChatPromptTemplate.from_messages(
        [("system",contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human","{input}")],
    )



    history_aware_retriever=create_history_aware_retriever(llm,retriever, contextualize_q_prompt)


    qa_prompt=ChatPromptTemplate.from_messages(
        [("system",system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human","{input}")],
    )
    
    question_answer=create_stuff_documents_chain(llm, qa_prompt)

    rag_chain=create_retrieval_chain(history_aware_retriever,question_answer)
    return rag_chain



if "chat_history" not in st.session_state:
    st.session_state.chat_history = []



def trim_messages(chat_history, max_messages=6):
    """Keep only the most recent N messages in chat history."""
    if len(chat_history) > max_messages:
        chat_history = chat_history[-max_messages:]
    return chat_history


def ask(question, rag_chain):
    st.session_state.chat_history = trim_messages(st.session_state.chat_history, max_messages=6)

    response = rag_chain.invoke(
        {"input": question, "chat_history": st.session_state.chat_history}
    )


    st.session_state.chat_history.extend([
        HumanMessage(content=question),
        AIMessage(content=response["answer"])
    ])
    answer=response["answer"]
    return answer



if file is None:
    st.info("📄 Please upload a PDF file to proceed.")
    text=None
else:
    st.success(f"✅ Uploaded: {file.name}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file.read())
        tmp_path = tmp_file.name  # store the path for PyPDFLoader
    if "retriever" not in st.session_state:
        with st.spinner("🔍 Building knowledge base (this may take a moment)..."):
            st.session_state.retriever = get_retriever(tmp_path, file.name)
            st.session_state.llm = model(api_key)
            st.session_state.rag_chain = chain(st.session_state.retriever, st.session_state.llm)
            st.success("✅ Knowledge base ready!")    
    text = st.text_input("Ask a question about Machine Learning", label_visibility="collapsed")
    if st.button("Get Answer"):
        if not api_key.strip() or text is None or not text.strip():
            st.error("Please enter your Groq API Key and ask a question about Machine Learning.")
        else:
            with st.spinner("Processing your question..."):
                answer = ask(text, st.session_state.rag_chain)
                st.success(f"✅ Here's your answer:\n\n {answer}")
                with st.expander("Chat History"):
                    for i in range(0, len(st.session_state.chat_history), 2):
                        user_msg = st.session_state.chat_history[i]
                        ai_msg = st.session_state.chat_history[i + 1]
                        st.markdown(f"**You:** {user_msg.content}")
                        st.markdown(f"**AI:** {ai_msg.content}")
    
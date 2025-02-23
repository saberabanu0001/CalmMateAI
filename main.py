import os
from dotenv import load_dotenv
import gradio as gr

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Load API keys securely
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not GROQ_API_KEY or not HUGGINGFACEHUB_API_TOKEN:
    raise ValueError("❌ Missing API keys! Make sure to set them in a .env file.")

def initialize_llm():
    """Initialize the LLaMA model using Groq API."""
    return ChatGroq(
        temperature=0,
        model_name='llama-3.3-70b-versatile',
        groq_api_key=GROQ_API_KEY
    )

def create_vector_db():
    """Load PDFs, process text, and create a Chroma vector database."""
    loader = DirectoryLoader('./sample_data', glob='*.pdf', loader_cls=PyPDFLoader)
    # Process texts and create embeddings here

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceBgeEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma.from_documents(texts, embeddings, persist_directory="chroma_db")
    db.persist()
    print("✅ ChromaDB created successfully!")
    return db

def setup_qa_chain(db, llm):
    """Set up the RetrievalQA chain."""
    retriever = db.as_retriever()

    prompt_template = """You are a mental health expert. Use the following information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    {context}
    
    user={question}
    chatbot:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# Initialize components
print("🔄 Initializing CalmMateAI...")
llm = initialize_llm()
db_path = "chroma_db"

if os.path.exists(db_path) and os.listdir(db_path):
    print("🟢 Loading existing ChromaDB...")
    embedding = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=db_path, embedding_function=embedding)
else:
    print("🚀 Creating a new ChromaDB...")
    vector_db = create_vector_db()

qa_chain = setup_qa_chain(vector_db, llm)

# Gradio Chat Function
def chat_response(user_input, history=[]):
    if not user_input.strip():
        return history + [("You", user_input), ("CalmMateAI", "Please enter a valid question. 🙏")], ""

    try:
        response = qa_chain.invoke({"query": user_input})["result"]
    except Exception as e:
        response = f"⚠️ Error: {str(e)}"

    history.append(("You", user_input))
    history.append(("CalmMateAI", response))
    
    return history, ""  # Clears input field after sending

# Gradio UI
theme = gr.themes.Soft(primary_hue="blue", secondary_hue="gray")

with gr.Blocks(theme=theme) as app:
    gr.Markdown("# 🌿 CalmMateAI: Your Mental Health Companion")
    gr.Markdown("**A safe space for mental well-being. Ask anything, and I'll help!**")

    with gr.Row():
        chatbot = gr.Chatbot(label="CalmMateAI Chat", height=400)
        user_input = gr.Textbox(placeholder="Type your question here...", show_label=False)

    send_button = gr.Button("Send", variant="primary")

    send_button.click(chat_response, inputs=[user_input, chatbot], outputs=[chatbot, user_input])
    user_input.submit(chat_response, inputs=[user_input, chatbot], outputs=[chatbot, user_input])

app.launch(debug=True)

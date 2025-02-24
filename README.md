🌿 CalmMateAI: Your Mental Health Companion
🤝 A Safe Space for Mental Well-Being
CalmMateAI is an AI-powered chatbot designed to provide compassionate and informative responses to mental health-related questions. Built with LangChain, Gradio, ChromaDB, and Groq's LLaMA model, this chatbot serves as a supportive tool for users seeking insights on mental well-being.

🚀 Features

✅ Conversational AI – Interact with CalmMateAI to get thoughtful responses on mental health topics.

✅ Document-Based Retrieval – Upload PDFs to create a vector database for personalized Q&A.

✅ Groq's LLaMA Model – Leverages the power of LLaMA 3.3-70B Versatile for intelligent responses.

✅ Embeddings with Hugging Face – Uses sentence-transformers/all-MiniLM-L6-v2 for efficient text processing.

✅ Gradio UI – A simple and user-friendly chatbot interface for easy communication.

🛠️ Tech Stack
Python 🐍
LangChain – For LLM integration and document retrieval
Gradio – For building the interactive chatbot UI
ChromaDB – For storing and retrieving vector embeddings
Groq API (LLaMA 3.3-70B) – For natural language understanding
Hugging Face Embeddings – For transforming text into vector representations


🔧 Installation & Setup

1️⃣ Clone the Repository
git clone https://github.com/saberabanu0001/CalmMateAI.git
cd CalmMateAI

2️⃣ Create and Activate a Virtual Environment
# For Windows (PowerShell)
python -m venv venv
venv\Scripts\activate

# For Mac/Linux
python3 -m venv venv
source venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Set Up API Keys
Create a .env file and add your Groq API Key and Hugging Face API Token:
GROQ_API_KEY=your_groq_api_key_here
HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_key_here

5️⃣ Run the Application
python main.py
The chatbot will launch, and you can start interacting with CalmMateAI. 🎉

🎨 UI Preview
🖥️ CalmMateAI features a minimalistic Gradio interface with a chatbot window and input box.

🏗️ Project Structure
CalmMateAI/
│── sample_data/           # Folder for storing uploaded PDFs  
│── chroma_db/             # Chroma database storage  
│── venv/                  # Virtual environment  
│── main.py                # Main application script  
│── requirements.txt        # Python dependencies  
│── .gitignore              # Ignoring unnecessary files  
│── .env                    # Stores API keys (DO NOT SHARE)  
│── README.md               # Project documentation  
🌟 Contributing
Want to improve CalmMateAI? Contributions are welcome! Open an issue or submit a pull request.

💬 Contact
🔹 Sabera Banu (Developer)
🔹 GitHub: saberabanu0001
🔹 Email: saberabanu677@example.com


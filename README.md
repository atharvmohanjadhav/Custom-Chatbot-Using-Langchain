# 🤖 Custom Chatbot using LangChain

This project is a **custom chatbot application** built with [LangChain](https://www.langchain.com/), [Streamlit](https://streamlit.io/), and [Groq's LLaMA 3.3](https://groq.com/) model. It allows users to enter any valid web URL, extract the content from the webpage, and interact with it using a conversational chatbot interface.

---

## 🚀 Features

- 🌐 Load and process content from any valid URL
- ✂️ Smart text chunking using `RecursiveCharacterTextSplitter`
- 🧠 Semantic search with Pinecone Vector DB and HuggingFace Embeddings
- 💬 Conversational interface using Groq’s `llama-3.3-70b-versatile` model
- 📜 Maintains chat history per session
- 🧪 Built with modern tools like Streamlit, LangChain, HuggingFace, Groq, Pinecone

---

## 🛠️ Tech Stack

- **Frontend/UI**: Streamlit
- **LLM**: Groq (LLaMA 3.3-70b)
- **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`)
- **Vector Store**: Pinecone
- **Data Loader**: LangChain's `WebBaseLoader`

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/custom-chatbot-langchain.git
cd custom-chatbot-langchain

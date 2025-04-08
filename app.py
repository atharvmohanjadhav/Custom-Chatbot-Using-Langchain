from dotenv import load_dotenv
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
import os
import validators

# Load environment variables
load_dotenv()

# Streamlit Page Config
st.set_page_config(page_title="Your Custom Chatbot", layout="wide")

# Title and Description
st.title("🤖 Your Custom Chatbot")
st.sidebar.write("Enter a valid URL, and chat with the content extracted from the webpage!")

# Initialize Session State for Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function to validate and load data from URL
def load_data(url):
    if not validators.url(url):  # Validate URL
        return None, "Invalid URL. Please enter a valid URL."
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        return docs, None
    except Exception as e:
        return None, f"Failed to load data: {e}"

# Sidebar Input for URL
user_url = st.sidebar.text_input("Enter Web URL", placeholder="https://example.com")

if user_url:
    st.sidebar.write("You entered:", user_url)
    
    # Load webpage content
    docs, error = load_data(user_url)
    if error:
        st.sidebar.error(error)
    else:
        st.sidebar.success("Data loaded successfully!")

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        documents = text_splitter.split_documents(docs)

        # Initialize Embeddings
        from sentence_transformers import SentenceTransformer

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Pinecone Setup (Make sure API Key is set in the environment)
        import pinecone

        os.environ["PINECONE_API_KEY"] = "pcsk_748Zfb_HciVT3heHenUizcDLrZfDZX5pj9F51bzqmCfy1M7LFY7uvyxLxFbmfSF1WNTiot" # Store API key safely
        if not os.environ["PINECONE_API_KEY"]:
            st.error("Pinecone API Key is missing!")
        else:
            pc = pinecone.Pinecone(api_key=os.environ["PINECONE_API_KEY"])

            index_name = "langchain"
            index = pc.Index(index_name)

            from langchain.vectorstores import Pinecone

            vectorstore = Pinecone(index,embeddings.embed_query,index_name)

            # Function to Retrieve Similar Documents
            def retrieve(query, k=2):
                res = vectorstore.similarity_search(query=query, k=k)
                return res

            # Initialize LLM (Ensure API Key is Set)
            GROQ_API_KEY = os.getenv("GROQ_API_KEY")
            if not GROQ_API_KEY:
                st.error("Groq API Key is missing!")
            else:
                llm = ChatGroq(model_name="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
                
                from langchain.chains.question_answering import load_qa_chain
                chain = load_qa_chain(llm, chain_type="stuff")

                # Function to Generate Answers
                def get_ai_response(query):
                    # Search relevant data in Pinecone
                    search_results = vectorstore.similarity_search(query, k=3)

                    # Extract relevant context
                    context = "\n".join([doc.page_content for doc in search_results])

                    # Generate response using LLaMA
                    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
                    res = llm.invoke(prompt)
                    
                    return res

                # Chat Interface
                for msg in st.session_state.chat_history:
                    with st.chat_message(msg['role']):
                        st.markdown(msg['content'])

                # User Input for Question
                user_question = st.chat_input("Ask your question about the webpage:")
                
                if user_question:
                    st.chat_message("User").markdown(user_question)
                    st.session_state.chat_history.append({'role': 'user', 'content': user_question})

                    # Get Answer
                    answer = get_ai_response(user_question)
                    st.session_state.chat_history.append({'role': 'assistant', 'content': answer.content})

                    # Display Chatbot Response
                    with st.chat_message("Assistant"):
                        st.markdown(answer.content)

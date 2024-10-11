import streamlit as st
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from config_env import embedding_endpoint, llm_endpoint, AZURE_OPENAI_VERSION,api_key
from langchain.schema import Document  # Add this import
from langchain.globals import set_verbose, get_verbose

set_verbose(True)
st.set_page_config(page_title="Hypothetical RAG Chatbot", layout="wide")

# Initialize models and vector store
embedding_model = AzureOpenAIEmbeddings(azure_endpoint=embedding_endpoint, api_key=api_key)
vector_store = InMemoryVectorStore(embedding_model).load('docint_vector_store', embedding_model)

# vector_store_files = ['vector_store_v1', 'vector_store_v2','vector_store','vector_store1','vector_store2']  # Add your file names here

open_ai_llm = AzureChatOpenAI(
    openai_api_version=AZURE_OPENAI_VERSION,
    azure_endpoint=llm_endpoint,
    temperature=0,
    api_key=api_key
)

# Chat prompt template
response_prompt = ChatPromptTemplate.from_template("""
Instructions: You are a customer service assistant chatbot at Bank Mandiri known as Mita. Your main role is to assist customers by providing accurate information, answering questions, and resolving issues related to our products and services. 
Always use bahasa indonesia in response.
You are an assistant for answering specific questions related to Bank Mandiri.
If a question is outside the topic of Bank Mandiri and our products or services, kindly state that the question is not relevant. If a user simply wants to try the service or asks who you are, introduce yourself and ask how you can assist them. 
Reformat your answers to be more straigtforward
Main Guidelines:
1. Polite and Professional Tone: Always communicate in a friendly and professional manner.
2. Empathy and Understanding: Acknowledge customer concerns and express understanding.
3. Clarity and Accuracy: Provide clear, concise, and accurate information.
4. Focus on Problem Resolution: Strive to resolve issues efficiently and effectively.
5. Escalation Protocol: If a customer's issue cannot be resolved, inform them that their question will be escalated to a human representative.
                                                   
You are designed to learn from interactions, so continuously improve your responses based on customer feedback.
Context: {context}

Question: {question}
""")


def create_hypothetical_rag_chain(vectorstore):
    def retrieve_docs(query: str) -> str:
        docs = vectorstore.similarity_search(query, k=3)
        return summarize_content("\n\n".join(doc.page_content for doc in docs))

    retriever = RunnableLambda(retrieve_docs)

    chain = (
        {
            "question": itemgetter("original_question"),
            "context": lambda x: retriever.invoke(x["question"]),
        }
        | response_prompt
        | open_ai_llm
    )
    
    return chain

def summarize_content(content):
    summary_prompt = ChatPromptTemplate.from_template("""
    Instruction: "Combine and group all the information gathered from the documents and tell the information in a clear and concise manner.
    Show the most relevant content good details. Add little information that might user want"  
    {content}
    """)
    
    summary_chain = summary_prompt | open_ai_llm
    summary_response = summary_chain.invoke({"content": content})
    return summary_response.content

# Predefined responses
RESPONSES = {
    'id': {
        'greeting': "👋 Hai! Saya Mita, asisten layanan pelanggan Bank Mandiri. Untuk memberikan pelayanan terbaik silahkan bertanya atau beri saya instruksi",
        'short_greeting': "👋 Hai! Saya Mita, asisten layanan pelanggan Bank Mandiri.",
        'language_prompt': "Untuk memberikan pelayanan terbaik, apakah Anda lebih nyaman menggunakan Bahasa Indonesia atau English?",
        'irrelevant': "Maaf, pertanyaan tersebut tidak relevan dengan produk atau layanan Bank Mandiri. Silakan ajukan pertanyaan lain seputar layanan kami!",
        'error': "Maaf, terjadi kesalahan dalam memproses permintaan Anda. Silakan coba lagi.",
        'clarification': "Mohon maaf, saya kurang memahami maksud Anda. Bisakah Anda menjelaskan lebih detail?",
    },
    'en': {
        'greeting': "👋 Hi! I'm Mita, Bank Mandiri's customer service assistant. To provide the best service, please give me instruction or ask anything",
        'short_greeting': "👋 Hi! I'm Mita, Bank Mandiri's customer service assistant.",
        'language_prompt': "To provide the best service, would you prefer to communicate in Bahasa Indonesia or English?",
        'irrelevant': "I apologize, but that question isn't relevant to Bank Mandiri's products or services. Please feel free to ask about our services!",
        'error': "I apologize, but there was an error processing your request. Please try again.",
        'clarification': "I'm sorry, I didn't quite understand. Could you please elaborate?",
    }
}

def detect_language_preference(text):
    """Simple language detection based on common words"""
    # For simplicity, we're always returning 'id' here. In a real implementation,
    # you'd want to implement actual language detection logic.
    return 'id'

def get_response(key, lang='id'):
    """Get response in the specified language"""
    return RESPONSES[lang][key]

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "language_set" not in st.session_state:
    st.session_state.language_set = False
if "selected_language" not in st.session_state:
    st.session_state.selected_language = None

# Initialize the RAG chain
rag_chain = create_hypothetical_rag_chain(vector_store)

# Streamlit UI
st.title("Your Truly Livin Partner")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])



# User input handling
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Detect language if not already set
                if not st.session_state.language_set:
                    detected_lang = detect_language_preference(prompt)
                    st.session_state.selected_language = detected_lang
                    st.session_state.language_set = True
                
                lang = st.session_state.selected_language
                
                # Generate response using RAG
                response = rag_chain.invoke({
                    "question": prompt,
                    "original_question": prompt
                })

               
                # Keyword + semantic similarity search
                similarity_threshold = 0.65  
                alpha = 0.7 
                docs = vector_store.similarity_search(prompt, k=7, similarity_threshold = similarity_threshold, alpha = alpha)
                
                summary = summarize_content(docs)
                full_response = f"{response.content}"

                # Check for irrelevant questions
                if "tidak relevan" in response.content.lower() or "not relevant" in response.content.lower():
                    st.markdown(get_response('irrelevant', lang))
                else:
                    st.markdown(full_response)

                # Add assistant's response to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response
                })

            except Exception as e:
                st.error(get_response('error', lang))
                print(f"Error: {str(e)}")  # For debugging

# # Display chat history (optional)
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])


import os
import pickle
from openai import OpenAI
from langchain.schema import Document
import os
import streamlit as st
import datetime

from openai import AzureOpenAI


from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
#embedding = OpenAIEmbeddings(openai_api_key=<>)
#embedding = OpenAIEmbeddings()

persist_directory = "C:/temp"

persist_directory = os.path.join(persist_directory, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(persist_directory, exist_ok=True)

st.title('GoGoAI Climate Insight Engine')

client = AzureOpenAI(
    api_key="511a71ca06ca408f84220b82d2451920",
    #api_key=getpass.getpass("OpenAI API Key:"),  
    api_version="2024-07-01-preview",
    azure_endpoint="https://genai-openai-gogoai.openai.azure.com/"
)


# Function to load the docs list from a file using pickle
def load_docs(filename="company_docs.pkl"):
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            docs = pickle.load(file)
        st.write(f"Documents loaded from {filename}")
        return docs
    else:
        st.write("No saved documents found.")
        return []



docs = load_docs("C:/git/GoGoAI-ClimateFin/company_docs_50_with_ratings_update.pkl")
# Split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)


splits = text_splitter.split_documents(docs)


embedding = OpenAIEmbeddings(api_key="YOUR_API_KEY")

vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)


llm_name = "gpt-4o"
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name=llm_name, temperature=0,
                api_key="YOUR_API_KEY")


from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)


from langchain.chains import ConversationalRetrievalChain
retriever=vectordb.as_retriever()
qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory
)


# Build prompt
from langchain.prompts import PromptTemplate
template = """You are a financial analyst and ESG expert. Your answers always contain quantitative information. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)


st.subheader('Query Financial Statement Information')
user_query = st.text_area('Enter your query here:')
if st.button('Submit'):
    result = qa({"question": user_query})
    st.write(result['answer'])
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI 
from htmlTemplates import css, bot_template, user_template
import os

name= 'Imanol Asolo'

def get_pdf_text(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPEN_AI_APIKEY"])
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vector_store):
    llm = ChatOpenAI(openai_api_key=st.secrets["OPEN_AI_APIKEY"])
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)

def main():
    st.set_page_config(page_title= name, page_icon=":wave:", layout="centered")
    st.write(css, unsafe_allow_html=True)

    st.title("AIProfileVCard")
   
    st.header(name)
    st.subheader("CEO of CodeCodix")

    col1,col2 = st.columns(2)
    with col1:

        st.image('picture_imanol.png', caption=name, width=200)
    with col2:    
        description = """
    ### About Me
    I am the CEO of CodeCodix, a company dedicated to developing artificial intelligence tools. Besides leading the company, I am a proud father who loves the sea. At CodeCodix, I work as a Full Stack Developer and Scrum Master, playing multiple roles to ensure the success of our projects. My passion for technology and versatility in my skills allow me to contribute significantly to various aspects of development and team management.
    """
        st.markdown(description)

    col1,col2 = st.columns(2)
    with col1:
        services = ["AI Tools Development", "Technology Consulting", "Project Management"]
        st.write("### Services Offered")
        for service in services:
            st.write(f"- {service}")

    with col2:
        projects = ["AI_Medicare", "Raptor_eye", "Botarmy_Hub"]
        st.write ('### Developed projects')
        for project in projects:
            st.write(f"- {project}")

    st.write("### Chat with Me, know me and let´s contact!")
    st.info("Doesn´t matter the language,ask anything you need!")

    pdf_path = os.path.join(os.getcwd(), "imanolpdf1.pdf")
    pdf_text = get_pdf_text(pdf_path)
    text_chunks = get_text_chunks(pdf_text)
    vector_store = get_vector_store(text_chunks)
    conversation_chain = get_conversation_chain(vector_store)
    st.session_state.conversation = conversation_chain
    st.session_state.chat_history = []

    user_question = st.text_input("Ask me anything:")
    if user_question:
        handle_user_input(user_question)

if __name__ == "__main__":
    main()

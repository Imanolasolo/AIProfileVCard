import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI 
from htmlTemplates import css, bot_template, user_template
import os
import base64

# Set your name
name = ':red[Imanol Asolo]'

# Function to extract text from a PDF file
def get_pdf_text(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split the extracted text
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter.split_text(text)

# Function to create the vector store
def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPEN_AI_APIKEY"])
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

# Function to get the conversation chain
def get_conversation_chain(vector_store):
    llm = ChatOpenAI(openai_api_key=st.secrets["OPEN_AI_APIKEY"])
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )

# Handle user input
def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)

# Main function
def main():
    st.set_page_config(page_title=name, page_icon=":wave:", layout="centered")

    def get_base64_of_bin_file(bin_file):
        with open(bin_file, 'rb') as f:
            return base64.b64encode(f.read()).decode()

    img_base64 = get_base64_of_bin_file('IA_background.jpg')
    st.markdown(f"""
    <style>
    .stApp {{
        background: url('data:image/jpeg;base64,{img_base64}') no-repeat center center fixed;
        background-size: cover;
    }}
    </style>
    """, unsafe_allow_html=True)

    st.write(css, unsafe_allow_html=True)
    st.title("AIProfileVCard")
    col1,col2,col3 = st.columns([1, 1, 2])
    with col1:
        lang = st.radio("Idioma / Language", ("Español", "English"))
    with col2:
        
        st.header(name)
    with col3:
        if lang == "Español":    
            st.subheader("IA-lizador | CEO de CodeCodix")
        else:
            st.subheader("AI-lizer | CEO of CodeCodix")

    

    col1, col2 = st.columns(2)
    with col1:
        st.image('picture_imanol.png', caption=name, width=200)
    with col2:
        if lang == "Español":
            st.markdown("""
            ### Sobre mí
            Soy Imanol Asolo, IA-lizador. Ayudo a empresas y proyectos a evolucionar, automatizar y escalar mediante herramientas de inteligencia artificial.
            Lidero CodeCodix, donde creamos soluciones prácticas de IA que funcionan desde la primera semana. Transformamos hospitales, autoescuelas, barcos atuneros y más.
            """)
        else:
            st.markdown("""
            ### About Me
            I'm Imanol Asolo, an AI-lizer. I help businesses and projects evolve, automate and scale through smart AI tools.
            As CEO of CodeCodix, I lead the creation of practical AI systems that deliver value from week one. We transform hospitals, driving schools, fishing vessels and beyond.
            """)

    col1, col2 = st.columns(2)
    with col1:
        services = ["Desarrollo de herramientas IA", "Consultoría tecnológica", "Gestión de proyectos"] if lang == "Español" else ["AI Tools Development", "Technology Consulting", "Project Management"]
        st.write("### Servicios / Services")
        for service in services:
            st.write(f"- {service}")

    with col2:
        projects = ["AI_Medicare", "Raptor_eye", "Botarmy_Hub"]
        st.write ('### Proyectos / Projects')
        for project in projects:
            st.write(f"- {project}")

    st.write("### ¡Conversa conmigo! / Chat with me!")
    st.info("Hazme una pregunta y descubre cómo puedo ayudarte | Ask me anything and see how I can help you")


# Cargar múltiples PDFs desde carpeta "pdfs"
    pdf_folder = os.path.join(os.getcwd(), "pdfs")
    pdf_text = ""
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_folder, filename)
            pdf_text += get_pdf_text(file_path)

    text_chunks = get_text_chunks(pdf_text)
    vector_store = get_vector_store(text_chunks)
    conversation_chain = get_conversation_chain(vector_store)

    st.session_state.conversation = conversation_chain
    st.session_state.chat_history = []

    user_question = st.text_input("Pregunta / Ask:")
    if user_question:
        handle_user_input(user_question)

if __name__ == "__main__":
    main()

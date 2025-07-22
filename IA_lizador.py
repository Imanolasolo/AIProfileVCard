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
            st.subheader("IA-lizador Estratégico | CEO de CodeCodix")
        else:
            st.subheader("Strategic AI-lizer | CEO of CodeCodix")

    col1, col2 = st.columns(2)
    with col1:
        st.image('picture_imanol.png', caption=name, width=200)
    with col2:
        if lang == "Español":
            st.markdown("""
            ### Sobre mí
            **Full Stack Developer | Scrum Master | AI & LLM Solutions Architect**
            
            Desarrollador Full Stack visionario y Scrum Master certificado con 7+ años de experiencia construyendo soluciones digitales de vanguardia que fusionan desarrollo web moderno con capacidades de IA y LLM. 
            
            Especializado en diseñar y escalar plataformas inteligentes en los sectores de salud, negocios y engagement de clientes. Reconocido por transformar desafíos complejos en soluciones escalables de alto impacto.
            
            📧 jjusturi@gmail.com | 📱 +593 099 351 3082
            """)
        else:
            st.markdown("""
            ### About Me
            **Full Stack Developer | Scrum Master | AI & LLM Solutions Architect**
            
            Visionary Full Stack Developer and Certified Scrum Master with 7+ years of experience building cutting-edge digital solutions that merge modern web development with AI and LLM capabilities.
            
            Specialized in designing and scaling intelligent platforms across healthcare, business, and customer engagement sectors. Recognized for transforming complex challenges into high-impact, scalable solutions.
            
            📧 jjusturi@gmail.com | 📱 +593 099 351 3082
            """)

    # Core Competencies Section
    st.write("### 🧠 Competencias Clave / Core Competencies")
    col1, col2 = st.columns(2)
    with col1:
        if lang == "Español":
            st.markdown("""
            **Herramientas IA & LLM:**
            - Prompt Engineering, LangChain
            - GPT-4, OpenAI API, Hugging Face
            
            **Desarrollo Full Stack:**
            - React, Vue, Streamlit, TypeScript
            - Node.js, Django, FastAPI
            
            **Cloud & DevOps:**
            - Docker, Kubernetes, AWS, Azure
            """)
        else:
            st.markdown("""
            **AI & LLM Tools:**
            - Prompt Engineering, LangChain
            - GPT-4, OpenAI API, Hugging Face
            
            **Full Stack Development:**
            - React, Vue, Streamlit, TypeScript
            - Node.js, Django, FastAPI
            
            **Cloud & DevOps:**
            - Docker, Kubernetes, AWS, Azure
            """)
    
    with col2:
        if lang == "Español":
            st.markdown("""
            **Liderazgo Ágil:**
            - Ceremonias Scrum, Coaching de Equipos
            - Sprint Planning, Kanban, Jira
            
            **Automatización & Testing:**
            - Selenium, Pytest, Unit Testing
            
            **Lenguajes:**
            - Python, JavaScript, Java, C++, PHP
            """)
        else:
            st.markdown("""
            **Agile Leadership:**
            - Scrum Ceremonies, Team Coaching
            - Sprint Planning, Kanban, Jira
            
            **Automation & Testing:**
            - Selenium, Pytest, Unit Testing
            
            **Languages:**
            - Python, JavaScript, Java, C++, PHP
            """)

    # Key Projects Section
    st.write("### 🚀 Proyectos Destacados / Key Projects")
    col1, col2 = st.columns(2)
    with col1:
        if lang == "Español":
            st.markdown("""
            **🩺 AI Medicare Suite**
            - Plataforma hospitalaria con IA integrada
            - Gestión de pacientes y flujos de emergencia
            - Agentes conversacionales con LLM
            
            **📇 AI Interactive Business Card**
            - Tarjeta de presentación inteligente
            - Chat en tiempo real y portfolio personal
            """)
        else:
            st.markdown("""
            **🩺 AI Medicare Suite**
            - AI-integrated hospital platform
            - Patient management & emergency workflows
            - LLM-powered conversational agents
            
            **📇 AI Interactive Business Card**
            - Smart business card with real-time chat
            - Personal portfolio and contact features
            """)
    
    with col2:
        if lang == "Español":
            st.markdown("""
            **🤖 Botarmy Hub**
            - Marketplace de asistentes virtuales
            - Sistema de afiliados con revenue sharing
            - Herramientas de despliegue para clientes
            
            **🌊 NautiAI (ERP Industria Atunera)**
            - ERP específico para flotas pesqueras
            - Planificación de recursos con IA
            """)
        else:
            st.markdown("""
            **🤖 Botarmy Hub**
            - Virtual assistant marketplace
            - Affiliate system with revenue sharing
            - Client deployment tools
            
            **🌊 NautiAI (Tuna Industry ERP)**
            - Industry-specific AI ERP tool
            - Resource planning for fishing fleets
            """)

    # Achievements Section
    st.write("### 🏆 Logros Clave / Key Achievements")
    if lang == "Español":
        achievements = [
            "🚀 Lanzó plataforma hospitalaria con IA en Manta, Ecuador - actualmente en operaciones reales",
            "🧠 Creó AI Business Cards con chat LLM integrado, redefiniendo el networking profesional",
            "📈 Construyó Botarmy Hub con incentivos de reventa (50% comisiones)",
            "📣 Desarrolló plataforma de IA para comunicación y relaciones públicas en salud",
            "🧩 Diseñó herramientas tipo ERP para pesca, agricultura y turismo bajo el modelo AI-lización"
        ]
    else:
        achievements = [
            "🚀 Launched AI-powered hospital platform in Manta, Ecuador - currently handling real operations",
            "🧠 Created AI Business Cards with embedded LLM chat, redefining professional networking",
            "📈 Built Botarmy Hub with built-in reseller incentives (50% commissions)",
            "📣 Developed AI-driven social media and PR platform for healthcare teams",
            "🧩 Designed ERP-like tools for fishing, agriculture, and tourism under AI-lización model"
        ]
    
    for achievement in achievements:
        st.write(f"• {achievement}")

    st.write("### ¡Conversa conmigo! / Chat with me!")
    st.info("Hazme una pregunta sobre mis proyectos, experiencia o cómo puedo ayudarte | Ask me about my projects, experience or how I can help you")

    # Load PDFs and setup conversation
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

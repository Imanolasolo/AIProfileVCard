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
            st.subheader("Arquitecto de Crecimiento | CEO de CodeCodix")
        else:
            st.subheader("Growth Architect | CEO of CodeCodix")

    col1, col2 = st.columns(2)
    with col1:
        st.image('picture_imanol.png', caption=name, width=200)
    with col2:
        if lang == "Español":
            st.markdown("""
            ### Sobre mí
            **Full Stack Developer | Scrum Master | AI & LLM Solutions Architect | Arquitecto de Crecimiento**
            
            Arquitecto de Crecimiento visionario y Scrum Master certificado con más de 7 años de experiencia construyendo soluciones digitales de vanguardia que fusionan el desarrollo web moderno con capacidades de IA y LLM.
            """)
        else:
            st.markdown("""
            ### About Me
            **Full Stack Developer | Scrum Master | AI & LLM Solutions Architect | Growth Architect**
            
            Growth Architect visionary and Certified Scrum Master with 7+ years of experience building cutting-edge digital solutions that merge modern web development with AI and LLM capabilities.            
                        
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
                        
            **Ingeniería de crecimiento:**
            - Estrategias de escalabilidad, optimización de procesos
            - Implementación de soluciones basadas en datos
            - Análisis de métricas y rendimiento
                        
            **Manejo de proyectos:**
            - Metodologías ágiles, Scrum, Kanban
            - Gestión de equipos y liderazgo
            - Herramientas: Jira, Trello, Confluence
            - Product Owner
            - Scrum Master
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
                        
            **Growth Engineering:**
            - Scalability strategies, process optimization
            - Data-driven solution implementation
            - Metrics and performance analysis
            
            **Project Management:**
            - Agile methodologies, Scrum, Kanban
            - Team management and leadership
            - Tools: Jira, Trello, Confluence
            - Product Owner
            - Scrum Master
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

    # Growth Engineering Section
    st.write("### 📈 Ingeniería de Crecimiento / Growth Engineering")
    if lang == "Español":
        st.markdown("""
        Como Arquitecto de Crecimiento, he liderado iniciativas estratégicas para escalar soluciones digitales mediante la optimización de procesos y la implementación de tecnologías basadas en datos. Mi enfoque se centra en identificar oportunidades de crecimiento sostenible, mejorar la eficiencia operativa y maximizar el impacto a través de estrategias innovadoras.
        
        - Desarrollo e implementación de estrategias de escalabilidad para plataformas digitales.
        - Optimización de procesos internos mediante automatización y análisis de datos.
        - Colaboración con equipos multifuncionales para alinear objetivos de negocio y tecnología.
        - Monitoreo y análisis continuo de métricas clave para evaluar el rendimiento y ajustar estrategias según sea necesario.
        """)
    else:
        st.markdown("""
        As a Growth Architect, I have led strategic initiatives to scale digital solutions by optimizing processes and implementing data-driven technologies. My focus is on identifying opportunities for sustainable growth, improving operational efficiency, and maximizing impact through innovative strategies.
        
        - Developing and implementing scalability strategies for digital platforms.
        - Optimizing internal processes through automation and data analysis.
        - Collaborating with cross-functional teams to align business and technology goals.
        - Continuously monitoring and analyzing key metrics to evaluate performance and adjust strategies as needed.
        """)

    # Customer benefits Section as growth architect
    st.write("### 🎯 Beneficios para el Cliente / Customer Benefits")
    if lang == "Español":
        st.markdown("""
        - **Crecimiento Sostenible:** Estrategias diseñadas para impulsar un crecimiento continuo y escalable.
        - **Eficiencia Operativa:** Optimización de procesos que reduce costos y mejora la productividad.
        - **Innovación Constante:** Implementación de tecnologías avanzadas para mantener la competitividad en el mercado.
        - **Resultados Medibles:** Enfoque en métricas clave para asegurar el éxito y el retorno de inversión.
        """)

    else:
        st.markdown("""
        - **Sustainable Growth:** Strategies designed to drive continuous and scalable growth.
        - **Operational Efficiency:** Process optimization that reduces costs and improves productivity.
        - **Constant Innovation:** Implementation of advanced technologies to stay competitive in the market.
        - **Measurable Results:** Focus on key metrics to ensure success and return on investment.
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

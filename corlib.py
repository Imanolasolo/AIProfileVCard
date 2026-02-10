import os
# Allow duplicate OpenMP runtimes to load (workaround for Windows libomp/libiomp conflicts).
# This is an unsafe workaround; prefer installing compatible packages or ensuring a single OpenMP runtime.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import os
import base64

# Set your name for the AIProfileVCard
name = 'Corazones liberados'

# Language dictionaries
TRANSLATIONS = {
    'en': {
        'title': 'AIProfileVCard',
        'ceo_title': 'Sponsorships 2026',
        'about_me_title': '### About Us',
        'about_me_text': '''We are a social development initiative that promotes violence prevention, economic autonomy, education, and holistic well-being among vulnerable populations in Manta.

We work within the San Juan community of Manta and in shelters partnered with churches and government institutions, coordinating efforts with academia, international cooperation, and the private sector.

👉 Our approach combines measurable social impact, sustainability, and strategic partnerships.''',
        'services_title': '### Programs by thematic axes',
        'eje1_title': 'Axis 1: Violence Prevention',
        'eje2_title': 'Axis 2: Economic Empowerment',
        'eje3_title': 'Axis 3: Education for the Future',
        'eje4_title': 'Axis 4: Health and Holistic Well-being',
        'eje1_description': '''"Circles of Peace and Protection"

✔ Psychosocial support
✔ Legal guidance
✔ Community protection networks

Impact: safer family environments and reduced risk of violence.''',
        'eje2_description': '''"Unleash Your Potential"

✔ Technical training
✔ Entrepreneurship
✔ Financial education
✔ Seed funds

Impact: women generating their own income and reducing economic dependency.''',
        'eje3_description': '''"STEM Hearts"

✔ Technology clubs
✔ STEM education
✔ Educational innovation

Impact: development of local talent and 21st-century skills.''',
        'eje4_description': '''"Taking care of myself is my power"

✔ Mental health
✔ Trauma management
✔ Sexual and reproductive education

Impact: increased self-esteem, resilience, and informed decision-making.''',
        'sponsorships_title': '### Sponsorship Options',
        'sponsorship1_title': 'Bronze Sponsorship',
        'sponsorship1_description': '''USD 10,000
✔ Funding for specific components
✔ Recognition as a social ally
✔ Results report''',
        'sponsorship2_title': 'Silver Sponsorship',
        'sponsorship2_description': '''USD 25,000
✔ Co-sponsorship of an axis
✔ Semi-annual impact report
✔ Visibility in events and communications
✔ Institutional recognition''',
        'sponsorship3_title': 'Gold Sponsorship',
        'sponsorship3_description': '''USD 50,000+
✔ Sponsorship of a complete axis
✔ Prominent logo placement in materials and events
✔ Exclusive impact report
✔ Visibility in media and social networks
✔ Participation in community events
Ideal for strategic CSR programs.''',
        'services_pdf': 'pdfs/BCS_PDFEN.pdf',
        'projects_title': '### Services',
        'projects': ['AI consulting', 'Software Development', 'Product Ownership', 'Startup Mentoring'],
        'chat_title': '### Chat with Me, know me and let\'s contact!',
        'chat_info': 'No matter the language, ask anything you need!',
        'ask_placeholder': 'Ask me anything:',
        'picture_caption': name
    },
    'es': {
        'title': 'AIProfileVCard',
        'ceo_title': 'Patrocinios 2026',
        'about_me_title': '### Sobre Nosotros',
        'about_me_text': '''Somos una iniciativa de desarrollo social que impulsa prevención de violencia, autonomía económica, educación y bienestar integral en poblaciones vulnerables de Manta.

Trabajamos en la comunidad de San Juan de Manta y en refugios aliados a iglesia y gobierno, articulando esfuerzos con academia, cooperación internacional y sector privado.

👉 Nuestro enfoque combina impacto social medible + sostenibilidad + alianzas estratégicas.''',
        'services_title': '### Programas por ejes',
        'eje1_title': 'Eje 1: Prevención de Violencia',
        'eje2_title': 'Eje 2: Empoderamiento Económico',
        'eje3_title': 'Eje 3: Educación para el futuro',
        'eje4_title': 'Eje 4: Salud yBienestar Integral',
        'eje1_description': '''"Círculos de paz y protección"

✔ Atención psicosocial
✔ Orientación jurídica
✔ Redes de protección comunitaria

Impacto: entornos familiares más seguros y reducción de riesgos de violencia.''',
        'eje2_description': '''"Libera tu potencial"

✔ Formación técnica
✔ Emprendimiento
✔ Educación financiera
✔ Fondos semilla

Impacto: mujeres generando ingresos propios y reduciendo dependencia económica.''',
        'eje3_description': '''"Corazones STEM"

✔ Clubes tecnológicos
✔ Educación STEM
✔ Innovación educativa

Impacto: desarrollo de talento local y habilidades del siglo XXI.''',
        'eje4_description': '''"Cuidarme es mi poder"

✔ Salud mental
✔ Manejo de trauma
✔ Educación sexual y reproductiva

Impacto: mayor autoestima, resiliencia y toma de decisiones informadas.''',

        'services_pdf': 'pdfs/BCS_PDF.pdf',
        'sponsorships_title': '### Opciones de auspicio',
        'sponsorship1_title': 'Auspicio Bronce',
        'sponsorship1_description': '''USD 10.000

✔ Financiamiento de componentes específicos
✔ Reconocimiento como aliado social
✔ Reporte de resultados''',
        'sponsorship2_title': 'Auspicio Plata',
        'sponsorship2_description': '''USD 25.000

✔ Co-auspicio de un eje
✔ Reporte semestral de impacto
✔ Visibilidad en eventos y comunicaciones
✔ Reconocimiento institucional''',
        'sponsorship3_title': 'Auspicio Oro',
        'sponsorship3_description': '''USD 50.000+

✔ Auspicio de un eje completo
✔ Logo destacado en materiales y eventos
✔ Reporte de impacto exclusivo
✔ Visibilidad en medios y redes
✔ Participación en eventos comunitarios

Ideal para programas de RSE estratégicos. ''',

        'projects': ['Consultoría de IA', 'Desarrollo de Software', 'Propiedad de Producto', 'Mentoría de Startups'],
        'chat_title': '### ¡Chatea conmigo, conóceme y contactemos!',
        'chat_info': '¡No importa el idioma, pregunta lo que necesites!',
        'ask_placeholder': 'Pregúntame lo que quieras:',
        'picture_caption': name
    }
}

# Function to extract text from a PDF file
def get_pdf_text(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    text = ""
    # Iterate through each page and extract text
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split the extracted text into manageable chunks for processing
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to generate a vector store using the text chunks
def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPEN_AI_APIKEY"])
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Function to create a conversational chain using the vector store
def get_conversation_chain(vector_store):
    llm = ChatOpenAI(openai_api_key=st.secrets["OPEN_AI_APIKEY"])
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Function to handle user input and generate responses
def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Display the conversation history
    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)

# Function to display PDF in Streamlit
def display_pdf_button(pdf_path, button_text):
    """Display a download button for the PDF file"""
    try:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        
        st.download_button(
            label=button_text,
            data=pdf_bytes,
            file_name=os.path.basename(pdf_path),
            mime="application/pdf"
        )
    except FileNotFoundError:
        st.error(f"PDF file not found: {pdf_path}")
    except Exception as e:
        st.error(f"Error loading PDF: {str(e)}")

# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title= name, page_icon=":wave:", layout="centered")

    # Initialize language in session state
    if 'language' not in st.session_state:
        st.session_state.language = 'en'

    # Get current language translations
    lang = TRANSLATIONS[st.session_state.language]

    # Function to encode image as base64 to set as background
    # def get_base64_of_bin_file(bin_file):
        #with open(bin_file, 'rb') as f:
            #data = f.read()
        #return base64.b64encode(data).decode()

    ## Encode the background image
    #img_base64 = get_base64_of_bin_file('logo_portfolio.jpg')

    ## Set the background image using the encoded base64 string
    #st.markdown(
    #f"""
    #<style>
    #.stApp {{
        #background: url('data:image/jpeg;base64,{img_base64}') no-repeat center center fixed;
        #background-size: cover;
    #}}
    #</style>
    #""",
    #unsafe_allow_html=True
#)

    # Apply custom CSS styles
    st.write(css, unsafe_allow_html=True)

    # Language selector buttons next to title
    col_title, col_lang = st.columns([3, 1])
    with col_title:
        # Title in a styled box
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50;">
            <h1 style="margin: 0; color: #1f1f1f;">{lang['title']}</h1>
        </div>
        """, unsafe_allow_html=True)
    with col_lang:
        col_en, col_es = st.columns(2)
        with col_en:
            if st.button("EN"):
                st.session_state.language = 'en'
                st.rerun()
        with col_es:
            if st.button("ES"):
                st.session_state.language = 'es'
                st.rerun()

    # Name in a styled box
    st.markdown(f"""
    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid #2196F3; margin-top: 20px;">
        <h2 style="margin: 0; color: #1f1f1f;">{name}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader(lang['ceo_title'])

    # Display the profile picture and description in two columns
    col1, col2 = st.columns(2)
    with col1:
        st.image('juguetes.jpeg', caption=lang['picture_caption'], width=400)
    with col2:
        st.markdown(lang['about_me_title'])
        st.markdown(lang['about_me_text'])

    # Display services offered and projects developed in two columns
    st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid #FF9800; margin-bottom: 20px;">
            <h3 style="margin: 0 0 10px 0; color: #1f1f1f;">{lang['services_title'].replace('### ', '')}</h3>
        </div>
        """, unsafe_allow_html=True)
    col1,col2,col3,col4 = st.columns(4)
    with col1:
        with st.expander(lang['eje1_title']):
            st.markdown(lang['eje1_description'])
    with col2:
        with st.expander(lang['eje2_title']):
            st.markdown(lang['eje2_description'])
    with col3:
        with st.expander(lang['eje3_title']):
            st.markdown(lang['eje3_description'])
    with col4:
        with st.expander(lang['eje4_title']):
            st.markdown(lang['eje4_description'])
    
    # Sponsorships section
    st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid #9C27B0; margin-bottom: 20px; margin-top: 20px;">
            <h3 style="margin: 0 0 10px 0; color: #1f1f1f;">{lang['sponsorships_title'].replace('### ', '')}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.expander(lang['sponsorship1_title']):
            st.markdown(lang['sponsorship1_description'])
    with col2:
        with st.expander(lang['sponsorship2_title']):
            st.markdown(lang['sponsorship2_description'])
    with col3:
        with st.expander(lang['sponsorship3_title']):
            st.markdown(lang['sponsorship3_description'])
    
    # Section for interacting with the AI chatbot
    st.write(lang['chat_title'])
    st.info(lang['chat_info'])
    # Process the PDF file to be used as context for the chatbot
    pdf_path = os.path.join(os.getcwd(), "pdfs/Brochure_CORLIB.pdf")
    pdf_text = get_pdf_text(pdf_path)
    text_chunks = get_text_chunks(pdf_text)
    vector_store = get_vector_store(text_chunks)
    conversation_chain = get_conversation_chain(vector_store)

    # Store the conversation chain and history in session state
    st.session_state.conversation = conversation_chain
    st.session_state.chat_history = []

    # Input box for user questions
    user_question = st.text_input(lang['ask_placeholder'])
    if user_question:
        handle_user_input(user_question)

# Run the main function if the script is executed
if __name__ == "__main__":
    main()
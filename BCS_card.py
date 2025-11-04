import os
# Allow duplicate OpenMP runtimes to load (workaround for Windows libomp/libiomp conflicts).
# This is an unsafe workaround; prefer installing compatible packages or ensuring a single OpenMP runtime.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
try:
    from langchain.memory import ConversationBufferMemory
except Exception:
    try:
        # some distributions may ship langchain integrations under a different package
        from langchain_community.memory import ConversationBufferMemory
    except Exception as e:
        raise ImportError(
            "ConversationBufferMemory not found. Install a compatible langchain package (e.g. add 'langchain>=0.1.0,<0.2.0' to requirements)"
        ) from e
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template
import os
import base64
# static dictionary-based translations (no runtime translation service)

# Set your name for the AIProfileVCard
name = 'BCS - Business Core Solutions'

# Original UI texts (kept in English here). We'll translate these on demand.
english_texts = {
    'app_title': 'AIProfileVCard',
    'company_name': name,
    'profile_caption': name,
    'description': """
### What we do at BCS
BCS builds intelligent, custom core software that empowers companies to automate, scale, and evolve their business. We also act as resellers of cold wallets (secure hardware storage) and provide cryptocurrency advisory and decision-support to help organizations adopt and leverage digital assets safely.
""",
    'services_header': '### Services Offered',
    'services': [
        "AI Tools Development",
        "Technology Consulting",
        "Project Management",
        "Cold wallets reseller (sale & distribution)",
        "Cryptocurrency advisory and decision-support",
    ],
    'projects_header': '### Developed projects',
    # canonical project identifiers (do NOT translate)
    'projects': ["AI_Medicare", "Raptor_eye", "Botarmy_Hub"],
    'section_prompt': "### Tell us about your projects and let´s discover how can we work together!",
    'info_text': "Doesn´t matter the language, ask anything you need!",
    'input_placeholder': "How we can help you today?",
}

# Spanish dictionary (static). Project identifiers are preserved and NOT translated.
spanish_texts = {
    'app_title': 'AIProfileVCard',
    'company_name': 'BCS - Soluciones centrales de negocio',
    'profile_caption': name,
    'description': """
### Qué hacemos en BCS
BCS crea un software central inteligente y personalizado que permite a las empresas automatizar, escalar y hacer evolucionar sus negocios. Además, actuamos como distribuidores de carteras frías (cold wallets) y ofrecemos asesoría y apoyo en la toma de decisiones con criptomonedas para ayudar a las organizaciones a adoptar y aprovechar los activos digitales de forma segura.
""",
    'services_header': '### Servicios ofrecidos',
    'services': [
        "Desarrollo de herramientas de IA",
        "Consultoría Tecnológica",
        "Gestión de proyectos",
        "Venta y distribución de carteras frías (cold wallets)",
        "Asesoría y apoyo en toma de decisiones con criptomonedas",
    ],
    'projects_header': '### Proyectos desarrollados',
    # keep original project identifiers
    'projects': ["AI_Medicare", "Raptor_eye", "Botarmy_Hub"],
    'section_prompt': '¡Cuéntanos tus proyectos y descubramos cómo podemos trabajar juntos!',
    'info_text': '¡No importa el idioma, pregunta lo que necesites!',
    'input_placeholder': '¿Cómo podemos ayudarte hoy?',
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

# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title= name, page_icon=":wave:", layout="centered")

    # Function to encode image as base64 to set as background
    def get_base64_of_bin_file(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()

    # Encode the background image
    img_base64 = get_base64_of_bin_file('17927.jpg')

    # Set the background image using the encoded base64 string
    st.markdown(
    f"""
    <style>
    .stApp {{
        background: url('data:image/jpeg;base64,{img_base64}') no-repeat center center fixed;
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

    # Apply custom CSS styles
    st.write(css, unsafe_allow_html=True)

    # Initialize language/session state containers
    if 'lang' not in st.session_state:
        st.session_state['lang'] = 'en'
    if 'translations' not in st.session_state:
        # cache translated versions: key -> lang code -> dict of texts
        st.session_state['translations'] = {}
        # store English as the base (original)
        st.session_state['translations']['en'] = english_texts
    if 'ui_texts' not in st.session_state:
        st.session_state['ui_texts'] = st.session_state['translations']['en']

    # Title + language buttons row
    title_col, btn_en_col, btn_es_col = st.columns([4, 2, 2])
    with title_col:
        # Use markdown so we can keep sizing consistent when placing buttons beside
        st.markdown(f"# {st.session_state['ui_texts']['app_title']}")
    with btn_en_col:
        if st.button('English'):
            translate_all('en')
    with btn_es_col:
        if st.button('Español'):
            translate_all('es')

    # Header/company name
    st.header(st.session_state['ui_texts']['company_name'])
    # st.subheader("BCS - Business Core Solutions")

    # Display the profile picture and description in two columns
    col1, col2 = st.columns(2)
    with col1:
        st.image('BCS_logo.png', caption=name, width=200)
    with col2:
        # Render description heading and body separately to preserve markdown heading formatting
        desc = st.session_state['ui_texts'].get('description', '')
        # If description is a single string that includes a heading, try to split it safely
        if isinstance(desc, str) and '\n' in desc:
            first, rest = desc.split('\n', 1)
            # Ensure heading has leading hashes
            heading = first.strip()
            # Remove any leading '#' characters and reapply a single H3 marker to be consistent
            heading_text = heading.lstrip('#').strip()
            st.markdown(f"### {heading_text}")
            # Render body as markdown paragraph (sanitize any leftover indentation)
            body = rest.strip()
            st.markdown(body)
        else:
            st.markdown(desc)

    # Display services offered and projects developed in two columns
    col1, col2 = st.columns(2)
    with col1:
        services = st.session_state['ui_texts']['services']
        st.write(st.session_state['ui_texts']['services_header'])
        for service in services:
            st.write(f"- {service}")

    with col2:
        projects = st.session_state['ui_texts']['projects']
        st.write(st.session_state['ui_texts']['projects_header'])
        for project in projects:
            st.write(f"- {project}")

    # Section for interacting with the AI chatbot
    st.write(st.session_state['ui_texts']['section_prompt'])
    st.info(st.session_state['ui_texts']['info_text'])

    # Process the PDF file to be used as context for the chatbot
    pdf_path = os.path.join(os.getcwd(), "pdfs/BCS_base.pdf")
    pdf_text = get_pdf_text(pdf_path)
    text_chunks = get_text_chunks(pdf_text)
    vector_store = get_vector_store(text_chunks)
    conversation_chain = get_conversation_chain(vector_store)

    # Store the conversation chain and history in session state
    st.session_state.conversation = conversation_chain
    st.session_state.chat_history = []

    # Input box for user questions
    user_question = st.text_input(st.session_state['ui_texts']['input_placeholder'])
    if user_question:
        handle_user_input(user_question)


def translate_all(target_lang: str):
    """Translate all original_texts into target_lang and cache the result in session_state.

    target_lang: language code like 'en' or 'es'
    """
    # Switch UI texts from static dictionaries
    if target_lang == 'en':
        st.session_state['translations']['en'] = english_texts
        st.session_state['ui_texts'] = english_texts
        st.session_state['lang'] = 'en'
    elif target_lang == 'es':
        st.session_state['translations']['es'] = spanish_texts
        st.session_state['ui_texts'] = spanish_texts
        st.session_state['lang'] = 'es'
    else:
        # fallback to English
        st.session_state['ui_texts'] = english_texts
        st.session_state['lang'] = 'en'

# Run the main function if the script is executed
if __name__ == "__main__":
    main()

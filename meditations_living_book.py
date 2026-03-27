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

# App identity
APP_TITLE = "Meditaciones — Marco Aurelio"
APP_SUBTITLE_EN = "A living book for reflection, practice, and dialogue"
APP_SUBTITLE_ES = "Un libro vivo para reflexión, práctica y diálogo"

# Language dictionaries
TRANSLATIONS = {
    'en': {
        'title': APP_TITLE,
        'subtitle': APP_SUBTITLE_EN,
        'intro_title': '### What this app is',
        'intro_text': (
            "A focused space to read, search, and talk with *Meditations* (as a knowledge base) "
            "to build practical, Stoic-friendly habits: attention, discipline of desire, and clarity in action."
        ),
        'howto_title': '### How to use it',
        'howto_bullets': [
            "Ask about a passage, theme, or virtue (justice, courage, temperance, wisdom).",
            "Request a short daily reflection or a journaling prompt.",
            "When you share a situation, ask for a Stoic reframing and a concrete next action.",
        ],
        'practice_title': '### Quick practices',
        'practice_bullets': [
            "Dichotomy of control: list what depends on you vs. what doesn’t.",
            "View from above: zoom out and describe the event from a wider perspective.",
            "Evening review: what did I do well, what did I resist, what will I do next time?",
        ],
        'chat_title': '### Dialogue with the book',
        'chat_info': "Ask in any language. Responses should stay grounded in the text and Stoic practice.",
        'ask_placeholder': 'Ask a question (e.g., “How would Marcus handle anger at work?”):',
        'download_title': '### Source text',
        'download_button': '📄 Download the PDF used as context',
    },
    'es': {
        'title': APP_TITLE,
        'subtitle': APP_SUBTITLE_ES,
        'intro_title': '### Qué es esta app',
        'intro_text': (
            "Un espacio intencional para leer, buscar y dialogar con *Meditaciones* (como base de conocimiento) "
            "y convertir ideas estoicas en práctica: atención, disciplina del deseo y claridad para actuar."
        ),
        'howto_title': '### Cómo usarla',
        'howto_bullets': [
            "Pregunta por un pasaje, tema o virtud (justicia, valentía, templanza, sabiduría).",
            "Pide una reflexión diaria breve o un prompt de journaling.",
            "Si compartes una situación, pide un reencuadre estoico y una acción concreta.",
        ],
        'practice_title': '### Prácticas rápidas',
        'practice_bullets': [
            "Dicotomía del control: lista lo que depende de ti vs. lo que no.",
            "Vista desde arriba: aléjate y describe el evento desde una perspectiva más amplia.",
            "Revisión nocturna: ¿qué hice bien, qué resistí, qué haré distinto mañana?",
        ],
        'chat_title': '### Diálogo con el libro',
        'chat_info': "Pregunta en cualquier idioma. Las respuestas deben anclarse en el texto y en la práctica estoica.",
        'ask_placeholder': 'Haz una pregunta (ej.: “¿Cómo manejaría Marco Aurelio la ira en el trabajo?”):',
        'download_title': '### Texto fuente',
        'download_button': '📄 Descargar el PDF usado como contexto',
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
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever()
    )
    return conversation_chain

# Function to handle user input and generate responses
def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question, 'chat_history': []})
    answer = response.get("answer") or response.get("result") or ""

    # Show ONLY the latest exchange (no history)
    st.write(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", answer), unsafe_allow_html=True)

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
    st.set_page_config(page_title=APP_TITLE, page_icon="📘", layout="centered")

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
            <p style="margin: 8px 0 0 0; color: #444;">{lang['subtitle']}</p>
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

    st.markdown(lang['intro_title'])
    st.markdown(lang['intro_text'])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(lang['howto_title'])
        for b in lang['howto_bullets']:
            st.markdown(f"- {b}")
    with col2:
        st.markdown(lang['practice_title'])
        for b in lang['practice_bullets']:
            st.markdown(f"- {b}")

    # Section for interacting with the AI chatbot
    st.write(lang['chat_title'])
    st.info(lang['chat_info'])

    # Process the PDF file to be used as context for the chatbot (cache in session)
    pdf_path = os.path.join(os.getcwd(), "pdfs/meditations.pdf")
    st.markdown(lang['download_title'])
    display_pdf_button(pdf_path, lang['download_button'])

    if 'conversation' not in st.session_state:
        with st.spinner("Indexing the book..." if st.session_state.language == 'en' else "Indexando el libro..."):
            pdf_text = get_pdf_text(pdf_path)
            text_chunks = get_text_chunks(pdf_text)
            vector_store = get_vector_store(text_chunks)
            st.session_state.conversation = get_conversation_chain(vector_store)

    # Placeholder so each new message replaces the last one
    chat_slot = st.empty()

    # Input box for user questions
    user_question = st.text_input(lang['ask_placeholder'])
    if user_question:
        with chat_slot:
            handle_user_input(user_question)

# Run the main function if the script is executed
if __name__ == "__main__":
    main()
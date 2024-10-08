# AIProfileVCard

AIProfileVCard is a Streamlit web application designed as an interactive business card, featuring an AI-powered chatbot. The app showcases personal information, services, and projects while allowing users to engage in a conversation with the AI, which has been trained on a PDF document about the owner.

## Features

- **Personal Profile**: Displays the user's profile picture, name, and a brief description.
- **Services Offered**: Lists the services provided by the user.
- **Projects Developed**: Showcases the projects the user has worked on.
- **AI Chatbot**: Users can ask questions and interact with the AI, which has been trained on a PDF document.
- **Custom Background**: The app features a custom background image.

## Technologies Used

- **Streamlit**: For building the interactive web application.
- **PyPDF2**: For extracting text from the PDF document.
- **Langchain**: For splitting text, embedding vectors, and creating a conversational retrieval chain.
- **FAISS**: For efficient similarity search and clustering of the text chunks.
- **OpenAI API**: For generating embeddings and enabling the AI chatbot.
- **Python**: The core programming language used for the app.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/AIProfileVCard.git
    cd AIProfileVCard
    ```

2. **Install the required Python packages**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up the environment variables**:
   Create a `.streamlit/secrets.toml` file to store your OpenAI API key:
   ```toml
   [secrets]
   OPEN_AI_APIKEY = "your-openai-api-key"
   ```

4. **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

## Usage

1. **Home Page**: Upon launching the app, you will see the user's profile information, services, and projects.
2. **Chatbot Interaction**: Enter your questions in the input box at the bottom of the page, and the AI will respond based on the content of the PDF document.
3. **Customization**: You can customize the profile information, services, projects, and background image by modifying the respective sections in the `app.py` file.

## Files in the Repository

- `app.py`: The main Streamlit application file.
- `requirements.txt`: Lists the Python dependencies required to run the app.
- `htmlTemplates.py`: Contains HTML templates for styling the chatbot interface.
- `logo_portfolio.jpg`: The background image for the app.
- `picture_imanol.png`: The profile picture displayed in the app.
- `imanolpdf1.pdf`: The PDF document used to train the AI chatbot.

## Customization

To customize this app for yourself:

1. **Profile Information**: Update the `name`, `description`, and profile image in the `main()` function.
2. **Services & Projects**: Modify the `services` and `projects` lists to reflect your offerings.
3. **Background Image**: Replace `logo_portfolio.jpg` with your desired background image.
4. **PDF Document**: Replace `imanolpdf1.pdf` with your own PDF file to train the chatbot.

## Contributing

Feel free to fork this repository and submit pull requests to improve the app or add new features.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

If you have any questions or suggestions, feel free to reach out:

- **Name**: Imanol Asolo
- **Company**: CodeCodix
- **Role**: CEO, Full Stack Developer, Scrum Master
- **Email**: [your-email@example.com]

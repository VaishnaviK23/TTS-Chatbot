# TTS-Chatbot
A chatbot that answers questions based on user-provided data using Langchain, OpenAI and Pinecone. Integrates text-to-speech capabilities with the open-source library Coqui TTS, converting the chatbot responses to audio for enhanced accessibility.

- Install all the requirements in the requirements.txt file
- Create an empty directory called 'audio'
- Create a .env file and add the following lines to it:
  OPENAI_API_KEY = "<your_api_key>"
  PINECONE_API_KEY = "<your_api_key>"
- Add the PDF you want to use as your data source to the 'data' directory. Change the filepath variable to that PDF. Two sample PDFs are already included.

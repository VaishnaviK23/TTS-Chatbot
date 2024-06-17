from dotenv import load_dotenv
import os
import pickle
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
import simpleaudio as sa
from TTS.api import TTS

load_dotenv()

def create_vector_store(filepath):

    # Extract base name of the file to use as index name
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    vector_store_path = f"vector_stores/{base_name}.pkl"
    index_name = base_name

    # Initialize pinecone
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)

    # Initialize embedding
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    # Check if index already exists
    if index_name in pc.list_indexes().names():
        # Index already exists, retrieve existing vector store
        vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)
        return vector_store

    # Create index in pinecone
    if index_name not in pc.list_indexes().names():
        pc.create_index(name=index_name, dimension=1536, metric="cosine",
                        spec=ServerlessSpec(
                            cloud="aws", region="us-east-1"
                        ))

    # Check if vector store data already exists locally
    if os.path.exists(vector_store_path):
        with open(vector_store_path, "rb") as f:
            documents = pickle.load(f)

    else:
        # Load data
        loader = PyPDFLoader(filepath)
        data = loader.load()

        # Split data into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
        documents = text_splitter.split_documents(data)


    # Create vector store
    vector_store = PineconeVectorStore.from_documents(documents=documents, embedding=embeddings, index_name=index_name)

    # Save vector store data locally
    with open(vector_store_path, "wb") as f:
        pickle.dump((documents), f)

    return vector_store

def get_response_to_query(query, vector_store, llm, k=4):
    docs = vector_store.similarity_search(query, k=k)
    docs_page_content = ''.join([d.page_content for d in docs])
    template = """
            You are a helpful assistant that can answer a question about the contents of a PDF.
            Answer the following question: {question}
            By searching the following PDF: {docs}
    
            Only use factual information from the PDF to answer the question.
            If you feel like you do not have enough information to answer the question, say "I don't know."
    
            Give detailed answers.
            """
    prompt = PromptTemplate(template=template, input_variables=['question', 'docs'])
    formatted_prompt = prompt.format(question=query, docs=docs_page_content)
    response = llm.invoke(formatted_prompt)

    return response.content


def speak_text(text):
    # Initialize TTS with a model name
    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DCA")

    # Convert text to speech and save to a file
    audio_path = "audio/output.wav"
    tts.tts_to_file(text=text, file_path=audio_path)

    # Play the audio file
    wave_obj = sa.WaveObject.from_wave_file(audio_path)
    play_obj = wave_obj.play()
    play_obj.wait_done()

filepath = "data/hp.pdf" # Replace with your PDF path
vector_store = create_vector_store(filepath)
llm = ChatOpenAI(model="gpt-3.5-turbo")

# ChatBot:
print("Type 'bye' or 'exit' to end the chat: ")
user_input = input("You: ")
while user_input.lower() != 'bye' and user_input.lower() != 'exit':
    bot_response = get_response_to_query(user_input, vector_store, llm, 4)
    print("Assistant:", bot_response)
    speak_text(bot_response)  # Speak the bot response
    user_input = input("You: ")

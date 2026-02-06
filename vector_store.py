import os
from dotenv import load_dotenv
from pinecone import Pinecone,ServerlessSpec
from src.helper import load_pdf,filter_documents,chunk_text,download_embeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY").strip()
openai_api_key = os.getenv("OPENAI_API_KEY").strip()

# Setting up the environment variables
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["PINECONE_API_KEY"] = pinecone_api_key


extracted_data = load_pdf("data")                      # Extract text from pdf
filtered_data = filter_documents(extracted_data)       # filter the extracted data in away to get source and page_content(text)
chunks = chunk_text(filtered_data)                     # split the text into chunks

embeddings = download_embeddings()                     # download the embeddings from the HuggingFace model


# Intialize Pinecone-client(pc)
pc = Pinecone(api_key=pinecone_api_key)


# create index if not exists
index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384, #Dimension of the embeddings
        metric="cosine", #Metric for the embeddings
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)

# Load the chunks into the vector store
doc_search = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=index_name
)

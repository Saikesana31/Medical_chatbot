from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_pinecone import Pinecone
from typing import List,Dict
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


# Extract text from PDF
def load_pdf(data):
    loader = DirectoryLoader(
        data,
        glob = "*.pdf",
        loader_cls = PyPDFLoader)

    documents = loader.load()
    return documents


# Filter the documents
def filter_documents(docs:List[Document]) -> List[Document]:
    """
    Filter out the above documents and extract:
    source for metadata and page content for text
    """
    filtered_docs = []
    for doc in docs:
        src = doc.metadata.get("source")
        text = doc.page_content
        filtered_docs.append(
            Document(
            page_content=text,
            metadata={"source":src}))
    return filtered_docs


# split the text into chunks
def chunk_text(docs:List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20)
    chunks = text_splitter.split_documents(docs)
    return chunks



# Intialize the embedding model
def download_embeddings():
    """
    Download the embeddings from the HuggingFace model
    """
    embeddings= HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

# format the retrived docs for our context so that it can reduce hallucinations
def format_docs(docs):
    return "\n".join([doc.page_content for doc in docs])

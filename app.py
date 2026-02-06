from flask import Flask, render_template, request
from src.helper import download_embeddings,format_docs
from src.template import template
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

# Intialize the Flask app
app = Flask(__name__)

pinecone_api_key = os.getenv("PINECONE_API_KEY").strip()
openai_api_key = os.getenv("OPENAI_API_KEY").strip()

# Setting up the environment variables
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["PINECONE_API_KEY"] = pinecone_api_key

embeddings = download_embeddings()  # download the embeddings from the HuggingFace model

# Intialize the vector store
doc_search = PineconeVectorStore.from_existing_index(
    index_name="medical-chatbot",
    embedding=embeddings
)

# create a retriver to get the most relevant documents from the vector store
retriever = doc_search.as_retriever(
    search_type = "similarity",
    search_kwargs={"k":5},
)

# Intialize the LLM
llm = ChatOpenAI(model="gpt-5-nano-2025-08-07")

# Intialize the prompt template
prompt = ChatPromptTemplate.from_template(template)
 
# create a RAG chain to answer the question based on the documents
rag_chain = (
    {"context" : retriever | format_docs, "question" : RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# Intialize the chatbot
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods = ["GET","POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)

    response = rag_chain.invoke(input)
    print(response)
    return str(response)



# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)  
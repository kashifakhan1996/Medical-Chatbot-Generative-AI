from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain,retrieval_qa
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = download_hugging_face_embeddings()


index_name = "medicalbot"

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})


llm = OpenAI(temperature=0.4, max_tokens=500)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompts),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = retryInvoke({"input": msg})
    print("Response : ", response)
    if "answer" in response:
        return str(response["answer"])
    return []

def retryInvoke(inputData):
    import time
    import openai

    # Assuming rag_chain is your RetrievalQA chain
    max_retries = 6
    delay = 2  # Start with 2 seconds
    result = []
    for attempt in range(max_retries):
        try:
            result = rag_chain.invoke(inputData)
            print("Got result:", result)
            break
        except openai.RateLimitError:
            print(f"Rate limit hit. Retrying in {delay} seconds... (Attempt {attempt + 1})")
            time.sleep(delay)
            delay *= 2  # Exponential backoff
        except Exception as e:
            print("Other error:", e)
            break
    return result





if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
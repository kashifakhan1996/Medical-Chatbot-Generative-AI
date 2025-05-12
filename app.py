from flask import Flask,render_template,jsonify,request
from src.helper import load_pdf_file,text_split,download_huggingface_embeddings
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import system_prompts
import os

load_dotenv()
app = Flask(__name__)

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

os.environ['PINECONE_API_KEY']=PINECONE_API_KEY
os.environ['OPENAI_API_KEY']=OPENAI_API_KEY

index_name = 'medicalbot'
embeddings = download_huggingface_embeddings()

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retreiver = docsearch.as_retriever(search_type="similarity",search_kwargs={"k":3})

llm = OpenAI(temperature=0.4,max_tokens=500)

prompt = ChatPromptTemplate.from_messages(
    [
    ('system',system_prompts),
    ('human',"{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm,prompt)
rag_chain=create_retrieval_chain(retreiver,question_answer_chain)

@app.route('/')
def index():
    render_template('chat.html')

@app.route('/get',methods=['GET','POST'])
def index():
    msg = request.form['msg']
    input = msg
    print(input)
    response = rag_chain.invoke({"input":msg})
    print(response)
    return str(response['answer'])

if __name__== 'main':
    app.run(host='0.0.0.0',port=8080,debug=True)

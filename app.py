from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os 
from pinecone import Pinecone,ServerlessSpec
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import io
from langchain.embeddings import OpenAIEmbeddings
from langchain_groq import ChatGroq 
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from flask_cors import CORS




load_dotenv()

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  
        chunk_overlap=200,  
    )

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def createIndex(index_name):
    dimension = 1536  
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pc.create_index(name=index_name, 
                    dimension=dimension, 
                    metric="cosine",
                    spec=ServerlessSpec(
                            cloud="aws",
                            region="us-east-1"
                        ),
                    deletion_protection="disabled")
    
    return "Index created successfully"
    








def getChunks(directory):
    all_chunks = []
    
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory, filename)
            print(f"Processing: {pdf_path}")
            
            # Load the PDF using PyPDFLoader
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Split the documents into chunks
            for doc in documents:
                chunks = text_splitter.split_text(doc.page_content)
                all_chunks.extend(chunks)
    
    return all_chunks
    
            

    

def saveToPinecone(chunks,index_name):
    from langchain.vectorstores import Pinecone
    Pinecone.from_texts(
        chunks,
        index_name=index_name,
        embedding=embeddings_model
    )

   
def uploadFiles(files):
    uploaded_files = []
    for file in files:
        
        if file.filename == '' or not file.filename.endswith('.pdf'):
            return jsonify({"error": f"Invalid file: {file.filename}. Only PDF files are allowed."}), 400
        
        # Save the file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        uploaded_files.append(file.filename)
    
    print(jsonify({"message": "Files uploaded successfully", "uploaded_files": uploaded_files}))


def deleteAllFiles(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):  # Check if it's a file
            os.remove(file_path)

@app.route('/createindex', methods=['POST'])
def create_index():
    data = request.get_json()
    if not data or 'index_name' not in data:
        return jsonify({"error": "index_name is required"}), 400
    
    index_name = data['index_name']
    try:
        createIndex(index_name)
        response_message = f"Index '{index_name}' has been created successfully."
        return jsonify({"message": response_message}), 201
    except Exception as e:
        print(e)
        return jsonify({"error": "Index not created"}), 400




@app.route('/getindexes',methods=['GET'])
def getIndexes():
    from pinecone import Pinecone

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("documents")
    indexes=[]
    for index in pc.list_indexes():
        indexes.append(index.name)
    return jsonify({"Indexes":indexes})


        


@app.route('/upload', methods=['POST'])
def upload_files():

    if 'files' not in request.files:
        return jsonify({"error": "No files part in the request"}), 400
    request.get_data()
    
    files = request.files.getlist('files')
    index_name=request.form.get('index_name')

    uploadFiles(files)
    chunks = getChunks(app.config['UPLOAD_FOLDER'])
    saveToPinecone(chunks,index_name)
    deleteAllFiles(app.config['UPLOAD_FOLDER'])

    return jsonify({"message": "Files uploaded successfully"}), 200

@app.route('/generate', methods=['POST'])
def generate_response():
    data = request.get_json()
    question=data['question']
    index_name=data['index_name']
    from langchain.vectorstores import Pinecone
    vectorstore = Pinecone.from_existing_index(index_name, embeddings_model)
    llm=ChatGroq(groq_api_key=GROQ_API_KEY,
             model_name="llama-3.1-70b-versatile")
    prompt=ChatPromptTemplate.from_template(
                """
                Answer the questions based on the provided context only.
                Please provide the most accurate response based on the question
                <context>
                {context}
                <context>
                Questions:{input}

                """
            )
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=vectorstore.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    response=retrieval_chain.invoke({'input':question})
    print(response['answer'])

    return jsonify({"answer":response['answer']}),200

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"message": "Server is running"}), 200

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080)

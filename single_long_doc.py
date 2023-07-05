import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

os.environ["OPENAI_API_KEY"] = "sk-"

# Load the document
loader = PyPDFLoader('./docs/suhail_parakkal.pdf')
documents = loader.load()

# Lets split the data into chunks of 1000 characters, wwith an 
# overlap of 200 characters between the chunks, which helps to give 
# better results

text_splitter = CharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(documents)

# Lets create our vectorDB, using the OpenAIEmbeddings tranformer to create
# embeddings from our text chunks. We set all the db information to be stored
# inside the ./data directory, so it doesn't clutter up our source files

vectordb = Chroma.from_documents(
    documents,
    embedding=OpenAIEmbeddings(),
    persist_directory='./data'
)
vectordb.persist()

qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    retriever=vectordb.as_retriever(search_kwargs = {'k':7}),
    return_source_documents=True
)

# We can now execute queries against our Q&A chain
result = qa_chain({'query':'''What is this file about? 
                   What professional experience does the person have?'''})
print(result['result'])
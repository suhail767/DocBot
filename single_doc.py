import os
from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

os.environ["OPENAI_API_KEY"] = "sk-"

pdf_loader = PyPDFLoader('./docs/suhail_parakkal.pdf')
documents = pdf_loader.load()


chain = load_qa_chain(llm=OpenAI(), verbose=True)
query = 'What is this file? Who is this about? Summarise it?'
response = chain.run(input_documents = documents, question=query)
print(response)
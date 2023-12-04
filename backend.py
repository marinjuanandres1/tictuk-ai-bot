from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import DirectoryLoader
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
import os
import openai
import pinecone
from dotenv import load_dotenv
load_dotenv()

# Get your API keys from openai, you will need to create an account.
# Here is the link to get the keys: https://platform.openai.com/account/billing/overview
os.environ["OPENAI_API_KEY"] = os.getenv("openai_apikey")

#root_dir = "/Users/juanandresmarin/Documents/Coding/STREAMLIT_BOT/DOCS/knowledge_base.pdf"
root_dir = "https://marinjuanandres1.github.io/tictuk-ai-bot/knowledge_base.pdf"

def load_docs(root_dir):

  loader = PyPDFLoader(root_dir)
  pages = loader.load_and_split()

  return pages

docs = load_docs(root_dir)

# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings()

pinecone.init(api_key= os.getenv("pinecone_apikey"), environment="gcp-starter")

index_name = "aibot"

index = Pinecone.from_documents(docs, embeddings, index_name=index_name)

chain = load_qa_chain(OpenAI(), chain_type="stuff")

def get_similiar_docs(query, k=2, score=False):
  if score:
    similar_docs = index.similarity_search_with_score(query, k=k)
  else:
    similar_docs = index.similarity_search(query, k=k)
  return similar_docs


def get_answer(query):
  similar_docs = get_similiar_docs(query)
  answer = chain.run(input_documents=similar_docs, question=query)
  return answer

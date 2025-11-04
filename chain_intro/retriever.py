import dotenv
import os
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
os.environ["OPENAI_API_KEY"] = ''
from langchain_openai import OpenAIEmbeddings
dotenv.load_dotenv()
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# file path 
REVIEWS_CSV_PATH = "data/reviews.csv"
REVIEWS_CHROMA_PATH = "chroma_data"
OS_CSV_PATH = "data/Operating System.pdf"
cancer_CSV_PATH = "data/cancer.pdf"
OS_CHROMA_PATH = "chroma_data"

# load the file 
loader = CSVLoader(file_path=REVIEWS_CSV_PATH, source_column="review")
reviews = loader.load()
loader = PyPDFLoader(file_path=OS_CSV_PATH)
OS = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
documents = text_splitter.split_documents(OS)

# cancer file upload 
loader = PyPDFLoader(file_path=cancer_CSV_PATH)
cancer = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
document2 = text_splitter.split_documents(cancer)

load= WebBaseLoader("https://travelfoodatlas.com/ethiopian-foods")
docs = load.load()
# embed to vector form 
reviews_vector_db = Chroma.from_documents(
    reviews, OpenAIEmbeddings(), persist_directory=REVIEWS_CHROMA_PATH
)
OS_vector_db = Chroma.from_documents(
    documents, OpenAIEmbeddings(), persist_directory=OS_CHROMA_PATH
)

cancer_vector_db = Chroma.from_documents(
    document2, OpenAIEmbeddings(), persist_directory=OS_CHROMA_PATH
)

# store into vector database 
reviews_vector = Chroma(
     persist_directory=REVIEWS_CHROMA_PATH,
     embedding_function=OpenAIEmbeddings(),
 )
OS_vector = Chroma(
     persist_directory=OS_CHROMA_PATH,
     embedding_function=OpenAIEmbeddings(),
 )
# Test 
# question = """Has anyone complained about
#             communication with the hospital staff?"""
question = """what is cancer?"""
relevant_docs = OS_vector.similarity_search(question, k=3)

print(relevant_docs[0].page_content)

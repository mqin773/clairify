

import streamlit as st
import os

OPENAI_API_KEY = 'sk-proj-QdHDJ3Ou2e7hvjtmAFvrT3BlbkFJbvocSkcjswz4dYp4SxLT'
PINECONE_API_KEY = "0f30edf2-0cf8-4a8f-a7c9-d460decc2a32"
#from streamlit_app import reponse

from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
import os
from langchain.document_loaders import PyPDFLoader #allows for all types of data


directory = 'C:/Users/jduan/Downloads/P2024/'

logo1 = st.image("200.PNG")
st.logo("200.png")

def load_docs(path):
  #loader = DirectoryLoader(dir)
  loader = PyPDFLoader(path)
  docs = loader.load()
  return docs

######################
#from model.py import load_docs

st.title(f"cl:green[{"ai"}]rify")
st.subheader(f"Never get lost in a research paper again")
response = ""
documents = []


def save_uploadedfile(uploadedfile, path):
     with open(os.path.join(path,uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())
     return st.success("Saved File:{} to path".format(uploadedfile.name))

input = st.file_uploader("Upload a 'pdf' file", ['pdf'], False)

#################
import validators

#Website
from langchain.document_loaders import WebBaseLoader

def load_website(url):
  loader = WebBaseLoader(url)

  docs = loader.load()
  print(docs[0].page_content[:500])

#url=""
num = 6
#if st.button("Add a URL"):
url = st.text_input('Type URL link...')

print(url)
if len(url) >= 5:
    input = load_website(url)
    #print(input)

#Youtube video 
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers.audio import (
    OpenAIWhisperParser,
    OpenAIWhisperParserLocal,
)
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader

def load_video(url):
  save_dir="docs/youtube/"

  local = False
  if local:
    loader = GenericLoader(
        YoutubeAudioLoader(url, save_dir), OpenAIWhisperParserLocal()
    )
  else:
      loader = GenericLoader(YoutubeAudioLoader(url, save_dir), OpenAIWhisperParser())
  docs = loader.load()

  print(docs[0].page_content[0:500])

url2=""
num = 7
#if st.button("Add a YouTube Video"):

url2 = st.text_input('Type YouTube link...')

if len(url2) >= 5:
    url_list=[url2]
    input = load_video(url_list)
    


_ ="""
#Website
url = ""
url_website = False
validation = True
if st.button("Add a URL"):
  validation = False

num = 5
print(validation)

#Youtube Video

url2=""
validation2 = True
num2 = 100
if st.button("Add a YouTube URL"):
  validation2 = False

#Website


while not validation:
  url = st.text_input('The URL link', key=str(num))
  st.text("URL is invalid")
  validation = validators.url(url)
  url_website=True
  num+=1

from langchain.document_loaders import WebBaseLoader

def load_website(url):
  loader = WebBaseLoader(url)

  docs = loader.load()
  print(docs[0].page_content[:500])

if url_website:
    load_website(url)

#Youtube Video
url2=""
validation2 = True
num2 = 100
if st.button("Add a YouTube URL"):
  validation2 = False

while not validation2:
  url2 = st.text_input('YouTube link', key = str(num2))
  st.text("URL is invalid")
  validation2 = validators.url(url2)
  num2+=1

from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers.audio import (
    OpenAIWhisperParser,
    OpenAIWhisperParserLocal,
)
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader

def load_video(url):
  save_dir="docs/youtube/"

  local = False
  if local:
    loader = GenericLoader(
        YoutubeAudioLoader(url, save_dir), OpenAIWhisperParserLocal()
    )
  else:
      loader = GenericLoader(YoutubeAudioLoader(url, save_dir), OpenAIWhisperParser())
  docs = loader.load()

  print(docs[0].page_content[0:500])

url_list=[url2]
#load_video(url+list)
"""

######################

if input:
    file_details = {"FileName":input.name,"FileType":input.type}
    save_uploadedfile(input, 'C:/Users/jduan/Downloads/P2024/')
    documents = load_docs(os.path.join('C:/Users/jduan/Downloads/P2024/',input.name))

    with st.chat_message("assistant"):
        st.markdown("Ask me any question about the contents of your file!")

    if "messages" not in st.session_state:
        st.session_state.messages = []


    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("What is your question?")

    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        #########
        def split_docs(documents, chunk_size=1000, chunk_overlap=20):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            docs = text_splitter.split_documents(documents)
            return docs

        docs = split_docs(documents)
        print(len(docs))

        os.environ['OPENAI_API_KEY'] = 'sk-proj-QdHDJ3Ou2e7hvjtmAFvrT3BlbkFJbvocSkcjswz4dYp4SxLT'

        embeddings = OpenAIEmbeddings()

        query_result = embeddings.embed_query("Hello world")
        #len(query_result)

        from pinecone import Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)

        index_name = 'langchain2'
        index = pc.Index(index_name)
        index.describe_index_stats()

        from langchain.embeddings.openai import OpenAIEmbeddings
        model_name = 'text-embedding-ada-002'
        embed = OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY)

        from langchain.vectorstores import Pinecone
        #vectorstore = Pinecone(index, embeddings.embed_query, "text")
        os.environ["PINECONE_API_KEY"] = "0f30edf2-0cf8-4a8f-a7c9-d460decc2a32"

        vectorstore = Pinecone.from_documents(
                docs,
                index_name='langchain2',
                embedding=embeddings
            )

        query = prompt
        vectorstore.similarity_search(query, k=3)

        from langchain.chains import RetrievalQAWithSourcesChain

        from langchain.chat_models import ChatOpenAI
        from langchain.chains import RetrievalQA


        # completion llm
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name='gpt-4',
            temperature=0.0
        )
        qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )


        response = qa_with_sources(query)['answer']
        
        ##########
    # response = response

        st.markdown("""
            <style>
                [data-testid="column"]:nth-child(2){
                    background-color: lightgrey;
                }
            </style>
            """, unsafe_allow_html=True
        )

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

######################################################


#documents = load_docs(file_path) ####TRANSFER


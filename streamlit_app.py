import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.document_loaders import PyMuPDFLoader
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain import PromptTemplate  


def load_embedding_model(model_path, normalize_embedding=True):
    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device':'cpu'}, # here we will run the model with CPU only
        encode_kwargs = {
            'normalize_embeddings': normalize_embedding # keep True to compute cosine similarity
        }
    )

def document_data(query, chat_history):
   pdf_path = './bau_zh.pdf'
   loader = PyMuPDFLoader(file_path=pdf_path)
   doc = loader.load()

   text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap= 100)
   #separators=["\n\n","\n"," ","","."]) 
   text = text_splitter.split_documents(documents= doc) 

   #Adding additional docs
   loader = PyMuPDFLoader("./bau_zh_2.pdf")
   #doc_new = loader.load()
   text_new = loader.load_and_split(text_splitter)
   text += text_new

   # creating embeddings using HuggingFace embeddings
   embeddings = load_embedding_model(model_path="all-MiniLM-L6-v2")

   vectorstore = FAISS.from_documents(text, embeddings)
   vectorstore.save_local("vectors")
   print("Embeddings successfully saved in vector Database and saved locally")

   # Loading the saved embeddings 
   loaded_vectors=FAISS.load_local("vectors", embeddings, allow_dangerous_deserialization=True )

   llm = Ollama(model="phi3:mini", temperature=0)

   #Define the Prompt template
   chatTemplate = """
      You are now Assistant, an AI expert in construction loaw in Zürich, 
      who speaks and answers questions in the german language. 
      The user speaks german and will ask question about 
      upcoming construction projects in Zürich in german. 
      The Goal is to assist the user in learning about building regulations by giving him 
      answers and guidance in german.
      **Important:** Answer the question based on the chat history(delimited by <hs></hs>) and context(delimited by <ctx> </ctx>) below.
      Don’t justify your answers. Don’t give any information not mentioned in the context delimited by <ctx> </ctx>.
      -----------
      <ctx>
      {context}
      </ctx>
      -----------
      <hs>
      {chat_history}
      </hs>
      -----------
      Question: {question}
      Answer: 
   """

   promptHist = PromptTemplate(
      input_variables=["context", "question", "chat_history"],
      template=chatTemplate
   ) 
   # ConversationalRetrievalChain 
   qa = ConversationalRetrievalChain.from_llm(
        llm=llm, #OpenAI()
        retriever= loaded_vectors.as_retriever(),
        return_source_documents=True,
        verbose=True,
        combine_docs_chain_kwargs={'prompt': promptHist}
    )

    
   return qa({"question":query, "chat_history":chat_history})

if __name__ == '__main__':

   st.header("Bauanfrage ChatBot")
   # ChatInput
   prompt = st.chat_input("Frage zu künftigen Baugesuchen eingeben")

   if "user_prompt_history" not in st.session_state:
       st.session_state["user_prompt_history"]=[]
   if "chat_answers_history" not in st.session_state:
       st.session_state["chat_answers_history"]=[]
   if "chat_history" not in st.session_state:
       st.session_state["chat_history"]=[]

   if prompt:
      with st.spinner("Antwort wird generiert..."):
         output=document_data(query=prompt, chat_history = st.session_state["chat_history"])

         # Storing the questions, answers and chat history

         st.session_state["chat_answers_history"].append(output['answer'])
         st.session_state["user_prompt_history"].append(prompt)
         st.session_state["chat_history"].append((prompt,output['answer']))

    # Displaying the chat history
  
   if st.session_state["chat_answers_history"]:
      for i, j in zip(st.session_state["chat_answers_history"],st.session_state["user_prompt_history"]):
         message1 = st.chat_message("user")
         message1.write(j)
         message2 = st.chat_message("assistant")
         message2.write(i)

import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import AzureChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
# from dotenv import load_dotenv
# from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader,TextLoader
from langchain.prompts import PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain

# load_dotenv()
# OPENAI_API_TYPE = os.environ['OPENAI_API_TYPE']
# OPENAI_API_BASE = os.environ['OPENAI_API_BASE']
# OPENAI_API_VERSION = os.environ['OPENAI_API_VERSION']
# OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
# DEPLOYEMENT_NAME = os.environ['DEPLOYEMENT_NAME']
# OPENAI_API_TYPE = 'gpt-3'
# OPENAI_API_BASE = '//api.openai.com'
# OPENAI_API_VERSION = 'v1'
GOOGLE_API_KEY = "AIzaSyC5jVGT9OHx4soEsliU60ByZsieobJPRms"
genai.configure(api_key=GOOGLE_API_KEY)
class chat_gen():
    def __init__(self):
        self.chat_history=[]

    def load_doc(self,document_path):
        loader = PyPDFLoader(document_path)
        documents = loader.load()
        # Split document in chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
        docs = text_splitter.split_documents(documents=documents)
        # embeddings = uggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # Create vectors
        vectorstore = FAISS.from_documents(docs, embeddings)
        # Persist the vectors locally on disk
        vectorstore.save_local("faiss_index_datamodel")

        # Load from local storage
        persisted_vectorstore = FAISS.load_local("faiss_index_datamodel", embeddings,allow_dangerous_deserialization=True)
        return persisted_vectorstore

 
    def load_model(self,):
        llm = ChatGoogleGenerativeAI(model = "gemini-pro",
                            temperature=0.0
                            )

        # Define your system instruction
        system_instruction = """ As an AI assistant, you must answer the query from the user from the retrieved content,
        if no relavant information is available, answer the question by using your knowledge about the topic"""

        # Define your template with the system instruction
        template = (
            f"{system_instruction} "
            "Combine the chat history{chat_history} and follow up question into "
            "a standalone question to answer from the {context}. "
            "Follow up question: {question}"
        )

        prompt = PromptTemplate.from_template(template)
        # chain = ConversationalRetrievalChain.from_llm(
        #     llm=llm,
        #     retriever=self.load_doc("GenAI.pdf").as_retriever(),
        #     #condense_question_prompt=prompt,
        #     combine_docs_chain_kwargs={'prompt': prompt},
        #     chain_type="stuff",
        # )
        chain = load_qa_chain(llm= llm,
            retriever=self.load_doc("GenAI.pdf").as_retriever(),
            #condense_question_prompt=prompt,
            combine_docs_chain_kwargs={'prompt': prompt},
            chain_type="stuff",
        )
        # chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
        return chain

    def ask_pdf(self,query):
        result = self.load_model()({"question":query,"chat_history": self.chat_history})
        self.chat_history.append((query, result["answer"]))
        #print(result)
        return result['answer']


if __name__ == "__main__":
    chat = chat_gen()
    # print(chat.ask_pdf("who is charlie chaplin"))
    # print(chat.ask_pdf("when did he die?"))

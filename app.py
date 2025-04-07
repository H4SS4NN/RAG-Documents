import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

def load_documents():
    """Charge tous les documents du dossier data"""
    documents = []
    data_dir = "data"
    
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_dir, filename)
            loader = TextLoader(file_path)
            documents.extend(loader.load())
    
    return documents

def create_vector_store(documents):
    """Crée un magasin de vecteurs à partir des documents"""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    texts = text_splitter.split_documents(documents)
    embeddings = OllamaEmbeddings(model="mistral")
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

def create_qa_chain(vectorstore):
    """Crée une chaîne de questions-réponses"""
    llm = Ollama(model="mistral")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    return qa_chain

def main():
    # Charger les documents
    print("Chargement des documents...")
    documents = load_documents()
    
    # Créer le magasin de vecteurs
    print("Création du magasin de vecteurs...")
    vectorstore = create_vector_store(documents)
    
    # Créer la chaîne de questions-réponses
    print("Création de la chaîne de questions-réponses...")
    qa_chain = create_qa_chain(vectorstore)
    
    # Interface utilisateur simple
    print("\nSystème RAG sur la Seconde Guerre mondiale")
    print("Tapez 'quit' pour quitter")
    
    while True:
        query = input("\nVotre question: ")
        if query.lower() == 'quit':
            break
            
        try:
            result = qa_chain.invoke({"query": query})
            print(f"\nRéponse: {result['result']}")
        except Exception as e:
            print(f"Erreur: {str(e)}")

if __name__ == "__main__":
    main() 
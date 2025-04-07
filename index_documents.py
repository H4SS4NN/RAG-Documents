from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS
from loaders import load_documents

def index_documents():
    """Indexe les documents une seule fois et sauvegarde l'index FAISS"""
    # Charger tous les documents
    print("Chargement des documents...")
    documents = load_documents()

    # Découper les documents
    print("Découpage des documents...")
    splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separator="\n",
        length_function=len
    )
    texts = splitter.split_documents(documents)

    # Créer les embeddings et l'index FAISS
    print("Création des embeddings et de l'index FAISS...")
    embeddings = OllamaEmbeddings(model="mistral")
    vectorstore = FAISS.from_documents(texts, embeddings)

    # Sauvegarder l'index
    print("Sauvegarde de l'index...")
    vectorstore.save_local("my_faiss_index")
    print("Indexation terminée et sauvegardée dans 'my_faiss_index'")

if __name__ == "__main__":
    index_documents() 
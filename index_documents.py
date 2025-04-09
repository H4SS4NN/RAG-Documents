from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from loaders import load_documents
from tqdm import tqdm
import os
import json
import hashlib
import time

def get_file_hash(file_path):
    """Calcule le hash d'un fichier pour détecter les modifications"""
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def load_indexed_files():
    """Charge la liste des fichiers déjà indexés"""
    if os.path.exists('indexed_files.json'):
        with open('indexed_files.json', 'r') as f:
            return json.load(f)
    return {}

def save_indexed_files(indexed_files):
    """Sauvegarde la liste des fichiers indexés"""
    with open('indexed_files.json', 'w') as f:
        json.dump(indexed_files, f)

def format_size(size_bytes):
    """Convertit la taille en octets en format lisible"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def format_time(seconds):
    """Convertit les secondes en format lisible"""
    if seconds < 60:
        return f"{seconds:.1f} secondes"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f} minutes"
    hours = minutes / 60
    return f"{hours:.1f} heures"

def index_documents():
    """Indexe les documents une seule fois et sauvegarde l'index FAISS"""
    start_time = time.time()
    
    # Charger les fichiers déjà indexés
    indexed_files = load_indexed_files()
    
    # Charger tous les documents
    print("\n🔍 Analyse des documents...")
    documents = []
    new_files = []
    total_size = 0
    
    for filename in os.listdir("data"):
        if filename.endswith(('.txt', '.pdf', '.docx', '.csv', '.html', '.json', '.md', '.ppt', '.pptx')):
            file_path = os.path.join("data", filename)
            file_size = os.path.getsize(file_path)
            total_size += file_size
            file_hash = get_file_hash(file_path)
            
            if filename not in indexed_files or indexed_files[filename] != file_hash:
                new_files.append((filename, file_size))
                indexed_files[filename] = file_hash
    
    if not new_files:
        print("✅ Tous les documents sont déjà indexés à jour.")
        return
    
    print(f"\n📚 Nouveaux documents à indexer ({len(new_files)}):")
    for filename, size in new_files:
        print(f"- {filename} ({format_size(size)})")
    print(f"📊 Taille totale des nouveaux documents: {format_size(total_size)}")
    
    # Charger les nouveaux documents
    print("\n📥 Chargement des nouveaux documents...")
    documents = load_documents()
    
    if not documents:
        print("❌ Aucun nouveau document à indexer.")
        return

    # Découper les documents
    print("\n✂️ Découpage des documents...")
    splitter = CharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separator="\n",
        length_function=len
    )
    texts = splitter.split_documents(documents)
    print(f"📝 Nombre total de chunks créés: {len(texts)}")
    print(f"📏 Taille moyenne des chunks: {sum(len(t.page_content) for t in texts) / len(texts):.0f} caractères")

    # Créer les embeddings et l'index FAISS
    print("\n🤖 Création des embeddings et de l'index FAISS...")
    print("⚡ Utilisation du modèle all-MiniLM-L6-v2 pour des embeddings rapides et efficaces")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Vérifier si l'index existe et est compatible
    if os.path.exists("my_faiss_index"):
        try:
            print("📂 Chargement de l'index existant...")
            vectorstore = FAISS.load_local(
                "my_faiss_index",
                embeddings,
                allow_dangerous_deserialization=True
            )
            print("✅ Index existant chargé avec succès")
        except Exception as e:
            print("⚠️ L'index existant n'est pas compatible avec le nouveau modèle d'embeddings")
            print("🔄 Recréation d'un nouvel index...")
            vectorstore = FAISS.from_documents(texts, embeddings)
    else:
        print("🆕 Création d'un nouvel index...")
        vectorstore = FAISS.from_documents(texts, embeddings)

    # Sauvegarder l'index
    print("\n💾 Sauvegarde de l'index...")
    vectorstore.save_local("my_faiss_index")
    save_indexed_files(indexed_files)
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"\n✅ Indexation terminée en {format_time(duration)}")
    print(f"📁 Index sauvegardé dans 'my_faiss_index'")
    print(f"📊 Taille de l'index: {format_size(os.path.getsize('my_faiss_index/index.faiss'))}")

if __name__ == "__main__":
    index_documents() 
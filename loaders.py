import os
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    CSVLoader,
    UnstructuredHTMLLoader,
    JSONLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader
)
from langchain.schema import Document

def load_documents(data_dir="data"):
    """Charge tous les documents du dossier data, quel que soit leur format"""
    documents = []
    
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        
        try:
            if filename.endswith(".txt"):
                loader = TextLoader(file_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = filename
                    doc.metadata["type"] = "text"
                documents.extend(docs)
            elif filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = filename
                    doc.metadata["type"] = "pdf"
                    doc.metadata["page"] = doc.metadata.get("page", "?")
                documents.extend(docs)
            elif filename.endswith((".doc", ".docx")):
                loader = UnstructuredWordDocumentLoader(file_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = filename
                    doc.metadata["type"] = "word"
                documents.extend(docs)
            elif filename.endswith(".csv"):
                loader = CSVLoader(file_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = filename
                    doc.metadata["type"] = "csv"
                documents.extend(docs)
            elif filename.endswith((".html", ".htm")):
                loader = UnstructuredHTMLLoader(file_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = filename
                    doc.metadata["type"] = "html"
                documents.extend(docs)
            elif filename.endswith(".json"):
                loader = JSONLoader(file_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = filename
                    doc.metadata["type"] = "json"
                documents.extend(docs)
            elif filename.endswith(".md"):
                loader = UnstructuredMarkdownLoader(file_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = filename
                    doc.metadata["type"] = "markdown"
                documents.extend(docs)
            elif filename.endswith((".ppt", ".pptx")):
                loader = UnstructuredPowerPointLoader(file_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = filename
                    doc.metadata["type"] = "powerpoint"
                documents.extend(docs)
            else:
                print(f"Format non supporté: {filename}")
        except Exception as e:
            print(f"Erreur lors du chargement de {filename}: {str(e)}")
    
    return documents 
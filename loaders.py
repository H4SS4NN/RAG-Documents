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

def load_documents(data_dir="data"):
    """Charge tous les documents du dossier data, quel que soit leur format"""
    documents = []
    
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        
        try:
            if filename.endswith(".txt"):
                documents.extend(TextLoader(file_path).load())
            elif filename.endswith(".pdf"):
                documents.extend(PyPDFLoader(file_path).load())
            elif filename.endswith((".doc", ".docx")):
                documents.extend(UnstructuredWordDocumentLoader(file_path).load())
            elif filename.endswith(".csv"):
                documents.extend(CSVLoader(file_path).load())
            elif filename.endswith((".html", ".htm")):
                documents.extend(UnstructuredHTMLLoader(file_path).load())
            elif filename.endswith(".json"):
                documents.extend(JSONLoader(file_path).load())
            elif filename.endswith(".md"):
                documents.extend(UnstructuredMarkdownLoader(file_path).load())
            elif filename.endswith((".ppt", ".pptx")):
                documents.extend(UnstructuredPowerPointLoader(file_path).load())
            else:
                print(f"Format non supporté: {filename}")
        except Exception as e:
            print(f"Erreur lors du chargement de {filename}: {str(e)}")
    
    return documents 
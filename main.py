from langchain.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

def main():
    # Vérifier si l'index existe
    if not os.path.exists("my_faiss_index"):
        print("L'index FAISS n'existe pas. Veuillez d'abord exécuter index_documents.py")
        return

    # Charger l'index FAISS
    print("Chargement de l'index FAISS...")
    embeddings = OllamaEmbeddings(model="mistral")
    vectorstore = FAISS.load_local(
        "my_faiss_index", 
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Créer le prompt personnalisé
    template = """Utilise uniquement les informations suivantes pour répondre à la question. 
    Si la réponse n'est pas dans les informations fournies, dis simplement "Je ne peux pas répondre à cette question avec les informations dont je dispose."

    Contexte: {context}

    Question: {question}

    Réponse: """

    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    # Créer la chaîne de questions-réponses
    print("Création de la chaîne de questions-réponses...")
    llm = Ollama(model="mistral")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    # Interface utilisateur
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
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
import os

def main():
    # Vérifier si l'index FAISS existe
    if not os.path.exists("my_faiss_index"):
        print("❌ L'index FAISS n'existe pas. Veuillez d'abord exécuter index_documents.py")
        return

    print("Chargement de l'index FAISS...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(
        "my_faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Créer la chaîne de question-réponse
    llm = OllamaLLM(model="mistral")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True
    )

    print("\n💬 Posez vos questions (tapez 'quit' pour quitter):")
    while True:
        query = input("\nQuestion: ")
        if query.lower() == 'quit':
            break

        # Récupérer les documents pertinents
        relevant_docs = vectorstore.similarity_search(query, k=5)
        
        # Afficher le contexte qui sera utilisé
        print("\n📚 Contexte utilisé pour la réponse:")
        for i, doc in enumerate(relevant_docs, 1):
            source = doc.metadata.get("source", "Inconnu")
            page = doc.metadata.get("page", "N/A")
            print(f"\n{i}. Document: {source}")
            if page != "N/A":
                print(f"   Page: {page}")
            print(f"   Extrait: {doc.page_content[:200]}...")
        
        # Générer la réponse
        print("\n🤖 Génération de la réponse...")
        result = qa_chain.invoke({"query": query})
        print("\nRéponse:", result["result"])
        
        # Afficher les sources utilisées
        if result["source_documents"]:
            print("\n📖 Sources utilisées:")
            for i, doc in enumerate(result["source_documents"], 1):
                source = doc.metadata.get("source", "Inconnu")
                page = doc.metadata.get("page", "N/A")
                print(f"\n{i}. Document: {source}")
                if page != "N/A":
                    print(f"   Page: {page}")
                print(f"   Extrait: {doc.page_content[:200]}...")

if __name__ == "__main__":
    main() 
# RAG-Mini : Système de Questions-Réponses sur la Seconde Guerre mondiale

Un système RAG (Retrieval-Augmented Generation) simple et efficace qui permet de poser des questions sur la Seconde Guerre mondiale en utilisant différents types de documents comme source d'information.

## 🚀 Fonctionnalités

- Support de multiples formats de documents (.txt, .pdf, .docx, .csv, .html, .json, .md, .ppt)
- Utilisation d'Ollama pour le traitement local
- Indexation efficace avec FAISS
- Interface simple en ligne de commande
- Réponses basées uniquement sur les documents fournis

## 📋 Prérequis

- Python 3.8+
- Ollama installé sur votre machine
- Modèle Mistral téléchargé via Ollama

## 🛠 Installation

1. Clonez le repository :
```bash
git clone https://github.com/votre-username/rag-mini.git
cd rag-mini
```

2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

3. Téléchargez le modèle Mistral via Ollama :
```bash
ollama pull mistral
```

## 📁 Structure du Projet

```
rag-mini/
├── data/                  # Dossier contenant les documents sources
├── loaders.py            # Gestionnaire de chargement des documents
├── index_documents.py    # Script d'indexation des documents
├── main.py              # Script principal pour les questions-réponses
└── requirements.txt     # Dépendances du projet
```

## 🚀 Utilisation

1. Placez vos documents dans le dossier `data/`. Les formats supportés sont :
   - .txt (fichiers texte)
   - .pdf (documents PDF)
   - .doc/.docx (documents Word)
   - .csv (fichiers CSV)
   - .html/.htm (pages web)
   - .json (fichiers JSON)
   - .md (fichiers Markdown)
   - .ppt/.pptx (présentations PowerPoint)

2. Indexez les documents (à faire une seule fois) :
```bash
python index_documents.py
```

3. Lancez le système de questions-réponses :
```bash
python main.py
```

4. Posez vos questions sur la Seconde Guerre mondiale !

## 🔧 Configuration

Le système utilise par défaut :
- Le modèle Mistral pour les embeddings et les réponses
- Une taille de chunk de 1000 caractères
- Un chevauchement de 200 caractères entre les chunks

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
- Ouvrir une issue pour signaler un bug
- Proposer une amélioration via une pull request
- Ajouter de nouveaux types de documents supportés

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 🙏 Remerciements

- [LangChain](https://github.com/langchain-ai/langchain) pour le framework
- [Ollama](https://ollama.ai/) pour les modèles locaux
- [FAISS](https://github.com/facebookresearch/faiss) pour l'indexation vectorielle 
import os
import shutil
from pathlib import Path
from itertools import chain
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma


# Configuration
DATA_DIR = "data"
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"   
COLLECTION_NAME = "rag_collection"

def load_documents(directory: str):
    """Load PDFs and text files from directory."""
    docs = []
    data_path = Path(directory)
    header_path = data_path / "header"
    education_path = data_path / "education"
    jobs_path = data_path / "jobs"
    projects_path = data_path / "projects"
    skills_path = data_path / "skills"
    resumes_path = data_path / "old_resumes"

    # header
    header_files = chain(header_path.rglob("*.txt"),header_path.rglob("*.md"))
    for header in header_files:
        loader = TextLoader(str(header), encoding="utf-8")
        loaded = loader.load()
        if not loaded:
            print(f"No header file detected. Resume will have no name, number, email, github unless old resume was provided.")
        for doc in loaded:
            doc.metadata["source"] = str(header)
            doc.metadata["doc_type"] = "header"
            docs.append(doc)

    # education
    ed_files = chain(education_path.rglob("*.txt"),education_path.rglob("*.md"))
    for edu_file in ed_files:
        loader = TextLoader(str(edu_file), encoding="utf-8")
        loaded = loader.load()
        if not loaded:
            print(f"No education file detected. Resume will have no education listed.")
        for doc in loaded:
            doc.metadata["source"] = str(edu_file)
            doc.metadata["doc_type"] = "education"
            docs.append(doc)
    
    # jobs
    job_files = chain(jobs_path.rglob("*.txt"),jobs_path.rglob("*.md"))
    for job in job_files:
        loader = TextLoader(str(job), encoding="utf-8")
        loaded = loader.load()
        if not loaded:
            print(f"No job files detected. Resume will have no related experience listed.")
        for doc in loaded:
            doc.metadata["source"] = str(job)
            doc.metadata["doc_type"] = "jobs"
            docs.append(doc)

    # projects
    project_files = chain(projects_path.rglob("*.txt"),projects_path.rglob("*.md"))
    for project in project_files:
        loader = TextLoader(str(project), encoding="utf-8")
        loaded = loader.load()
        if not loaded:
            print(f"No project files detected. Resume will have no projects listed.")
        for doc in loaded:
            doc.metadata["source"] = str(project)
            doc.metadata["doc_type"] = "projects"
            docs.append(doc)

    # skills
    skill_files = chain(skills_path.rglob("*.txt"),skills_path.rglob("*.md"))
    for skill in skill_files:
        loader = TextLoader(str(skill), encoding="utf-8")
        loaded = loader.load()
        if not loaded:
            print(f"No skill files detected. Resume will have no related skills listed.")
        for doc in loaded:
            doc.metadata["source"] = str(skill)
            doc.metadata["doc_type"] = "skills"
            docs.append(doc)
            
    
    # old resume loader
    for pdf_file in resumes_path.rglob("*.pdf"):
        loader = PyPDFLoader(str(pdf_file))
        loaded = loader.load()
        for doc in loaded:
            doc.metadata["source"] = str(pdf_file)
            doc.metadata["doc_type"] = "resume"
            docs.append(doc)
     
    print(f"Loaded {len(docs)} documents from {directory}")
    return docs


def main():
    # Clear existing DB if you want a fresh start (comment out if updating)
    if os.path.exists(CHROMA_DIR):
        print("Removing old Chroma DB for fresh ingest...")
        shutil.rmtree(CHROMA_DIR)  # Uncomment if you want full reset
    
    # Load and split documents
    docs = load_documents(DATA_DIR)
    if not docs:
        print("No documents found in data/ folder!")
        return
    
    splitter = RecursiveCharacterTextSplitter(
            chunk_size=6000,
            chunk_overlap=0,          
            separators=["\n\n## ", "\n\n", "\n",  ". "]
        )


    chunks = splitter.split_documents(docs)


    # Embed and store
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME
    )
    print(f"Successfully ingested into Chroma at {CHROMA_DIR}")

if __name__ == "__main__":
    main()
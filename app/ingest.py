from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.vector_store import get_chroma_db_collection
from app.config import DATA_PATH


# loading the document
def load_documents():
    file_loader = PyPDFDirectoryLoader(DATA_PATH)
    return file_loader.load()


# splitting the document
def split_document_to_chunks(raw_documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(raw_documents)
 
    documents = []
    metadata = []
    ids = []  

    for i, chunk in enumerate(chunks):
        documents.append(chunk.page_content)
        ids.append("ID"+str(i))
        metadata.append(chunk.metadata)
 

    return {
        "documents": documents,
        "ids": ids,
        "metadata": metadata
    }


def create_collection_chroma_db():
    collection = get_chroma_db_collection()
    raw_documents = load_documents()
    data = split_document_to_chunks(raw_documents)
 
    collection.upsert(
        documents=data["documents"],
        metadatas=data["metadata"],
        ids=data["ids"]
    )

if __name__ == "__main__":
    create_collection_chroma_db()
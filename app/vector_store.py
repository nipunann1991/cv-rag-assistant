import chromadb
from app.config import CHROMA_PATH, COLLECTION_NAME

def get_chroma_db_collection():
    chroma_db_client = chromadb.PersistentClient(path=CHROMA_PATH)
    return chroma_db_client.get_or_create_collection(name=COLLECTION_NAME)

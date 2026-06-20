import os
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")
CHROMA_PATH = os.getenv("CHROMA_PATH")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

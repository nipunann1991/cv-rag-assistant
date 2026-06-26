# CV RAG Assistant

A RAG chatbot that answers questions about Nirmal Nipuna Nanayakkara's professional profile, skills, work experience, education, certifications, and projects using LLM.

🤖 **Live Demo:** https://rag-assistant-chatbot.vercel.app/

## Features

- Read the CV/PDF documents
- Extract text from PDF files
- Split content into searchable chunks
- Generate embeddings using OpenAI
- Store vectors in ChromaDB
- Ask questions and retrieve relevant CV context
- FastAPI backend with REST endpoints

## Tech Stack

- Python
- FastAPI
- OpenAI
- LangChain
- ChromaDB
- PyPDF
- Railway

### Install dependencies

```bash
pip install -r requirements.txt
```

### Create a `.env` file

```env
OPENAI_API_KEY=your_openai_api_key
```

### Run the application

```bash
uvicorn main:app --reload
```

The API will be available at:

```
http://127.0.0.1:8000
```

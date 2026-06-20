from openai import OpenAI
from dotenv import load_dotenv
from app.vector_store import get_chroma_db_collection
from app.config import OPENAI_API_KEY

load_dotenv()

print(OPENAI_API_KEY)

def main():
    user_query = input("What do you want to know about?\n\n")
    context = ask_question(user_query)

    print("\n\n---------------------\n\n")
    print(context)


def ask_question(user_query: str) -> str:
    collection = get_chroma_db_collection()

    print(OPENAI_API_KEY)

    results = collection.query(
        query_texts=[user_query],
        n_results=10
    )
 
    return call_openai(str(results['documents']), user_query)


def call_openai(results, user_query) -> str:

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        system_prompt = f"""
        You are an advanced RAG assistant for my CV and professional profile.

        Rules:
        - Answer only using the provided context.
        - Do not make up facts.
        - If the answer is not available, say: "I don't know."
        - Mention relevant source numbers when useful.
        - Keep answers professional and concise.

        Context:
        {results}
        """

        response = client.chat.completions.create(
            model="gpt-4o",
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}    
            ],
            temperature=0.2
        )
        
        return response.choices[0].message.content
    
    except Exception as error:
        return f"Unexpected error: {error}"


if __name__ == "__main__":
    main()
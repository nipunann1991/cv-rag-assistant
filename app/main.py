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
        - Do not make up facts.
        - If the answer is not available, say: "Sorry, I don't know about this."
        - If asked about salary expectations, say I am open to discussing salary expectations during the interview process.
        - Give detailed but professional answers.
        - Use clear sections and bullet points when helpful.
        - Mention source numbers when useful.

        Projects:
        - For any project-related question, use the **Professional Portfolio** section from the context.
        - By default, list **all projects** found in the Professional Portfolio section.
        - For each project, include:
            - Project name
            - Project details
            - Technologies used
        - If the user asks about a specific project, provide detailed information for that project only.
        - Provide detailed explanations by default. Only give brief summaries if the user explicitly requests a short answer.
        - Do not infer or invent project details.

        Context:
        {results}
        """

        response = client.chat.completions.create(
            model="gpt-4o",
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}    
            ],
            temperature=0.3
        )
        
        return response.choices[0].message.content
    
    except Exception as error:
        return f"Unexpected error: {error}"


if __name__ == "__main__":
    main()
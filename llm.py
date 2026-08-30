import os
from dotenv import load_dotenv
from openai import OpenAI


# =========================================================
# 1. LOAD API KEY
# =========================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is not set.")


# =========================================================
# 2. OPENAI CLIENT
# =========================================================

client = OpenAI(api_key=api_key)


# =========================================================
# 3. CHECK IF QUESTION IS ABOUT GENAI
# =========================================================

def is_genai_question(user_input):
    print("====================================")
    print("CLASSIFIER INPUT:", user_input)

    response = client.responses.create(
        model="gpt-4o-mini",
        input=f"""
Classify the following question.

Return ONLY YES or NO.

YES = the question is directly related to Generative AI,
LLMs, ChatGPT, OpenAI, prompt engineering, RAG, embeddings,
AI agents, fine-tuning, tokens, transformers used in GenAI,
multimodal GenAI, text/image/audio generation, or GenAI applications.

NO = anything unrelated to Generative AI.

Question:
{user_input}
"""
    )

    result = response.output_text.strip().upper()

    print("CLASSIFIER RESULT:", result)
    print("====================================")

    return result == "YES"


# =========================================================
# 4. GENERATE GENAI ANSWER
# =========================================================

def get_response(messages):
    print("GET_RESPONSE CALLED")

    user_input = messages[-1]["content"]

    print("USER INPUT:", user_input)

    if not is_genai_question(user_input):
        print("NON-GENAI QUESTION - BLOCKED")
        return (
            "Sorry, I can only help with Generative AI topics. "
            "Please ask me a question about GenAI, LLMs, "
            "prompt engineering, RAG, AI agents, or another "
            "GenAI topic."
        )

    print("GENAI QUESTION - SENDING TO TUTOR")

    response = client.responses.create(
        model="gpt-4o-mini",
        instructions="""
You are a Generative AI tutor.

Only answer questions related to Generative AI.

Relevant topics include:
- Generative AI
- AI
- LLMs
- ChatGPT
- OpenAI
- Prompt engineering
- Tokens
- Context windows
- Embeddings
- Vector databases
- RAG
- Fine-tuning
- AI agents
- Transformers in GenAI
- Multimodal AI
- Text generation
- Image generation
- Audio generation
- GenAI applications
- GenAI safety

Answer GenAI questions using simple language,
clear explanations, practical examples, and best practices.

The topic filter has already determined that the user's
question is related to Generative AI.
""",
        input=messages,
        temperature=0.5,
    )

    return response.output_text

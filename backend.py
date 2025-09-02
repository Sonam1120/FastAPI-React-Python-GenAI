# if you dont use pipenv uncomment the following:
# from dotenv import load_dotenv
# load_dotenv()

# Step1: Setup Pydantic Model (Schema Validation)
from pydantic import BaseModel
from typing import List
from fastapi import FastAPI
from ai_agent import get_response_from_ai_agent

class RequestState(BaseModel):
    model_name: str
    model_provider: str
    system_prompt: str
    messages: List[str]
    allow_search: bool

# Allowed models
ALLOWED_MODEL_NAMES = [
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
    "llama-3.3-70b-versatile",
    "gpt-4o-mini"
]

# FastAPI App
app = FastAPI(title="LangGraph AI Agent")

@app.post("/chat")
def chat_endpoint(request: RequestState):
    """
    API Endpoint to interact with the Chatbot using LangGraph and search tools.
    It dynamically selects the model specified in the request.
    """

    if request.model_name not in ALLOWED_MODEL_NAMES:
        return {"status": "error", "data": "❌ Invalid model name. Kindly select a valid AI model."}

    # Take first message as query (avoid list issue for Groq/OpenAI/Gemini APIs)
    if not request.messages or not request.messages[0].strip():
        return {"status": "error", "data": "❌ Messages list is empty or invalid."}

    llm_id = request.model_name
    query = request.messages[0]   # ✅ Fix: send string only
    allow_search = request.allow_search
    system_prompt = request.system_prompt
    provider = request.model_provider

    try:
        # Create AI Agent and get response
        response = get_response_from_ai_agent(
            llm_id, query, allow_search, system_prompt, provider
        )
        return {"data": response}

    except Exception as e:
        return { "data": f"⚠️ Internal Error: {str(e)}"}


# Step3: Run app & Explore Swagger UI Docs
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9999)

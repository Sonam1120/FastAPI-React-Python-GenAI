from dotenv import load_dotenv
import os

# Load keys from .env
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
# from langchain_openai import ChatOpenAI   # OpenAI साठी पण

# Gemini LLM
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GEMINI_API_KEY
)

# Groq LLM
groq_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY
)

# Search Tool
search_tool = TavilySearch(max_results=2, api_key=TAVILY_API_KEY)

# Step3: Setup AI Agent with Search tool functionality
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

system_prompt = "Act as an AI chatbot who is smart and friendly"

def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    if provider == "Groq":
        llm = ChatGroq(model=llm_id, api_key=GROQ_API_KEY)
    elif provider == "OpenAI":
        llm = ChatOpenAI(model=llm_id)
    elif provider == "Gemini":
        llm = ChatGoogleGenerativeAI(model=llm_id, google_api_key=GEMINI_API_KEY)
    else:
        raise ValueError("⚠️ Unknown provider")

    tools = [search_tool] if allow_search else []

    # ❌ system_prompt ला param म्हणून देण्याऐवजी
    # ✅ SystemMessage मध्ये add करा
    agent = create_react_agent(
        model=llm,
        tools=tools
    )

    # Messages मध्ये system + human दोन्ही टाका
    state = {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]
    }

    response = agent.invoke(state)

    messages = response.get("messages", [])
    ai_messages = [msg.content for msg in messages if isinstance(msg, AIMessage)]

    # शेवटचा AI message print
    if ai_messages:
        print("AI Response:", ai_messages[-1])
        return ai_messages[-1]
    else:
        print("⚠️ No AI message returned")
        return None


# Example Run
if __name__ == "__main__":
    get_response_from_ai_agent(
        llm_id="gemini-1.5-flash",
        query="Hi Gemini! Tell me a fun fact about space 🚀",
        allow_search=False,
        system_prompt=system_prompt,
        provider="Gemini"
    )

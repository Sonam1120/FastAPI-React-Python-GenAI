# if you dont use pipenv uncomment the following:
# from dotenv import load_dotenv
# load_dotenv()

#Step1: Setup UI with streamlit (model provider, model, system prompt, web_search, query)
import streamlit as st
import requests

st.set_page_config(page_title="LangGraph Agent UI", layout="centered")
st.title("AI Chatbot Agents")
st.write("Create and Interact with the AI Agents!")

# System prompt
system_prompt = st.text_area(
    "Define your AI Agent: ", 
    height=70, 
    placeholder="Type your system prompt here..."
)

# Model options
MODEL_NAMES_GROQ = ["llama-3.3-70b-versatile"]   # ✅ removed deprecated mixtral
MODEL_NAMES_OPENAI = ["gpt-4o-mini"]

provider = st.radio("Select Provider:", ("Groq", "OpenAI"))

if provider == "Groq":
    selected_model = st.selectbox("Select Groq Model:", MODEL_NAMES_GROQ)
elif provider == "OpenAI":
    selected_model = st.selectbox("Select OpenAI Model:", MODEL_NAMES_OPENAI)

# Web search toggle
allow_web_search = st.checkbox("Allow Web Search")

# User query
user_query = st.text_area(
    "Enter your query: ", 
    height=150, 
    placeholder="Ask Anything!"
)

API_URL = "http://127.0.0.1:9999/chat"

# Step2: Send request to backend
if st.button("Ask Agent!"):
    if user_query.strip():
        payload = {
            "model_name": selected_model,
            "model_provider": provider,
            "system_prompt": system_prompt,
            "messages": [user_query],
            "allow_search": allow_web_search
        }

        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                response_data = response.json()

                if "error" in response_data:
                    st.error(response_data["error"])
                else:
                    st.subheader("Agent Response")

                    # 🔥 Handle response format
                    final_text = response_data.get("data", str(response_data))

                    if isinstance(final_text, dict):
                        # If backend returns dict, pretty print
                        for key, val in final_text.items():
                            st.write(f"**{key}:** {val}")
                    elif "\n" in final_text:
                        # If multiline response, show nicely
                        st.markdown("**Final Response:**")
                        for line in final_text.split("\n"):
                            if line.strip():
                                st.write(line)
                    else:
                        # Single line response
                        st.markdown(f"**Final Response:** {final_text}")
            else:
                st.error(f"❌ Backend Error: {response.status_code}")
        except Exception as e:
            st.error(f"⚠️ Could not connect to backend: {e}")

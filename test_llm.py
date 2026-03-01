import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

def test_llm():
    api_key = os.getenv("OPENROUTER_API_KEY")
    print(f"Testing with API Key: {api_key[:10]}...")
    
    llm = ChatOpenAI(
        model="openai/gpt-oss-120b:free", 
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
    )
    
    try:
        print("Sending request to OpenRouter...")
        response = llm.invoke([HumanMessage(content="Return a JSON with key 'test' and value 'success'")])
        print("Response received:")
        print(response.content)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_llm()

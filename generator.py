import os
from typing import TypedDict, List, Dict, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import json

load_dotenv()

class GraphState(TypedDict):
    resume_text: str
    structured_data: Dict
    portfolio_content: Dict
    error: str
    api_key: Optional[str]

def get_llm(api_key: str = None):
    return ChatOpenAI(
        model="openai/gpt-oss-120b:free",
        openai_api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        timeout=45,  # 45 second timeout for complex resumes
        max_retries=3,
    )

def extract_resume_node(state: GraphState, api_key: str = None):
    """Parses resume text into structured JSON."""
    try:
        llm = get_llm(api_key)
        prompt = f"""
        Extract the following information from this resume and return it as a JSON object:
        - full_name
        - contact_info (email, phone, linkedin)
        - summary (a professional 2-3 sentence bio)
        - skills (list of strings)
        - experience (list of objects with keys: job_role, company, dates, description)
        - projects (list of objects with keys: name, tech_stack, description)
        - education (list of objects: degree, school, dates)

        Resume Text:
        {state['resume_text']}
        """
        
        response = llm.invoke([
            SystemMessage(content="You are an expert resume parser. Respond ONLY with raw JSON. Do not include markdown formatting like ```json."),
            HumanMessage(content=prompt)
        ])
        
        # Robust JSON extraction
        content = response.content.strip()
        if "{" in content:
            content = content[content.find("{"):content.rfind("}")+1]
        
        data = json.loads(content)
        return {"structured_data": data, "error": ""}
    except Exception as e:
        print(f"Error in extract_resume_node: {e}")
        return {"error": f"Extraction Failed: {str(e)}"}

def write_portfolio_node(state: GraphState, api_key: str = None):
    """Expands structured data into creative web content."""
    if state.get("error"):
        return state

    try:
        llm = get_llm(api_key)
        data = state.get('structured_data', {})
        
        prompt = f"""
        Based on this professional data, generate high-quality web content for a multi-page portfolio:
        
        Data: {json.dumps(data)}
        
        Generate content for 4 sections:
        1. Home Page: A catchy hero title and a compelling personal introduction.
        2. Skills Page: Group the skills into categories (e.g. Core Tech, Tools, Soft Skills).
           CRITICAL: 'skills' MUST be a DICTIONARY where keys are category names and values are LISTS of strings.
        3. Experience Page: Rewrite the experience descriptions to be more impact-oriented. 
           CRITICAL: 'experience' MUST be a LIST of objects. Each object MUST have: 'job_role', 'company', 'dates', 'description'.
        4. Projects Page: Enriched descriptions for each project.
           CRITICAL: 'projects' MUST be a LIST of objects. Each object MUST have: 'name', 'tech_stack', 'description'.
        
        Return as a JSON object with keys: 'home', 'skills', 'experience', 'projects'.
        """
        
        response = llm.invoke([
            SystemMessage(content="You are a professional website copywriter. Respond ONLY with raw JSON. Do not include markdown formatting like ```json."),
            HumanMessage(content=prompt)
        ])
        
        content = response.content.strip()
        if "{" in content:
            content = content[content.find("{"):content.rfind("}")+1]
            
        portfolio_content = json.loads(content)
        return {"portfolio_content": portfolio_content, "error": ""}
    except Exception as e:
        print(f"Error in write_portfolio_node: {e}")
        return {"error": f"Generation Failed: {str(e)}"}

# Build Workflow
workflow = StateGraph(GraphState)

def extract_wrapper(state):
    return extract_resume_node(state, state.get("api_key"))

def write_wrapper(state):
    return write_portfolio_node(state, state.get("api_key"))

workflow.add_node("extract", extract_wrapper)
workflow.add_node("write", write_wrapper)

workflow.set_entry_point("extract")
workflow.add_edge("extract", "write")
workflow.add_edge("write", END)

app_graph = workflow.compile()

def generate_portfolio(resume_text: str, api_key: str = None):
    initial_state = {
        "resume_text": resume_text, 
        "structured_data": {}, 
        "portfolio_content": {}, 
        "error": "",
        "api_key": api_key
    }
    final_state = app_graph.invoke(initial_state)
    return {
        "structured_data": final_state.get("structured_data"),
        "portfolio_content": final_state.get("portfolio_content"),
        "error": final_state.get("error")
    }

from generator import generate_portfolio
import os
from dotenv import load_dotenv

load_dotenv()

def simulate_generation():
    # Simulate a small resume text
    resume_text = """
    Arijit Kar
    Software Engineer
    Email: arijit@example.com
    Skills: Python, JavaScript, AI
    Experience: 
    - AI Developer at Tech Corp (2020-2024): Built RAG systems.
    - Intern at Code Inc (2019): Developed web apps.
    Projects:
    - Portfoli-AI: An AI portfolio generator.
    """
    
    print("Starting simulation...")
    try:
        result = generate_portfolio(resume_text)
        print("Result received:")
        import json
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    simulate_generation()

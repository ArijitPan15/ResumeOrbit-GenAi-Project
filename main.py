import os
import io
from fastapi import FastAPI, Depends, HTTPException, status, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
import pypdf
import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from generator import generate_portfolio
from database import PortfolioVectorDB
from dotenv import load_dotenv
from models import init_db, User, Portfolio
from auth import get_db, get_current_user, create_access_token, verify_password, get_password_hash

# Initialize
load_dotenv()
init_db()
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
db_vector = PortfolioVectorDB()
chat_llm = ChatOpenAI(
    model="openai/gpt-oss-120b:free", 
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
)

# --- PUBLIC ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("landing.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.hashed_password):
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"})
    
    access_token = create_access_token(data={"sub": user.username})
    response = RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
    response.set_cookie(key="access_token", value=f"Bearer {access_token}", httponly=True)
    return response

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register")
async def register(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.username == username).first()
    if existing_user:
        return templates.TemplateResponse("register.html", {"request": request, "error": "Username taken"})
    
    new_user = User(username=username, hashed_password=get_password_hash(password))
    db.add(new_user)
    db.commit()
    return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

# --- PROTECTED ADMIN ROUTES ---
async def get_current_user_from_cookie(request: Request, db: Session = Depends(get_db)):
    token = request.cookies.get("access_token")
    if not token or not token.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    token = token.split(" ")[1]
    return await get_current_user(token, db)

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, user: User = Depends(get_current_user_from_cookie)):
    return templates.TemplateResponse("dashboard.html", {"request": request, "user": user})

@app.post("/update-key")
async def update_key(
    key: str = Form(...),
    user: User = Depends(get_current_user_from_cookie),
    db: Session = Depends(get_db)
):
    user.openrouter_key = key
    db.commit()
    return {"status": "success"}

@app.post("/generate")
async def process_resume(
    file: UploadFile = File(...), 
    user: User = Depends(get_current_user_from_cookie),
    db: Session = Depends(get_db)
):
    # 1. Extract Text
    content = await file.read()
    pdf_reader = pypdf.PdfReader(io.BytesIO(content))
    resume_text = ""
    for page in pdf_reader.pages:
        resume_text += page.extract_text()
    
    # Truncate to avoid LLM token limits and hangs
    resume_text = resume_text[:15000]
    print(f"Generating portfolio for user {user.username}. Resume length: {len(resume_text)}")
    
    # 2. RAG Indexing
    chunks = [resume_text[i:i+1000] for i in range(0, len(resume_text), 1000)]
    metadatas = [{"source": "resume"} for _ in chunks]
    db_vector.upsert_resume(user.id, "main_resume", chunks, metadatas)
    
    # 3. AI Generation (using User's Key if available)
    result = generate_portfolio(resume_text, api_key=user.openrouter_key)
    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])
    
    # 4. Save to DB
    portfolio = db.query(Portfolio).filter(Portfolio.user_id == user.id).first()
    if portfolio:
        portfolio.content = result["portfolio_content"]
    else:
        portfolio = Portfolio(user_id=user.id, content=result["portfolio_content"])
        db.add(portfolio)
    db.commit()
    
    return {"status": "success", "portfolio_id": portfolio.id}

# --- VISITOR / PUBLIC VIEW ---
@app.get("/p/{username}", response_class=HTMLResponse)
async def public_portfolio(request: Request, username: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user or not user.portfolios:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    portfolio_data = user.portfolios[0].content
    return templates.TemplateResponse("portfolio_public.html", {
        "request": request,
        "p": portfolio_data,
        "username": username,
        "user_id": user.id
    })

@app.post("/chat/{user_id}")
async def public_chat(user_id: int, request: Request, db: Session = Depends(get_db)):
    data = await request.json()
    query = data.get("query")
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    context_chunks = db_vector.query_resume(user_id, query)
    context = "\n".join(context_chunks)
    
    sys_prompt = f"""
    ROLE: You are the professional Digital Representative for {user.username}.
    MISSION: Your mission is to answer questions about {user.username}'s professional background, skills, and experience only.
    
    STRICT BOUNDARIES:
    1. ONLY answer questions related to {user.username}'s career, projects, skills, education, and professional interests.
    2. DECLINE all off-topic, personal, political, or philosophical questions.
    3. If asked about something NOT in the Resume Context, say: "I'm sorry, I don't have that information. I am specifically trained to answer questions about {user.username}'s professional background."
    4. NEVER provide personal advice, life tips, or general trivia.
    5. Maintain a professional, helpful, and representative tone at all times.
    
    RESUME CONTEXT:
    {context}
    """
    
    # Dynamic LLM for Chat
    user_llm = ChatOpenAI(
        model="openai/gpt-oss-120b:free", 
        openai_api_key=user.openrouter_key or os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
    )
    
    response = user_llm.invoke([
        SystemMessage(content=sys_prompt),
        HumanMessage(content=query)
    ])
    
    return {"response": response.content}

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from openai import OpenAI
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = FastAPI(title="Resume AI Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenRequest(BaseModel):
    resume_text: str
    job_text: str
    tone: Optional[str] = "Confident"
    model_name: Optional[str] = None

class GenResponse(BaseModel):
    tailored_resume: str
    cover_letter: str

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/generate", response_model=GenResponse)
def generate(req: GenRequest):
    if not OPENAI_API_KEY or client is None:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set on server")

    model = req.model_name or DEFAULT_MODEL

    try:
        r1 = client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content":"You are an ATS resume optimizer."},
                {"role":"user","content":f"JOB DESCRIPTION:\n{req.job_text}\n\nCANDIDATE RESUME:\n{req.resume_text}\n\nReturn an ATS-optimized resume with sections and bullet points."}
            ],
            temperature=0.4,
        )
        tailored_resume = r1.choices[0].message.content.strip()

        r2 = client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content":"You write concise, sincere one-page cover letters."},
                {"role":"user","content":f"Tone: {req.tone}\n\nJOB DESCRIPTION:\n{req.job_text}\n\nCANDIDATE RESUME:\n{req.resume_text}\n\nReturn a one-page cover letter."}
            ],
            temperature=0.5,
        )
        cover_letter = r2.choices[0].message.content.strip()

        return GenResponse(tailored_resume=tailored_resume, cover_letter=cover_letter)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {e}")

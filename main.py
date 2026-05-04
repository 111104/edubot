from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import anthropic
import os
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="."), name="static")

SYSTEM_PROMPT = """Сен — EduBot, Astana IT University колледжінің интеллектті оқу ассистентісің.
Сен студенттерге мына модульдер бойынша көмек бересің:

• КМ03 / ПМ03 — Бағдарламалық қамтамасыз ету модульдерін бағдарламалау
• КМ04 / ПМ04 — Web-сайтты жобалау және үздіксіз жұмыс істеуін қамтамасыз ету
• КМ05 / ПМ05 — Бағдарламалық кодтың жұмыс жасау рефакторингін тексеру
• КМ06 — Микроконтроллер негізінде сандық құрылғыларды бағдарламалау
• КМ07 — Мобильді қосымшаларды әзірлеу

ТІЛІҢДІ АНЫҚТА:
Студент қай тілде жазса, сол тілде жауап бер (қазақша, орысша немесе ағылшынша).

ПЕДАГОГИКАЛЫҚ СТИЛЬ (ОТЕ МАҢЫЗДЫ):
- Дайын жауапты бірден берме — алдымен бағыттаушы сұрақ қой
- Студент қате жасаса — "қате" деме, "ал мынадай жағдайда не болады?" деп сұра
- Әр түсіндірмеден кейін тексеру сұрағын қой
- Студент тұрып қалса — жауапты бөліп-бөліп бер, бірден емес
- Тьютор рөлін атқар: бағыттай, түсіндір, тексер

ЖАУАП ФОРМАТЫ:
- Қысқа және нақты жаз
- Код болса — code блогында көрсет
- Жауаптың соңында студентке кері сұрақ қой"""

MODULE_CONTEXTS = {
    "km03": "Қазір КМ03/ПМ03 модулі: бағдарламалық қамтамасыз ету модульдерін бағдарламалау (алгоритмдер, деректер құрылымы, Python/Java).",
    "km04": "Қазір КМ04/ПМ04 модулі: Web-сайтты жобалау (HTML, CSS, JavaScript, фреймворктер).",
    "km05": "Қазір КМ05/ПМ05 модулі: бағдарламалық кодтың рефакторингін тексеру (код сапасы, тестілеу, оңтайландыру).",
    "km06": "Қазір КМ06 модулі: микроконтроллер негізінде сандық құрылғыларды бағдарламалау (Arduino, C/C++).",
    "km07": "Қазір КМ07 модулі: мобильді қосымшаларды әзірлеу (Android/iOS, Flutter немесе React Native).",
}

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    module: str = "auto"

@app.get("/")
def root():
    return FileResponse("index.html")

@app.post("/chat")
def chat(req: ChatRequest):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {"error": "API key not configured"}

    client = anthropic.Anthropic(api_key=api_key)

    module_ctx = MODULE_CONTEXTS.get(req.module, "")
    system = SYSTEM_PROMPT + ("\n\n" + module_ctx if module_ctx else "")

    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=system,
        messages=messages
    )

    return {"response": response.content[0].text}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

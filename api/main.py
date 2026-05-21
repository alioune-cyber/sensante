from xml.parsers.expat import model

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np




import os
from dotenv import load_dotenv
from groq import Groq

from pydantic import BaseModel, Field


from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


#---Schemas Pydantic--
class PatientInput(BaseModel):
    age: int = Field(..., ge=0, le=120)
    sexe: str = Field(...)
    temperature: float = Field(..., ge=35.0, le=42.0)
    tension_sys: int = Field(..., ge=60, le=250)
    toux: bool = Field(...)
    fatigue: bool = Field(...)
    maux_tete: bool = Field(...)
    region: str = Field(...)

class DiagnosticOutput(BaseModel):
    diagnostic: str
    probabilite: float
    confiance: str
    message: str

#---Application FastAPI--
app = FastAPI(
    title="SenSante API",
    description="Assistant pre-diagnostic medical pour le Senegal",
    version="0.2.0"
)

#---Chargement du modele (une seule fois)--
"""print("Chargement du modele...")
    
model = joblib.load("models/model.pkl")
le_sexe = joblib.load("models/encoder_sexe.pkl")
le_region = joblib.load("models/encoder_region.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")
    
print(f"Modele charge : {list(model.classes_)}")
"""



#---Chargement du modele depuis Hugging Face Hub--
from huggingface_hub import hf_hub_download

print("Chargement du modele...")

try:
    # Télécharger les fichiers du repo modèle HF
    model_path = hf_hub_download(
        repo_id="cyberamn/sensante_model",
        filename="model.pkl"
    )

    sexe_path = hf_hub_download(
        repo_id="cyberamn/sensante_model",
        filename="encoder_sexe.pkl"
    )

    region_path = hf_hub_download(
        repo_id="cyberamn/sensante_model",
        filename="encoder_region.pkl"
    )

    feature_cols_path = hf_hub_download(
        repo_id="cyberamn/sensante_model",
        filename="feature_cols.pkl"
    )

    # Charger les objets
    model = joblib.load(model_path)
    le_sexe = joblib.load(sexe_path)
    le_region = joblib.load(region_path)
    feature_cols = joblib.load(feature_cols_path)

    print(f"Modele charge : {list(model.classes_)}")

except Exception as e:
    print(f"Erreur chargement modele : {e}")
    raise e





#---Routes--
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "SenSante API is running"}



@app.post("/predict", response_model=DiagnosticOutput)
def predict(patient: PatientInput):
    # Encoder
    try:
        sexe_enc = le_sexe.transform([patient.sexe])[0]
    except ValueError:
        return DiagnosticOutput(
            diagnostic="erreur", probabilite=0.0,
            confiance="aucune",
            message=f"Sexe invalide : {patient.sexe}"
        )
    
    try:
        region_enc = le_region.transform([patient.region])[0]

    except ValueError:
        
        return DiagnosticOutput(
            diagnostic="erreur", probabilite=0.0,
            confiance="aucune",
            message=f"Region inconnue : {patient.region}"
        )
        
        
    features = np.array([[
        patient.age, sexe_enc, patient.temperature,
        patient.tension_sys, int(patient.toux),
        int(patient.fatigue), int(patient.maux_tete),
        region_enc
    ]])

    # Prediction
    diagnostic = model.predict(features)[0]
        
    proba_max = float(model.predict_proba(features)[0].max())
        
    confiance = ("haute" if proba_max >= 0.7 else "moyenne" if proba_max >= 0.4 else "faible")
        
    messages = {
        "palu": "Suspicion de paludisme. Consultez rapidement.",
        "grippe": "Suspicion de grippe. Repos et hydratation.",
        "typh": "Suspicion de typhoide. Consultation necessaire.",
        "sain": "Pas de pathologie detectee."
    }
        
    return DiagnosticOutput(
        diagnostic=diagnostic,
        probabilite=round(proba_max, 2),
        confiance=confiance,
        message=messages.get(diagnostic, "Consultez un medecin.")
    )





# Charger les variables d'environnement
load_dotenv()

# Client Groq (charge au demarrage)
groq_client = None

groq_api_key = os.getenv("GROQ_API_KEY")

if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
    print("Client Groq initialise.")
else:
    print(
        "ATTENTION : GROQ_API_KEY non trouvee. "
        "/explain sera desactive."
    )














class ExplainInput(BaseModel):
    diagnostic: str = Field(
        ...,
        description="Diagnostic predit par le modele"
    )

    probabilite: float = Field(
        ...,
        description="Probabilite du diagnostic"
    )

    age: int = Field(...)

    sexe: str = Field(...)

    temperature: float = Field(...)

    region: str = Field(...)


class ExplainOutput(BaseModel):
    explication: str = Field(
        ...,
        description="Explication en francais"
    )

    modele_llm: str = Field(
        default="llama-3.1-8b-instant",
        description="Modele LLM utilise"
    )



"""SYSTEM_PROMPT = (
    "Tu es un assistant medical senegalais. "
    "Tu recois un diagnostic et des donnees patient. "

    "mais tu as le droit d'utiliser quelques mots français simples si tu ne trouves pas le mot wolof exact. "

    "Mots wolof à utiliser absolument : "
    "'dégg naa' (je comprends), 'wér na' (ça va / doucement), 'yaram' (corps / santé), "
    "'fajkat' (médecin), 'jarama' (merci), 'ndax' (peut-être / parce que), "
    "'baax na' (c'est bien / d'accord), 'tangoor' (fièvre), 'xare' (toux), "
    "'bàyyi' (laisser / arrêter), 'jëmm' (le corps), 'fànni' (fatigué), "
    "'tollu' (suffit / maximum), 'yore' (avoir / souffrir de). "

    
    "Explique le resultat en francais avec des mots wolofs simple, "
    "comme un medecin parlerait a son patient. "
    "Sois rassurant mais recommande toujours "
    "une consultation medicale. "
    "Maximum 3 phrases. "
    "Ne fais JAMAIS de diagnostic toi-meme. "
    "Tu expliques uniquement le diagnostic fourni."
)"""


SYSTEM_PROMPT = (
    "Tu es un assistant medical senegalais. "
    "Tu recois un diagnostic et des donnees patient. "
    "Explique le resultat en francais simple, "
    "comme un medecin parlerait a son patient. "
    "Sois rassurant mais recommande toujours "
    "une consultation medicale. "
    "Maximum 3 phrases. "
    "Ne fais JAMAIS de diagnostic toi-meme. "
    "Tu expliques uniquement le diagnostic fourni."
)



@app.post("/explain", response_model=ExplainOutput)
def explain(data: ExplainInput):
    """
    Expliquer un diagnostic en francais avec un LLM.
    """

    if not groq_client:
        return ExplainOutput(
            explication=(
                "Service d'explication indisponible. "
                "Cle API non configuree."
            ),
            modele_llm="aucun"
        )

    # Construire le user prompt
    user_prompt = (
        f"Patient : {data.sexe}, {data.age} ans, "
        f"region {data.region}\n"
        f"Temperature : {data.temperature} C\n"
        f"Diagnostic du modele : {data.diagnostic} "
        f"(probabilite {data.probabilite:.0%})\n"
        f"Explique ce resultat au patient."
    )

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            max_tokens=200,
            temperature=0.3
        )

        explication = response.choices[0].message.content

    except Exception as e:
        explication = (
            f"Erreur lors de l'appel au LLM : {str(e)}"
        )

    return ExplainOutput(
        explication=explication,
        modele_llm="llama-3.1-8b-instant"
    )



# Autoriser les requetes depuis le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # En dev : tout accepter
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)







# Servir le frontend comme fichier statique
app.mount("/static", StaticFiles(directory="frontend"),name="static")

@app.get("/")
def serve_frontend():
    """Servir la page d'accueil."""
    return FileResponse("frontend/index.html")



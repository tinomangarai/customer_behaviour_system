from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
import threading
import webbrowser
import uvicorn

app = FastAPI()

# Configure templates and static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load models
scaler = joblib.load('models/cluster_scaler.pkl')
cluster_model = joblib.load('models/cluster_model.pkl')
churn_model = joblib.load('models/churn_model.pkl')

def open_browser():
    """Open the default browser to the prediction form"""
    webbrowser.open("http://localhost:8000/predict-form")

@app.on_event("startup")
async def startup_event():
    """Open browser when the app starts"""
    threading.Timer(1.5, open_browser).start()

@app.get("/predict-form")
async def prediction_form(request: Request):
    """Serve prediction form"""
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict-result")
async def predict_from_form(
    request: Request,
    interaction_count: int = Form(...),
    days_since_last: int = Form(...),
    avg_menu_steps: float = Form(...),
    avg_sentiment: float = Form(...),
    resolution_rate: float = Form(...)
):
    """Handle form submission and return results"""
    try:
        # Churn prediction
        churn_proba = churn_model.predict_proba(
            [[interaction_count, days_since_last, avg_menu_steps]]
        )[0][1]
        
        # Persona clustering
        cluster_features = scaler.transform(
            [[interaction_count, days_since_last, avg_sentiment]]
        )
        persona = cluster_model.predict(cluster_features)[0]
        
        # NPS classification
        nps_group = "Passive"
        if avg_sentiment > 0.5:
            nps_group = "Promoter"
        elif avg_sentiment < -0.3:
            nps_group = "Detractor"
            
        # UX Score
        ux_score = (resolution_rate * 0.7) + (1 - (avg_menu_steps / 10)) * 0.3
        
        # Persona mapping
        persona_map = {
            0: 'Low-Engagement Detractor',
            1: 'Active Promoter',
            2: 'At-Risk Passive',
            3: 'Frequent but Neutral'
        }
        
        return templates.TemplateResponse("result.html", {
            "request": request,
            "churn_prob": f"{churn_proba:.2%}",
            "persona": persona_map.get(persona, "Unknown"),
            "nps_group": nps_group,
            "ux_score": f"{ux_score:.2f}"
        })
        
    except Exception as e:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": str(e)
        })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

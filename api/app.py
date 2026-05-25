
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from predict import load_model_and_meta, score
import uvicorn

app = FastAPI()

# Load the active model at startup
model = None
model_meta = None
features = None

@app.on_event("startup")
def load_active_model():
	global model, model_meta, features
	model, model_meta = load_model_and_meta()
	trained_on = model_meta.trained_on
	# Get features from versioned_data_mapping via model_meta
	from config import versioned_data_mapping
	from pathlib import Path
	trained_on_key = Path(trained_on).name
	features = versioned_data_mapping[trained_on_key].features

class PredictionRequest(BaseModel):
	data: dict
	

@app.post("/predict")
async def get_prediction(request: PredictionRequest):
	global model, model_meta, features
	application = request.data
	try:
		result = score(application, model, features, model_meta.version)
		return result
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))


# Home landing page (pretty HTML)
from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
async def home():
	return """
	<html>
	<head>
		<title>Fraud Detection ML Model Inference Portal</title>
		<style>
			body { font-family: 'Segoe UI', Arial, sans-serif; background: #f7f9fa; color: #222; margin: 0; padding: 0; }
			.container { max-width: 700px; margin: 60px auto; background: #fff; border-radius: 12px; box-shadow: 0 4px 24px rgba(0,0,0,0.07); padding: 40px 32px; }
			h1 { color: #1a73e8; font-size: 2.2em; margin-bottom: 0.2em; }
			.subtitle { color: #555; font-size: 1.1em; margin-bottom: 1.5em; }
			code, pre { background: #f3f3f3; border-radius: 6px; padding: 2px 6px; font-size: 1em; }
			.example { background: #f8fafc; border-left: 4px solid #1a73e8; padding: 16px; margin: 24px 0; font-size: 1.05em; }
			.footer { color: #888; font-size: 0.95em; margin-top: 2em; text-align: center; }
		</style>
	</head>
	<body>
		<div class="container">
			<h1>Fraud Detection ML Model Inference Portal</h1>
			<div class="subtitle">
				Welcome! This API lets you score job applications for fraud risk using the latest production ML model.<br>
				<b>POST</b> your application data as JSON to <code>/predict</code> to get a prediction.
			</div>
			<div class="example">
				<b>Example usage:</b><br>
				<pre>curl -X POST 'http://127.0.0.1:8000/predict' \
                        -H 'Content-Type: application/json' \
                        -d '{
                            "data": {
                            "application_completion_seconds": 45.0,
                            "hour_of_day": 3,
                            "email_domain_risk_score": 0.7,
                            "account_age_days": 4,
                            "num_applications_last_24h": 9,
                            "ip_location_mismatch_km": 3200.0,
                            "is_vpn_or_proxy": 1,
                            "profile_trust_score": 0.2
                            }
                        }'</pre>
			</div>
			<div>
				<b>API Documentation:</b> <a href="/docs">Swagger UI</a>
			</div>
			<div class="footer">
				&copy; 2026 Fraud Detection ML Team
			</div>
		</div>
	</body>
	</html>
	"""


# Optional: GET endpoint for health check
@app.get("/health")
async def health():
	return {"status": "ok"}

if __name__ == "__main__":
	uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)

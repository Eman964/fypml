from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
sales_model = joblib.load("model/yearly_sales_model.pkl")
profit_model = joblib.load("model/profit_model.pkl")

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()

    input_df = pd.DataFrame([data], columns=[
        "Category", "Used For", "Selling Price (USD)",
        "COG + Expenses (USD)", "likes", "dislikes"
    ])

    predicted_sales = sales_model.predict(input_df)[0]
    input_df["Predicted Yearly Sales"] = predicted_sales
    predicted_profit = profit_model.predict(input_df)[0]

    return {
        "Predicted Yearly Sales": predicted_sales,
        "Predicted Profit": predicted_profit
    }

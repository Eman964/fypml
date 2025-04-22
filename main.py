from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for input validation
class InputData(BaseModel):
    Category: str
    Used_For: str = Field(..., alias="Used For")
    Selling_Price_USD: float = Field(..., alias="Selling Price (USD)")
    COG_Expenses_USD: float = Field(..., alias="COG + Expenses (USD)")
    likes: int
    dislikes: int

    class Config:
        allow_population_by_field_name = True  # Allow using original field names (optional)

# Load models
sales_model = joblib.load("model/yearly_sales_model.pkl")
profit_model = joblib.load("model/profit_model.pkl")

@app.post("/predict")
async def predict(data: InputData):
    # Convert the Pydantic model to a DataFrame
    input_df = pd.DataFrame([data.dict(by_alias=True)])

    # Predict sales and profit
    predicted_sales = sales_model.predict(input_df)[0]
    input_df["Predicted Yearly Sales"] = predicted_sales
    predicted_profit = profit_model.predict(input_df)[0]

    return {
        "Predicted Yearly Sales": predicted_sales,
        "Predicted Profit": predicted_profit
    }

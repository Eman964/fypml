from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

# Load the pre-trained model
model = joblib.load('sales_model.pkl')

# Create a FastAPI app instance
app = FastAPI()

# Define input data model
class ProductData(BaseModel):
    Category: str
    Used_For: str
    Selling_Price: float
    COG_Expenses: float
    Likes: int
    Dislikes: int

@app.post("/predict-sales")
async def predict_sales(data: ProductData):
    # Convert input data to DataFrame for model prediction
    input_data = pd.DataFrame([data.dict()])
    prediction = model.predict(input_data)
    return {"Yearly_Sales": prediction[0]}

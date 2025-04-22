from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (you can restrict this for security)
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Load the trained models
sales_model = joblib.load('model/yearly_sales_model.pkl')
profit_model = joblib.load('model/profit_model.pkl')

# Define preprocessing steps for prediction (matching the original preprocessing)
numerical_features = ['Selling Price (USD)', 'COG + Expenses (USD)', 'likes', 'dislikes']
categorical_features = ['Category', 'Used For']

numerical_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Full preprocessor for prediction
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Helper function to compute derived features and predict profit
def predict_profit_sample(category, used_for, price, cost, likes, dislikes):
    # Create input DataFrame
    sample = pd.DataFrame([{
        'Category': category,
        'Used For': used_for,
        'Selling Price (USD)': price,
        'COG + Expenses (USD)': cost,
        'likes': likes,
        'dislikes': dislikes
    }])

    # Derived features
    sample['Price_to_Cost_Ratio'] = price / cost if cost != 0 else 0
    sample['Margin'] = price - cost
    total_engagement = likes + dislikes
    sample['Like_Ratio'] = likes / total_engagement if total_engagement != 0 else 0.5
    sample['Net_Sentiment'] = likes - dislikes
    sample['Engagement'] = total_engagement
    sample['Item_Type'] = 'Unknown'  # This can be adjusted if there is a rule

    # Predict Yearly Sales
    predicted_yearly_sales = sales_model.predict(sample)[0]

    # Add predicted sales for Profit prediction
    sample['Predicted_Yearly_Sales'] = predicted_yearly_sales

    # Predict Profit
    predicted_profit = profit_model.predict(sample)[0]

    return predicted_yearly_sales, predicted_profit

@app.post("/predict")
async def predict(request: Request):
    # Parse input data
    data = await request.json()

    # Extract values from the incoming request
    category = data['Category']
    used_for = data['Used For']
    price = data['Selling Price (USD)']
    cost = data['COG + Expenses (USD)']
    likes = data['likes']
    dislikes = data['dislikes']

    # Call the helper function to make predictions
    predicted_yearly_sales, predicted_profit = predict_profit_sample(
        category, used_for, price, cost, likes, dislikes
    )

    # Return the predictions in JSON format
    return {
        "Predicted Yearly Sales": predicted_yearly_sales,
        "Predicted Profit": predicted_profit
    }

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# 1. Sensor Data Simulation
def load_sensor_data():
    np.random.seed(42)
    data = {
        'Pressure_Drop': np.random.normal(1.2, 0.1, 200),
        'Flow_Rate': np.random.normal(50.0, 1.5, 200)
    }
    df = pd.DataFrame(data)
    
    scaling_phase = {
        'Pressure_Drop': [1.8, 2.1, 2.5], 
        'Flow_Rate': [42.0, 38.0, 32.0]
    }
    df = pd.concat([df, pd.DataFrame(scaling_phase)], ignore_index=True)
    return df

# 2. Data Preprocessing & Scaling
def preprocess_data(df):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[['Pressure_Drop', 'Flow_Rate']])
    return scaled_data, scaler

# 3. AI Model Training (Isolation Forest)
def train_detection_model(scaled_data):
    model = IsolationForest(contamination=0.02, random_state=42)
    model.fit(scaled_data)
    return model

# 4. Real-time Prediction for Dashboard
def predict_membrane_status(model, scaler, new_pressure, new_flow):
    new_data = scaler.transform(pd.DataFrame([[new_pressure, new_flow]], columns=['Pressure_Drop', 'Flow_Rate']))
    prediction = model.predict(new_data)
    
    if prediction[0] == 1:
        return "[NORMAL] Membrane is healthy."
    else:
        return "[WARNING] Scaling Detected - Activate Piezo Cleaning!"

# --- Execution ---
if __name__ == "__main__":
    print("--- Eco Wave AI System Started ---\n")
    
    raw_df = load_sensor_data()
    scaled_data, scaler = preprocess_data(raw_df)
    ai_model = train_detection_model(scaled_data)
    
    status_1 = predict_membrane_status(ai_model, scaler, new_pressure=1.25, new_flow=49.5)
    print(f"Sensor Reading (1): {status_1}")
    
    status_2 = predict_membrane_status(ai_model, scaler, new_pressure=2.2, new_flow=39.0)
    print(f"Sensor Reading (2): {status_2}")
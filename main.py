Developing a smart commodity tracker involves real-time data retrieval and prediction using machine learning models. Below is a complete Python program outline for a smart commodity tracker that includes data retrieval, preprocessing, training, and prediction. It's important to note that for a production-ready system, additional components like a database for storage, more sophisticated error handling, security measures, and a web interface might be required.

Here's a simplified version:

```python
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import datetime
import logging

# Setup logging
logging.basicConfig(filename='commodity_tracker.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

API_URL = 'https://api.example.com/commodity_prices'  # Hypothetical API URL

def fetch_data(commodity):
    try:
        logging.info(f"Fetching data for {commodity}")
        response = requests.get(API_URL, params={'commodity': commodity})
        response.raise_for_status()  # Raises an HTTPError for bad responses
        data = response.json()
        # Convert JSON to DataFrame
        df = pd.DataFrame(data)
        return df
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data: {e}")
        return None

def preprocess_data(df):
    try:
        logging.info("Preprocessing data")
        # Check if the necessary columns exist
        if 'date' not in df or 'price' not in df:
            raise ValueError("DataFrame missing required columns")

        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        logging.error(f"Error in preprocessing data: {e}")
        return None

def train_model(X, y):
    try:
        logging.info("Training model")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        logging.info(f"Model trained with MSE: {mse}")
        return model
    except Exception as e:
        logging.error(f"Error in training model: {e}")
        return None

def predict_future_prices(model, last_known_data_point, num_days=30):
    try:
        logging.info(f"Predicting commodity prices for next {num_days} days")
        future_dates = [last_known_data_point + datetime.timedelta(days=i) for i in range(1, num_days+1)]
        future_dates_ordinal = np.array([date.toordinal() for date in future_dates]).reshape(-1, 1)
        predictions = model.predict(future_dates_ordinal)
        return pd.DataFrame({'date': future_dates, 'predicted_price': predictions})
    except Exception as e:
        logging.error(f"Error in predicting future prices: {e}")
        return None

if __name__ == "__main__":
    commodity = 'gold'
    df = fetch_data(commodity)
    
    if df is not None:
        preprocessed_df = preprocess_data(df)
        
        if preprocessed_df is not None:
            preprocessed_df['day_ordinal'] = preprocessed_df.index.map(datetime.datetime.toordinal)
            X = preprocessed_df[['day_ordinal']]
            y = preprocessed_df['price']
            model = train_model(X, y)
            
            if model is not None:
                last_known_date = preprocessed_df.index[-1]
                predictions = predict_future_prices(model, last_known_date)
                if predictions is not None:
                    print(predictions)
```

### Explanation

1. **Data Retrieval**: Uses a fictional API to fetch commodity price data. Replace `API_URL` with a real API endpoint and ensure APIs allow for such requests.

2. **Error Handling**: Wraps critical sections like API calls and data processing in try-except blocks to catch and log errors.

3. **Preprocessing**: Converts dates to a numerical format suitable for modeling and sorts the data chronologically.

4. **Model Training**: Uses a simple linear regression from `scikit-learn` to demonstrate the concept. In a real-world scenario, look into more complex models or choose a model more tailored to the data's characteristics.

5. **Prediction**: Predicts future prices for a specified number of days using the trained model.

6. **Logging**: Utilizes the logging module for saving error messages and other important information to a log file.

This program assumes the availability of historical data and a consistent structure of input data. Adaptations may be needed to fit specific data and requirements.
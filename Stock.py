import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from datetime import timedelta

# -----------------------------
# Streamlit Title and Sidebar
# -----------------------------
st.set_page_config(page_title="Stock Market Prediction", layout="wide")
st.title("📊 Stock Market Prediction App")

st.sidebar.header("⚙️ Model Configuration")
model_choice = st.sidebar.selectbox(
    "Select Prediction Model",
    ["Linear Regression", "Random Forest", "LSTM (Deep Learning)"]
)

st.sidebar.markdown("---")
st.sidebar.info("Upload your stock dataset containing **Date** and **Close** columns.")

# -----------------------------
# File Upload Section
# -----------------------------
uploaded_file = st.file_uploader("📤 Upload Stock Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Read data
    data = pd.read_csv(uploaded_file)

    st.subheader("📋 Dataset Preview")
    st.write(data.head())

    # Check required columns
    if 'Date' not in data.columns or 'Close' not in data.columns:
        st.error("❌ CSV must contain 'Date' and 'Close' columns!")
    else:
        # Process dates
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values('Date')

        # -----------------------------
        # Common Features
        # -----------------------------
        data['Day'] = data['Date'].dt.day
        data['Month'] = data['Date'].dt.month
        data['Year'] = data['Date'].dt.year

        # Select features and target
        X = data[['Day', 'Month', 'Year']]
        y = data['Close']

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        # -----------------------------
        # MODEL SELECTION
        # -----------------------------
        if model_choice == "Linear Regression":
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        elif model_choice == "Random Forest":
            model = RandomForestRegressor(
                n_estimators=200,
                random_state=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        else:
            # -----------------------------
            # Prepare data for LSTM
            # -----------------------------
            scaler = MinMaxScaler(feature_range=(0, 1))

            scaled_data = scaler.fit_transform(
                np.array(y).reshape(-1, 1)
            )

            train_size = int(len(scaled_data) * 0.8)

            train_data = scaled_data[:train_size]

            test_data = scaled_data[train_size - 60:]

            x_train, y_train = [], []

            for i in range(60, len(train_data)):
                x_train.append(
                    train_data[i-60:i, 0]
                )

                y_train.append(
                    train_data[i, 0]
                )

            x_train, y_train = np.array(x_train), np.array(y_train)

            x_train = np.reshape(
                x_train,
                (
                    x_train.shape[0],
                    x_train.shape[1],
                    1
                )
            )

            # -----------------------------
            # Build LSTM model
            # -----------------------------
            model = Sequential()

            model.add(
                LSTM(
                    32,
                    return_sequences=True,
                    input_shape=(x_train.shape[1], 1)
                )
            )

            model.add(
                LSTM(32)
            )

            model.add(
                Dense(16)
            )

            model.add(
                Dense(1)
            )

            model.compile(
                optimizer='adam',
                loss='mean_squared_error'
            )

            # -----------------------------
            # Train LSTM model
            # -----------------------------
            with st.spinner(
                "Training LSTM model... this may take a moment ⏳"
            ):
                model.fit(
                    x_train,
                    y_train,
                    epochs=5,
                    batch_size=32,
                    verbose=1
                )

            st.success(
                "✅ LSTM model training completed!"
            )

            # -----------------------------
            # Prepare test data
            # -----------------------------
            x_test, y_test_scaled = [], scaled_data[train_size:]

            for i in range(60, len(test_data)):
                x_test.append(
                    test_data[i-60:i, 0]
                )

            x_test = np.array(x_test)

            x_test = np.reshape(
                x_test,
                (
                    x_test.shape[0],
                    x_test.shape[1],
                    1
                )
            )

            # -----------------------------
            # LSTM Prediction
            # -----------------------------
            y_pred_scaled = model.predict(
                x_test,
                verbose=0
            )

            y_pred = scaler.inverse_transform(
                y_pred_scaled
            )

            y_test = y[len(y) - len(y_pred):]

        # -----------------------------
        # Model Evaluation
        # -----------------------------
        r2 = r2_score(y_test, y_pred)

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                y_pred
            )
        )

        st.subheader("📈 Model Performance")

        st.write(
            f"**Model Used:** {model_choice}"
        )

        st.write(
            f"**R² Score:** {r2:.3f}"
        )

        st.write(
            f"**RMSE:** {rmse:.3f}"
        )

        # -----------------------------
        # Visualization
        # -----------------------------
        st.subheader("📉 Actual vs Predicted Prices")

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.plot(
            y_test.values
            if model_choice != "LSTM (Deep Learning)"
            else y_test,
            label='Actual',
            color='blue'
        )

        ax.plot(
            y_pred,
            label='Predicted',
            color='orange'
        )

        ax.set_xlabel(
            "Index / Date"
        )

        ax.set_ylabel(
            "Close Price"
        )

        ax.legend()

        st.pyplot(fig)

        # -----------------------------
        # Future Forecast
        # -----------------------------
        st.subheader("🔮 Future Price Prediction")

        n_days = st.slider(
            "Select number of future days to predict:",
            1,
            60,
            7
        )

        last_date = data['Date'].max()

        future_dates = [
            last_date + timedelta(days=i+1)
            for i in range(n_days)
        ]

        if model_choice in [
            "Linear Regression",
            "Random Forest"
        ]:

            future_df = pd.DataFrame({
                'Date': future_dates,

                'Day': [
                    d.day
                    for d in future_dates
                ],

                'Month': [
                    d.month
                    for d in future_dates
                ],

                'Year': [
                    d.year
                    for d in future_dates
                ]
            })

            future_pred = model.predict(
                future_df[
                    ['Day', 'Month', 'Year']
                ]
            )

        else:
            # LSTM future forecast

            last_60_days = scaled_data[-60:]

            future_input = list(
                last_60_days.flatten()
            )

            future_pred_scaled = []

            for _ in range(n_days):

                x_future = np.array(
                    future_input[-60:]
                ).reshape(
                    (1, 60, 1)
                )

                next_pred = model.predict(
                    x_future,
                    verbose=0
                )[0, 0]

                future_input.append(
                    next_pred
                )

                future_pred_scaled.append(
                    next_pred
                )

            future_pred = scaler.inverse_transform(
                np.array(
                    future_pred_scaled
                ).reshape(-1, 1)
            ).flatten()

            future_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted Close Price': future_pred
            })

        # Display and plot
        future_df['Predicted Close Price'] = future_pred

        st.write(future_df)

        fig2, ax2 = plt.subplots(
            figsize=(10, 5)
        )

        ax2.plot(
            future_df['Date'],
            future_df['Predicted Close Price'],
            marker='o',
            color='green'
        )

        ax2.set_title(
            f"Future {n_days}-Day Forecast ({model_choice})"
        )

        ax2.set_xlabel(
            "Date"
        )

        ax2.set_ylabel(
            "Predicted Close Price"
        )

        st.pyplot(fig2)

        # -----------------------------
        # Download Option
        # -----------------------------
        csv = future_df.to_csv(
            index=False
        ).encode('utf-8')

        st.download_button(
            label="📥 Download Future Predictions as CSV",
            data=csv,
            file_name=(
                f'future_predictions_'
                f'{model_choice.lower().replace(" ", "_")}.csv'
            ),
            mime='text/csv'
        )

else:
    st.info(
        "👆 Please upload a CSV file to start the prediction."
    )

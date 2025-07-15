import streamlit as st
import pandas as pd
from datetime import date, datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import statsmodels.api as sm
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="USD Strength Dashboard", layout="wide")

#---------- CUSTOM CSS ----------
st.markdown("""
    <style>
    /* Metric box styling (unchanged) */
    [data-testid="stMetric"] {
        background-color: #1c1c1c;
        color: white;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0px 0px 8px rgba(255, 255, 255, 0.05);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    /* Style for st.metric() boxes */
    [data-testid="stMetric"] {
        background-color: #000000; /* pure black */
        color: #d3d3d3;  /* light grey text */
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0px 0px 8px rgba(255, 255, 255, 0.05);
        text-align: center;
    }

    /* Ensure inner text like delta and value is also grey */
    [data-testid="stMetric"] div {
        color: #d3d3d3 !important;
    }
    </style>
""", unsafe_allow_html=True)



# ---------- SIDEBAR INPUTS ----------
st.sidebar.image("D:\Raghav\Georgetown\microsoft_logo.png", use_container_width=True)
st.sidebar.title("⚙️ Model Configuration")

country_currency = st.sidebar.selectbox(
    "Country and Currency",
    options=["Japan - JPY", "Australia - AUD", "Canada - CAD", "Eurozone - EUR", "United Kingdom - GBP"]
)

if country_currency == "Japan - JPY":
    data = pd.read_csv(r"C:\Users\Raghav\Downloads\Japan Dataset(Exchange Rate (2)).csv")
    currency_symbol = "¥"
    base_quote = "USD/JPY"
# elif country_currency == "Australia - AUD":
#     data = pd.read_csv(r"C:\Users\Raghav\Downloads\Australia Dataset(Exchange Rate (2)).csv")
#     currency_symbol = "A$"
#     base_quote = "USD/AUD"

# elif country_currency == "Canada - CAD":
#     data = pd.read_csv(r"C:\Users\Raghav\Downloads\Canada Dataset(Exchange Rate (2)).csv")
#     currency_symbol = "C$"
#     base_quote = "USD/CAD"

# elif country_currency == "United Kingdom - GBP":
#     data = pd.read_csv(r"C:\Users\Raghav\Downloads\UK Dataset(Exchange Rate (2)).csv")
#     currency_symbol = "£"
#     base_quote = "USD/GBP"    

else:
    st.warning("Dataset for this country is not loaded yet.")
    st.stop()

    
data[f"Lagged {base_quote}"] = data[f"{base_quote} (Monthly)"].shift(1)
data[f"{base_quote} (Monthly)"] = np.log(data[f"{base_quote} (Monthly)"])
data[f"Lagged {base_quote}"] = np.log(data[f"Lagged {base_quote}"])

data = data.dropna()

target_col = f"{base_quote} (Monthly)"
features_all = [col for col in data.columns if col not in ["Date", target_col]]


# Separate raw X and y
X_raw_all = data[features_all]
y_raw = data[target_col]

# Sidebar: Variable selector
independent_vars = st.sidebar.multiselect(
    "Independent Variables",
    options=X_raw_all.columns.tolist(),
    default=X_raw_all.columns.tolist()
)

# Handle empty variable selection
if len(independent_vars) == 0:
    st.error("Please select at least one independent variable.")
    st.stop()

# Drop zero-variance features
X_std_check = X_raw_all[independent_vars].std()
zero_std_vars = X_std_check[X_std_check == 0].index.tolist()

if zero_std_vars:
    st.warning(f"The following variables had no variation and were removed: {', '.join(zero_std_vars)}")
    independent_vars = [var for var in independent_vars if var not in zero_std_vars]

# Final predictor set
X_raw = X_raw_all[independent_vars]


if len(independent_vars) == 0:
    st.error("Please select at least one independent variable.")
    st.stop()


# 🔍 Drop any predictors with 0 standard deviation (flat series)
X_std_check = X_raw_all.std()
zero_std_vars = X_std_check[X_std_check == 0].index.tolist()

if zero_std_vars:
    st.warning(f"These variables have no variation and were removed: {', '.join(zero_std_vars)}")
    X_raw = X_raw_all.drop(columns=zero_std_vars)
    independent_vars = [v for v in independent_vars if v not in zero_std_vars]


model_choice = st.sidebar.radio(
    "Select Model Type:",
    ["Scorecard", "Gradient Boosting"]
)

chart_placeholder = st.empty()

def create_exchange_rate_chart(actual_series, prediction, prediction_date):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual_series.index, y=actual_series.values, name="Actual FX Rate"))
    fig.add_trace(go.Scatter(x=[prediction_date], y=[prediction], mode='markers', name="Prediction", marker=dict(color='red', size=10)))
    fig.update_layout(title="USD Exchange Rate with Prediction")
    return fig


# Helper function to categorize residual jumps (same as your model)
def categorize_resid_jump(residual, threshold):
    if residual > threshold:
        return 1
    elif residual < -threshold:
        return -1
    else:
        return 0


# Gradient Boosting Residual Model

def rolling_linear_plus_xgb_corrected(
    dataset,
    features,
    target_col,
    date_col="Date",
    initial_train_size=180,
    test_window=1,
    max_lag=1,
    min_history_for_lags=10  # Minimum residuals before using lagged features
):
    n_obs = len(dataset)
    n_splits = n_obs - initial_train_size - test_window + 1

    # Store residuals from previous predictions
    residuals_full = np.full(n_obs, np.nan)

    # Store ALL predictions and actuals for final metric calculation
    all_preds = []
    all_actuals = []
    direction_correct = []
    dates_tested = []
    previous_actual = None

    base_features = features.copy()
    
    print(f"Starting corrected rolling window forecasting...")
    print(f"Dataset size: {n_obs}")
    print(f"Initial train size: {initial_train_size}")
    print(f"Number of splits: {n_splits}")
    print(f"Will start using lagged features after {min_history_for_lags} iterations")
    print("=" * 60)

    for i in range(n_splits):
        train_start = 0
        train_end = initial_train_size + i
        test_start = train_end
        test_end = test_start + test_window

        # Check if we have enough residual history to use lagged features
        valid_residuals_count = np.sum(~np.isnan(residuals_full[:test_start]))
        use_lagged_features = valid_residuals_count >= min_history_for_lags

        # Calculate threshold from past residuals
        if i > 0:
            past_residuals = residuals_full[:test_start]
            valid_residuals = past_residuals[~np.isnan(past_residuals)]
            threshold = np.mean(np.abs(valid_residuals)) if len(valid_residuals) > 0 else 0
            # Use mean of valid residuals as default instead of 0
            default_lag_value = np.mean(valid_residuals) if len(valid_residuals) > 0 else 0
        else:
            threshold = 0
            default_lag_value = 0

        # Prepare training data
        df_train = dataset.iloc[train_start:train_end].copy()
        
        # Determine which features to use
        if use_lagged_features:
            # Create lagged residual features for training data
            for lag in range(1, max_lag + 1):
                lag_col = f"resid_lag{lag}"
                lagged_values = []
                
                for idx in range(train_start, train_end):
                    lag_idx = idx - lag
                    if (lag_idx >= 0 and 
                        lag_idx < len(residuals_full) and 
                        not np.isnan(residuals_full[lag_idx])):
                        lagged_values.append(residuals_full[lag_idx])
                    else:
                        # Use mean of available residuals instead of 0
                        lagged_values.append(default_lag_value)
                
                df_train[lag_col] = lagged_values

            # Create jump categorical feature for training data
            jump_cat_train = []
            for idx in range(train_start, train_end):
                lag_idx = idx - 1
                if (lag_idx >= 0 and 
                    lag_idx < len(residuals_full) and 
                    not np.isnan(residuals_full[lag_idx])):
                    jump_cat_train.append(categorize_resid_jump(residuals_full[lag_idx], threshold))
                else:
                    jump_cat_train.append(0)  # Keep 0 for jump categories
            
            df_train["resid_jump_lag1"] = jump_cat_train
            
            # Features include lagged features
            current_features = base_features + [f"resid_lag{lag}" for lag in range(1, max_lag + 1)] + ["resid_jump_lag1"]
        else:
            # Only use base features
            current_features = base_features

        # Fill any remaining NaN values
        df_train = df_train.fillna(0)

        X_train = df_train[current_features]
        y_train = df_train[target_col]

        # Step 1: Linear regression
        lin_model = LinearRegression()
        lin_model.fit(X_train, y_train)
        lin_preds_train = lin_model.predict(X_train)
        residuals_train = y_train - lin_preds_train

        # Step 2: XGBoost on residuals
        xgb_model = XGBRegressor(
            n_estimators=9,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.85,
            reg_alpha=0.01581,
            reg_lambda=0.7,
            gamma=0.00100,
            random_state=8,
            verbosity=0,
        )
        xgb_model.fit(X_train, residuals_train)

        # Prepare test data
        df_test = dataset.iloc[test_start:test_end].copy()

        if use_lagged_features:
            # Add lagged features for test data
            for lag in range(1, max_lag + 1):
                lag_col = f"resid_lag{lag}"
                lag_idx = test_start - lag
                if (lag_idx >= 0 and 
                    lag_idx < len(residuals_full) and 
                    not np.isnan(residuals_full[lag_idx])):
                    df_test[lag_col] = residuals_full[lag_idx]
                else:
                    df_test[lag_col] = default_lag_value

            # Add jump categorical feature for test data
            lag_idx = test_start - 1
            if (lag_idx >= 0 and 
                lag_idx < len(residuals_full) and 
                not np.isnan(residuals_full[lag_idx])):
                jump_val = categorize_resid_jump(residuals_full[lag_idx], threshold)
            else:
                jump_val = 0
            df_test["resid_jump_lag1"] = jump_val

        df_test = df_test.fillna(0)

        X_test = df_test[current_features]
        y_test = df_test[target_col]

        # Make predictions
        lin_pred = lin_model.predict(X_test)
        xgb_pred = xgb_model.predict(X_test)
        preds = lin_pred + xgb_pred

        # Store residuals for future iterations
        resid = y_test.values - preds
        residuals_full[test_start:test_end] = resid

        # Store results for final metric calculation
        all_preds.extend(preds.tolist())
        all_actuals.extend(y_test.tolist())
        dates_tested.extend(dataset[date_col].iloc[test_start:test_end].tolist())

        # Calculate direction accuracy
        current_actual = y_test.iloc[0]
        current_pred = preds[0]
        
        if previous_actual is not None:
            actual_direction = current_actual - previous_actual
            pred_direction = current_pred - previous_actual
            
            if actual_direction != 0:
                direction_correct.append(np.sign(pred_direction) == np.sign(actual_direction))
        
        previous_actual = current_actual

        # Print progress for key iterations
        if i < 5 or i == min_history_for_lags or i % 10 == 0:
            lag_status = "WITH LAGS" if use_lagged_features else "BASE ONLY"
            print(f"Iteration {i+1:2d}: {lag_status} | "
                  f"Valid residuals: {valid_residuals_count:2d} | "
                  f"Features: {len(current_features):2d} | "
                  f"Pred: {preds[0]:.4f} | "
                  f"Actual: {current_actual:.4f} | "
                  f"Error: {resid[0]:+.4f}")

    # Calculate final metrics across ALL predictions
    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)
    
    final_rmse = np.sqrt(mean_squared_error(all_actuals, all_preds))
    final_mae = mean_absolute_error(all_actuals, all_preds)
    direction_acc = np.mean(direction_correct) if direction_correct else None

    print("\n" + "=" * 60)
    print("FINAL RESULTS:")
    print("=" * 60)
    print(f"Total predictions: {len(all_preds)}")
    print(f"Predictions using base features only: {min_history_for_lags}")
    print(f"Predictions using lagged features: {len(all_preds) - min_history_for_lags}")
    print(f"RMSE: {final_rmse:.6f}")
    print(f"MAE: {final_mae:.6f}")
    print(f"RMSE - MAE: {final_rmse - final_mae:.6f}")
    
    if direction_acc is not None:
        print(f"Direction Accuracy: {direction_acc * 100:.2f}%")
    
    # Show feature importance from the last model (if lagged features were used)
    if use_lagged_features:
        feature_importance = list(zip(current_features, xgb_model.feature_importances_))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        print(f"\nTop 5 Feature Importances (from last model):")
        for feat, imp in feature_importance[:5]:
            print(f"  {feat}: {imp:.4f}")
    
    return {
        "rmse": final_rmse,
        "mae": final_mae,
        "direction_accuracy": direction_acc,
        "dates_tested": dates_tested,
        "predictions": all_preds.tolist(),
        "actuals": all_actuals.tolist(),
        "residuals": residuals_full,
        "lagged_features_start": min_history_for_lags
    }


results = rolling_linear_plus_xgb_corrected(
    dataset=data,
    features=independent_vars,        # clean variable list selected via sidebar
    target_col=target_col,            # consistently defined earlier
    date_col="Date",                  # assumed available in your dataset
    min_history_for_lags=10           # you can adjust this if needed
)

# Create a DataFrame of predictions vs actuals
pred_df = pd.DataFrame({
    "Date": results["dates_tested"],
    "Actual FX Rate (log)": results["actuals"],
    "Predicted FX Rate (log)": results["predictions"],
    "Residual": np.array(results["actuals"]) - np.array(results["predictions"])
})

# Convert Date to datetime if needed
pred_df["Date"] = pd.to_datetime(pred_df["Date"])

# Optional: Convert log FX rates back to level
pred_df["Actual FX Rate"] = np.exp(pred_df["Actual FX Rate (log)"])
pred_df["Predicted FX Rate"] = np.exp(pred_df["Predicted FX Rate (log)"])

# Reorder columns
pred_df = pred_df[["Date", "Actual FX Rate", "Predicted FX Rate", "Residual"]]

# Display in Streamlit
st.subheader("📈 Prediction Results")
st.dataframe(pred_df, use_container_width=True)
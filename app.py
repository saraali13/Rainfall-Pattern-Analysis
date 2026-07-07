import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Rainfall System", layout="wide")

st.title("🌦 Rainfall Analysis & Prediction System")
st.markdown("### District: Kohat")

# ---------------------------
# SIDEBAR
# ---------------------------
option = st.sidebar.radio(
    "Select Module",
    ["📈 Report Analysis (2024)", "📊 AI Prediction"]
)

# =====================================================
# 🟩 MODULE 1: REPORT ANALYSIS
# =====================================================
if option == "📈 Report Analysis (2024)":

    st.header("📈 Monthly Rainfall Analysis - 2024")

    # Data from your report
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    rainfall = [28.5, 40, 120, 161.38, 21.3, 50, 190, 195, 55.3, 25, 14.4, 7.9]

    df = pd.DataFrame({
        "Month": months,
        "Rainfall (mm)": rainfall
    })

    col1, col2 = st.columns(2)

    # Table
    with col1:
        st.subheader("📄 Data Table")
        st.dataframe(df)

    # Plot
    with col2:
        fig, ax = plt.subplots()
        ax.plot(months, rainfall, marker='o')
        ax.set_title("Monthly Rainfall Pattern (2024)")
        ax.set_ylabel("Rainfall (mm)")
        ax.set_xlabel("Months")
        ax.grid(True)

        st.pyplot(fig)

    # Insights
    st.subheader("Key Findings")

    st.markdown("""
    - 🌧️ **Peak rainfall** observed in **July & August**
    - 🌨 **Secondary peak** in **April**
    - ☀️ **Dry months**: May, November, December
    - 🔄 **Bimodal rainfall pattern** detected
    """)

    st.subheader("📘 Interpretation")

    st.markdown("""
    The rainfall distribution in Kohat shows a bimodal pattern, indicating two distinct rainy periods.
    This pattern is important for agricultural planning and water resource management.
    """)

# =====================================================
# 🟦 MODULE 2: AI PREDICTION
# =====================================================
elif option == "📊 AI Prediction":

    st.header("📊 AI-Based Rainfall Prediction")

    st.info("""
    🔄 **Recursive Forecasting Method:**
    - **2026**: Trained on historical data only
    - **2027**: Trained on historical data + 2026 predictions
    - **2028**: Trained on historical data + 2026 & 2027 predictions

    This creates a more realistic forecast where each year influences the next.
    """)

    data_path = Path(__file__).parent / "rainfall data.csv"

    if not data_path.exists():
        st.error(f"Dataset not found at: {data_path}")
        st.stop()

    # Load historical data
    historical_data = pd.read_csv(data_path)

    # Accept either 'rainfall' or the provided 'precipitation' column
    if 'rainfall' in historical_data.columns:
        target_col = 'rainfall'
    elif 'precipitation' in historical_data.columns:
        target_col = 'precipitation'
    else:
        st.error("CSV must contain a 'rainfall' or 'precipitation' column")
        st.stop()

    if 'system:time_start' in historical_data.columns:
        historical_data['system:time_start'] = pd.to_datetime(historical_data['system:time_start'], errors='coerce')

    # Prepare historical data
    historical_data = historical_data.dropna(subset=[target_col]).copy()
    historical_data['Month'] = historical_data['system:time_start'].dt.month
    historical_data['Year'] = historical_data['system:time_start'].dt.year
    historical_data = historical_data.dropna(subset=['Year', 'Month'])

    months_range = np.arange(1, 13)
    months_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Store predictions for each year
    predictions_dict = {}


    # Function to train model and predict next year
    def predict_next_year(training_data, target_year):
        """
        Train a Random Forest model on training data and predict for target_year
        """
        # Prepare features
        X_train = training_data[['Year', 'Month']]
        y_train = training_data[target_col]

        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predict for target year
        future_df = pd.DataFrame({
            'Year': [target_year] * 12,
            'Month': months_range
        })
        predictions = model.predict(future_df)

        # Ensure no negative values
        predictions = np.maximum(predictions, 0)

        return predictions, model


    # Step 1: Predict 2026 using only historical data
    st.subheader("📊 Step 1: Predicting 2026")
    st.markdown("Training on historical data only...")

    predictions_2026, model_2026 = predict_next_year(historical_data, 2026)
    predictions_dict[2026] = predictions_2026

    # Create dataframe for 2026 predictions
    pred_2026_df = pd.DataFrame({
        'Month': months_labels,
        'Predicted Rainfall (mm)': [round(x, 2) for x in predictions_2026]
    })

    # Display 2026 predictions
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(pred_2026_df, use_container_width=True)
    with col2:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(months_labels, predictions_2026, marker='o', color='blue')
        ax.set_title("2026 Predictions")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    # Step 2: Create training data for 2027 (historical + 2026 predictions)
    st.subheader("📊 Step 2: Predicting 2027")
    st.markdown("Training on historical data + 2026 predictions...")

    # Create extended dataset for 2027 training
    train_2027_data = historical_data.copy()

    # Add 2026 predictions as if they were actual observations
    for month in months_range:
        new_row = pd.DataFrame({
            'Year': [2026],
            'Month': [month],
            target_col: [predictions_2026[month - 1]],
            'system:time_start': [pd.Timestamp(f'2026-{month}-01')]
        })
        train_2027_data = pd.concat([train_2027_data, new_row], ignore_index=True)

    # Predict 2027
    predictions_2027, model_2027 = predict_next_year(train_2027_data, 2027)
    predictions_dict[2027] = predictions_2027

    # Display 2027 predictions
    pred_2027_df = pd.DataFrame({
        'Month': months_labels,
        'Predicted Rainfall (mm)': [round(x, 2) for x in predictions_2027]
    })

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(pred_2027_df, use_container_width=True)
    with col2:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(months_labels, predictions_2027, marker='s', color='orange')
        ax.set_title("2027 Predictions")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    # Step 3: Create training data for 2028 (historical + 2026 + 2027 predictions)
    st.subheader("📊 Step 3: Predicting 2028")
    st.markdown("Training on historical data + 2026 & 2027 predictions...")

    # Create extended dataset for 2028 training
    train_2028_data = train_2027_data.copy()

    # Add 2027 predictions
    for month in months_range:
        new_row = pd.DataFrame({
            'Year': [2027],
            'Month': [month],
            target_col: [predictions_2027[month - 1]],
            'system:time_start': [pd.Timestamp(f'2027-{month}-01')]
        })
        train_2028_data = pd.concat([train_2028_data, new_row], ignore_index=True)

    # Predict 2028
    predictions_2028, model_2028 = predict_next_year(train_2028_data, 2028)
    predictions_dict[2028] = predictions_2028

    # Display 2028 predictions
    pred_2028_df = pd.DataFrame({
        'Month': months_labels,
        'Predicted Rainfall (mm)': [round(x, 2) for x in predictions_2028]
    })

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(pred_2028_df, use_container_width=True)
    with col2:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(months_labels, predictions_2028, marker='^', color='green')
        ax.set_title("2028 Predictions")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    # Create combined dataframe for all years
    combined_data = []
    for target_year in [2026, 2027, 2028]:
        for i, month in enumerate(months_labels):
            combined_data.append({
                'Year': target_year,
                'Month': month,
                'Predicted Rainfall (mm)': round(predictions_dict[target_year][i], 2)
            })

    combined_df = pd.DataFrame(combined_data)

    # Save predictions to CSV
    combined_df.to_csv("predicted_rainfall_recursive_2026_2028.csv", index=False)

    # Year selector for detailed visualization
    st.subheader("📈 Detailed Analysis by Year")
    selected_year = st.selectbox(
        "Select Year to View Detailed Visualizations",
        [2026, 2027, 2028],
        format_func=lambda x: f"{x}"
    )

    # Create dataframe for selected year
    selected_pred_df = pd.DataFrame({
        "Month": months_labels,
        "Predicted Rainfall (mm)": [round(predictions_dict[selected_year][i], 2) for i in range(12)]
    })

    # Detailed visualizations for selected year
    col1, col2 = st.columns(2)

    # Line Chart
    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(months_labels, predictions_dict[selected_year], marker='o', linewidth=2, markersize=8)
        ax1.set_title(f"Monthly Rainfall Trend - {selected_year}", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Months")
        ax1.set_ylabel("Rainfall (mm)")
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.tick_params(axis='x', rotation=45)

        # Add value labels
        for i, value in enumerate(predictions_dict[selected_year]):
            ax1.annotate(f'{value:.1f}', (i, value), textcoords="offset points", xytext=(0, 10), ha='center')

        st.pyplot(fig1)

    # Bar Chart
    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        bars = ax2.bar(months_labels, predictions_dict[selected_year], alpha=0.8, edgecolor='black')
        ax2.set_title(f"Rainfall Distribution - {selected_year}", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Months")
        ax2.set_ylabel("Rainfall (mm)")
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        ax2.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, value in zip(bars, predictions_dict[selected_year]):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 2, f'{value:.1f}', ha='center', va='bottom')

        st.pyplot(fig2)

    # Interactive area chart for selected year
    st.markdown(f"### 📈 Interactive Rainfall Pattern - {selected_year}")
    chart_data = selected_pred_df.set_index("Month")
    st.area_chart(chart_data)

    # Year-over-Year Comparison
    st.subheader("📊 Year-over-Year Comparison (2026-2028)")

    # Line chart comparison
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.plot(months_labels, predictions_dict[2026], marker='o', label='2026 (Historical trained)', linewidth=2,
             markersize=8)
    ax3.plot(months_labels, predictions_dict[2027], marker='s', label='2027 (Trained on +2026)', linewidth=2,
             markersize=8)
    ax3.plot(months_labels, predictions_dict[2028], marker='^', label='2028 (Trained on +2026+2027)', linewidth=2,
             markersize=8)
    ax3.set_title("Recursive Forecasting Comparison (2026-2028)", fontsize=14, fontweight='bold')
    ax3.set_xlabel("Months", fontsize=12)
    ax3.set_ylabel("Rainfall (mm)", fontsize=12)
    ax3.legend(fontsize=11)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.tick_params(axis='x', rotation=45)
    st.pyplot(fig3)

    # Bar chart comparison
    fig4, ax4 = plt.subplots(figsize=(14, 6))
    x = np.arange(len(months_labels))
    width = 0.25

    bars1 = ax4.bar(x - width, predictions_dict[2026], width, label='2026', alpha=0.8, color='#1f77b4')
    bars2 = ax4.bar(x, predictions_dict[2027], width, label='2027', alpha=0.8, color='#ff7f0e')
    bars3 = ax4.bar(x + width, predictions_dict[2028], width, label='2028', alpha=0.8, color='#2ca02c')

    ax4.set_xlabel("Months", fontsize=12)
    ax4.set_ylabel("Rainfall (mm)", fontsize=12)
    ax4.set_title("Rainfall Distribution Comparison (Recursive Method)", fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(months_labels, fontsize=10)
    ax4.legend(fontsize=11)
    ax4.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig4)

    # Summary Statistics
    st.subheader("📊 Summary Statistics")

    summary_data = []
    for year in [2026, 2027, 2028]:
        predictions = predictions_dict[year]
        training_info = "Historical only" if year == 2026 else (
            "Historical + 2026" if year == 2027 else "Historical + 2026 + 2027")

        summary_data.append({
            'Year': year,
            'Training Data': training_info,
            'Total Annual Rainfall (mm)': round(np.sum(predictions), 2),
            'Average Monthly (mm)': round(np.mean(predictions), 2),
            'Maximum (mm)': round(np.max(predictions), 2),
            'Minimum (mm)': round(np.min(predictions), 2),
            'Peak Month': months_labels[np.argmax(predictions)],
            'Std Deviation': round(np.std(predictions), 2)
        })

    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)

    # Training Data Growth Visualization
    st.subheader("📈 Impact of Recursive Training")

    fig5, ax5 = plt.subplots(figsize=(10, 5))
    years = [2026, 2027, 2028]
    annual_totals = [np.sum(predictions_dict[year]) for year in years]

    ax5.plot(years, annual_totals, marker='o', linewidth=2, markersize=10, color='purple')
    ax5.set_title("Effect of Recursive Training on Annual Rainfall", fontsize=14, fontweight='bold')
    ax5.set_xlabel("Year", fontsize=12)
    ax5.set_ylabel("Total Annual Rainfall (mm)", fontsize=12)
    ax5.grid(True, linestyle='--', alpha=0.7)

    # Add value labels
    for i, (year, total) in enumerate(zip(years, annual_totals)):
        ax5.annotate(f'{total:.1f} mm', (year, total), textcoords="offset points", xytext=(0, 10), ha='center')

    st.pyplot(fig5)

    # Show data table for all years
    st.subheader("📄 Complete Prediction Table (2026-2028)")
    st.dataframe(combined_df, use_container_width=True)

    # Download button for combined predictions
    csv = combined_df.to_csv(index=False)
    st.download_button(
        label="📥 Download All Predictions (2026-2028)",
        data=csv,
        file_name="rainfall_predictions_recursive_2026_2028.csv",
        mime="text/csv"
    )

    # Feature importance explanation
    with st.expander("ℹ️ About Recursive Forecasting"):
        st.markdown("""
        ### How Recursive Forecasting Works:

        1. **2026 Prediction**: 
           - Trained ONLY on historical data
           - Most reliable as it's based on actual observations

        2. **2027 Prediction**: 
           - Trained on historical data + 2026 predictions
           - 2026 predictions are treated as "actual" data points
           - Shows how the model evolves with new information

        3. **2028 Prediction**: 
           - Trained on historical data + 2026 + 2027 predictions
           - Demonstrates cumulative effect of recursive learning

        ### Advantages:
        - More realistic for time series forecasting
        - Shows how predictions evolve over time
        - Captures long-term patterns and trends
        - Each year's prediction influences the next

        ### Limitations:
        - Errors can compound over time
        - Predictions become less certain further into the future
        - Requires careful validation with actual data when available
        """)

    st.success(
        "✅ Predictions completed! Each year's predictions were used to train the model for the next year.")
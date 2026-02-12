import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import re
from datetime import datetime

# CONFIGURATION
LOG_FILE = "aimlFin2026_1_baklaga/task_3/l_baklaga_89734_server.log"

def analyze():
    print("1. Parsing Log File...")
    timestamps = []
    # REGEX: Precise match for format [2024-03-22 18:01:31+04:00]
    # We capture only the date and time part: 2024-03-22 18:01:31
    time_pattern = re.compile(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')

    try:
        with open(LOG_FILE, 'r') as f:
            for line in f:
                match = time_pattern.search(line)
                if match:
                    dt_obj = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
                    timestamps.append(dt_obj)
    except FileNotFoundError:
        print("❌ Error: Log file not found. Please download it first.")
        return

    if not timestamps:
        print("❌ Error: No timestamps parsed. The regex did not match the file format.")
        return

    # Create DataFrame & Sort
    df = pd.DataFrame(timestamps, columns=['time'])
    df = df.sort_values('time')

    # 2. Aggregate Traffic (Requests Per Second)
    traffic = df.groupby('time').size().reset_index(name='requests')
    start_time = traffic['time'].min()
    traffic['seconds'] = (traffic['time'] - start_time).dt.total_seconds()

    # 3. Robust Regression (Two-Pass) to Improve Fitting
    X = traffic[['seconds']]
    y = traffic['requests']
    
    # Pass 1: Initial Fit
    model_init = LinearRegression()
    model_init.fit(X, y)
    initial_pred = model_init.predict(X)
    
    # Identify "Normal" Traffic (Filter out massive spikes for training)
    # We define normal as traffic within the 90th percentile of residuals
    residuals_init = y - initial_pred
    threshold_init = residuals_init.quantile(0.90) 
    normal_traffic = traffic[residuals_init < threshold_init]
    
    # Pass 2: Fit on ONLY Normal Traffic (Robust Baseline)
    X_clean = normal_traffic[['seconds']]
    y_clean = normal_traffic['requests']
    
    model_robust = LinearRegression()
    model_robust.fit(X_clean, y_clean)
    
    # Predict on ALL data using the Robust Baseline
    traffic['predicted'] = model_robust.predict(X)
    traffic['residual'] = traffic['requests'] - traffic['predicted']

    # 4. Anomaly Detection (Statistical Z-Score)
    # Calculate stats based on NORMAL traffic variance
    clean_std = (y_clean - model_robust.predict(X_clean)).std()
    clean_mean = (y_clean - model_robust.predict(X_clean)).mean()
    
    # Calculate Z-Score for every point
    traffic['z_score'] = (traffic['residual'] - clean_mean) / clean_std
    
    # Strict Threshold: > 5 Sigma (Extremely high confidence)
    # Since we have a clean baseline, we can use a very high threshold to avoid false positives
    anomalies = traffic[traffic['z_score'] > 5]

    # 5. Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(traffic['time'], traffic['requests'], label='Actual Traffic', color='blue', alpha=0.4)
    plt.plot(traffic['time'], traffic['predicted'], label='Robust Regression Baseline', color='green', linestyle='--', linewidth=2)
    
    if not anomalies.empty:
        plt.scatter(anomalies['time'], anomalies['requests'], color='red', label=f'DDoS Attack (Max Z-Score: {traffic["z_score"].max():.1f})', zorder=5, s=15)
    
    plt.title('Task 3: Precision DDoS Detection (Robust Regression)')
    plt.xlabel('Time')
    plt.ylabel('Requests/Sec')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('aimlFin2026_1_baklaga/task_3/ddos_viz.png')
    print("✅ High-precision graph saved as ddos_viz.png")

    # 6. Report Generation
    print("\n" + "="*50)
    print("   RESULTS FOR YOUR REPORT (ddos.md)")
    print("="*50)
    
    if not anomalies.empty:
        print(f"Normal Baseline Traffic: ~{model_robust.intercept_:.1f} req/sec")
        print(f"Max Attack Traffic:      {anomalies['requests'].max()} req/sec")
        print(f"Statistical Confidence:  {traffic['z_score'].max():.1f} Sigma (Extremely High)")
        print(f"\n[DETECTED INTERVAL]")
        print(f"Start: {anomalies['time'].min()}")
        print(f"End:   {anomalies['time'].max()}")
    else:
        print("No anomalies detected.")

if __name__ == "__main__":
    analyze()

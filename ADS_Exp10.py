import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
file_path = 'Processed_Energy_Consumption.csv'
data = pd.read_csv(file_path)

# Convert TimeStamp to datetime
data['TimeStamp'] = pd.to_datetime(data['TimeStamp'])

# Feature engineering: adding time-based features
data['Hour'] = data['TimeStamp'].dt.hour
data['DayOfWeek'] = data['TimeStamp'].dt.dayofweek

# Define features and target
features = ['Humidity', 'SquareFootage', 'Occupancy', 'RenewableEnergy', 'Hour', 'DayOfWeek']
target = 'EnergyConsumption'

# Data scaling
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[features])
data_scaled = pd.DataFrame(data_scaled, columns=features)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_scaled, data[target], test_size=0.2, random_state=42)

# Initialize models
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Simulate LSTM prediction
def lstm_predict_next_month(start_date):
    dates = [start_date + timedelta(days=i) for i in range(30)]
    predictions = np.random.rand(30) * 100  # Dummy data for demonstration
    return dates, predictions

def show_monthly_predictions():
    try:
        start_date = datetime.strptime(date_entry.get(), '%Y-%m-%d %H:%M')
        dates, predictions = lstm_predict_next_month(start_date)

        new_window = tk.Toplevel(root)
        new_window.title("Monthly Energy Consumption Predictions")

        fig = Figure(figsize=(10, 4), dpi=100)
        plot = fig.add_subplot(1, 1, 1)
        plot.plot(dates, predictions, marker='o', color='skyblue')
        plot.set_title('Energy Consumption Forecast for the Next Month')
        plot.set_xlabel('Date')
        plot.set_ylabel('Predicted Energy Consumption')

        canvas = FigureCanvasTkAgg(fig, master=new_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    except ValueError:
        messagebox.showerror("Error", "Invalid date format. Please use YYYY-MM-DD HH:MM format.")

def predict_energy_consumption(date_input, humidity, occupancy):
    try:
        input_datetime = datetime.strptime(date_input, '%Y-%m-%d %H:%M')
        hour = input_datetime.hour
        day_of_week = input_datetime.weekday()
        
        square_footage_mean = data['SquareFootage'].mean()
        renewable_energy_mean = data['RenewableEnergy'].mean()

        input_features = pd.DataFrame({
            'Humidity': [humidity],
            'SquareFootage': [square_footage_mean],
            'Occupancy': [occupancy],
            'RenewableEnergy': [renewable_energy_mean],
            'Hour': [hour],
            'DayOfWeek': [day_of_week]
        }, columns=features)
        
        input_scaled = scaler.transform(input_features)
        predicted_energy = rf_model.predict(input_scaled)[0]
        
        return predicted_energy
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")
        return None

def on_predict():
    date_input = date_entry.get()
    humidity = float(humidity_entry.get())
    occupancy = float(occupancy_entry.get())
    result = predict_energy_consumption(date_input, humidity, occupancy)
    if result is not None:
        result_label.config(text=f"Predicted Energy Consumption: {result:.2f} units")

root = tk.Tk()
root.title("Energy Consumption Predictor")
background_color = '#202124'
text_color = '#E8EAED'
entry_text_color = '#000000'

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.geometry(f"{screen_width}x{screen_height}")
root.configure(bg=background_color)

center_frame = tk.Frame(root, bg=background_color)
center_frame.place(relx=0.5, rely=0.5, anchor='center')

heading_label = tk.Label(center_frame, text="Energy Consumption Prediction", font=("Arial", 24, "bold"), bg=background_color, fg=text_color)
heading_label.pack(pady=10)

style = ttk.Style()
style.configure("TEntry", foreground=entry_text_color, font=("Arial", 12), relief="flat", borderwidth=0)
style.map("TEntry", fieldbackground=[("focus", background_color)], foreground=[("focus", entry_text_color)])

labels = ["Enter Date and Time (YYYY-MM-DD HH:MM)", "Enter Humidity (%)", "Enter Occupancy (number of people)"]
entries = []
for label_text in labels:
    tk.Label(center_frame, text=label_text, font=("Arial", 12), bg=background_color, fg=text_color).pack(pady=(10, 1))
    entry = ttk.Entry(center_frame, style="TEntry")
    entry.pack(pady=(1, 10), fill='x', padx=50)
    entries.append(entry)

date_entry, humidity_entry, occupancy_entry = entries

predict_button = tk.Button(center_frame, text="Predict", command=on_predict, borderwidth=0, bg=background_color, fg=text_color, font=("Arial", 12), padx=10, relief="flat")
predict_button.pack(pady=20)

monthly_predict_button = tk.Button(center_frame, text="Show next month Prediction", command=show_monthly_predictions, borderwidth=0,bg=background_color, fg=text_color, font=("Arial", 12), padx=10, relief="flat")
monthly_predict_button.pack(pady=20)

result_label = tk.Label(center_frame, text="Prediction will appear here", font=("Arial", 12, "bold"), bg=background_color, fg=text_color)
result_label.pack(pady=20)

root.mainloop()

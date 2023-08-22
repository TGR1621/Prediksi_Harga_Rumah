import streamlit as st
import pickle
import locale
import os
import numpy as np
import pandas as pd

try:
    with open('linreg_model.pkl', 'rb') as f:
        linreg_model = pickle.load(f)

    with open('rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
except Exception as e:
    print("Error loading models:", e)

dataset = pd.read_csv('DATA RUMAH.csv', sep=';')
dataset = dataset.rename(columns={'NAMA RUMAH': 'NAMA_RUMAH'})
dataset = dataset.drop(columns='NO')

os.environ['LC_ALL'] = 'en_US.UTF-8'  # Replace with the appropriate locale
locale.setlocale(locale.LC_ALL, '')

# Define functions as before...

def main():
    st.title('Rumah Recommendation App')

    # Input form
    lb = st.number_input('Lebar Bangunan (lb)')
    lt = st.number_input('Luas Tanah (lt)')
    kt = st.number_input('Jumlah Kamar Tidur (kt)', step=1)
    km = st.number_input('Jumlah Kamar Mandi (km)', step=1)
    GRS = st.number_input('Luas Garasi (GRS)', step=1)

    if st.button('Predict'):
        input_data = [[lb, lt, kt, km, GRS]]

        # Get predictions for each model
        linreg_prediction = linreg_model.predict(input_data)[0]
        rf_prediction = rf_model.predict(input_data)[0]

        # Convert predictions to currency format
        linreg_prediction_str = currency_format(linreg_prediction, grouping=True)
        rf_prediction_str = currency_format(rf_prediction, grouping=True)

        # Get recommended houses
        rekomendasi = get_rekomendasi(linreg_prediction_str, rf_prediction_str, num_samples=5)

        # Display predictions and recommendations
        st.subheader('Predictions:')
        st.write(f'Linear Regression Prediction: {linreg_prediction_str}')
        st.write(f'Random Forest Prediction: {rf_prediction_str}')

        st.subheader('Recommended Houses:')
        for house in rekomendasi:
            st.write(f"House: {house['NAMA_RUMAH']}, Price: {house['HARGA']}")

if __name__ == '__main__':
    main()


import streamlit as st
from joblib import load
import numpy as np

loaded_rf_model = load('C:/Users/pro/Desktop/random_forest_model.joblib')
def predict_price(year, distance, car_type):
    # Convert car_type to numerical value (if needed)
    # Perform any additional preprocessing if necessary
    
    # Make predictions using the model
    l=np.zeros(12)
    l[0]=year
    l[1]=distance
    l[int(car_type)+2]=1
    


    prediction = loaded_rf_model.predict([l])

    return prediction[0]
# Streamlit app
def main():
    # Set the app title and description
    st.title('Car Price Prediction App')
    st.write("Welcome to the Car Price Prediction App! Enter the details below to get a price estimate.")
    car_type_mapping = {'برلیانس': 0, 'تیبا': 1, 'زوتی': 2, 'ساینا': 3, 'سیتروئن': 4,
                        'شاهین': 5, 'پراید': 6, 'چانگان': 7, 'کوییک': 8, 'کیا': 9}

    # Input components for user input
    #number = st.number_input("Insert a number", value=None, placeholder="Type a number...")
    year = st.number_input("year", value=None, placeholder="Type build year...")
    distance = st.number_input("Distance", value=None, placeholder="Type total distance that you car ride...")
    selected_car_type = st.selectbox('Select the Type of Car',
                                  {'برلیانس': 0, 'تیبا': 1, 'زوتی': 2, 'ساینا': 3, 'سیتروئن': 4,
                                   'شاهین': 5, 'پراید': 6, 'چانگان': 7, 'کوییک': 8, 'کیا': 9})
    #print(car_type_index)
    car_type_index = car_type_mapping[selected_car_type]
    

    #print(car_type)

    # Convert car_type to numerical value (if needed)

    # Button to trigger the prediction
    if st.button('Predict Price'):
        # Make prediction
        price_prediction = predict_price(year, distance, car_type_index)

        # Display the prediction
        st.success(f'Predicted Car Price: toman{price_prediction:,.2f}')

if __name__ == '__main__':
    main()
# Fire_Prediction_Models
The first model will determine if the alert is a red flag warning using XGBoost. The second model uses historical data to predict a fire in the location given the coordinates of such location. 

red_flag_warning.py will take alerts from NWS in California and classify them as a red flag warning or not.

fire_prediction_warning_classifier.py is a classifier that will divide map data into a grid and predict where the next fire is most likely to occur in the grid. 

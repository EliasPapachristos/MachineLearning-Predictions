# Import Library
from sklearn.externals import joblib

# Load the Model that we Trained Previously
model = joblib.load('trained_house_classifier_model.pkl')

# Provide Features of a New House. They Must be in the Same Order!
house_for_valuing = [
    # Features of the House
    1981,  # year_built
    1,  # stories
    2,  # num_bedrooms
    1,  # full_bathrooms
    1,  # half_bathrooms
    129,  # livable_sqft
    129,  # total_sqft
    0,  # garage_sqft
    0,  # carport_sqft
    False,  # has_fireplace
    False,  # has_pool
    True,  # has_central_heating
    True,  # has_central_cooling

    # Garage Type: We choose ONLY One of them
    0,  # attached
    0,  # detached
    1,  # none

    # City: We choose ONLY One of them
    0,  # Amystad
    1,  # Brownport
    0,  # Chadstad
    0,  # Clarkberg
    0,  # Coletown
    0,  # Davidfort
    0,  # Davidtown
    0,  # East Amychester
    0,  # East Janiceville
    0,  # East Justin
    0,  # East Lucas
    0,  # Fosterberg
    0,  # Hallfort
    0,  # Jeffreyhaven
    0,  # Jenniferberg
    0,  # Joshuafurt
    0,  # Julieberg
    0,  # Justinport
    0,  # Lake Carolyn
    0,  # Lake Christinaport
    0,  # Lake Dariusborough
    0,  # Lake Jack
    0,  # Lake Jennifer
    0,  # Leahview
    0,  # Lewishaven
    0,  # Martinezfort
    0,  # Morrisport
    0,  # New Michele
    0,  # New Robinton
    0,  # North Erinville
    0,  # Port Adamtown
    0,  # Port Andrealand
    0,  # Port Daniel
    0,  # Port Jonathanborough
    0,  # Richardport
    0,  # Rickytown
    0,  # Scottberg
    0,  # South Anthony
    0,  # South Stevenfurt
    0,  # Toddshire
    0,  # Wendybury
    0,  # West Ann
    0,  # West Brittanyview
    0,  # West Gerald
    0,  # West Gregoryview
    0,  # West Lydia
    0  # West Terrence
]
# scikit - learn assumes you want to predict the values for lots of houses at once, so it expects an array.
# But we want to look at a single house, so it will be the only item in our array.

homes_for_valuing = [
    house_for_valuing
]

# Run the Model and Make a Prediction for Each House in the "homes_for_valuing"
predicted_home_values = model.predict(homes_for_valuing)

# Let's see the Predicting of the house
predicted_value = predicted_home_values[0]
print('This House has Estimated Value of ${:,.2f}'.format(predicted_value))


# Import the Libraries
import pandas as pd
import webbrowser
import os

# Read the Dataset into a Data Table using Pandas
df = pd.read_csv('ml_house_data_set.csv')

# Let's see the first 5
df.head()

# Create a Web Page View of the Data for Easy Viewing
html = df[0 : 100].to_html()

# Save the html to a temporary file
with open("data.html", "w") as f:
    f.write(html)

# Saving the HTML to a Temporary File
full_file = os.path.abspath('data.html')
webbrowser.open('file://{}'.format(full_file))


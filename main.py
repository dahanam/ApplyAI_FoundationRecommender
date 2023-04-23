import pandas as pd
import numpy as np
import cv

# Load the dataset
df = pd.read_csv("C:\Users\dahan\OneDrive\Documents\shades.csv")

# Convert hexadecimal color codes to RGB values
df['R'] = df['hex'].apply(lambda x: int(x[1:3], 16))
df['G'] = df['hex'].apply(lambda x: int(x[3:5], 16))
df['B'] = df['hex'].apply(lambda x: int(x[5:7], 16))

# Split the data into training and testing sets
msk = np.random.rand(len(df)) < 0.8
train_df = df[msk]
test_df = df[~msk]

#Train a machine learning model

from sklearn.neighbors import KNeighborsClassifier

# Train the model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_df[['R', 'G', 'B']], train_df['L'])

from sklearn.metrics import accuracy_score

# Evaluate the model
y_pred = knn.predict(test_df[['R', 'G', 'B']])
accuracy = accuracy_score(test_df['L'], y_pred) #group replaced 'L'
print('Accuracy:', accuracy)

#temp --> basic implemintation
# Get user input
skin_tone = input('Enter your skin tone: ').lower()  ##getting the color

#in the new new model the prog gets the shade from the user's camera
#instead of asking

# Convert skin tone to RGB value
if skin_tone == 'fair':
    rgb = [255, 214, 186] ##use clusters instead
#    group=0
elif skin_tone == 'light':
    rgb = [255, 235, 199]
elif skin_tone == 'medium':
    rgb = [255, 159, 102]
elif skin_tone == 'tan':
    rgb = [255, 128, 64]
elif skin_tone == 'deep':
    rgb = [255, 102, 0]
else:
    print('Invalid skin tone')
    exit()

# Predict foundation shade based on RGB value
shade = knn.predict([rgb])[0]

print('Recommended foundation shade:', shade)
#not accurate --> recommended shade is not right
#fix the rgb ranges
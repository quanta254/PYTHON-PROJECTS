import zipfile
import os
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification


# Replace 'c:\\Users\\Richwanga\\Downloads\\rice+cammeo+and+osmancik.zip' with the actual path to your downloaded zip file
zip_file_path = 'c:\\Users\\Richwanga\\Downloads\\rice+cammeo+and+osmancik.zip'
extraction_path = 'c:\\Users\\Richwanga\\Downloads\\extracted_data'

# Create the extraction directory if it doesn't exist
os.makedirs(extraction_path, exist_ok=True)

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extraction_path)

import os

# Replace 'c:\\Users\\Richwanga\\Downloads\\extracted_data' with the actual path to your extracted folder
extracted_folder_path = 'c:\\Users\\Richwanga\\Downloads\\extracted_data'

# List all files in the extracted folder
files_in_extracted_folder = os.listdir(extracted_folder_path)

# Display the list of files
print("Files in the extracted folder:")
print(files_in_extracted_folder)


import os
import pandas as pd
from scipy.io import arff

# Replace 'c:\\Users\\Richwanga\\Downloads\\extracted_data' with the actual path to your extracted folder
extracted_folder_path = 'c:\\Users\\Richwanga\\Downloads\\extracted_data'

# Replace 'Rice_Cammeo_Osmancik.arff' with the actual ARFF filename
arff_file_path = os.path.join(extracted_folder_path, 'Rice_Cammeo_Osmancik.arff')

# Load the ARFF data into a Pandas DataFrame
data, meta = arff.loadarff(arff_file_path)
df = pd.DataFrame(data)

# Now you can explore and analyze the data using Pandas
# For example, you can display the first few rows of the DataFrame
print(df.head())


class_labels = ["b'Cammeo'"]  
label_mapping = {"b'Cammeo'":0, "b'Osmancik'":1}
integer_labels =[label_mapping[labels] for labels in class_labels]
print(integer_labels)
import pandas as pd


import pandas as pd


# Define a mapping for each class
class_mapping = {"b'Cammeo'": 0, "b'Osmansik'": 1}

# Map the values in the 'Class' column based on the defined mapping
df['Class'] = df['Class'].map(lambda x: class_mapping.get(x, x))

# Now the 'Class' column contains numerical values
print(df)
df["Class"] = (df["Class"] == "b'Cammeo'").astype(int)



# Filter data for the specific classes
selected_classes = ["b'Cammeo'", "b'Osmancik'"]
filtered_df = df[df['Class'].isin(selected_classes)]

# Plot histograms for 'Area' and 'Perimeter' columns
plt.figure(figsize=(10, 6))

# Plot histogram for 'Extent'
plt.subplot(2, 1, 1)
plt.hist(filtered_df[filtered_df['Class'] == "b'Cammeo'"]['Extent'], bins=30, alpha=0.5, label="Cammeo")
plt.hist(filtered_df[filtered_df['Class'] == "b'Osmancik'"]['Extent'], bins=30, alpha=0.5, label="Osmancik")
plt.title('Histogram of Extent')
plt.legend()

# Plot histogram for ' Area'
plt.subplot(2, 1, 2)
plt.hist(filtered_df[filtered_df['Class'] == "b'Cammeo'"]['Area'], bins=100, alpha=0.7, label="Cammeo")
plt.hist(filtered_df[filtered_df['Class'] == "b'Osmancik'"]['Area'], bins=100, alpha=0.7, label="Osmancik")
plt.title('Histogram of Area')
plt.legend()

plt.tight_layout()
plt.show()

import numpy as np
train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])


from sklearn.preprocessing import StandardScaler

print(df.columns)

# Extract features (X) and target variable (y), if applicable
# For example, assuming the last column is the target variable
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Use StandardScaler to scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# If you want to convert the scaled features back to a DataFrame
df_scaled = pd.DataFrame(X_scaled, columns=df.columns[:-1])

# Now 'df_scaled' contains the scaled features, and you can use it for further analysis
print(df_scaled.head())
print(len(train[train["Class"]==0  ]))
print(len(train[train["Class"]==0]))
print(train["Class"].unique())
print(train["Class"].dtype)
unique_values = train["Class"].unique()
print(unique_values)




X, y = make_classification(n_samples=1000, n_features=8, n_classes=2, random_state=42)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# You can print the shapes of the resulting arrays to verify the sizes
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)



knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
y_pred =knn_model.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB() 
nb_model=nb_model.fit(X_train, y_train)
y_pred =nb_model.predict(X_test)
print(classification_report(y_test, y_pred))


from sklearn.linear_model import LogisticRegression
lg_model = LogisticRegression() 
lg_model=lg_model.fit(X_train, y_train)
y_pred =lg_model.predict(X_test)
print(classification_report(y_test, y_pred))



from sklearn.svm import SVC
svm_model = SVC() 
svm_model=svm_model.fit(X_train, y_train)
y_pred =svm_model.predict(X_test)
print(classification_report(y_test, y_pred))


# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

# Generate a sample dataset (replace this with your own dataset)
X, y = make_classification(n_samples=3810, n_features=8, n_classes=2, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the neural network model
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(8,)))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0, validation_data=(X_test, y_test))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')


import matplotlib.pyplot as plt

# Assuming your model is already defined and compiled

# Train the model and get the history
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Plot training and validation loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
print(classification_report(y_test, y_pred))




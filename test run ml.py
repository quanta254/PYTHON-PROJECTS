import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
import os
import pandas as pd
zip_file_path = r"c:\Users\Richwanga\Downloads\rice+cammeo+and+osmancik.zip"

# Directory to extract the contents
extract_to = r"c:\Users\Richwanga\Downloads\extracted_dataset"

# Create the extraction directory if it doesn't exist
os.makedirs(extract_to, exist_ok=True)

# Extract the contents of the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

# List the files in the extracted directory
extracted_files = os.listdir(extract_to)
# Assuming the extracted dataset contains a CSV file
# You may need to adjust the file name based on the actual contents
csv_file_name = [file for file in extracted_files if file.endswith('.csv')]
csv_file_path = os.path.join(extract_to, csv_file_name)

# Load the CSV file into a DataFrame
data = pd.read_csv(csv_file_path)



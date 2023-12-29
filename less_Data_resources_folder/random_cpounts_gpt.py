import pandas as pd
import numpy as np
import random

# Load the dataset
file_path = '/Users/sreebhargavibalija/Desktop/flower_self_driving_vehciles_fed_fineutnig/sentiment_analysis_self_driving_vehicles.csv'
data = pd.read_csv(file_path)

# Analyzing the sentiment distribution
sentiment_distribution = data['Sentiment'].value_counts(normalize=True)

# Number of records to generate
num_records_to_generate = 200

# Generate sentiments based on the existing distribution
generated_sentiments = np.random.choice(sentiment_distribution.index, 
                                        size=num_records_to_generate, 
                                        p=sentiment_distribution.values)

# Function to generate text by randomly combining words or phrases
def generate_text_corrected(data):
    random_row = data.sample().iloc[0]
    words = random_row['Text'].split()
    num_words = random.randint(5, min(15, len(words)))
    return ' '.join(random.sample(words, num_words))

# Generate texts with the corrected function
generated_texts_corrected = [generate_text_corrected(data) for _ in range(num_records_to_generate)]

# Creating the new DataFrame with the generated data
generated_data_corrected = pd.DataFrame({
    'Text': generated_texts_corrected,
    'Sentiment': generated_sentiments
})

# Save to CSV
output_file_path = 'output_file_path.csv'
generated_data_corrected.to_csv(output_file_path, index=False)

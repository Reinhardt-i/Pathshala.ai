import json
import csv
import random

# File paths
input_file = '/Users/abraar/Downloads/Betav1/dataset 2/final_json_formate_84K.json'
output_file = '/Users/abraar/Downloads/Betav1/dataset 2/random_5000_questions.csv'

# Load the JSON data
with open(input_file, 'r', encoding='utf-8-sig') as json_file:
    data = json.load(json_file)

# Randomly select 5,000 entries
random_sample = random.sample(data, 7000)

# Save to a new CSV with the format: Question,AnsText
with open(output_file, 'w', encoding='utf-8', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Question", "AnsText"])  # Write the header

    for entry in random_sample:
        question = entry.get('input', '').strip()
        answer = entry.get('output', '').strip()
        csv_writer.writerow([question, answer])

print(f"Random sample of 5,000 entries saved to {output_file}")
import json
import csv

# File paths
input_file = '/Users/abraar/Downloads/Betav1/dataset 2/final_json_formate_84K.json'
output_file = '/Users/abraar/Downloads/Betav1/dataset 2/output_csv.csv'

# Open and load the JSON data with utf-8-sig encoding to handle BOM
with open(input_file, 'r', encoding='utf-8-sig') as json_file:
    data = json.load(json_file)

# Open a CSV file for writing
with open(output_file, 'w', encoding='utf-8', newline='') as csv_file:
    # Create a CSV writer
    csv_writer = csv.writer(csv_file)

    # Write the header
    csv_writer.writerow(["Input", "Output"])

    # Write the data rows
    for entry in data:
        csv_writer.writerow([entry.get('input', '').strip(), entry.get('output', '').strip()])

print(f"CSV file has been saved to {output_file}")
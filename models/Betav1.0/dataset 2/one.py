import pandas as pd

# Load the input CSV file
input_file = "Betav1/dataset 2/Textbook Dataset from NCTB.csv"  # Replace with your file name
data = pd.read_csv(input_file)

# Separate the data
passage_data = data[['Passage']]
qa_data = data[['Question', 'AnsText']]

# Save to separate CSV files
passage_csv_path = "passages_only.csv"
qa_csv_path = "questions_answers.csv"

passage_data.to_csv(passage_csv_path, index=False, encoding='utf-8-sig')
qa_data.to_csv(qa_csv_path, index=False, encoding='utf-8-sig')

print(f"Passages saved to {passage_csv_path}")
print(f"Questions and Answers saved to {qa_csv_path}")
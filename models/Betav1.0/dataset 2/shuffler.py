import pandas as pd
import random

def shuffle_csv(input_file: str, output_file: str) -> None:
    """
    Reads a CSV file, shuffles its rows, and writes the shuffled rows to a new CSV file.
    
    :param input_file: Path to the input CSV file.
    :param output_file: Path to the output shuffled CSV file.
    """
    df = pd.read_csv(input_file)
    df = df.sample(frac=1, random_state=random.randint(0, 100000)).reset_index(drop=True)
    df.to_csv(output_file, index=False)

# Example usage:
# shuffle_csv("input.csv", "shuffled_output.csv")


shuffle_csv("Betav1/dataset 2/questions_answers.csv", "Betav1/dataset 2/questions_answers_sf.csv")
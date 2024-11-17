import csv

input_file = 'dataset/Bangla Song Lyrics/BanglaSongLyrics.csv'
output_file = 'dataset/Bangla Song Lyrics/out.txt'

try:
    with open(input_file, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')  # Tab-delimited CSV
        lyrics = []

        for row in csv_reader:
            lyrics.append(row['lyrics'])

    with open(output_file, 'w', encoding='utf-8') as txt_file:
        for lyric in lyrics:
            txt_file.write(lyric + '\n\n\n')

    print(f"Lyrics have been successfully written to {output_file}.")

except FileNotFoundError:
    print(f"Error: The file {input_file} was not found.")
except KeyError as e:
    print(f"Error: Missing column in CSV file - {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
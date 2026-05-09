import os
import csv

input_folder = "data"
output_folder = "csv"

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.endswith(".txt"):
        txt_path = os.path.join(input_folder, file)
        csv_path = os.path.join(output_folder, file.replace(".txt", ".csv"))

        with open(txt_path, "r") as txt_file, open(csv_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)

            for line in txt_file:
                # Split by any whitespace
                row = line.strip().split()
                if row:
                    writer.writerow(row)

        print(f"Converted: {file} → {csv_path}")

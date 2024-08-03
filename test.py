import csv
from datetime import datetime


# Function to append a single row to the CSV file
rows_names = ["time", "value", "status"]


def append_to_csv(file_name, row):
    with open(file_name, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=rows_names)

        # Check if the file is empty to write the header
        if file.tell() == 0:
            writer.writeheader()

        print(row.keys())
        print(row)
        writer.writerow(row)


# Define the CSV file name
csv_file = 'stats_log.csv'

# Sample data to log
stats = [
    {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "value": 23, "status": "OK"},
    {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "status": "FAIL", "value": 45},
    {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "value": 67, "status": "OK"},
    {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "value": 67, "status": "OK"},
]

# Append each row to the CSV file
for stat in stats:
    append_to_csv(csv_file, stat)

print(f"Stats appended to {csv_file}")

import csv
import sys
import os
from openpyxl import Workbook

def csv_to_xlsx(csv_file_path, xlsx_file_path):
    # Create a new workbook and access the active worksheet
    wb = Workbook()
    ws = wb.active
    ws.title = "Parameters"

    # Open the CSV file and read its contents
    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        # Append each row from the CSV into the worksheet
        for row in reader:
            ws.append(row)

    # Save the workbook as an XLSX file
    wb.save(xlsx_file_path)
    print(f"Converted {csv_file_path} to {xlsx_file_path}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python alaki.py <csv_file_path> [<xlsx_file_path>]")
        sys.exit(1)
    csv_file = sys.argv[1]
    if len(sys.argv) > 2:
        xlsx_file = sys.argv[2]
    else:
        base, _ = os.path.splitext(csv_file)
        xlsx_file = base + '.xlsx'
    csv_to_xlsx(csv_file, xlsx_file)
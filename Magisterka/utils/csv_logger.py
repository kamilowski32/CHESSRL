import csv
from pathlib import Path

class CSVLogger:
    def __init__(self, path, header):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(header)

    def log(self, row):
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()

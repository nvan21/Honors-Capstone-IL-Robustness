import os
from pathlib import Path

starting_path = os.path.join(".", "logs")
path = Path(starting_path)
print(os.walk(starting_path))
print(path.rglob("*"))
for f in path.rglob("*"):
    print(f)

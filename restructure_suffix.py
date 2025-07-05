import os
import re
from collections import defaultdict

def rename_suffixes():
    input_dir = 'data/data_v1'
    pattern = re.compile(r'^(.*?)-(\d+)\.json$')

    # Gruppiere Dateien nach Basisnamen
    groups = defaultdict(list)
    for filename in os.listdir(input_dir):
        match = pattern.match(filename)
        if match:
            base_name = match.group(1)
            suffix = int(match.group(2))
            groups[base_name].append((suffix, filename))

    # Verarbeite jede Gruppe
    for base_name, files in groups.items():
        # Sortiere Dateien nach Original-Suffix (aufsteigend)
        files.sort()
        sorted_suffixes = [suffix for suffix, _ in files]
        
        # Erstelle Mapping: Alter Suffix → Neuer Suffix (1,2,3...)
        suffix_mapping = {
            old: new + 1
            for new, old in enumerate(sorted_suffixes)
        }

        # Umbenennung in aufsteigender Reihenfolge
        for old_suffix, filename in files:
            new_suffix = suffix_mapping[old_suffix]
            new_filename = f"{base_name}-{new_suffix}.json"
            old_path = os.path.join(input_dir, filename)
            new_path = os.path.join(input_dir, new_filename)
            
            if old_path != new_path:
                os.rename(old_path, new_path)
                print(f"Umbenannt: {filename} → {new_filename}")

if __name__ == '__main__':
    rename_suffixes()
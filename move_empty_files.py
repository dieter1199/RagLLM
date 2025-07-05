import os
import json
import re
import shutil

def process_files():
    input_dir = '/home/dihoit00/test/data50'
    empty_dir = os.path.join('/home/dihoit00/test/data50/empty')
    
    # Ordnerstruktur erstellen
    os.makedirs(empty_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        
        # Nur JSON-Dateien verarbeiten
        if filename.endswith('.json') and os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    continue  # Bei fehlerhaften Dateien überspringen

            # Prüfe auf leere vulnerable_lines
            vulnerable_lines = data.get('vulnerable_lines', [])
            
            if not vulnerable_lines:
                # Verschiebe in empty-Ordner
                dest_path = os.path.join(empty_dir, filename)
                shutil.move(file_path, dest_path)
                print(f"Verschoben {filename} -> empty/")

if __name__ == '__main__':
    process_files()
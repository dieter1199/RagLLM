import os
import json
import re

def process_files():
    input_dir = '/home/dihoit00/test/data8000'
    output_dir = os.path.join('/home/dihoit00/test/data8000/only_code')
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Bereinige die Code-Zeilen
            cleaned_code = [
                re.sub(r'^// Line_Reference \d+:\s*', '', line)
                for line in data.get('vulnerable_lines', [])
            ]
            
            # Schreibe die Code-Zeilen direkt in die Datei (kein JSON-Format)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(cleaned_code))

if __name__ == '__main__':
    process_files()
import json
import os
import shutil
from glob import glob

def process_json_files(directory):
    # Erstelle empty-Ordner falls nicht vorhanden
    empty_dir = os.path.join(directory, "empty")
    os.makedirs(empty_dir, exist_ok=True)
    
    # Finde alle JSON-Dateien
    json_files = glob(os.path.join(directory, "*.json"))
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Filtere alle Changes ohne vulnerable_lines
        filtered_changes = [
            change for change in data.get("changes", [])
            if change.get("vulnerable_lines") and len(change["vulnerable_lines"]) > 0
        ]
        
        # Prüfe ob Datei leer ist
        if len(filtered_changes) == 0:
            # Verschiebe in empty-Ordner
            dest_path = os.path.join(empty_dir, os.path.basename(json_file))
            shutil.move(json_file, dest_path)
        else:
            # Aktualisiere und speichere
            data["changes"] = filtered_changes
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
                f.write('\n')

if __name__ == "__main__":
    directory_path = '/home/dihoit00/test/data8000'
    process_json_files(directory_path)
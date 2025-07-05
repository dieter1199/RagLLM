import json
import os
from glob import glob

def split_json_files(directory):
    json_files = glob(os.path.join(directory, "*.json"))
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Überspringe Dateien ohne changes-Array
        if 'changes' not in data:
            continue
            
        changes = data.get('changes', [])
        if not changes:
            continue
        
        # Vorbereite Basis-Dateinamen
        base_name = os.path.splitext(json_file)[0]
        ext = os.path.splitext(json_file)[1]
        
        # Erstelle für jeden Change-Eintrag eine neue Datei
        for index, change in enumerate(changes, 1):
            new_data = {
                key: value for key, value in data.items() if key != 'changes'
            }
            new_data.update(change)
            
            new_filename = f"{base_name}-{index}{ext}"
            with open(new_filename, 'w', encoding='utf-8') as f_out:
                json.dump(new_data, f_out, indent=4, ensure_ascii=False)
                f_out.write('\n')
        
        # Lösche Originaldatei nach der Verarbeitung
        os.remove(json_file)

if __name__ == "__main__":
    directory_path = '/home/dihoit00/test/data8000'
    split_json_files(directory_path)
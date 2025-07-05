import json
import os
from glob import glob

def process_json_files(directory):
    # Alle JSON-Dateien im angegebenen Ordner finden
    json_files = glob(os.path.join(directory, '*.json'))
    
    for json_file in json_files:
        # JSON-Datei öffnen und laden
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Durch jede Änderung in 'changes' iterieren
        for change in data.get('changes', []):
            # 'code_changes_summary' extrahieren und verarbeiten
            ccs = change.get('code_changes_summary', {})
            
            # 'added_mitigation_lines' entfernen
            if 'added_mitigation_lines' in ccs:
                del ccs['added_mitigation_lines']
            
            # 'deleted_vulnerable_lines' in 'vulnerable_lines' umbenennen
            vulnerable_lines = ccs.get('deleted_vulnerable_lines', [])
            
            # Altes Feld 'code_changes_summary' entfernen
            if 'code_changes_summary' in change:
                del change['code_changes_summary']
            
            # Neues Feld 'vulnerable_lines' hinzufügen
            change['vulnerable_lines'] = vulnerable_lines
        
        # Modifizierte Daten zurück in die Datei schreiben
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
            f.write('\n')  # Für eine neue Zeile am Ende der Datei

if __name__ == "__main__":
    directory_path = '/home/dihoit00/test/data8000'
    process_json_files(directory_path)
import os

def remove_empty_lines():
    code_dir = os.path.join('/home/dihoit00/test/data8000/only_code')
    
    for filename in os.listdir(code_dir):
        file_path = os.path.join(code_dir, filename)
        
        if os.path.isfile(file_path) and filename.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Filtere leere Zeilen (einschließlich Zeilen mit nur Whitespace)
            cleaned_lines = [line.rstrip('\n') + '\n' for line in lines if line.strip()]
            
            # Überschreibe die Datei nur wenn Änderungen vorliegen
            if len(cleaned_lines) != len(lines):
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(cleaned_lines)
                print(f"Bereinigt: {filename}")

if __name__ == '__main__':
    remove_empty_lines()
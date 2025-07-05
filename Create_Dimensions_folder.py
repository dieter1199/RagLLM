import os
import shutil

def create_dimension_folders():
    base_dir = "/beegfs/scratch/workspace/es_dihoit00-RAG/data/Dimensions"
    source_dir = os.path.join(base_dir, "data22842")
    source_code_dir = os.path.join(source_dir, "only_code")
    
    # Erstelle Basisordner falls nicht existent
    os.makedirs(source_dir, exist_ok=True)
    
    # Liste der gewünschten Größen
    sizes = [50, 100, 500] + list(range(1000, 22501, 500))
    
    # Holen der ersten 22500 Dateien aus dem Quellordner
    all_files = sorted(os.listdir(source_dir))
    max_files = 22500  # Maximale Anzahl zu kopierender Dateien
    
    for size in sizes:
        # Begrenze die Größe auf die maximale verfügbare Anzahl
        current_size = min(size, max_files)
        
        # Zielordner erstellen
        target_dir = os.path.join(base_dir, f"data{current_size}")
        target_code_dir = os.path.join(target_dir, "only_code")
        
        os.makedirs(target_code_dir, exist_ok=True)
        
        # Dateien kopieren
        for file in all_files[:current_size]:
            # Kopiere Hauptdatei
            src_file = os.path.join(source_dir, file)
            dst_file = os.path.join(target_dir, file)
            if not os.path.exists(dst_file):
                shutil.copy(src_file, dst_file)
            
            # Kopiere Code-Datei
            code_src = os.path.join(source_code_dir, file)
            code_dst = os.path.join(target_code_dir, file)
            if os.path.exists(code_src) and not os.path.exists(code_dst):
                shutil.copy(code_src, code_dst)
        
        print(f"Ordner {target_dir} mit {current_size} Dateien erstellt.")

if __name__ == "__main__":
    create_dimension_folders()
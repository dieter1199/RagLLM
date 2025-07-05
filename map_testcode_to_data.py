import os
import re

TESTCODE_DIR = "/home/dihoit00/test/TestCode_Dimensions"
JSONCODE_DIR = "/home/dihoit00/test/data/Dimensions/data22842/only_code"

def extract_core_name_for_testcode(filename: str) -> str:
    """
    Entfernt typische Präfixe/Suffixe, z.B.:
      - "36_NVD-CWE-noinfo_CVE-2023-44452-1_(0_20).h"
      => "NVD-CWE-noinfo_CVE-2023-44452"
    """
    # Endung entfernen (z.B. .h, .c etc.)
    base = re.sub(r'\.\w+$', '', filename)
    # Führende Ziffern+_ entfernen (z.B. "36_")
    base = re.sub(r'^\d+_', '', base)
    # Letztes Muster '-X_(...)' entfernen (z.B. '-1_(0_20)')
    base = re.sub(r'-\d+_\(.*\)$', '', base)
    return base

def extract_core_name_for_json(filename: str) -> str:
    """
    Entfernt typische Präfixe/Suffixe, z.B.:
      - "149_NVD-CWE-noinfo_CVE-2023-44452-4.json"
      => "NVD-CWE-noinfo_CVE-2023-44452"
    """
    # Endung entfernen (z.B. .json)
    base = re.sub(r'\.\w+$', '', filename)
    # Führende Ziffern+_ entfernen
    base = re.sub(r'^\d+_', '', base)
    # Am Ende '-Zahl' entfernen (z.B. '-4')
    base = re.sub(r'-\d+$', '', base)
    return base

def get_lines_from_file(filepath: str) -> list:
    """
    Liest den Inhalt einer Datei zeilenweise ein und gibt eine Liste zurück.
    Hier einfach als Liste von Strings (ge-`strip()`-t).
    """
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [line.strip() for line in f]
    # Optional: Leerzeilen aussortieren
    # lines = [l for l in lines if l]
    return lines

def main():
    # 1. Alle TestCode-Dateien in TESTCODE_DIR auflisten
    test_files = [
        f for f in os.listdir(TESTCODE_DIR)
        if os.path.isfile(os.path.join(TESTCODE_DIR, f))
    ]
    
    # 2. Alle JSON-Dateien in JSONCODE_DIR auflisten
    json_files = [
        f for f in os.listdir(JSONCODE_DIR)
        if f.endswith(".json") and os.path.isfile(os.path.join(JSONCODE_DIR, f))
    ]
    
    # 3. JSON-Dateien in Dictionary: "core_name" -> Liste[json_filename]
    from collections import defaultdict
    json_map = defaultdict(list)
    for jfile in json_files:
        c = extract_core_name_for_json(jfile)
        json_map[c].append(jfile)
    
    # 4. Für jede TestCode-Datei:
    for test_file in test_files:
        # a) Datei-Inhalt lesen
        test_path = os.path.join(TESTCODE_DIR, test_file)
        test_lines = get_lines_from_file(test_path)
        test_lines_set = set(test_lines)
        
        # b) Core-Name extrahieren
        core_name = extract_core_name_for_testcode(test_file)
        
        # c) Kandidaten (JSON-Dateien) besorgen
        candidates = json_map.get(core_name, [])
        
        # d) Check, ob es einen JSON-Kandidaten gibt, dessen alle Zeilen im TestCode enthalten sind
        matched_file = None
        for cand_json in candidates:
            json_path = os.path.join(JSONCODE_DIR, cand_json)
            json_lines = get_lines_from_file(json_path)
            json_lines_set = set(json_lines)
            
            # Prüfen, ob alle JSON-Zeilen im Test-Code vorhanden sind:
            if json_lines_set.issubset(test_lines_set):
                matched_file = cand_json
                break  # da nur eine Datei passen kann (laut deiner Beschreibung)
        countr = 0
        # e) Ausgabe
        if matched_file:
            print(f"{countr}, {test_file} -> {matched_file}")
            countr = countr + 1
        else:
            print(f"{countr}, {test_file} -> NO MATCH")
            countr = countr + 1

if __name__ == "__main__":
    main()

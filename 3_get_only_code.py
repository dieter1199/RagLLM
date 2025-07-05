import os
import json
import re

def extract_line_text(line: str) -> str:
    """
    Entfernt vorn das '// Line_Reference <num>:'-Muster und
    gibt den reinen Code-Text zurück.
    Beispiel:
      "// Line_Reference 123:   entry->storageType = ST_NONVOLATILE;"
    wird zu
      "entry->storageType = ST_NONVOLATILE;"
    """
    pattern = r'^\s*//\s*Line_Reference\s+\d+:\s*'
    return re.sub(pattern, '', line).strip()

def main():
    # 1) Ordner mit den JSON-Dateien
    data_dir = r"/home/dihoit00/test/data8000"  # Anpassen

    # 2) Unterordner für die Ausgabe der neuen Dateien
    output_subdir = os.path.join(data_dir, "only_code")
    os.makedirs(output_subdir, exist_ok=True)

    # Durch alle .json-Dateien iterieren
    for filename in os.listdir(data_dir):
        if not filename.lower().endswith(".json"):
            continue
        filepath = os.path.join(data_dir, filename)

        if not os.path.isfile(filepath):
            continue

        # JSON-Datei einlesen
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Fehler beim Einlesen von {filename}: {e}")
            continue

        if not isinstance(data, dict):
            continue

        changes_list = data.get("changes", [])
        if not isinstance(changes_list, list):
            changes_list = []

        # Alle vorhandenen Sets von deleted_vulnerable_lines sammeln
        sets_of_lines = []
        for chg in changes_list:
            code_summary = chg.get("code_changes_summary", {})
            deleted_lines = code_summary.get("deleted_vulnerable_lines", [])
            if not deleted_lines:
                continue

            # Extrahiere den reinen Text jeder Zeile
            line_texts = []
            for dl in deleted_lines:
                txt = extract_line_text(dl)
                if txt:
                    line_texts.append(txt)

            if line_texts:
                # Entduplizieren innerhalb desselben Blocks
                unique_lines = set(line_texts)
                sets_of_lines.append(frozenset(unique_lines))

        if not sets_of_lines:
            # Keine nichtleeren deleted_vulnerable_lines => nichts ausgeben
            continue

        # Mehrere identische Sets auf unique reduzieren
        unique_sets = []
        seen_sets = set()
        for s in sets_of_lines:
            if s not in seen_sets:
                seen_sets.add(s)
                unique_sets.append(s)

        # Pro eindeutiges Set eine neue Datei anlegen
        base_name, ext = os.path.splitext(filename)
        index_num = 1
        for s in unique_sets:
            if not s:
                continue

            new_filename = f"{base_name}-{index_num}{ext}"
            new_path = os.path.join(output_subdir, new_filename)

            sorted_lines = sorted(s)  # optional sortiert
            try:
                with open(new_path, "w", encoding="utf-8") as out_f:
                    for line_text in sorted_lines:
                        out_f.write(line_text + "\n")
                print(f"Erzeugt: {new_filename} => {len(s)} Zeilen")
            except Exception as e:
                print(f"Fehler beim Schreiben von {new_filename}: {e}")

            index_num += 1

if __name__ == "__main__":
    main()

import os
import shutil
import random

def main():
    # -- 1. Ordnerpfade definieren (bei Bedarf anpassen) --
    BASE_DIR = "/beegfs/scratch/workspace/es_dihoit00-RAG/data/Dimensions"
    DATA22842_DIR = os.path.join(BASE_DIR, "data22842")
    DATA22842_ONLYCODE_DIR = os.path.join(DATA22842_DIR, "only_code")
    
    MATCH_DIR = "/beegfs/scratch/workspace/es_dihoit00-RAG/match"
    MATCH_ONLYCODE_DIR = os.path.join(MATCH_DIR, "only_code")
    
    # Diese Dateien sollen gesucht, kopiert und in jedem Unterordner sichergestellt werden:
    needed_files = [
        "263_NVD-CWE-noinfo_CVE-2024-27930-3.json",
        "257_NVD-CWE-noinfo_CVE-2024-27915-3.json",
        "299_NVD-CWE-noinfo_CVE-2024-28239-1.json",
        "112_CWE-79_CVE-2024-21515-1.json",
        "355_CWE-23_CVE-2024-3025-1.json",
        "195_NVD-CWE-noinfo_CVE-2024-25410-1.json",
        "58_CWE-284_CVE-2023-50257-5.json",
        "311_NVD-CWE-noinfo_CVE-2024-29023-4.json",
        "313_NVD-CWE-noinfo_CVE-2024-29028-1.json",
        "196_CWE-287_CVE-2024-25618-2.json",
        "246_NVD-CWE-noinfo_CVE-2024-27304-15.json",
        "318_NVD-CWE-noinfo_CVE-2024-29036-4.json",
        "429_NVD-CWE-noinfo_CVE-2024-32481-5.json",
        "146_CWE-290_CVE-2024-23832-6.json",
        "645_CWE-20_CVE-2024-4941-3.json",
        "53_NVD-CWE-noinfo_CVE-2023-49508-2.json",
        "553_NVD-CWE-noinfo_CVE-2024-36404-2.json",
        "16_CWE-1333_CVE-2021-4437-1.json",
        "599_NVD-CWE-noinfo_CVE-2024-39697-2.json",
        "116_NVD-CWE-noinfo_CVE-2024-21583-1.json",
        "579_NVD-CWE-noinfo_CVE-2024-38360-4.json",
        "306_NVD-CWE-noinfo_CVE-2024-28866-1.json",
        "432_NVD-CWE-noinfo_CVE-2024-32489-1.json",
        "227_NVD-CWE-noinfo_CVE-2024-27088-1.json",
        "605_NVD-CWE-noinfo_CVE-2024-39903-3.json",
        "102_NVD-CWE-noinfo_CVE-2024-2035-3.json",
        "390_NVD-CWE-noinfo_CVE-2024-31455-10.json",
        "343_NVD-CWE-noinfo_CVE-2024-29889-1.json",
        "241_NVD-CWE-noinfo_CVE-2024-27296-1.json",
        "333_NVD-CWE-noinfo_CVE-2024-29199-9.json",
        "333_NVD-CWE-noinfo_CVE-2024-29199-25.json",
        "291_NVD-CWE-noinfo_CVE-2024-28195-1.json",
        "291_NVD-CWE-noinfo_CVE-2024-28195-5.json",
        "159_NVD-CWE-noinfo_CVE-2024-24749-1.json",
        "641_CWE-20_CVE-2024-4287-4.json",
        "201_NVD-CWE-noinfo_CVE-2024-25638-64.json",
        "629_NVD-CWE-noinfo_CVE-2024-4068-1.json",
        "49_NVD-CWE-noinfo_CVE-2023-48709-1.json",
        "107_NVD-CWE-noinfo_CVE-2024-21507-1.json",
        "189_CWE-79_CVE-2024-25122-1.json",
        "36_NVD-CWE-noinfo_CVE-2023-44452-1.json",
        "560_CWE-79_CVE-2024-37160-1.json",
        "165_CWE-22_CVE-2024-24756-5.json",
        "553_NVD-CWE-noinfo_CVE-2024-36404-4.json",
        "624_NVD-CWE-noinfo_CVE-2024-40631-1.json",
        "188_NVD-CWE-noinfo_CVE-2024-25117-1.json",
        "316_NVD-CWE-noinfo_CVE-2024-29034-1.json"
    ]
    
    # -- Ordner "matches" vorbereiten --
    os.makedirs(MATCH_DIR, exist_ok=True)
    os.makedirs(MATCH_ONLYCODE_DIR, exist_ok=True)
    
    # -- 2. Dateien in data22842 / data22842/only_code suchen und kopieren --
    if not os.path.isdir(DATA22842_DIR):
        print(f"FEHLER: {DATA22842_DIR} existiert nicht.")
        return
    
    data22842_files = set(os.listdir(DATA22842_DIR))
    data22842_onlycode_files = set()
    
    if os.path.isdir(DATA22842_ONLYCODE_DIR):
        data22842_onlycode_files = set(os.listdir(DATA22842_ONLYCODE_DIR))
    else:
        print(f"Warnung: {DATA22842_ONLYCODE_DIR} existiert nicht. Überspringe only_code.")
    
    print("---- KOPIERE AUS data22842 NACH matches ----")
    for needed in needed_files:
        src_22842 = os.path.join(DATA22842_DIR, needed)
        src_22842_only = os.path.join(DATA22842_ONLYCODE_DIR, needed)
        
        # Kopie aus data22842
        if needed in data22842_files:
            dst_match = os.path.join(MATCH_DIR, needed)
            shutil.copy2(src_22842, dst_match)
            print(f"Kopiert {needed} von data22842 nach matches.")
        else:
            print(f"{needed} NICHT gefunden in data22842.")
        
        # Kopie aus data22842/only_code
        if needed in data22842_onlycode_files:
            dst_match_only = os.path.join(MATCH_ONLYCODE_DIR, needed)
            shutil.copy2(src_22842_only, dst_match_only)
            print(f"Kopiert {needed} von data22842/only_code nach matches/only_code.")
        else:
            print(f"{needed} NICHT gefunden in data22842/only_code.")
    
    # -- 3. Alle weiteren dataX-Unterordner in Dimensions durchgehen --
    #    (außer data22842 und matches selbst)
    all_subdirs = [
        d for d in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, d))
    ]
    
    print("\n---- DURCHLAUFE ALLE dataX-ORDNER (außer data22842 und match) ----")
    for subdir in all_subdirs:
        if subdir in ("data22842", "match"):
            continue  # Überspringen
        
        dataX_path = os.path.join(BASE_DIR, subdir)
        dataX_onlycode_path = os.path.join(dataX_path, "only_code")
        
        print(f"\n*** Bearbeite {subdir} ***")
        
        # Falls das Verzeichnis oder sein only_code-Subfolder fehlen, machen wir weiter
        if not os.path.isdir(dataX_path):
            print(f"Ordner {dataX_path} existiert nicht, überspringe.")
            continue
        
        # only_code kann existieren, muss aber nicht
        has_only_code = os.path.isdir(dataX_onlycode_path)
        
        current_files_main = set(os.listdir(dataX_path))
        current_files_only = set(os.listdir(dataX_onlycode_path)) if has_only_code else set()
        
        # Für jede Datei in needed_files prüfen, ob sie da ist
        for needed in needed_files:
            # -- Hauptordner checken --
            if needed not in current_files_main:
                # Datei fehlt -> aus matches kopieren
                src_needed = os.path.join(MATCH_DIR, needed)
                if os.path.isfile(src_needed):
                    shutil.copy2(src_needed, os.path.join(dataX_path, needed))
                    print(f"Kopiere fehlende Datei {needed} nach {dataX_path}")
                    
                    # Eine zufällige andere Datei entfernen, die nicht in needed_files ist
                    current_files_main = set(os.listdir(dataX_path))
                    non_matched = [
                        f for f in current_files_main
                        if f not in needed_files
                        and os.path.isfile(os.path.join(dataX_path, f))
                    ]

                    if non_matched:
                        to_remove = random.choice(non_matched)
                        os.remove(os.path.join(dataX_path, to_remove))
                        print(f"  -> Lösche zufällig nicht benötigte Datei {to_remove} aus {subdir}")
                        
                        # Auch im only_code entfernen, falls vorhanden
                        if has_only_code and to_remove in current_files_only:
                            full_path_only = os.path.join(dataX_onlycode_path, to_remove)
                            if os.path.isfile(full_path_only):
                                os.remove(full_path_only)
                                print(f"     und lösche {to_remove} auch aus {subdir}/only_code")
                            
                        # aktualisiere current_files_main/only
                        current_files_main = set(os.listdir(dataX_path))
                        if has_only_code:
                            current_files_only = set(os.listdir(dataX_onlycode_path))
                    else:
                        print("  Achtung: Keine nicht-benoetigte Datei mehr zum Löschen gefunden!")
            
            # -- only_code checken --
            if has_only_code and needed not in current_files_only:
                # Datei fehlt -> aus matches/only_code kopieren
                src_needed_only = os.path.join(MATCH_ONLYCODE_DIR, needed)
                if os.path.isfile(src_needed_only):
                    shutil.copy2(src_needed_only, os.path.join(dataX_onlycode_path, needed))
                    print(f"Kopiere fehlende Datei {needed} nach {subdir}/only_code")
                    
                    # Random-Datei entfernen (selbes Vorgehen wie oben) – hier jedoch nur im only_code-Ordner
                    current_files_only = set(os.listdir(dataX_onlycode_path))
                    non_matched_only = [
                        f for f in current_files_only
                        if f not in needed_files
                        and os.path.isfile(os.path.join(dataX_onlycode_path, f))
                    ]

                    if non_matched_only:
                        to_remove_only = random.choice(non_matched_only)
                        os.remove(os.path.join(dataX_onlycode_path, to_remove_only))
                        print(f"  -> Lösche zufällig nicht benötigte Datei {to_remove_only} aus {subdir}/only_code")
                        
                        # Auch im Hauptordner entfernen, falls es denselben Namen gibt
                        full_path_main = os.path.join(dataX_path, to_remove_only)
                        if to_remove_only in current_files_main and os.path.isfile(full_path_main):
                            os.remove(full_path_main)
                            print(f"     und lösche {to_remove_only} auch aus {subdir}")
                        
                        # aktualisiere Sets
                        current_files_only = set(os.listdir(dataX_onlycode_path))
                        current_files_main = set(os.listdir(dataX_path))
                    else:
                        print("  Achtung: Keine nicht-benoetigte Datei mehr zum Löschen gefunden (only_code)!")
        
        print(f"*** Fertig mit {subdir} ***")
    
    print("\n---- Skript beendet. ----")


if __name__ == "__main__":
    main()

import os
import re
import requests
import json
import chardet
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from langchain.schema import Document
from langchain.llms.base import LLM
from langchain.schema import Generation, LLMResult
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st
import shutil
import pandas as pd
from collections import defaultdict

# ------------------ Globale Variablen ------------------
selectedLLM = "deepseek-r1:14b"
selectedFileFormat = None
documents = []
document_embeddings = None
metadatas = []
embeddings = None


# Zu testende Embedding Modelle. 
# Um weitere Embedding Modelle zu testen, diese einfach Auskommentieren oder die jeweilige Bezeichnung aus HuggingFace einfügen.
# Achtung: Auf Kommasetzung nach der Bezeichnung Achten.

embedding_models = [
        # "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        # "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
        # "Alibaba-NLP/gte-large-en-v1.5",
        # "Alibaba-NLP/gte-multilingual-base",
        # "avsolatorio/GIST-Embedding-v0",
        # "avsolatorio/GIST-large-Embedding-v0",
        # "BAAI/bge-base-en-v1.5",
        # "BAAI/bge-large-en-v1.5",
        # "BASF-AI/nomic-embed-text-v1",
        # "BASF-AI/nomic-embed-text-v1.5",
        # "dunzhang/stella_en_1.5B_v5",
        # "dunzhang/stella_en_400M_v5",
        # "HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1",
        # "HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1.5",
        # "HIT-TMG/KaLM-embedding-multilingual-mini-v1",
        # "ibm-granite/granite-embedding-125m-english",
        # "infgrad/jasper_en_vision_language_v1", 
        "intfloat/e5-large-v2",     # <------------------------------Best Model so far
        #"intfloat/multilingual-e5-large",
        #"intfloat/multilingual-e5-large-instruct",
        # "jinaai/jina-embeddings-v3",
        # "jxm/cde-small-v1",
        # "Labib11/MUG-B-1.6",
        # "llmrails/ember-v1",
        # "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-supervised",
        # "mixedbread-ai/mxbai-embed-2d-large-v1",
        # "mixedbread-ai/mxbai-embed-large-v1",
        # "nomic-ai/modernbert-embed-base",
        # "nomic-ai/nomic-embed-text-v1",
        # "nomic-ai/nomic-embed-text-v1.5",
        # "nomic-ai/nomic-embed-text-v1-ablated",
        # "nvidia/NV-Embed-v2",
        # "PaDaS-Lab/arctic-l-bge-small",
        # "pingkeest/learning2_model",
        # "sam-babayev/sf_model_e5",
        # "sentence-transformers/all-MiniLM-L6-v2",
        # "Snowflake/snowflake-arctic-embed-l"
        # "microsoft/graphcodebert-base"
        # "Snowflake/snowflake-arctic-embed-l-v2.0",
        # "Snowflake/snowflake-arctic-embed-m-v1.5",
        # "Snowflake/snowflake-arctic-embed-m-v2.0",
        # "thenlper/gte-large",
        # "tsirif/BinGSE-Meta-Llama-3-8B-Instruct",
        # "voyageai/voyage-3-m-exp",
        # "voyageai/voyage-lite-02-instruct",
        # "w601sxs/b1ade-embed",
        "WhereIsAI/UAE-Large-V1"
    ]
    

# Zu testende LLMs. 
# Um weitere LLMs zu testen, diese einfach Auskommentieren oder die jeweilige Bezeichnung aus Ollama einfügen
# Achtung: Auf Kommasetzung nach der Bezeichnung Achten.
llm_models = [
        "deepseek-r1:14b"
        # "codegemma:7b",
        # "codellama:13b",
        # "deepseek-coder-v2:16b",
        # "gemma",
        # "gemma2:9b",
        # "llama3.2:latest",
        # "phi4:latest",
        # "phi3:14b",
        # "qwen2.5-coder:14b",
        # "qwen2.5-coder:0.5b-instruct-q3_K_S",
        # "starcoder2:15b"
    ]


def clear_gpu_memory():
    """
    GPU-Speicher bereinigen.
    Diese Funktion sorgt dafür, dass der GPU-Speicher freigegeben wird,
    falls Torch verfügbar ist. Dient der Vermeidung von Speicherproblemen.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("GPU-Speicher wurde bereinigt.")


# Einmal direkt beim Skript-Start ausführen, um GPU-Speicher aufzuräumen.
clear_gpu_memory()


# ------------------ LLM-Klasse (Ollama) ------------------
class OllamaLLM(LLM):
    """
    Eine angepasste LLM-Klasse zur Anbindung an den Ollama-Server.
    Wir erben von der Basisklasse 'LLM' aus LangChain und überschreiben
    die benötigten Methoden.
    """

    def __init__(self, llm_name):
        super().__init__()
        self._llm_name = llm_name

    def _call(self, prompt: str, stop=None):
        """
        Sendet das gegebene 'prompt' an den Ollama-Server (per HTTP-POST)
        und parst die Antwort. Erwartet JSON-Zeilen, die jeweils einen 'response'-Schlüssel haben.
        """
        print("--------------------------------------------------------------")
        print(f"Modellname: {self._llm_name}")
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": self._llm_name,
            "prompt": prompt,
            "temperature": 0,
            "top_p": 0.85,
            "top_k": 3
        }
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(url, headers=headers, json=payload)
        except requests.exceptions.RequestException as re:
            raise RuntimeError(f"Fehler beim LLM-Aufruf (Netzwerk/Connection): {re}") from re

        try:
            raw_content = response.content.decode('utf-8')
        except UnicodeDecodeError:
            raw_content = response.content.decode('latin-1')

        try:
            responses = []
            for line in raw_content.splitlines():
                if line.strip():
                    if "error" in line.lower():
                        raise ValueError(f"LLM gab eine Fehlermeldung zurück: {line}")
                    json_line = json.loads(line)
                    if "response" in json_line:
                        responses.append(json_line["response"])
            return "".join(responses)
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
            return f"Fehler bei der Verarbeitung des JSON: {e}"

    def generate(self, prompts, stop=None, callbacks=None, **kwargs):
        """
        Diese Methode implementiert das generate-Interface von LangChain.
        Sie ruft für jedes Prompt intern _call(...) auf und sammelt die Ergebnisse.
        """
        generations = []
        for prompt in prompts:
            result = self._call(prompt, stop)
            generations.append([Generation(text=result)])
        return LLMResult(generations=generations)

    @property
    def _llm_type(self):
        # Kennzeichnet den Typ dieses LLM in LangChain.
        return "ollama_llm"

    @property
    def _identifying_params(self):
        # Gibt die Parameter zurück, anhand derer dieses Modell identifiziert werden kann.
        return {"name_of_llm": self._llm_name}


# ------------------ Hilfsfunktionen ------------------
def load_data_from_directory(directory):
    """
    Lädt .json oder .txt-Dateien aus einem Verzeichnis und gibt eine Liste von LangChain-Documents zurück.
    Dabei wird die Datei zunächst im Binary-Modus eingelesen (um Encoding zu erkennen),
    anschließend mit dem ermittelten Encoding als JSON oder Text interpretiert.
    """
    global selectedFileFormat
    docs = []
    if not os.path.exists(directory):
        print(f"Fehler: Das Verzeichnis '{directory}' existiert nicht.")
        return docs

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        print(f"Verarbeite Datei: {filename}")
        try:
            # Encoding bestimmen
            with open(filepath, 'rb') as raw_file:
                raw_data = raw_file.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']
        except Exception as e:
            print(f"Fehler beim Öffnen der Datei {filepath}: {e}")
            st.write("---")
            continue

        # JSON-Dateien
        if filename.endswith(".json"):
            print(f"Lade JSON-Datei: {filename}")
            try:
                with open(filepath, 'r', encoding=encoding, errors='ignore') as json_file:
                    json_data = json.load(json_file)
                    docs.append(Document(page_content=json.dumps(json_data), metadata={"source": filename}))
                    if selectedFileFormat is None:
                        selectedFileFormat = "json"
            except Exception as e:
                print(f"Fehler beim Laden der JSON-Datei {filename}: {e}")
                st.write("---")
        # Text-Dateien
        elif filename.endswith(".txt"):
            print(f"Lade Text-Datei: {filename}")
            try:
                with open(filepath, 'r', encoding=encoding, errors='ignore') as text_file:
                    text_content = text_file.read()
                    docs.append(Document(page_content=text_content, metadata={"source": filename}))
                    if selectedFileFormat is None:
                        selectedFileFormat = "txt"
            except Exception as e:
                print(f"Fehler beim Laden der Text-Datei {filename}: {e}")
                st.write("---")
        else:
            # Ignoriere andere Dateitypen
            print(f"Überspringe Datei {filename}: Nicht unterstützter Dateityp.")
    print(f"Geladene Dokumente insgesamt: {len(docs)}")
    return docs


def normalize_test_filename(test_file: str) -> str:
    """
    Entfernt z. B. die Endung '.php' und Suffixe wie '_(0_20)'.
    So wird sichergestellt, dass wir eine einheitliche Basis für den Dokument-Namen bekommen.
    """
    filename_noext, _ = os.path.splitext(test_file)
    normalized = re.sub(r"_\(\d+_\d+\)$", "", filename_noext)
    return normalized


def manual_retrieval_with_top3(prompt):
    """
    Ermittelt die Top-3 Dokumente (und deren Ähnlichkeiten) anhand der Anfrage
    und gibt diese zurück (zusätzlich die gesamte Similarities-Liste und den Query-Embedding).
    """
    query_embedding = embeddings.embed_query(prompt)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)  # L2-Normierung
    similarities = document_embeddings @ query_embedding

    sorted_indices = np.argsort(similarities)[::-1]
    top3_indices = sorted_indices[:3]
    top3_docs = [documents[i] for i in top3_indices]
    top3_sims = [similarities[i] for i in top3_indices]

    return top3_docs, top3_sims, similarities, query_embedding


def plot_file_specific(similarities, prompt, title_prefix=""):
    """
    Erstellt zwei Diagramme für den Zusammenhang eines Prompts zu allen Dokumenten:
      1) Scatter-Plot der Cosine-Similarities mit Hervorhebung von Outliers (> 0.9).
      2) Balken-Chart für die Top-20 Dokumente mit Farbabstufung entsprechend der Ähnlichkeit.
    """
    if len(document_embeddings) > 0 and prompt:
        os.makedirs("GlobalPlots", exist_ok=True)

        # -- Scatter + Outliers --
        threshold = 0.9
        outlier_indices = np.where(similarities > threshold)[0]
        outlier_sims = similarities[outlier_indices]

        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(similarities)), similarities, color='red', edgecolor='black')
        plt.scatter(outlier_indices, outlier_sims, color='green', edgecolor='black', zorder=5)
        for idx in outlier_indices:
            doc_name = documents[idx].metadata.get("source", f"Doc_{idx}")
            plt.text(idx + 1, similarities[idx], doc_name, fontsize=9,
                     verticalalignment='bottom', horizontalalignment='left')

        plt.ylim(0, 1)
        plt.title(f"{title_prefix}Cosinus-Ähnlichkeit (einzelne Anfrage)")
        plt.xlabel("Dokument Index")
        plt.ylabel("Cosinus-Ähnlichkeit")
        plt.tight_layout()
        plt.savefig("GlobalPlots/query_to_documents_cosine_similarity_scatter_outliers.png")
        plt.close()

        # -- Top-20 Ranking (Balken-Chart) --
        top_indices = np.argsort(similarities)[::-1][:20]
        top_sims = similarities[top_indices]
        top_docs = [documents[i].metadata.get("source", f"Doc_{i}") for i in top_indices]

        from matplotlib.colors import LinearSegmentedColormap
        colors = [(1, 0, 0), (0, 1, 0)]
        cm = LinearSegmentedColormap.from_list('RedGreenGradient', colors, N=256)
        norm = plt.Normalize(vmin=0.5, vmax=1.0)
        bar_colors = [cm(norm(val)) for val in top_sims]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(top_sims)), top_sims, edgecolor='black', color=bar_colors)
        ax.invert_yaxis()
        ax.set_yticks(range(len(top_sims)))
        ax.set_yticklabels(top_docs)
        ax.set_xlim(0, 1)
        ax.set_title(f"{title_prefix}Top 20 Dokumente (einzelne Anfrage)")
        ax.set_xlabel("Cosinus-Ähnlichkeit")

        sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Cosinus-Ähnlichkeit')
        fig.tight_layout()
        fig.savefig("GlobalPlots/top_20_similar_documents_gradient.png")
        plt.close(fig)


def plot_global_matrices(emb_model, title_suffix=""):
    """
    Erstellt drei Darstellungen für das gesamte Dokumenten-Embedding:
      1) Heatmap der Cosine-Similarities
      2) Heatmap der euklidischen Distanzen
      3) Histogramm der Cosine-Similarities (Verteilung)
    """
    if len(document_embeddings) > 1:
        # Cosine-Similarity-Matrix
        similarity_matrix_cosine = cosine_similarity(document_embeddings)
        # Euklidische Distanz-Matrix
        similarity_matrix_euclidean = euclidean_distances(document_embeddings)

        # -- Cosine-Sim Heatmap --
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix_cosine, cmap="coolwarm", annot=False, vmin=0, vmax=1)
        plt.title(f"Cosinus-Ähnlichkeitsmatrix\nEmbedding-Modell: {emb_model}")
        plt.xlabel("Dok Index")
        plt.ylabel("Dok Index")
        plt.savefig("GlobalPlots/cosine_similarity_matrix.png")
        plt.close()

        # -- Euklidische Distanz Heatmap --
        euclidean_max = np.max(similarity_matrix_euclidean)
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix_euclidean, cmap="coolwarm", annot=False, vmin=0, vmax=euclidean_max)
        plt.title(f"Euklidische Distanzmatrix {title_suffix}")
        plt.xlabel("Dok Index")
        plt.ylabel("Dok Index")
        plt.savefig("GlobalPlots/euclidean_distance_matrix.png")
        plt.close()

        # -- Histogram Cosine-Sim-Verteilung --
        plt.figure(figsize=(10, 6))
        plt.hist(similarity_matrix_cosine.flatten(), bins=50, color='skyblue', edgecolor='black',
                 range=(0, 1), density=True)
        plt.title(f"Verteilung der Cosinus-Ähnlichkeitswerte {title_suffix}")
        plt.xlabel("Ähnlichkeitswert")
        plt.ylabel("Prozentuale Häufigkeit")
        plt.xlim(0, 1)
        plt.savefig("GlobalPlots/cosine_similarity_histogram.png")
        plt.close()


# ---------------------------
#   Hybrid-Response (LLM)
# ---------------------------
def hybrid_response(prompt, llm_name, emb_model=""):
    """
    Ruft das LLM auf, indem erst über manual_retrieval_with_top3 die Top-3-Dokumente zum Prompt geholt werden.
    Dann wird ein spezieller Prompt zusammengebaut (inkl. CVE-Infos aus den Top3),
    ans LLM gesendet und das Ergebnis ausgegeben. Abschließend werden Plots erzeugt.
    """
    top3_docs, top3_sims, similarities, _ = manual_retrieval_with_top3(prompt)

    # Inhalt der Top-Dokumente extrahieren
    doc_content_top1 = top3_docs[0].page_content if len(top3_docs) > 0 else ""
    doc_content_top2 = top3_docs[1].page_content if len(top3_docs) > 1 else ""
    doc_content_top3 = top3_docs[2].page_content if len(top3_docs) > 2 else ""

    # CVE-IDs, sofern in den JSONs vorhanden
    cve_id1, cve_id2, cve_id3 = "N/A", "N/A", "N/A"
    try:
        j1 = json.loads(doc_content_top1)
        cve_id1 = j1.get("cve_id", "N/A")
    except:
        pass
    if len(top3_docs) > 1:
        try:
            j2 = json.loads(doc_content_top2)
            cve_id2 = j2.get("cve_id", "N/A")
        except:
            pass
    if len(top3_docs) > 2:
        try:
            j3 = json.loads(doc_content_top3)
            cve_id3 = j3.get("cve_id", "N/A")
        except:
            pass

# Prompt an das LLM #------ Top-3 Dokument Prompt ---------

#     llama_prompt = f"""You are a security analyst specializing in identifying code vulnerabilities (CVEs). 
# You have access to retrieved CVE-related information and must rely solely on the provided context. Do not use external knowledge.

# Your objectives:
# 1. Read the user's question carefully.
# 2. Examine the retrieved CVE-related context below. This context contains descriptions, code snippets, and patch details related to certain vulnerabilities.
#    - The context is ranked from most relevant (Top-1) to less relevant (Top-3).
# 3. Determine if the given user's code or scenario matches any known CVE based on your retrieved context.
#    - The code may differ in variable names or structure; rely on conceptual similarity.
# 4. If you find a matching CVE, provide its ID and relevant details. If it's secure, say so.
# 5. If unsure, say you are not sure.
# 6. DO NOT use external knowledge.

# Additional instructions for handling multiple contexts:
# 1. Check if the answer is likely contained in Top-1. If yes, use Top-1.
# 2. If still unsure, check Top-2, then Top-3 in that order.
# 3. If no vulnerability is found in these three documents, say "no vulnerability found."

# User question: '{prompt}'

# Retrieved top-1 context with CVE ID {cve_id1}:
# [Top-1, similarity={top3_sims[0]:.3f}]
# {doc_content_top1}

# Retrieved top-2 context with CVE ID {cve_id2}:
# [Top-2, similarity={top3_sims[1]:.3f}]
# {doc_content_top2}

# Retrieved top-3 context with CVE ID {cve_id3}:
# [Top-3, similarity={top3_sims[2]:.3f}]
# {doc_content_top3}

# Based on the above context, provide the Retrieved CVE ID ({cve_id1} or {cve_id2} or {cve_id3}) and related info, 
# or state that no vulnerability was found.

# Begin your response with “Vulnerability Found” if a relevant CVE is identified in the User question based on the retrieved Information, or “Secure” if the retrieved vulnerability is not present.
# """


#------ Top-1 Dokument Prompt ---------

    llama_prompt = f"""You are a security analyst specializing in identifying code vulnerabilities (CVEs). 
You have access to retrieved CVE-related information and must rely solely on the provided context. Do not use external knowledge.

Your objectives:
1. Read the user's question carefully.
2. Examine the retrieved CVE-related context below. This context contains descriptions, code snippets, and patch details related to certain vulnerabilities.
3. Determine if the given user's code or scenario matches any known CVE based on your retrieved context.
    - The code may differ in variable names or structure, rely on conceptual similarity.
4. If you find a matching CVE, provide its ID and relevant details. If it's secure, say so.
5. If unsure, say you are not sure.
6. DO NOT use external knowledge.

User question: '{prompt}'

Retrieved context with CVE ID {cve_id1}:
{doc_content_top1}

Based on the above context, provide the Retrieved CVE ID {cve_id1} and related info, or state that no vulnerability was found.
Begin your response with “Vulnerability Found” if a relevant CVE is identified in the User question based on the retrieved Information, or “Secure” if the retrieved vulnerability is not present.
""" # YES NO

    print("---------------------------------------------------")
    print(f"PROMPT für LLM {llm_name}:\n{llama_prompt}")
    print("---------------------------------------------------")

    current_llm = OllamaLLM(llm_name)
    try:
        llama_response = current_llm._call(llama_prompt)
    except ValueError as ve:
        st.markdown(f"<span style='color:red;'>Fehler: LLM \"{llm_name}\" schlug fehl.</span>", unsafe_allow_html=True)
        llama_response = f"Fehler: {ve}"
    except RuntimeError as re:
        st.markdown(f"<span style='color:red;'>Fehler beim LLM-Aufruf: {re}</span>", unsafe_allow_html=True)
        llama_response = f"Fehler: {re}"
    except Exception as e:
        st.markdown(f"<span style='color:red;'>Allgemeiner Fehler: {e}</span>", unsafe_allow_html=True)
        llama_response = f"Fehler: {e}"

    if not llama_response.strip():
        llama_response = "Keine Antwort vom LLM."

    # Plot der Similarities (Scatter + Top20-Balken)
    plot_file_specific(similarities, prompt, title_prefix=f"LLM {llm_name}: ")

    return llama_response, top3_docs[0] if top3_docs else None, similarities


# ------------------ Datei-Stats (Tests) ------------------
def get_file_stats():
    """
    In Streamlit-Session State wird ein Dictionary abgelegt,
    das pro Embedding-Modell und Datei die Anzahl an Tests, Top1-Erfolgen usw. speichert.
    """
    if "file_stats" not in st.session_state:
        st.session_state.file_stats = {}
    return st.session_state.file_stats


def update_file_stats(embedding_model, filename, is_top1, is_top3):
    """
    Aktualisiert in st.session_state.file_stats[embedding_model][filename]:
      - top1_count (wie oft war es Top1)
      - top3_count (wie oft war es Top3)
      - tests_done (insgesamt wie viele Tests für diese Datei)
    """
    file_stats_global = get_file_stats()
    if embedding_model not in file_stats_global:
        file_stats_global[embedding_model] = {}
    if filename not in file_stats_global[embedding_model]:
        file_stats_global[embedding_model][filename] = {
            "top1_count": 0,
            "top3_count": 0,
            "tests_done": 0
        }
    file_stats_global[embedding_model][filename]["tests_done"] += 1
    if is_top1:
        file_stats_global[embedding_model][filename]["top1_count"] += 1
    if is_top3:
        file_stats_global[embedding_model][filename]["top3_count"] += 1


def show_file_stats_tables():
    """
    Zeigt je Embedding-Modell eine kompakte Auswertungstabelle:
    Wir sortieren die Testdateien nach Kategorien (z.B. _(0_20), _(0_40), etc.).
    Dann zeigen wir pro Kategorie die prozentualen Top1-/Top3-Raten an.
    
    Zusätzlich wird eine "ALL_MODELS"-Zusammenfassung generiert,
    in der die Werte aller Embeddings summiert werden.
    """
    if "file_stats" not in st.session_state:
        st.write("Keine Datei-Statistiken vorhanden.")
        return

    categories = ["_(0_20)", "_(0_40)", "_(1_40)", "_(2_40)", "_(0_60)", "_(0_80)", "_(0_100)", "_(0_200)", "_(0_300)", "_(0_400)", "_(0_500)"]
    suffix_regex = re.compile(r"_\(\d+_\d+\)$")

    file_stats_global = st.session_state.file_stats

    # "ALL_MODELS" neu anlegen und kumulative Statistiken bilden
    if "ALL_MODELS" in file_stats_global:
        del file_stats_global["ALL_MODELS"]
    file_stats_global["ALL_MODELS"] = {}

    for emb_model, files_map in list(file_stats_global.items()):
        if emb_model == "ALL_MODELS":
            continue
        for fname, stats in files_map.items():
            if fname not in file_stats_global["ALL_MODELS"]:
                file_stats_global["ALL_MODELS"][fname] = {
                    "top1_count": 0,
                    "top3_count": 0,
                    "tests_done": 0
                }
            file_stats_global["ALL_MODELS"][fname]["top1_count"] += stats["top1_count"]
            file_stats_global["ALL_MODELS"][fname]["top3_count"] += stats["top3_count"]
            file_stats_global["ALL_MODELS"][fname]["tests_done"] += stats["tests_done"]

    def produce_category_table(emb_model, files_map):
        cat_stats = defaultdict(lambda: [0, 0, 0])

        for fname, stt in files_map.items():
            base, ext = os.path.splitext(fname)
            match = suffix_regex.search(base)
            if match:
                bracket = match.group(0)
            else:
                bracket = None

            if bracket in categories:
                cat_stats[bracket][0] += stt["top1_count"]
                cat_stats[bracket][1] += stt["top3_count"]
                cat_stats[bracket][2] += stt["tests_done"]

        row_list = []
        for cat in categories:
            sum_t1, sum_t3, sum_tests = cat_stats[cat]
            if sum_tests == 0:
                continue
            top1_pct = sum_t1 / sum_tests * 100
            top3_pct = sum_t3 / sum_tests * 100
            row_list.append({
                "Kategorie": cat,
                "Top-1": f"{top1_pct:.1f}%",
                "Top-3": f"{top3_pct:.1f}%"
            })

        if not row_list:
            st.write(f"(Keine Dateien in den bekannten Kategorien für {emb_model})")
            return

        df = pd.DataFrame(row_list, columns=["Kategorie", "Top-1", "Top-3"])
        st.write(f"**Modell: {emb_model}**")
        st.write(df)
        st.write("-----")

    # Tabellen für jedes Embedding-Modell und ALL_MODELS zeigen
    for emb_model, files_map in file_stats_global.items():
        produce_category_table(emb_model, files_map)


# ------------------ Tests (Embedding+LLM) ------------------
def run_tests_for_embedding_model(embedding_model_name, testdata_dir, base_output_dir):
    """
    Führt automatische Tests für ein bestimmtes Embedding-Modell durch:
     - Lädt die Daten (JSON) in LangChain-Dokumente
     - Embedding dieser Dokumente
     - Testet sie gegen die Testdateien in testdata_dir
     - Speichert Plots und Statistiken in base_output_dir
    """
    try:
        clear_gpu_memory()
        if os.path.exists("data.pkl"):
            os.remove("data.pkl")

        global embeddings, documents, document_embeddings, metadatas, selectedFileFormat
        selectedFileFormat = None

        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

        data_dir = "data/data_v1/only_code"
        docs = load_data_from_directory(data_dir)
        documents = docs
        metadatas = [d.metadata for d in documents]
        texts = [d.page_content for d in docs]
        document_embeddings = embeddings.embed_documents(texts)
        document_embeddings = np.array(document_embeddings, dtype=np.float32)
        document_embeddings = normalize(document_embeddings, axis=1)

        # Speichern für späteren LLM-Test
        with open("data.pkl", "wb") as f:
            pickle.dump((documents, document_embeddings, metadatas), f)

        # Ersetze ':' durch '_' im Ordnernamen, falls vorhanden um Probleme zu vermeiden
        safe_emb_model = embedding_model_name.replace(":", "_")
        model_output_dir = os.path.join(base_output_dir, safe_emb_model)
        os.makedirs(model_output_dir, exist_ok=True)

        test_files = [f for f in os.listdir(testdata_dir)
                      if os.path.isfile(os.path.join(testdata_dir, f))]

        relevant_similarities = []
        top1_count = 0
        top3_count = 0
        total_tests = len(test_files)

        for test_file in test_files:
            test_file_path = os.path.join(testdata_dir, test_file)
            with open(test_file_path, "r", encoding="utf-8", errors="ignore") as f:
                test_code = f.read()

            st.markdown(
                f"<span style='color:#7bd1ed;'>**Verarbeite Datei:** </span>"
                f"<span style='color:#7bd1ed;'>{test_file}</span>",
                unsafe_allow_html=True
            )

            # Top-3 Retrieval
            top3_docs, top3_sims, similarities, _ = manual_retrieval_with_top3(test_code)
            plot_file_specific(similarities, test_code, title_prefix=f"Emb {embedding_model_name}: ")

            # Unterordner für jede Testdatei
            output_subdir = os.path.join(model_output_dir, test_file)
            os.makedirs(output_subdir, exist_ok=True)

            # Kopiere erstellte Plots
            plots_dir = "GlobalPlots"
            for p in ["top_20_similar_documents_gradient.png", "query_to_documents_cosine_similarity_scatter_outliers.png"]:
                src = os.path.join(plots_dir, p)
                if os.path.exists(src):
                    dst = os.path.join(output_subdir, p)
                    shutil.copyfile(src, dst)

            # Rangbestimmung
            normalized_name = normalize_test_filename(test_file)
            target_doc_name = normalized_name + ".json"

            query_embedding = embeddings.embed_query(test_code)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            sims = document_embeddings @ query_embedding

            target_index = None
            for i, meta in enumerate(metadatas):
                if meta.get("source", "") == target_doc_name:
                    target_index = i
                    break

            if target_index is not None:
                target_similarity = sims[target_index]
                sorted_indices = np.argsort(sims)[::-1]
                rank = np.where(sorted_indices == target_index)[0][0] + 1
                relevant_similarities.append(target_similarity)

                is_top1 = (rank == 1)
                is_top3 = (rank <= 3)

                if rank == 1:
                    color = "#6ff261"  # Grün
                elif rank <= 3:
                    color = "#f5e551"  # Gelb
                else:
                    color = "#f26161"  # Rot

                if is_top1:
                    top1_count += 1
                if is_top3:
                    top3_count += 1

                st.markdown(
                    f"<span style='color:{color};'>**Ziel-Datei Rang**: {rank}, Similarity: {target_similarity:.4f}</span>",
                    unsafe_allow_html=True
                )

                # Similarity-Infos in Datei
                sim_path = os.path.join(output_subdir, "similarity.txt")
                with open(sim_path, "w", encoding="utf-8") as sf:
                    sf.write(f"Rank: {rank}\nSimilarity: {target_similarity:.4f}\n")

                # Globale Stats updaten
                update_file_stats(embedding_model_name, test_file, is_top1, is_top3)
            else:
                st.write(f"**Keine passende .json-Datei für {normalized_name} gefunden!**")

            st.write("---")

        # Globale Plots
        plot_global_matrices(embedding_model_name, title_suffix=f"(Embedding: {embedding_model_name})")
        for p in ["cosine_similarity_matrix.png", "euclidean_distance_matrix.png", "cosine_similarity_histogram.png"]:
            src = os.path.join("GlobalPlots", p)
            if os.path.exists(src):
                dst = os.path.join(model_output_dir, p)
                shutil.copyfile(src, dst)

        if relevant_similarities:
            avg_similarity = sum(relevant_similarities) / len(relevant_similarities)
            top1_str = f"{top1_count}/{total_tests}"
            top3_str = f"{top3_count}/{total_tests}"

            st.markdown(
                f"<span style='color:yellow;'>**Durchschnittliche Cosinus-Ähnlichkeit: {avg_similarity:.4f}**</span>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<span style='color:yellow;'>**Trefferquote (Top-1)**: {top1_str}</span>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<span style='color:yellow;'>**Indirekte Trefferquote (Top-3)**: {top3_str}</span>",
                unsafe_allow_html=True
            )
            st.write("---")

            avg_file = os.path.join(model_output_dir, "average_similarity.txt")
            with open(avg_file, "w", encoding="utf-8") as af:
                af.write(f"AvgSimilarity: {avg_similarity:.4f}\nTop-1: {top1_str}\nTop-3: {top3_str}\n")
        else:
            st.write("Keine relevanten Similarities berechnet.")

        # Zusammenfassung in session_state
        st.session_state.results_overall.append({
            "embedding_model": embedding_model_name,
            "avg_similarity": avg_similarity if relevant_similarities else 0.0,
            "top1_ratio": top1_count / total_tests if total_tests else 0.0,
            "top3_ratio": top3_count / total_tests if total_tests else 0.0
        })

    except Exception as e:
        st.write(f"Fehler in run_tests_for_embedding_model mit Modell {embedding_model_name}: {e}")


def run_tests_for_llm(best_embedding_model, llm_name, testdata_dir, llm_output_base):
    """
    Zweiter Durchlauf: Der Code testet nun ein LLM-Modell (best_embedding_model wird
    für das Retrieval genutzt). Mit den gleichen Testdateien ermitteln wir die top-Dokumente
    und fragen das LLM, ob es eine CVE findet. Die Rangposition der passenden JSON-Datei
    wird ebenfalls ausgegeben. Ergebnisse werden in llm_output_base abgelegt.
    """
    try:
        clear_gpu_memory()
        global embeddings, documents, document_embeddings, metadatas

        if not os.path.exists("data.pkl"):
            st.write("data.pkl nicht gefunden – breche ab.")
            return

        # data.pkl laden
        with open("data.pkl", "rb") as f:
            documents, document_embeddings, metadatas = pickle.load(f)

        embeddings = HuggingFaceEmbeddings(model_name=best_embedding_model)

        # Auch hier ':' durch '_' ersetzen
        safe_emb_model = best_embedding_model.replace(":", "_")
        safe_llm_model = llm_name.replace(":", "_")
        llm_output_dir = os.path.join(llm_output_base, f"{safe_emb_model}_LLM_{safe_llm_model}")
        os.makedirs(llm_output_dir, exist_ok=True)

        test_files = [f for f in os.listdir(testdata_dir) if os.path.isfile(os.path.join(testdata_dir, f))]

        relevant_similarities = []
        top1_count = 0
        top3_count = 0
        total_tests = len(test_files)

        for test_file in test_files:
            st.markdown(
                f"<span style='color:#7bd1ed; font-size:18px;'>**Verarbeite Datei (LLM):** </span>"
                f"<span style='color:#7bd1ed; font-size:18px;'>{llm_name} => {test_file}</span>",
                unsafe_allow_html=True
            )

            test_file_path = os.path.join(testdata_dir, test_file)
            with open(test_file_path, "r", encoding="utf-8", errors="ignore") as f:
                test_code = f.read()

            try:
                llama_response, top_doc, similarities = hybrid_response(test_code, llm_name, best_embedding_model)
            except Exception as e:
                st.markdown(f"<span style='color:red;'>Fehler beim LLM-Aufruf: {e}</span>", unsafe_allow_html=True)
                llama_response = f"LLM-Fehler: {e}"

            st.write(llama_response)

            output_subdir = os.path.join(llm_output_dir, test_file)
            os.makedirs(output_subdir, exist_ok=True)

            # LLM-Antwort in .txt
            response_file = os.path.join(output_subdir, f"{safe_llm_model}_response.txt")
            with open(response_file, "w", encoding="utf-8") as rf:
                rf.write(llama_response)

            # Plots
            plot_file_specific(similarities, test_code, title_prefix=f"LLM {llm_name}: ")
            for p in ["top_20_similar_documents_gradient.png", "query_to_documents_cosine_similarity_scatter_outliers.png"]:
                src = os.path.join("GlobalPlots", p)
                if os.path.exists(src):
                    dst = os.path.join(output_subdir, p)
                    shutil.copyfile(src, dst)

            # Rangbestimmung
            normalized_name = normalize_test_filename(test_file)
            target_doc_name = normalized_name + ".json"
            query_embedding = embeddings.embed_query(test_code)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            sims = document_embeddings @ query_embedding

            target_index = None
            for i, meta in enumerate(metadatas):
                if meta.get("source", "") == target_doc_name:
                    target_index = i
                    break

            if target_index is not None:
                target_similarity = sims[target_index]
                sorted_indices = np.argsort(sims)[::-1]
                rank = np.where(sorted_indices == target_index)[0][0] + 1
                relevant_similarities.append(target_similarity)

                is_top1 = (rank == 1)
                is_top3 = (rank <= 3)

                if rank == 1:
                    color = "#6ff261"
                elif rank <= 3:
                    color = "#f5e551"
                else:
                    color = "#f26161"

                if is_top1:
                    top1_count += 1
                if is_top3:
                    top3_count += 1

                st.markdown(
                    f"<span style='color:{color};'>Rang: {rank}, Similarity: {target_similarity:.4f}</span>",
                    unsafe_allow_html=True
                )
            else:
                st.write(f"**Keine passende .json-Datei für {normalized_name} gefunden**")

            st.write("---")

        # Zusammenfassung
        if relevant_similarities:
            avg_similarity = sum(relevant_similarities) / len(relevant_similarities)
            ratio1_str = f"{top1_count}/{total_tests}"
            ratio3_str = f"{top3_count}/{total_tests}"
            st.markdown(
                f"<span style='color:yellow;'>**Tests für LLM {llm_name} abgeschlossen**</span>",
                unsafe_allow_html=True
            )
            st.write(f"Top-1: {ratio1_str}, Top-3: {ratio3_str}")
            st.write("---")
            avg_file = os.path.join(llm_output_dir, f"{safe_llm_model}_average_similarity.txt")
            with open(avg_file, "w", encoding="utf-8") as af:
                af.write(f"AverageSim: {avg_similarity:.4f}\nTop1: {ratio1_str}\nTop3: {ratio3_str}\n")
        else:
            st.write("Keine relevanten Ähnlichkeiten berechnet.")

    except Exception as e:
        st.write(f"Fehler im 2. Durchlauf LLM {llm_name}: {e}")


# ------------------ Dimensions-Test ------------------
def run_tests_for_dimensions():
    """
    Testet das erste Embedding-Modell mit dem Ordner in "data/Dimensions".
    Dadurch wird die Leistung bei zunehmender Wissensbasis getestet.
    Nutzt "TestCode" als Prompt, berechnet Top-1/-3 und speichert Results.
    """
    try:
        if "dimensions_results" not in st.session_state:
            st.session_state.dimensions_results = []
        else:
            st.session_state.dimensions_results = []

        embedding_model_name = embedding_models[0]
        st.write(f"Starte Dimension-Tests für Embedding-Modell: {embedding_model_name}")

        dimension_dir = "data/Dimensions"
        testdata_dir = "TestCode"
        base_output_dir = os.path.join("TestResults", "Dimensions")
        os.makedirs(base_output_dir, exist_ok=True)

        dimension_folders = [
            d for d in os.listdir(dimension_dir)
            if os.path.isdir(os.path.join(dimension_dir, d))
        ]

        def extract_number(folder_name):
            return int(re.sub("[^0-9]", "", folder_name)) if re.search(r"\d+", folder_name) else 0

        dimension_folders.sort(key=extract_number)

        for dim_folder in dimension_folders:
            st.write(f"---\n**Teste Ordner:** {dim_folder}")
            clear_gpu_memory()

            if os.path.exists("data.pkl"):
                os.remove("data.pkl")

            global embeddings, documents, document_embeddings, metadatas
            embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
            current_data_dir = os.path.join(dimension_dir, dim_folder)
            docs = load_data_from_directory(current_data_dir)
            documents = docs
            metadatas = [d.metadata for d in docs]
            texts = [d.page_content for d in docs]

            if len(texts) == 0:
                st.write(f"Keine Dateien im Ordner {dim_folder}, fahre fort.")
                continue

            document_embeddings = embeddings.embed_documents(texts)
            document_embeddings = np.array(document_embeddings, dtype=np.float32)
            document_embeddings = normalize(document_embeddings, axis=1)

            with open("data.pkl", "wb") as f:
                pickle.dump((documents, document_embeddings, metadatas), f)

            test_files = [f for f in os.listdir(testdata_dir) if os.path.isfile(os.path.join(testdata_dir, f))]

            total_tests = len(test_files)
            top1_count = 0
            top3_count = 0

            for test_file in test_files:
                st.markdown(
                    f"<span style='color:#7bd1ed;'>**Verarbeite Datei:** </span>"
                    f"<span style='color:#7bd1ed;'>{test_file}</span>",
                    unsafe_allow_html=True
                )
                test_file_path = os.path.join(testdata_dir, test_file)
                with open(test_file_path, "r", encoding="utf-8", errors="ignore") as f:
                    test_code = f.read()

                top3_docs, top3_sims, similarities, _ = manual_retrieval_with_top3(test_code)

                normalized_name = normalize_test_filename(test_file)
                target_doc_name = normalized_name + ".json"
                query_embedding = embeddings.embed_query(test_code)
                query_embedding = query_embedding / np.linalg.norm(query_embedding)
                sims = document_embeddings @ query_embedding

                target_index = None
                for i, meta in enumerate(metadatas):
                    if meta.get("source", "") == target_doc_name:
                        target_index = i
                        break

                if target_index is not None:
                    rank = np.where(np.argsort(sims)[::-1] == target_index)[0][0] + 1

                    is_top1 = (rank == 1)
                    is_top3 = (rank <= 3)

                    if rank == 1:
                        color = "#6ff261"
                    elif rank <= 3:
                        color = "#f5e551"
                    else:
                        color = "#f26161"

                    if is_top1:
                        top1_count += 1
                    if is_top3:
                        top3_count += 1

                    st.markdown(
                        f"<span style='color:{color};'>**Ziel-Datei Rang**: {rank}, Similarity: {sims[target_index]:.4f}</span>",
                        unsafe_allow_html=True
                    )
                else:
                    st.write(f"**Keine passende .json-Datei für {normalized_name} gefunden**")

                st.write("---")

            top1_ratio = top1_count / total_tests if total_tests else 0
            top3_ratio = top3_count / total_tests if total_tests else 0

            st.session_state.dimensions_results.append({
                "embedding_model": embedding_model_name,
                "folder": dim_folder,
                "top1_ratio": top1_ratio,
                "top3_ratio": top3_ratio
            })

            df_single = pd.DataFrame([{
                "Embedding-Modell": embedding_model_name,
                "Ordner": dim_folder,
                "Top-1 (Prozent)": f"{top1_ratio*100:.1f}%",
                "Top-3 (Prozent)": f"{top3_ratio*100:.1f}%"
            }])
            st.write("**Ergebnis-Tabelle für diesen Ordner**:")
            st.write(df_single)

            single_csv_path = os.path.join(base_output_dir, f"results_{dim_folder}.csv")
            df_single.to_csv(single_csv_path, index=False)
            st.write(f"Tabelle für {dim_folder} gespeichert als {single_csv_path}")

        # Gesamtergebnis
        st.write("---")
        st.write("**Gesamtergebnisse aller getesteten Ordner:**")
        if len(st.session_state.dimensions_results) > 0:
            df_all = pd.DataFrame(st.session_state.dimensions_results)
            df_all["Top-1 (Prozent)"] = (df_all["top1_ratio"] * 100).round(1)
            df_all["Top-3 (Prozent)"] = (df_all["top3_ratio"] * 100).round(1)
            df_all = df_all[["embedding_model", "folder", "Top-1 (Prozent)", "Top-3 (Prozent)"]]

            st.write(df_all)
            all_csv_path = os.path.join(base_output_dir, "all_dimensions_results.csv")
            df_all.to_csv(all_csv_path, index=False)
            st.write(f"Gesamttabelle gespeichert als {all_csv_path}")
        else:
            st.write("Keine Ergebnisse in st.session_state.dimensions_results vorhanden.")
    except Exception as e:
                print(f"Fehler: {e}")


# ------------------ LINES-Test (Up to 500) ------------------
def run_tests_for_lines():
    """
    Führt für JEDES Embedding-Modell Tests mit Unterordnern in "TestCode/TestCode_lines" (z.B. 100, 200, 300, 400, 500).
    Dadurch wird die Leistung bei zunehmender Codegröße getestet.
    Jede Unterordner-Zahl => separate Tests, color-coded Rang etc.
    Am Ende wird eine Gesamttabelle ausgegeben.
    """
    try:
        if "lines_results" not in st.session_state:
            st.session_state.lines_results = []
        else:
            st.session_state.lines_results = []

        base_test_folder = "TestCode/TestCode_lines"
        if not os.path.exists(base_test_folder):
            st.write(f"Ordner '{base_test_folder}' existiert nicht – Abbruch.")
            return

        # Subordner z.B. "100", "200", "300", "400", "500"
        line_folders = [
            d for d in os.listdir(base_test_folder)
            if os.path.isdir(os.path.join(base_test_folder, d))
        ]

        # Sortieren nach Zahl
        def extract_number(folder_name):
            return int(re.sub("[^0-9]", "", folder_name)) if re.search(r"\d+", folder_name) else 0

        line_folders.sort(key=extract_number)

        # Test für jedes Embedding
        for emb_model in embedding_models:
            st.markdown(
            f"<span style='color:white; font-size:20px;'>**Starte Up-to-500-Lines-Test für Embedding:** </span>"
            f"<span style='color:#d99a69; font-size:20px;'>{emb_model}</span>",
            unsafe_allow_html=True
            )
            st.write("---")

            # data.pkl neu
            clear_gpu_memory()
            if os.path.exists("data.pkl"):
                os.remove("data.pkl")

            global embeddings, documents, document_embeddings, metadatas, selectedFileFormat
            selectedFileFormat = None
            embeddings = HuggingFaceEmbeddings(model_name=emb_model)

            # "data" laden, Embedding berechnen
            docs = load_data_from_directory("data/data_v1/only_code")
            documents = docs
            metadatas = [d.metadata for d in documents]
            texts = [d.page_content for d in docs]
            document_embeddings = embeddings.embed_documents(texts)
            document_embeddings = np.array(document_embeddings, dtype=np.float32)
            document_embeddings = normalize(document_embeddings, axis=1)

            with open("data.pkl", "wb") as f:
                pickle.dump((documents, document_embeddings, metadatas), f)

            # pro Subfolder (100,200,300...)
            for folder_name in line_folders:
                folder_path = os.path.join(base_test_folder, folder_name)
                st.write(f"\n**Teste Subfolder:** {folder_path}")

                test_files = [
                    f for f in os.listdir(folder_path)
                    if os.path.isfile(os.path.join(folder_path, f))
                ]

                total_tests = len(test_files)
                top1_count = 0
                top3_count = 0

                for test_file in test_files:
                    st.markdown(
                        f"<span style='color:#7bd1ed;'>**Verarbeite Datei:** </span>"
                        f"<span style='color:#7bd1ed;'>{test_file}</span>",
                        unsafe_allow_html=True
                    )
                    file_path = os.path.join(folder_path, test_file)
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        test_code = f.read()

                    # Nur Retrieval
                    top3_docs, top3_sims, similarities, _ = manual_retrieval_with_top3(test_code)

                    normalized_name = normalize_test_filename(test_file)
                    target_doc_name = normalized_name + ".json"

                    query_embedding = embeddings.embed_query(test_code)
                    query_embedding = query_embedding / np.linalg.norm(query_embedding)
                    sims = document_embeddings @ query_embedding

                    target_index = None
                    for i, meta in enumerate(metadatas):
                        if meta.get("source", "") == target_doc_name:
                            target_index = i
                            break

                    if target_index is not None:
                        rank = np.where(np.argsort(sims)[::-1] == target_index)[0][0] + 1
                        is_top1 = (rank == 1)
                        is_top3 = (rank <= 3)

                        if rank == 1:
                            color = "#6ff261"
                        elif rank <= 3:
                            color = "#f5e551"
                        else:
                            color = "#f26161"

                        if is_top1:
                            top1_count += 1
                        if is_top3:
                            top3_count += 1

                        st.markdown(
                            f"<span style='color:{color};'>**Ziel-Datei Rang**: {rank}, Similarity: {sims[target_index]:.4f}</span>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.write(f"**Keine passende .json-Datei für {normalized_name} gefunden**")

                    st.write("---")

                top1_ratio = top1_count / total_tests if total_tests else 0
                top3_ratio = top3_count / total_tests if total_tests else 0

                # Merken in lines_results
                st.session_state.lines_results.append({
                    "embedding_model": emb_model,
                    "lines_folder": folder_name,
                    "top1_ratio": top1_ratio,
                    "top3_ratio": top3_ratio
                })

                # Sofortige Teil-Tabelle
                df_single = pd.DataFrame([{
                    "Embedding-Modell": emb_model,
                    "Folder": folder_name,
                    "Top-1 (Prozent)": f"{top1_ratio*100:.1f}%",
                    "Top-3 (Prozent)": f"{top3_ratio*100:.1f}%"
                }])
                st.write("**Ergebnis-Tabelle für diesen Ordner**:")
                st.write(df_single)

        # Am Ende: Gesamtergebnis
        st.write("---")
        st.write("**Gesamtergebnisse (Up-to-500-Lines) aller Embedding-Modelle:**")
        if len(st.session_state.lines_results) > 0:
            df_all = pd.DataFrame(st.session_state.lines_results)
            df_all["Top-1 (%)"] = (df_all["top1_ratio"]*100).round(1)
            df_all["Top-3 (%)"] = (df_all["top3_ratio"]*100).round(1)

            df_all = df_all[["embedding_model", "lines_folder", "Top-1 (%)", "Top-3 (%)"]]
            st.write(df_all)

            # ggf. speichern
            base_output_dir = os.path.join("TestResults", "EmbeddingModels")
            lines_csv_path = os.path.join(base_output_dir, "results_lines_100_500.csv")
            df_all.to_csv(lines_csv_path, index=False)
            st.write(f"Gesamttabelle (Up to 500 Lines) gespeichert als {lines_csv_path}")
        else:
            st.write("Keine Ergebnisse in st.session_state.lines_results vorhanden.")

    except Exception as e:
        st.write(f"Fehler in run_tests_for_lines: {e}")


# ------------------ CHANGED-CODES-Test ------------------
def run_tests_for_changed_codes():
    """
    Führt Tests mit dem Ordner "TestCode/TestCode_Changed" durch, analog zum Standard-Test.
    Dadurch wird die Embedding Leistung bei abgeänderten Codebeispielen mit der selben CVE getestet.
    - Lädt die "data/data_v1", berechnet Embedding
    - Testet alle Embedding-Modelle gegen die geänderten Codebeispiele
    - Zeigt Rang-Farben, Summaries, wählt bestes Embedding => testet LLM
    """
    try:
        st.session_state.results_overall = []
    except:  
        pass

    testdata_dir = os.path.join(os.path.dirname(__file__), "TestCode", "TestCode_Changed")
    base_output_dir = os.path.join(os.path.dirname(__file__), "TestResults", "EmbeddingModels")
    llm_output_base = os.path.join(os.path.dirname(__file__), "TestResults", "LLMs")

    # EMBEDDING-Tests
    for emb_model in embedding_models:
        st.markdown(
            f"<span style='color:white; font-size:20px;'>**Starte Tests (Changed Codes) für Embedding Modell:** </span>"
            f"<span style='color:#d99a69; font-size:20px;'>{emb_model}</span>",
            unsafe_allow_html=True
        )
        st.write("---")
        run_tests_for_embedding_model(emb_model, testdata_dir, base_output_dir)

    st.write("**Alle Tests (Changed Codes) für alle Embedding-Modelle abgeschlossen.**")

    df = pd.DataFrame(st.session_state.results_overall)
    if df.empty:
        st.write("Keine Ergebnisse (Changed Codes) verfügbar.")
        return

    df.sort_values(by=["top3_ratio", "top1_ratio", "avg_similarity"],
                   ascending=[False, False, False],
                   inplace=True)
    df["Top-1 (Prozent)"] = (df["top1_ratio"] * 100).round(1).astype(str) + "%"
    df["Top-3 (Prozent)"] = (df["top3_ratio"] * 100).round(1).astype(str) + "%"

    st.write("**Zusammenfassung (Changed Codes) aller Embedding-Modelle:**")
    st.write(df[["embedding_model", "avg_similarity", "Top-1 (Prozent)", "Top-3 (Prozent)"]])

    results_summary_file = os.path.join(base_output_dir, "results_summary_embedding_models_changed.csv")
    df.to_csv(results_summary_file, index=False)
    st.write("Ergebnisse gespeichert (Changed Codes).")

    # Kategorie-Tabelle
    st.write("---")
    st.write("**Kategorien (0_20), (0_40), (0_60), (0_80), (0_100), (0_200), (0_300), (0_400), (0_500)** – Übersicht je Modell:")
    show_file_stats_tables()

    # Bestes Embedding
    best_emb_model = df.iloc[0]["embedding_model"]
    st.markdown(
        f"<span style='color:white;'>**Bestes Embedding (nach Top-3-Ratio, Changed Codes):** </span>"
        f"<span style='color:#d99a69;'>{best_emb_model}</span>",
        unsafe_allow_html=True
    )

    clear_gpu_memory()
    if os.path.exists("data.pkl"):
        os.remove("data.pkl")

    embeddings = HuggingFaceEmbeddings(model_name=best_emb_model)
    docs = load_data_from_directory("data/data_v1/only_code")
    global documents, document_embeddings, metadatas
    documents = docs
    metadatas = [d.metadata for d in documents]
    texts = [d.page_content for d in docs]
    document_embeddings = embeddings.embed_documents(texts)
    document_embeddings = np.array(document_embeddings, dtype=np.float32)
    document_embeddings = normalize(document_embeddings, axis=1)
    with open("data.pkl", "wb") as f:
        pickle.dump((documents, document_embeddings, metadatas), f)

    st.markdown(
        f"<span style='color:yellow;'>**data.pkl** </span>"
        f"<span style='color:white;'>**wurde neu erstellt mit** </span>"
        f"<span style='color:#d99a69;'>{best_emb_model}</span>",
        unsafe_allow_html=True
    )

    # LLM-Tests
    st.write("---")
    st.markdown(
        f"<span style='color:white; font-size:20px;'>**Zweiter Durchlauf LLMs (Changed Codes)** </span>"
        f"<span style='color:#d99a69; font-size:20px;'>{best_emb_model}</span>",
        unsafe_allow_html=True
    )
    st.write("---")

    for llm_model in llm_models:
        st.markdown(
            f"<span style='color:white; font-size:18px;'>**Starte Tests LLM (Changed Codes):** </span>"
            f"<span style='color:#d99a69; font-size:18px;'>{llm_model}</span> "
            f"<span style='color:white;'>(Embedding: {best_emb_model})</span>",
            unsafe_allow_html=True
        )
        st.write("---")
        run_tests_for_llm(best_emb_model, llm_model, testdata_dir, llm_output_base)

    st.write("**Alle Tests (Changed Codes) abgeschlossen.**\n")
    st.write("**Getestete LLM-Modelle:**\n")
    for llm_model in llm_models:
        st.markdown(
            f"<span style='color:white; font-size:18px;'>- </span>"
            f"<span style='color:#d99a69; font-size:18px;'>{llm_model}</span>",
            unsafe_allow_html=True
        )

    st.write(df)


def main():
    """
    Streamlit-Einstiegspunkt:
     - Chat-Eingabebereich
     - Buttons für unterschiedliche Tests
       1) "Run Tests" => nutzt TestCode/TestCode_0_Changes
       2) "Test Up to 500 Lines" => Ordner TestCode/TestCode_lines/{100,200,300,400,500}
       3) "Test Changed Codes" => Ordner TestCode/TestCode_Changed
       4) "Test Dimensions" => Ordner data/Dimensions
    """
    global documents, document_embeddings, metadatas, embeddings

    st.title("Forschungsprojekt RAG-LLM")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "file_stats" not in st.session_state:
        st.session_state.file_stats = {}

    # Chat-Eingabe
    prompt = st.chat_input("Frage hier eingeben")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Standard-Embedding wählen (erstes aus der Liste)
        emb_model = embedding_models[0]
        embeddings = HuggingFaceEmbeddings(model_name=emb_model)

        # Falls data.pkl nicht existiert, neu anlegen
        if not os.path.exists("data.pkl"):
            docs = load_data_from_directory("data/data_v1/only_code")
            documents = docs
            metadatas = [d.metadata for d in documents]
            texts = [d.page_content for d in docs]
            document_embeddings = embeddings.embed_documents(texts)
            document_embeddings = np.array(document_embeddings, dtype=np.float32)
            document_embeddings = normalize(document_embeddings, axis=1)
            with open("data.pkl", "wb") as f:
                pickle.dump((documents, document_embeddings, metadatas), f)
        else:
            with open("data.pkl", "rb") as f:
                documents, document_embeddings, metadatas = pickle.load(f)

        llama_response, top_doc, similarities = hybrid_response(prompt, selectedLLM, emb_model)

        st.session_state.messages.append({"role": "assistant", "content": llama_response})

    # Chatverlauf anzeigen
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)

    st.write("---")

    if "results_overall" not in st.session_state:
        st.session_state.results_overall = []

    # 1) Button: Standard Tests => Ordner TestCode/TestCode_0_Changes
    if st.button("Run Tests"):
        testdata_dir = os.path.join(os.path.dirname(__file__), "TestCode", "TestCode_0_Changes")
        base_output_dir = os.path.join(os.path.dirname(__file__), "TestResults", "EmbeddingModels")
        llm_output_base = os.path.join(os.path.dirname(__file__), "TestResults", "LLMs")

        st.session_state.results_overall = []

        # Embedding-Modelle testen
        for emb_model in embedding_models:
            st.markdown(
                f"<span style='color:white; font-size:20px;'>**Starte Tests für Embedding Modell:** </span>"
                f"<span style='color:#d99a69; font-size:20px;'>{emb_model}</span>",
                unsafe_allow_html=True
            )
            st.write("---")
            run_tests_for_embedding_model(emb_model, testdata_dir, base_output_dir)

        st.write("**Alle Tests für alle Embedding-Modelle abgeschlossen.**")

        df = pd.DataFrame(st.session_state.results_overall)
        if not df.empty:
            # Sortierung nach Top3-Ratio, Top1-Ratio, AvgSimilarity
            df.sort_values(by=["top3_ratio", "top1_ratio", "avg_similarity"],
                           ascending=[False, False, False],
                           inplace=True)
            df["Top-1 (Prozent)"] = (df["top1_ratio"] * 100).round(1).astype(str) + "%"
            df["Top-3 (Prozent)"] = (df["top3_ratio"] * 100).round(1).astype(str) + "%"

            st.write("**Zusammenfassung aller Embedding-Modelle:**")
            st.write(df[["embedding_model", "avg_similarity", "Top-1 (Prozent)", "Top-3 (Prozent)"]])

            # Ergebnisse auch als CSV
            results_summary_file = os.path.join(base_output_dir, "results_summary_embedding_models.csv")
            df.to_csv(results_summary_file, index=False)
            st.write(f"Ergebnisse gespeichert")

            # Kategorie-Tabelle
            st.write("---")
            st.write("**Kategorien (0_20), (0_40), (0_60), (0_80), (0_100)** – Übersichts-Tabelle je Modell:")
            show_file_stats_tables()

            # Bestes Embedding wählen (Zeile 0 in sortiertem DF)
            best_emb_model = df.iloc[0]["embedding_model"]
            st.markdown(
                f"<span style='color:white;'>**Bestes Embedding Modell (nach Top-3-Ratio):** </span>"
                f"<span style='color:#d99a69;'>{best_emb_model}</span>",
                unsafe_allow_html=True
            )

            clear_gpu_memory()
            if os.path.exists("data.pkl"):
                os.remove("data.pkl")

            # Erneut data.pkl mit dem besten Embedding erzeugen
            embeddings = HuggingFaceEmbeddings(model_name=best_emb_model)
            docs = load_data_from_directory("data/data_v1/only_code")
            documents = docs
            metadatas = [d.metadata for d in documents]
            texts = [d.page_content for d in docs]
            document_embeddings = embeddings.embed_documents(texts)
            document_embeddings = np.array(document_embeddings, dtype=np.float32)
            document_embeddings = normalize(document_embeddings, axis=1)
            with open("data.pkl", "wb") as f:
                pickle.dump((documents, document_embeddings, metadatas), f)

            st.markdown(
                f"<span style='color:yellow;'>**data.pkl** </span>"
                f"<span style='color:white;'>**wurde neu erstellt mit** </span>"
                f"<span style='color:#d99a69;'>{best_emb_model}</span>",
                unsafe_allow_html=True
            )

            st.write("---")
            st.markdown(
                f"<span style='color:white; font-size:20px;'>**Zweiter Durchlauf für LLMs** </span>"
                f"<span style='color:#d99a69; font-size:20px;'>{best_emb_model}</span>",
                unsafe_allow_html=True
            )
            st.write("---")

            # LLM-Tests
            for llm_model in llm_models:
                st.markdown(
                    f"<span style='color:white; font-size:18px;'>**Starte Tests für LLM:** </span>"
                    f"<span style='color:#d99a69; font-size:18px;'>{llm_model}</span> "
                    f"<span style='color:white;'>(Embedding: {best_emb_model})</span>",
                    unsafe_allow_html=True
                )
                st.write("---")
                run_tests_for_llm(best_emb_model, llm_model, testdata_dir, llm_output_base)

            st.write("**Alle Tests abgeschlossen.**\n")
            st.write("**Getestete LLM-Modelle:**\n")
            for llm_model in llm_models:
                st.markdown(
                    f"<span style='color:white; font-size:18px;'>- </span>"
                    f"<span style='color:#d99a69; font-size:18px;'>{llm_model}</span>",
                    unsafe_allow_html=True
                )

            st.write(df)
        else:
            st.write("Keine Ergebnisse verfügbar. Kann kein bestes Embedding-Modell auswählen.")

    # 2) Button: "Test Up to 500 Lines"
    if st.button("(Test Up to 500 Lines)"):
        run_tests_for_lines()

    # 3) Button: "Test Changed Codes"
    if st.button("Test Changed Codes"):
        run_tests_for_changed_codes()

    # 4) Button: "Test Dimensions"
    if st.button("Test Dimensions"):
        run_tests_for_dimensions()


# --------------------------------
#   Streamlit-Einstiegspunkt
# --------------------------------
if __name__ == '__main__':
    main()

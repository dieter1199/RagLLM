# import os
# import re
# import requests
# import json
# import chardet
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pickle
# from sklearn.preprocessing import normalize
# from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
# from langchain.schema import Document
# from langchain.llms.base import LLM
# from langchain.schema import Generation, LLMResult
# from langchain_huggingface import HuggingFaceEmbeddings
# import streamlit as st
# import shutil
# import pandas as pd
# from collections import defaultdict

# # ------------------ Globale Variablen ------------------
# selectedLLM = "deepseek-r1:14b"
# selectedFileFormat = None
# documents = []
# document_embeddings = None
# metadatas = []
# embeddings = None

# import torch
# #st.text(f"CUDA verfügbar: {torch.cuda.is_available()}")
# if torch.cuda.is_available():
#     print(f"GPU: {torch.cuda.get_device_name(0)}")
#     #st.text(f"GPU: {torch.cuda.get_device_name(0)}")


# # Zu testende Embedding Modelle.
# embedding_models = [
#     "intfloat/e5-large-v2"  # <------------------------------Best Model so far
#     #"intfloat/multilingual-e5-large",
#     #"WhereIsAI/UAE-Large-V1",
#     #"BAAI/bge-large-en-v1.5",
#     #"microsoft/graphcodebert-base"
# ]

# # Zu testende LLMs.
# llm_models = [
#     # "deepseek-r1:14b"
# ]


# def clear_gpu_memory():
#     """
#     GPU-Speicher bereinigen.
#     """
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         torch.cuda.synchronize()
#         print("[INFO] GPU-Speicher wurde bereinigt.")
#         print("[INFO] ---------.")
#         #st.text(f"CUDA verfügbar: {torch.cuda.is_available()}")
#         #st.text(f"GPU: {torch.cuda.get_device_name(0)}")


# # Einmal direkt beim Skript-Start ausführen, um GPU-Speicher aufzuräumen.
# clear_gpu_memory()


# # ------------------ LLM-Klasse (Ollama) ------------------
# class OllamaLLM(LLM):
#     """
#     Eine angepasste LLM-Klasse zur Anbindung an den Ollama-Server.
#     """

#     def __init__(self, llm_name):
#         super().__init__()
#         self._llm_name = llm_name

#     def _call(self, prompt: str, stop=None):
#         url = "http://localhost:11434/api/generate"
#         payload = {
#             "model": self._llm_name,
#             "prompt": prompt,
#             "temperature": 0,
#             "top_p": 0.85,
#             "top_k": 3
#         }
#         headers = {"Content-Type": "application/json"}

#         print(f"[DEBUG] _call() => Modellname: {self._llm_name}, Prompt length: {len(prompt)} chars")

#         try:
#             response = requests.post(url, headers=headers, json=payload)
#         except requests.exceptions.RequestException as re:
#             raise RuntimeError(f"Fehler beim LLM-Aufruf (Netzwerk/Connection): {re}") from re

#         try:
#             raw_content = response.content.decode('utf-8')
#         except UnicodeDecodeError:
#             raw_content = response.content.decode('latin-1')

#         try:
#             responses = []
#             for line in raw_content.splitlines():
#                 if line.strip():
#                     if "error" in line.lower():
#                         raise ValueError(f"LLM gab eine Fehlermeldung zurück: {line}")
#                     json_line = json.loads(line)
#                     if "response" in json_line:
#                         responses.append(json_line["response"])
#             return "".join(responses)
#         except json.JSONDecodeError as e:
#             return f"Fehler bei der Verarbeitung des JSON: {e}"

#     def generate(self, prompts, stop=None, callbacks=None, **kwargs):
#         generations = []
#         for prompt in prompts:
#             result = self._call(prompt, stop)
#             generations.append([Generation(text=result)])
#         return LLMResult(generations=generations)

#     @property
#     def _llm_type(self):
#         return "ollama_llm"

#     @property
#     def _identifying_params(self):
#         return {"name_of_llm": self._llm_name}


# # ------------------ Hilfsfunktionen ------------------
# def load_data_from_directory(directory):
#     global selectedFileFormat
#     docs = []
#     print(f"[INFO] load_data_from_directory() => Scanne Ordner: {directory}")
#     if not os.path.exists(directory):
#         print(f"Fehler: Das Verzeichnis '{directory}' existiert nicht.")
#         return docs

#     for filename in os.listdir(directory):
#         filepath = os.path.join(directory, filename)
#         try:
#             with open(filepath, 'rb') as raw_file:
#                 raw_data = raw_file.read()
#                 result = chardet.detect(raw_data)
#                 encoding = result['encoding']
#         except Exception:
#             st.write("---")
#             continue

#         if filename.endswith(".json"):
#             try:
#                 with open(filepath, 'r', encoding=encoding, errors='ignore') as json_file:
#                     content = json_file.read()
#                 parsed_data = json.loads(content)
#                 docs.append(Document(page_content=json.dumps(parsed_data), metadata={"source": filename}))
#                 if selectedFileFormat is None:
#                     selectedFileFormat = "json"
#             except json.JSONDecodeError:
#                 try:
#                     with open(filepath, 'r', encoding=encoding, errors='ignore') as fallback_file:
#                         fallback_content = fallback_file.read()
#                     docs.append(Document(page_content=fallback_content, metadata={"source": filename}))
#                     if selectedFileFormat is None:
#                         selectedFileFormat = "txt"
#                 except Exception:
#                     continue

#         elif filename.endswith(".txt"):
#             try:
#                 with open(filepath, 'r', encoding=encoding, errors='ignore') as text_file:
#                     text_content = text_file.read()
#                     docs.append(Document(page_content=text_content, metadata={"source": filename}))
#                     if selectedFileFormat is None:
#                         selectedFileFormat = "txt"
#             except Exception:
#                 st.write("---")
#         else:
#             pass

#     print(f"[INFO] load_data_from_directory() => Fertig, {len(docs)} Dateien geladen.")
#     return docs


# def normalize_test_filename(test_file: str) -> str:
#     filename_noext, _ = os.path.splitext(test_file)
#     normalized = re.sub(r"_\(\d+_\d+\)$", "", filename_noext)
#     return normalized


# def manual_retrieval_with_top3(prompt):
#     """
#     Verschiedene Ansätze auf Dokument-Ebene oder Zeilen-Ebene.
#     Du wählst genau EINEN, indem du den entsprechenden Block einkommentierst
#     und die anderen auskommentierst.
#     """

#     print("[DEBUG] manual_retrieval_with_top3() => Starte Einbettung des Prompt...")

#     # =========================================================================
#     # ========== ANSATZ #1: Doc-Level Single-Vektor pro Dokument  =============
#     # =========================================================================
#     # => Dies ist der Standard (doc_embeddings)
#     #
#     # query_embedding = embeddings.embed_query(prompt)
#     # query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
#     # similarities = document_embeddings @ query_embedding
#     # sorted_indices = np.argsort(similarities)[::-1]
#     # top3_indices = sorted_indices[:3]
#     # top3_docs = [documents[i] for i in top3_indices]
#     # top3_sims = [similarities[i] for i in top3_indices]
    
#     # return top3_docs, top3_sims, similarities, query_embedding

#     # =========================================================================
#     # ========== ANSATZ #2: Zeilenbasiert (MAX) ===============================
#     # =========================================================================
#     #
#     # query_embedding = embeddings.embed_query(prompt)
#     # query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
#     # doc_scores = []
#     # for i, doc in enumerate(documents):
#     #     lines = doc.page_content.splitlines()
#     #     if not lines:  # Falls Dokument leer
#     #         doc_scores.append(0.0)
#     #         continue
#     #     line_embs = embeddings.embed_documents(lines)
#     #     line_embs = normalize(line_embs, axis=1)
    
#     #     line_sims = line_embs @ query_embedding
#     #     if line_sims.shape[0] == 0:
#     #         doc_scores.append(0.0)
#     #         continue
    
#     #     doc_score = np.max(line_sims)
#     #     doc_scores.append(doc_score)
    
#     # doc_scores = np.array(doc_scores)
#     # sorted_indices = np.argsort(doc_scores)[::-1]
#     # top3_indices = sorted_indices[:3]
#     # top3_docs = [documents[i] for i in top3_indices]
#     # top3_sims = [doc_scores[i] for i in top3_indices]
    
#     # return top3_docs, top3_sims, doc_scores, query_embedding

#     # =========================================================================
#     # ========== ANSATZ #3: Zeilenbasiert (MEAN) ==============================
#     # =========================================================================
#     # query_embedding = embeddings.embed_query(prompt)
#     # query_embedding = query_embedding / np.linalg.norm(query_embedding)

#     # doc_scores = []
#     # for i, doc in enumerate(documents):
#     #     lines = doc.page_content.splitlines()
#     #     if not lines:
#     #         doc_scores.append(0.0)
#     #         continue

#     #     line_embs = embeddings.embed_documents(lines)
#     #     line_embs = normalize(line_embs, axis=1)
#     #     line_sims = line_embs @ query_embedding
#     #     if line_sims.shape[0] == 0:
#     #         doc_scores.append(0.0)
#     #         continue

#     #     doc_score = np.mean(line_sims)
#     #     doc_scores.append(doc_score)

#     # doc_scores = np.array(doc_scores)
#     # sorted_indices = np.argsort(doc_scores)[::-1]
#     # top3_indices = sorted_indices[:3]
#     # top3_docs = [documents[i] for i in top3_indices]
#     # top3_sims = [doc_scores[i] for i in top3_indices]

#     # return top3_docs, top3_sims, doc_scores, query_embedding


#     # =========================================================================
#     # ========== ANSATZ #4: Zeilenbasiert (SUM) ===============================
#     # =========================================================================
#     # query_embedding = embeddings.embed_query(prompt)
#     # query_embedding = query_embedding / np.linalg.norm(query_embedding)
#     #
#     # doc_scores = []
#     # for i, doc in enumerate(documents):
#     #     lines = doc.page_content.splitlines()
#     #     if not lines:
#     #         doc_scores.append(0.0)
#     #         continue
#     #
#     #     line_embs = embeddings.embed_documents(lines)
#     #     line_embs = normalize(line_embs, axis=1)
#     #     line_sims = line_embs @ query_embedding
#     #
#     #     if line_sims.shape[0] == 0:
#     #         doc_scores.append(0.0)
#     #         continue
#     #
#     #     doc_score = np.sum(line_sims)
#     #     doc_scores.append(doc_score)
#     #
#     # doc_scores = np.array(doc_scores)
#     # sorted_indices = np.argsort(doc_scores)[::-1]
#     # top3_indices = sorted_indices[:3]
#     # top3_docs = [documents[i] for i in top3_indices]
#     # top3_sims = [doc_scores[i] for i in top3_indices]
#     #
#     # return top3_docs, top3_sims, doc_scores, query_embedding

#     # =========================================================================
#     # ========== ANSATZ #5: Zeilenbasiert (Top-K-Mean) ========================
#     # =========================================================================

#     print("[DEBUG] manual_retrieval_with_top3() => Starte Einbettung des Prompt...")

#     # 1) Prompt in Zeilen aufsplitten und embeddet
#     prompt_lines = prompt.splitlines()
#     if not prompt_lines:
#         print("   -> Prompt ist leer, gebe 0-Scores zurück.")
#         # Falls Prompt leer, hat jedes Dokument Score=0
#         doc_scores = np.zeros(len(documents), dtype=np.float32)
#         sorted_indices = np.argsort(doc_scores)[::-1]
#         top3_indices = sorted_indices[:3]
#         top3_docs = [documents[idx] for idx in top3_indices]
#         top3_sims = [doc_scores[idx] for idx in top3_indices]
#         return top3_docs, top3_sims, doc_scores, None

#     # Promptzeilen einbetten
#     prompt_line_embs = embeddings.embed_documents(prompt_lines)
#     prompt_line_embs = normalize(prompt_line_embs, axis=1)

#     doc_scores = []

#     for i, doc in enumerate(documents):
#         lines = doc.page_content.splitlines()
#         print(f"\n[DEBUG] Dokument-Index {i} => Quelle: {doc.metadata.get('source','?')}, Zeilen: {len(lines)}")

#         if not lines:
#             doc_scores.append(0.0)
#             print("   -> Dokument ist leer, Score=0.0")
#             continue

#         # 2) Doc-Zeilen einbetten
#         doc_line_embs = embeddings.embed_documents(lines)
#         doc_line_embs = normalize(doc_line_embs, axis=1)

#         # 3) Für jede Doc-Zeile: max Similarity zu einer Prompt-Zeile
#         line_max_sims = []
#         for j, doc_line_emb in enumerate(doc_line_embs):
#             # Cosine-Sim zur *jeden* Prompt-Zeile
#             sims_to_prompt = doc_line_emb @ prompt_line_embs.T
#             # nimm das Maximum
#             best_line_sim = np.max(sims_to_prompt)
#             line_max_sims.append(best_line_sim)

#         # 4) Mittelwert aller Zeilen-Maxima = doc_score
#         doc_score = np.mean(line_max_sims)
#         doc_scores.append(doc_score)

#         # Debug-Ausgabe
#         print(f"   -> line_max_sims[:5] (Beispiele): {line_max_sims[:7]} ...")
#         print(f"   -> doc_score (Avg of line-wise maxima): {doc_score:.4f}")

#     doc_scores = np.array(doc_scores)
#     # Ranking
#     sorted_indices = np.argsort(doc_scores)[::-1]
#     top3_indices = sorted_indices[:3]
#     top3_docs = [documents[idx] for idx in top3_indices]
#     top3_sims = [doc_scores[idx] for idx in top3_indices]

#     return top3_docs, top3_sims, doc_scores, None




# def plot_file_specific(similarities, prompt, title_prefix=""):
#     """
#     Zeichnet einen Scatterplot und einen Barplot für die 'similarities',
#     die aus manual_retrieval_with_top3() kommen.
#     """
#     if len(similarities) > 0 and prompt:
#         os.makedirs("GlobalPlots", exist_ok=True)

#         import matplotlib.pyplot as plt
#         from matplotlib.colors import LinearSegmentedColormap

#         threshold = 0.9
#         outlier_indices = np.where(similarities > threshold)[0]

#         plt.figure(figsize=(10, 6))
#         plt.scatter(range(len(similarities)), similarities, color='red', edgecolor='black')
#         plt.scatter(outlier_indices, similarities[outlier_indices], color='green', edgecolor='black', zorder=5)
#         plt.ylim(0, 1)
#         plt.title(f"{title_prefix}Cosinus-Ähnlichkeit (einzelne Anfrage)")
#         plt.xlabel("Dokument Index")
#         plt.ylabel("Cosinus-Ähnlichkeit")
#         plt.tight_layout()
#         plt.savefig("GlobalPlots/query_to_documents_cosine_similarity_scatter_outliers.png")
#         plt.close()

#         top_indices = np.argsort(similarities)[::-1][:20]
#         top_sims = similarities[top_indices]
#         top_docs = [documents[i].metadata.get("source", f"Doc_{i}") for i in top_indices]
#         colors = [(1, 0, 0), (0, 1, 0)]
#         cm = LinearSegmentedColormap.from_list('RedGreenGradient', colors, N=256)
#         norm = plt.Normalize(vmin=0.5, vmax=1.0)
#         bar_colors = [cm(norm(val)) for val in top_sims]

#         fig, ax = plt.subplots(figsize=(10, 6))
#         ax.barh(range(len(top_sims)), top_sims, edgecolor='black', color=bar_colors)
#         ax.invert_yaxis()
#         ax.set_yticks(range(len(top_sims)))
#         ax.set_yticklabels(top_docs)
#         ax.set_xlim(0, 1)
#         ax.set_title(f"{title_prefix}Top 20 Dokumente (einzelne Anfrage)")
#         ax.set_xlabel("Cosinus-Ähnlichkeit")

#         sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
#         sm.set_array([])
#         cbar = fig.colorbar(sm, ax=ax)
#         cbar.set_label('Cosinus-Ähnlichkeit')
#         fig.tight_layout()
#         fig.savefig("GlobalPlots/top_20_similar_documents_gradient.png")
#         plt.close(fig)


# def plot_global_matrices(emb_model, title_suffix=""):
#     """
#     Zeichnet eine globale Cosine Similarity Matrix & Distanzmatrix
#     für die Embeddings, die mit doc-level-Berechnung (document_embeddings)
#     entstanden sind. (Reines Logging.)
#     """
#     import matplotlib.pyplot as plt
#     import seaborn as sns

#     if len(document_embeddings) > 1:
#         sim_matrix_cosine = cosine_similarity(document_embeddings)
#         dist_matrix_euclidean = euclidean_distances(document_embeddings)

#         plt.figure(figsize=(10, 8))
#         sns.heatmap(sim_matrix_cosine, cmap="coolwarm", annot=False, vmin=0, vmax=1)
#         plt.title(f"Cosinus-Ähnlichkeitsmatrix\nEmbedding-Modell: {emb_model}")
#         plt.xlabel("Dok Index")
#         plt.ylabel("Dok Index")
#         plt.savefig("GlobalPlots/cosine_similarity_matrix.png")
#         plt.close()
        
#         euc_max = np.max(dist_matrix_euclidean)
#         plt.figure(figsize=(10, 8))
#         sns.heatmap(dist_matrix_euclidean, cmap="coolwarm", annot=False, vmin=0, vmax=euc_max)
#         plt.title(f"Euklidische Distanzmatrix {title_suffix}")
#         plt.xlabel("Dok Index")
#         plt.ylabel("Dok Index")
#         plt.savefig("GlobalPlots/euclidean_distance_matrix.png")
#         plt.close()

#         plt.figure(figsize=(10, 6))
#         plt.hist(sim_matrix_cosine.flatten(), bins=50, color='skyblue', edgecolor='black',
#                  range=(0, 1), density=True)
#         plt.title(f"Verteilung der Cosine-Sim-Werte {title_suffix}")
#         plt.xlabel("Ähnlichkeitswert")
#         plt.ylabel("Prozentuale Häufigkeit")
#         plt.xlim(0, 1)
#         plt.savefig("GlobalPlots/cosine_similarity_histogram.png")
#         plt.close()


# def hybrid_response(prompt, llm_name, emb_model=""):
#     """
#     Wird für Live-Prompts genutzt – holt sich Top3 aus 'manual_retrieval_with_top3()'.
#     """
#     print(f"[DEBUG] hybrid_response() => LLM: {llm_name}, PromptLen={len(prompt)}")
#     top3_docs, top3_sims, similarities, _ = manual_retrieval_with_top3(prompt)

#     # Holen wir uns mal Content & cve_id:
#     doc_content_top1 = top3_docs[0].page_content if len(top3_docs) > 0 else ""
#     cve_id1 = "N/A"
#     try:
#         j1 = json.loads(doc_content_top1)
#         cve_id1 = j1.get("cve_id", "N/A")
#     except:
#         pass

#     llama_prompt = f"""You are a security analyst specializing in identifying code vulnerabilities (CVEs).

# User question: '{prompt}'

# Retrieved context with CVE ID {cve_id1}:
# {doc_content_top1}

# Begin your response with “Vulnerability Found” if a relevant CVE is identified or “Secure” if not.
# """

#     current_llm = OllamaLLM(llm_name)
#     try:
#         llama_response = current_llm._call(llama_prompt)
#     except Exception as e:
#         llama_response = f"Fehler LLM: {e}"

#     if not llama_response.strip():
#         llama_response = "Keine Antwort vom LLM."

#     plot_file_specific(similarities, prompt, title_prefix=f"LLM {llm_name}: ")

#     return llama_response, (top3_docs[0] if len(top3_docs) > 0 else None), similarities


# def get_file_stats():
#     if "file_stats" not in st.session_state:
#         st.session_state.file_stats = {}
#     return st.session_state.file_stats


# def update_file_stats(embedding_model, filename, is_top1, is_top3, similarity=0.0):
#     """
#     Erweitert um 'similarity', damit wir später einen 
#     Durchschnitt pro Kategorie ausgeben können.
#     """
#     file_stats_global = get_file_stats()
#     if embedding_model not in file_stats_global:
#         file_stats_global[embedding_model] = {}
#     if filename not in file_stats_global[embedding_model]:
#         file_stats_global[embedding_model][filename] = {
#             "top1_count": 0,
#             "top3_count": 0,
#             "tests_done": 0,
#             "sum_sims": 0.0  # NEU: hier sammeln wir die Similarities an
#         }
#     file_stats_global[embedding_model][filename]["tests_done"] += 1
#     if is_top1:
#         file_stats_global[embedding_model][filename]["top1_count"] += 1
#     if is_top3:
#         file_stats_global[embedding_model][filename]["top3_count"] += 1

#     # NEU: Summe der Similarities hochzählen
#     file_stats_global[embedding_model][filename]["sum_sims"] += similarity


# def show_file_stats_tables():
#     if "file_stats" not in st.session_state:
#         st.write("Keine Datei-Statistiken vorhanden.")
#         return

#     categories = ["_(0_20)", "_(0_40)", "_(1_40)", "_(2_40)", "_(0_60)", "_(0_80)", "_(0_100)", "_(0_200)", "_(0_300)", "_(0_400)", "_(0_500)"]
#     suffix_regex = re.compile(r"_\(\d+_\d+\)$")

#     file_stats_global = st.session_state.file_stats

#     if "ALL_MODELS" in file_stats_global:
#         del file_stats_global["ALL_MODELS"]
#     file_stats_global["ALL_MODELS"] = {}

#     for emb_model, files_map in list(file_stats_global.items()):
#         if emb_model == "ALL_MODELS":
#             continue
#         for fname, stats in files_map.items():
#             if fname not in file_stats_global["ALL_MODELS"]:
#                 file_stats_global["ALL_MODELS"][fname] = {
#                     "top1_count": 0,
#                     "top3_count": 0,
#                     "tests_done": 0
#                 }
#             file_stats_global["ALL_MODELS"][fname]["top1_count"] += stats["top1_count"]
#             file_stats_global["ALL_MODELS"][fname]["top3_count"] += stats["top3_count"]
#             file_stats_global["ALL_MODELS"][fname]["tests_done"] += stats["tests_done"]

#     def produce_category_table(emb_model, files_map):
#         from collections import defaultdict
#         # top1_count, top3_count, tests_done, sum_sims
#         cat_stats = defaultdict(lambda: [0, 0, 0, 0.0])

#         for fname, stt in files_map.items():
#             base, ext = os.path.splitext(fname)
#             match = suffix_regex.search(base)
#             if match:
#                 bracket = match.group(0)
#             else:
#                 bracket = None

#             if bracket in categories:
#                 cat_stats[bracket][0] += stt["top1_count"]
#                 cat_stats[bracket][1] += stt["top3_count"]
#                 cat_stats[bracket][2] += stt["tests_done"]
#                 # NEU: sum_sims aufsummieren
#                 cat_stats[bracket][3] += stt.get("sum_sims", 0.0)

#         row_list = []
#         for cat in categories:
#             sum_t1, sum_t3, sum_tests, sum_sims = cat_stats[cat]
#             if sum_tests == 0:
#                 continue

#             # Prozentwerte für Top-1 / Top-3
#             top1_pct = sum_t1 / sum_tests * 100
#             top3_pct = sum_t3 / sum_tests * 100

#             # NEU: Durchschnittliche Similarity in dieser Kategorie
#             avg_sim = sum_sims / sum_tests

#             row_list.append({
#                 "Kategorie": cat,
#                 "Top-1": f"{top1_pct:.1f}%",
#                 "Top-3": f"{top3_pct:.1f}%",
#                 "AvgSim": f"{avg_sim:.4f}"  # neue Spalte
#             })

#         if not row_list:
#             st.write(f"(Keine Dateien in den bekannten Kategorien für {emb_model})")
#             return

#         df = pd.DataFrame(row_list, columns=["Kategorie", "Top-1", "Top-3", "AvgSim"])
#         st.write(f"**Modell: {emb_model}**")
#         st.write(df)
#         st.write("-----")


#     for emb_model, files_map in file_stats_global.items():
#         produce_category_table(emb_model, files_map)


# # ------------------ TEST-Funktionen (Embedding+LLM) ------------------
# def run_tests_for_embedding_model(embedding_model_name, testdata_dir, base_output_dir):
#     """
#     Führt automatische Tests für ein bestimmtes Embedding-Modell durch:
#      - Lädt die Daten aus only_code
#      - Speichert doc-level Embeddings (zur Doku)
#      - ABER: Das eigentliche Ranking kommt aus manual_retrieval_with_top3()!
#     """
#     global embeddings, documents, document_embeddings, metadatas, selectedFileFormat
#     try:
#         clear_gpu_memory()
#         if os.path.exists("data.pkl"):
#             os.remove("data.pkl")

#         selectedFileFormat = None

#         data_dir = os.path.join("data", "data_v1", "only_code")
#         print(f"[INFO] Lade Dokumente aus => {data_dir}")
#         docs = load_data_from_directory(data_dir)

#         # Print Info: mit welchem Pfad wir arbeiten, zur Sicherheit
#         print("=== Documents loaded for embedding from only_code ===")
#         for i, d in enumerate(docs):
#             print(f"Doc index {i}: {d.metadata['source']} => path: {d.metadata.get('source_folder','')}")
#         print("=====================================================")

#         for d in docs:
#             d.metadata["source_folder"] = data_dir

#         documents = docs
#         metadatas = [d.metadata for d in documents]
#         texts = [d.page_content for d in docs]

#         embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
#         # doc-level embeddings (nur zum Logging/Plotten)
#         doc_emb = embeddings.embed_documents(texts)
#         doc_emb = np.array(doc_emb, dtype=np.float32)
#         doc_emb = normalize(doc_emb, axis=1)
#         # Globale Variable, nur für Plot Global
#         global document_embeddings
#         document_embeddings = doc_emb

#         with open("data.pkl", "wb") as f:
#             pickle.dump((documents, document_embeddings, metadatas), f)

#         safe_emb_model = embedding_model_name.replace(":", "_")
#         model_output_dir = os.path.join(base_output_dir, safe_emb_model)
#         os.makedirs(model_output_dir, exist_ok=True)

#         test_files = [f for f in os.listdir(testdata_dir)
#                       if os.path.isfile(os.path.join(testdata_dir, f))]

#         relevant_similarities = []
#         top1_count = 0
#         top3_count = 0
#         total_tests = len(test_files)

#         for test_file in test_files:
#             test_file_path = os.path.join(testdata_dir, test_file)
#             with open(test_file_path, "r", encoding="utf-8", errors="ignore") as f:
#                 test_code = f.read()

#             print("=== CODE SNIPPET FROM TestFile BEFORE EMBEDDING ===")
#             print(test_code)
#             print("==================================================")

#             st.markdown(
#                 f"<span style='color:#7bd1ed;'>**Verarbeite Testdatei:** </span>"
#                 f"<span style='color:#7bd1ed;'>{test_file}</span>",
#                 unsafe_allow_html=True
#             )

#             # >>>>>> Hier Ranking per manual_retrieval_with_top3 (Zeilenbasiert oder doc-level)
#             top3_docs, top3_sims, doc_scores, _ = manual_retrieval_with_top3(test_code)
#             # doc_scores = enten doc-level sims oder das aggregatorische "doc_score" je Document
#             plot_file_specific(doc_scores, test_code, title_prefix=f"Emb {embedding_model_name}: ")

#             output_subdir = os.path.join(model_output_dir, test_file)
#             os.makedirs(output_subdir, exist_ok=True)

#             plots_dir = "GlobalPlots"
#             for p in ["top_20_similar_documents_gradient.png", "query_to_documents_cosine_similarity_scatter_outliers.png"]:
#                 src = os.path.join(plots_dir, p)
#                 if os.path.exists(src):
#                     dst = os.path.join(output_subdir, p)
#                     shutil.copyfile(src, dst)

#             normalized_name = normalize_test_filename(test_file)
#             target_doc_base = normalized_name + ".json"

#             # Bestimme Rang/Similarity NICHT mehr über doc-level, sondern über doc_scores!
#             target_index = None
#             for i, meta in enumerate(metadatas):
#                 doc_src = meta.get("source", "")
#                 if doc_src == target_doc_base:
#                     target_index = i
#                     break

#             if target_index is not None:
#                 target_similarity = doc_scores[target_index]
#                 sorted_indices = np.argsort(doc_scores)[::-1]
#                 rank = np.where(sorted_indices == target_index)[0][0] + 1

#                 is_top1 = (rank == 1)
#                 is_top3 = (rank <= 3)

#                 if rank == 1:
#                     color = "#6ff261"
#                 elif rank <= 3:
#                     color = "#f5e551"
#                 else:
#                     color = "#f26161"

#                 if is_top1:
#                     top1_count += 1
#                 if is_top3:
#                     top3_count += 1

#                 relevant_similarities.append(target_similarity)

#                 st.markdown(
#                     f"<span style='color:{color};'>**Ziel-Datei Rang**: {rank}, Similarity: {target_similarity:.4f}</span>",
#                     unsafe_allow_html=True
#                 )

#                 sim_path = os.path.join(output_subdir, "similarity.txt")
#                 with open(sim_path, "w", encoding="utf-8") as sf:
#                     sf.write(f"Rank: {rank}\nSimilarity: {target_similarity:.4f}\n")

#                 update_file_stats(embedding_model_name, test_file, is_top1, is_top3, target_similarity)
#             else:
#                 st.write(f"**Keine passende .json-Datei für {normalized_name} gefunden**")

#             st.write("---")

#         # Nur zum Logging: Plot der doc-level Embeddings
#         plot_global_matrices(embedding_model_name, title_suffix=f"(Embedding: {embedding_model_name})")
#         for p in ["cosine_similarity_matrix.png", "euclidean_distance_matrix.png", "cosine_similarity_histogram.png"]:
#             src = os.path.join("GlobalPlots", p)
#             if os.path.exists(src):
#                 dst = os.path.join(model_output_dir, p)
#                 shutil.copyfile(src, dst)

#         if relevant_similarities:
#             avg_similarity = sum(relevant_similarities) / len(relevant_similarities)
#             top1_str = f"{top1_count}/{total_tests}"
#             top3_str = f"{top3_count}/{total_tests}"

#             st.markdown(
#                 f"<span style='color:yellow;'>**Durchschnittliche Cosinus-Ähnlichkeit: {avg_similarity:.4f}**</span>",
#                 unsafe_allow_html=True
#             )
#             st.markdown(
#                 f"<span style='color:yellow;'>**Trefferquote (Top-1)**: {top1_str}</span>",
#                 unsafe_allow_html=True
#             )
#             st.markdown(
#                 f"<span style='color:yellow;'>**Trefferquote (Top-3)**: {top3_str}</span>",
#                 unsafe_allow_html=True
#             )
#             st.write("---")

#             avg_file = os.path.join(model_output_dir, "average_similarity.txt")
#             with open(avg_file, "w", encoding="utf-8") as af:
#                 af.write(f"AvgSimilarity: {avg_similarity:.4f}\nTop-1: {top1_str}\nTop-3: {top3_str}\n")

#             st.session_state.results_overall.append({
#                 "embedding_model": embedding_model_name,
#                 "avg_similarity": avg_similarity,
#                 "top1_ratio": top1_count / total_tests if total_tests else 0.0,
#                 "top3_ratio": top3_count / total_tests if total_tests else 0.0
#             })
#         else:
#             st.write("Keine relevanten Similarities berechnet.")

#     except Exception as e:
#         st.write(f"Fehler in run_tests_for_embedding_model mit Modell {embedding_model_name}: {e}")


# def run_tests_for_llm(best_embedding_model, llm_name, testdata_dir, llm_output_base):
#     """
#     LLM-Test mit bereits erstelltem Embedding (data.pkl).
#     Achtung: Auch hier nutzen wir manual_retrieval_with_top3() für das Ranking.
#     """
#     global embeddings, documents, document_embeddings, metadatas
#     try:
#         clear_gpu_memory()

#         if not os.path.exists("data.pkl"):
#             st.write("data.pkl nicht gefunden – Abbruch.")
#             return

#         with open("data.pkl", "rb") as f:
#             documents, document_embeddings, metadatas = pickle.load(f)

#         embeddings = HuggingFaceEmbeddings(model_name=best_embedding_model)

#         safe_emb_model = best_embedding_model.replace(":", "_")
#         safe_llm_model = llm_name.replace(":", "_")
#         llm_output_dir = os.path.join(llm_output_base, f"{safe_emb_model}_LLM_{safe_llm_model}")
#         os.makedirs(llm_output_dir, exist_ok=True)

#         test_files = [f for f in os.listdir(testdata_dir) if os.path.isfile(os.path.join(testdata_dir, f))]

#         relevant_similarities = []
#         top1_count = 0
#         top3_count = 0
#         total_tests = len(test_files)

#         for test_file in test_files:
#             st.markdown(
#                 f"<span style='color:#7bd1ed; font-size:18px;'>**Verarbeite Datei (LLM):** </span>"
#                 f"<span style='color:#7bd1ed; font-size:18px;'>{llm_name} => {test_file}</span>",
#                 unsafe_allow_html=True
#             )

#             test_file_path = os.path.join(testdata_dir, test_file)
#             with open(test_file_path, "r", encoding="utf-8", errors="ignore") as f:
#                 test_code = f.read()

#             # Erneut manual_retrieval_with_top3()
#             try:
#                 llama_response, top_doc, doc_scores = hybrid_response(test_code, llm_name, best_embedding_model)
#             except Exception as e:
#                 st.markdown(f"<span style='color:red;'>Fehler beim LLM-Aufruf: {e}</span>", unsafe_allow_html=True)
#                 llama_response = f"LLM-Fehler: {e}"

#             st.write(llama_response)

#             output_subdir = os.path.join(llm_output_dir, test_file)
#             os.makedirs(output_subdir, exist_ok=True)

#             response_file = os.path.join(output_subdir, f"{safe_llm_model}_response.txt")
#             with open(response_file, "w", encoding="utf-8") as rf:
#                 rf.write(llama_response)

#             plot_file_specific(doc_scores, test_code, title_prefix=f"LLM {llm_name}: ")
#             for p in ["top_20_similar_documents_gradient.png", "query_to_documents_cosine_similarity_scatter_outliers.png"]:
#                 src = os.path.join("GlobalPlots", p)
#                 if os.path.exists(src):
#                     dst = os.path.join(output_subdir, p)
#                     shutil.copyfile(src, dst)

#             # Ranking analog wie in run_tests_for_embedding_model
#             normalized_name = normalize_test_filename(test_file)
#             target_doc_base = normalized_name + ".json"

#             target_index = None
#             for i, meta in enumerate(metadatas):
#                 doc_src = meta.get("source", "")
#                 if doc_src == target_doc_base:
#                     target_index = i
#                     break

#             if target_index is not None:
#                 target_similarity = doc_scores[target_index]
#                 sorted_indices = np.argsort(doc_scores)[::-1]
#                 rank = np.where(sorted_indices == target_index)[0][0] + 1

#                 is_top1 = (rank == 1)
#                 is_top3 = (rank <= 3)

#                 if rank == 1:
#                     color = "#6ff261"
#                 elif rank <= 3:
#                     color = "#f5e551"
#                 else:
#                     color = "#f26161"

#                 if is_top1:
#                     top1_count += 1
#                 if is_top3:
#                     top3_count += 1

#                 relevant_similarities.append(target_similarity)

#                 st.markdown(
#                     f"<span style='color:{color};'>Rang: {rank}, Similarity: {target_similarity:.4f}</span>",
#                     unsafe_allow_html=True
#                 )
#             else:
#                 st.write(f"**Keine passende .json-Datei für {normalized_name} gefunden**")

#             st.write("---")

#         if relevant_similarities:
#             avg_similarity = sum(relevant_similarities) / len(relevant_similarities)
#             ratio1_str = f"{top1_count}/{total_tests}"
#             ratio3_str = f"{top3_count}/{total_tests}"
#             st.markdown(
#                 f"<span style='color:yellow;'>**Tests für LLM {llm_name} abgeschlossen**</span>",
#                 unsafe_allow_html=True
#             )
#             st.write(f"Top-1: {ratio1_str}, Top-3: {ratio3_str}")
#             st.write("---")
#             avg_file = os.path.join(llm_output_dir, f"{safe_llm_model}_average_similarity.txt")
#             with open(avg_file, "w", encoding="utf-8") as af:
#                 af.write(f"AverageSim: {avg_similarity:.4f}\nTop1: {ratio1_str}\nTop3: {ratio3_str}\n")
#         else:
#             st.write("Keine relevanten Ähnlichkeiten berechnet.")

#     except Exception as e:
#         st.write(f"Fehler im 2. Durchlauf LLM {llm_name}: {e}")


# def run_tests_for_dimensions():
#     """
#     Testet das erste Embedding-Modell mit dem Ordner in "data/Dimensions".
#     (Intern weiter doc-level Embeddings, man kann aber auch Zeilen-Ansatz 
#     für das Ranking wählen.)
#     Jetzt zusätzlich mit Average Similarity pro Ordner im CSV.
#     """
#     global embeddings, documents, document_embeddings, metadatas
#     try:
#         if "dimensions_results" not in st.session_state:
#             st.session_state.dimensions_results = []
#         else:
#             st.session_state.dimensions_results = []

#         embedding_model_name = embedding_models[0]
#         st.write(f"Starte Dimension-Tests für Embedding-Modell: {embedding_model_name}")

#         dimension_dir = "/beegfs/scratch/workspace/es_dihoit00-RAG/data/Dimensions"
#         testdata_dir = "/beegfs/scratch/workspace/es_dihoit00-RAG/TestCode/TestCode_Dimensions"
#         base_output_dir = os.path.join("/beegfs/scratch/workspace/es_dihoit00-RAG/TestResults/Dimensions")
#         os.makedirs(base_output_dir, exist_ok=True)

#         dimension_folders = [
#             d for d in os.listdir(dimension_dir)
#             if os.path.isdir(os.path.join(dimension_dir, d))
#         ]

#         def extract_number(folder_name):
#             return int(re.sub("[^0-9]", "", folder_name)) if re.search(r"\d+", folder_name) else 0

#         dimension_folders.sort(key=extract_number)

#         for dim_folder in dimension_folders:
#             st.write(f"---\n**Teste Ordner:** {dim_folder}")
#             clear_gpu_memory()

#             if os.path.exists("data.pkl"):
#                 os.remove("data.pkl")

#             embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

#             current_data_dir = os.path.join(dimension_dir, dim_folder, "only_code")
#             st.write(f"[Dimension Info] Embedding-Ordner: {current_data_dir}")
#             docs = load_data_from_directory(current_data_dir)
#             for d in docs:
#                 d.metadata["source_folder"] = current_data_dir

#             documents = docs
#             metadatas = [d.metadata for d in documents]
#             texts = [d.page_content for d in docs]

#             if len(texts) == 0:
#                 st.write(f"Keine Dateien im Ordner {dim_folder}, fahre fort.")
#                 continue

#             doc_emb = embeddings.embed_documents(texts)
#             doc_emb = np.array(doc_emb, dtype=np.float32)
#             doc_emb = normalize(doc_emb, axis=1)
#             document_embeddings = doc_emb

#             with open("data.pkl", "wb") as f:
#                 pickle.dump((documents, document_embeddings, metadatas), f)

#             test_files = [f for f in os.listdir(testdata_dir) if os.path.isfile(os.path.join(testdata_dir, f))]

#             total_tests = len(test_files)
#             top1_count = 0
#             top3_count = 0
#             relevant_sims = []  # <--- NEU: Sammeln für Average Similarity

#             for test_file in test_files:
#                 st.markdown(
#                     f"<span style='color:#7bd1ed;'>**Verarbeite Datei:** </span>"
#                     f"<span style='color:#7bd1ed;'>{test_file}</span>",
#                     unsafe_allow_html=True
#                 )
#                 test_file_path = os.path.join(testdata_dir, test_file)
#                 with open(test_file_path, "r", encoding="utf-8", errors="ignore") as f:
#                     test_code = f.read()

#                 top3_docs, top3_sims, doc_scores, _ = manual_retrieval_with_top3(test_code)

#                 normalized_name = normalize_test_filename(test_file)
#                 target_doc_base = normalized_name + ".json"

#                 target_index = None
#                 for i, meta in enumerate(metadatas):
#                     doc_src = meta.get("source", "")
#                     if doc_src == target_doc_base:
#                         target_index = i
#                         break

#                 if target_index is not None:
#                     target_similarity = doc_scores[target_index]
#                     sorted_indices = np.argsort(doc_scores)[::-1]
#                     rank = np.where(sorted_indices == target_index)[0][0] + 1

#                     if rank == 1:
#                         color = "#6ff261"
#                         top1_count += 1
#                     elif rank <= 3:
#                         color = "#f5e551"
#                         top3_count += 1
#                     else:
#                         color = "#f26161"

#                     # Fürs spätere Averaging merken
#                     relevant_sims.append(target_similarity)

#                     st.markdown(
#                         f"<span style='color:{color};'>**Ziel-Datei Rang**: {rank}, Similarity: {target_similarity:.4f}</span>",
#                         unsafe_allow_html=True
#                     )
#                 else:
#                     st.write(f"**Keine passende .json-Datei für {normalized_name} gefunden**")

#                 st.write("---")

#             top1_ratio = top1_count / total_tests if total_tests else 0
#             top3_ratio = top3_count / total_tests if total_tests else 0
#             if relevant_sims:
#                 avg_similarity = sum(relevant_sims) / len(relevant_sims)
#             else:
#                 avg_similarity = 0.0

#             # In Session-State ablegen (für Gesamtauswertung)
#             st.session_state.dimensions_results.append({
#                 "embedding_model": embedding_model_name,
#                 "folder": dim_folder,
#                 "top1_ratio": top1_ratio,
#                 "top3_ratio": top3_ratio,
#                 "avg_similarity": avg_similarity
#             })

#             df_single = pd.DataFrame([{
#                 "Embedding-Modell": embedding_model_name,
#                 "Ordner": dim_folder,
#                 "Top-1 (Prozent)": f"{top1_ratio*100:.1f}%",
#                 "Top-3 (Prozent)": f"{top3_ratio*100:.1f}%",
#                 "AvgSimilarity": f"{avg_similarity:.4f}"
#             }])
#             st.write("**Ergebnis-Tabelle für diesen Ordner**:")
#             st.write(df_single)

#             single_csv_path = os.path.join(base_output_dir, f"results_{dim_folder}.csv")
#             df_single.to_csv(single_csv_path, index=False)
#             st.write(f"Tabelle für {dim_folder} gespeichert als {single_csv_path}")

#         st.write("---")
#         st.write("**Gesamtergebnisse aller getesteten Ordner:**")
#         if len(st.session_state.dimensions_results) > 0:
#             df_all = pd.DataFrame(st.session_state.dimensions_results)
#             df_all["Top-1 (Prozent)"] = (df_all["top1_ratio"] * 100).round(1)
#             df_all["Top-3 (Prozent)"] = (df_all["top3_ratio"] * 100).round(1)
#             df_all["AvgSimilarity"] = df_all["avg_similarity"].round(4)

#             df_all = df_all[["embedding_model", "folder", "Top-1 (Prozent)", "Top-3 (Prozent)", "AvgSimilarity"]]
#             st.write(df_all)

#             all_csv_path = os.path.join(base_output_dir, "all_dimensions_results.csv")
#             df_all.to_csv(all_csv_path, index=False)
#             st.write(f"Gesamttabelle gespeichert als {all_csv_path}")
#         else:
#             st.write("Keine Ergebnisse in st.session_state.dimensions_results vorhanden.")
#     except Exception as e:
#         print(f"Fehler: {e}")


# def run_tests_for_lines():
#     """
#     Führt für JEDES Embedding-Modell Tests mit Unterordnern in "TestCode/TestCode_lines".
#     Auch hier verwenden wir manual_retrieval_with_top3() (Zeilenbasiert oder doc-level).
#     Jetzt zusätzlich mit Average Similarity im finalen CSV.
#     """
#     global embeddings, documents, document_embeddings, metadatas, selectedFileFormat
#     try:
#         if "lines_results" not in st.session_state:
#             st.session_state.lines_results = []
#         else:
#             st.session_state.lines_results = []

#         base_test_folder = "TestCode/TestCode_lines"
#         if not os.path.exists(base_test_folder):
#             st.write(f"Ordner '{base_test_folder}' existiert nicht – Abbruch.")
#             return

#         line_folders = [
#             d for d in os.listdir(base_test_folder)
#             if os.path.isdir(os.path.join(base_test_folder, d))
#         ]

#         def extract_number(folder_name):
#             return int(re.sub("[^0-9]", "", folder_name)) if re.search(r"\d+", folder_name) else 0

#         line_folders.sort(key=extract_number)

#         for emb_model in embedding_models:
#             st.markdown(
#                 f"<span style='color:white; font-size:20px;'>**Starte Up-to-500-Lines-Test für Embedding:** </span>"
#                 f"<span style='color:#d99a69; font-size:20px;'>{emb_model}</span>",
#                 unsafe_allow_html=True
#             )
#             st.write("---")

#             clear_gpu_memory()
#             if os.path.exists("data.pkl"):
#                 os.remove("data.pkl")

#             selectedFileFormat = None
#             embeddings = HuggingFaceEmbeddings(model_name=emb_model)

#             data_dir = os.path.join("data", "data_v1", "only_code")
#             st.write(f"[Lines Info] Embedding-Ordner: {data_dir}")
#             docs = load_data_from_directory(data_dir)
#             for d in docs:
#                 d.metadata["source_folder"] = data_dir

#             documents = docs
#             metadatas = [d.metadata for d in documents]
#             texts = [d.page_content for d in docs]
#             doc_emb = embeddings.embed_documents(texts)
#             doc_emb = np.array(doc_emb, dtype=np.float32)
#             doc_emb = normalize(doc_emb, axis=1)
#             document_embeddings = doc_emb

#             with open("data.pkl", "wb") as f:
#                 pickle.dump((documents, document_embeddings, metadatas), f)

#             for folder_name in line_folders:
#                 folder_path = os.path.join(base_test_folder, folder_name)
#                 st.write(f"\n**Teste Subfolder:** {folder_path}")

#                 test_files = [
#                     f for f in os.listdir(folder_path)
#                     if os.path.isfile(os.path.join(folder_path, f))
#                 ]

#                 total_tests = len(test_files)
#                 top1_count = 0
#                 top3_count = 0
#                 relevant_sims = []  # <--- NEU: Für Avg Similarity

#                 for test_file in test_files:
#                     st.markdown(
#                         f"<span style='color:#7bd1ed;'>**Verarbeite Datei:** </span>"
#                         f"<span style='color:#7bd1ed;'>{test_file}</span>",
#                         unsafe_allow_html=True
#                     )
#                     file_path = os.path.join(folder_path, test_file)
#                     with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
#                         test_code = f.read()

#                     top3_docs, top3_sims, doc_scores, _ = manual_retrieval_with_top3(test_code)

#                     normalized_name = normalize_test_filename(test_file)
#                     target_doc_base = normalized_name + ".json"

#                     target_index = None
#                     for i, meta in enumerate(metadatas):
#                         doc_src = meta.get("source", "")
#                         if doc_src == target_doc_base:
#                             target_index = i
#                             break

#                     if target_index is not None:
#                         target_similarity = doc_scores[target_index]
#                         relevant_sims.append(target_similarity)  # Sammeln
#                         sorted_indices = np.argsort(doc_scores)[::-1]
#                         rank = np.where(sorted_indices == target_index)[0][0] + 1

#                         is_top1 = (rank == 1)
#                         is_top3 = (rank <= 3)

#                         if rank == 1:
#                             color = "#6ff261"
#                             top1_count += 1
#                         elif rank <= 3:
#                             color = "#f5e551"
#                             top3_count += 1
#                         else:
#                             color = "#f26161"

#                         st.markdown(
#                             f"<span style='color:{color};'>**Ziel-Datei Rang**: {rank}, Similarity: {target_similarity:.4f}</span>",
#                             unsafe_allow_html=True
#                         )
#                     else:
#                         st.write(f"**Keine passende .json-Datei für {normalized_name} gefunden**")

#                     st.write("---")

#                 top1_ratio = top1_count / total_tests if total_tests else 0
#                 top3_ratio = top3_count / total_tests if total_tests else 0

#                 if relevant_sims:
#                     avg_similarity = sum(relevant_sims) / len(relevant_sims)
#                 else:
#                     avg_similarity = 0.0

#                 st.session_state.lines_results.append({
#                     "embedding_model": emb_model,
#                     "lines_folder": folder_name,
#                     "top1_ratio": top1_ratio,
#                     "top3_ratio": top3_ratio,
#                     "avg_similarity": avg_similarity
#                 })

#                 df_single = pd.DataFrame([{
#                     "Embedding-Modell": emb_model,
#                     "Folder": folder_name,
#                     "Top-1 (Prozent)": f"{top1_ratio*100:.1f}%",
#                     "Top-3 (Prozent)": f"{top3_ratio*100:.1f}%",
#                     "AvgSimilarity": f"{avg_similarity:.4f}"
#                 }])
#                 st.write("**Ergebnis-Tabelle für diesen Ordner**:")
#                 st.write(df_single)

#         st.write("---")
#         st.write("**Gesamtergebnisse (Up-to-500-Lines) aller Embedding-Modelle:**")
#         if len(st.session_state.lines_results) > 0:
#             df_all = pd.DataFrame(st.session_state.lines_results)
#             df_all["Top-1 (%)"] = (df_all["top1_ratio"]*100).round(1)
#             df_all["Top-3 (%)"] = (df_all["top3_ratio"]*100).round(1)
#             df_all["AvgSimilarity"] = df_all["avg_similarity"].round(4)

#             df_all = df_all[["embedding_model", "lines_folder", "Top-1 (%)", "Top-3 (%)", "AvgSimilarity"]]
#             st.write(df_all)

#             base_output_dir = os.path.join("TestResults", "EmbeddingModels")
#             lines_csv_path = os.path.join(base_output_dir, "results_lines_100_500.csv")
#             df_all.to_csv(lines_csv_path, index=False)
#             st.write(f"Gesamttabelle (Up to 500 Lines) gespeichert als {lines_csv_path}")
#         else:
#             st.write("Keine Ergebnisse in st.session_state.lines_results vorhanden.")

#     except Exception as e:
#         st.write(f"Fehler in run_tests_for_lines: {e}")



# def run_tests_for_changed_codes():
#     """
#     Führt Tests mit dem Ordner "TestCode/TestCode_Changed" durch.
#     """
#     global documents, document_embeddings, metadatas
#     try:
#         st.session_state.results_overall = []
#     except:
#         pass

#     testdata_dir = os.path.join(os.path.dirname(__file__), "TestCode", "TestCode_Changed")
#     base_output_dir = os.path.join(os.path.dirname(__file__), "TestResults", "EmbeddingModels")
#     llm_output_base = os.path.join(os.path.dirname(__file__), "TestResults", "LLMs")

#     for emb_model in embedding_models:
#         st.markdown(
#             f"<span style='color:white; font-size:20px;'>**Starte Tests (Changed Codes) für Embedding Modell:** </span>"
#             f"<span style='color:#d99a69; font-size:20px;'>{emb_model}</span>",
#             unsafe_allow_html=True
#         )
#         st.write("---")
#         run_tests_for_embedding_model(emb_model, testdata_dir, base_output_dir)

#     st.write("**Alle Tests (Changed Codes) für alle Embedding-Modelle abgeschlossen.**")

#     df = pd.DataFrame(st.session_state.results_overall)
#     if df.empty:
#         st.write("Keine Ergebnisse (Changed Codes) verfügbar.")
#         return

#     df.sort_values(by=["top3_ratio", "top1_ratio", "avg_similarity"],
#                    ascending=[False, False, False],
#                    inplace=True)
#     df["Top-1 (Prozent)"] = (df["top1_ratio"] * 100).round(1).astype(str) + "%"
#     df["Top-3 (Prozent)"] = (df["top3_ratio"] * 100).round(1).astype(str) + "%"

#     st.write("**Zusammenfassung (Changed Codes) aller Embedding-Modelle:**")
#     st.write(df[["embedding_model", "avg_similarity", "Top-1 (Prozent)", "Top-3 (Prozent)"]])

#     results_summary_file = os.path.join(base_output_dir, "results_summary_embedding_models_changed.csv")
#     df.to_csv(results_summary_file, index=False)
#     st.write("Ergebnisse gespeichert (Changed Codes).")

#     st.write("---")
#     st.write("**Kategorien (0_20), (0_40), (0_60), (0_80), (0_100), (0_200), (0_300), (0_400), (0_500)** – Übersicht je Modell:")
#     show_file_stats_tables()

#     best_emb_model = df.iloc[0]["embedding_model"]
#     st.markdown(
#         f"<span style='color:white;'>**Bestes Embedding (nach Top-3-Ratio, Changed Codes):** </span>"
#         f"<span style='color:#d99a69;'>{best_emb_model}</span>",
#         unsafe_allow_html=True
#     )

#     clear_gpu_memory()
#     if os.path.exists("data.pkl"):
#         os.remove("data.pkl")

#     embeddings = HuggingFaceEmbeddings(model_name=best_emb_model)
#     data_dir = os.path.join("data", "data_v1", "only_code")
#     docs = load_data_from_directory(data_dir)
#     for d in docs:
#         d.metadata["source_folder"] = data_dir

#     documents = docs
#     metadatas = [d.metadata for d in documents]
#     texts = [d.page_content for d in docs]
#     doc_emb = embeddings.embed_documents(texts)
#     doc_emb = np.array(doc_emb, dtype=np.float32)
#     doc_emb = normalize(doc_emb, axis=1)
#     document_embeddings = doc_emb

#     with open("data.pkl", "wb") as f:
#         pickle.dump((documents, document_embeddings, metadatas), f)

#     st.markdown(
#         f"<span style='color:yellow;'>**data.pkl** </span>"
#         f"<span style='color:white;'>**wurde neu erstellt mit** </span>"
#         f"<span style='color:#d99a69;'>{best_emb_model}</span>",
#         unsafe_allow_html=True
#     )

#     st.write("---")
#     st.markdown(
#         f"<span style='color:white; font-size:20px;'>**Zweiter Durchlauf LLMs (Changed Codes)** </span>"
#         f"<span style='color:#d99a69; font-size:20px;'>{best_emb_model}</span>",
#         unsafe_allow_html=True
#     )
#     st.write("---")

#     for llm_model in llm_models:
#         st.markdown(
#             f"<span style='color:white; font-size:18px;'>**Starte Tests LLM (Changed Codes):** </span>"
#             f"<span style='color:#d99a69; font-size:18px;'>{llm_model}</span> "
#             f"<span style='color:white;'>(Embedding: {best_emb_model})</span>",
#             unsafe_allow_html=True
#         )
#         st.write("---")
        
#         run_tests_for_llm(best_emb_model, llm_model, testdata_dir, os.path.join(os.path.dirname(__file__), "TestResults", "LLMs"))

#     st.write("**Alle Tests (Changed Codes) abgeschlossen.**\n")
#     st.write("**Getestete LLM-Modelle:**\n")
#     for llm_model in llm_models:
#         st.markdown(
#             f"<span style='color:white; font-size:18px;'>- </span>"
#             f"<span style='color:#d99a69; font-size:18px;'>{llm_model}</span>",
#             unsafe_allow_html=True
#         )

#     st.write(df)


# # ------------------ NEU: Zeilenbasierte Erklärung (Explain) ------------------
# def explain_decision(test_filename: str, chosen_emb: str):
#     """
#     1) clear_gpu_memory
#     2) data.pkl NEU anlegen mit chosen_emb
#     3) manual_retrieval_with_top3() für test_filename
#     4) Erzeugung Heatmaps (TestCode vs. Ziel, TestCode vs. Top1).
#     """
#     print("[DEBUG] explain_decision() aufgerufen.")
#     print(f"[DEBUG] Eingabe => test_filename: {test_filename}, chosen_emb: {chosen_emb}")

#     clear_gpu_memory()
#     print("[DEBUG] Neuberechnung der data.pkl mit Embedding:", chosen_emb)

#     data_dir = os.path.join("data", "data_v1", "only_code")
#     new_docs = load_data_from_directory(data_dir)
#     for d in new_docs:
#         d.metadata["source_folder"] = data_dir

#     global documents, document_embeddings, metadatas, embeddings
#     documents = new_docs
#     metadatas = [d.metadata for d in documents]
#     texts = [d.page_content for d in new_docs]
#     embeddings = HuggingFaceEmbeddings(model_name=chosen_emb)

#     doc_emb = embeddings.embed_documents(texts)
#     doc_emb = np.array(doc_emb, dtype=np.float32)
#     doc_emb = normalize(doc_emb, axis=1)
#     document_embeddings = doc_emb

#     with open("data.pkl", "wb") as f:
#         pickle.dump((documents, document_embeddings, metadatas), f)
#     print("[DEBUG] data.pkl erstellt.")

#     if not os.path.exists(test_filename):
#         st.write(f"Datei {test_filename} wurde nicht gefunden. Bitte vollständigen Pfad angeben.")
#         print("[DEBUG] test_filename nicht gefunden:", test_filename)
#         return

#     with open(test_filename, "r", encoding="utf-8", errors="ignore") as f:
#         test_code = f.read()

#     print("[DEBUG] Starte manual_retrieval_with_top3() für diese Testdatei.")
#     top3_docs, top3_sims, similarities, query_emb = manual_retrieval_with_top3(test_code)

#     plot_file_specific(similarities, test_code, title_prefix=f"Explain Decision - {chosen_emb}: ")

#     normalized_name = normalize_test_filename(os.path.basename(test_filename))
#     target_doc_base = normalized_name + ".json"
#     sims = similarities  # Wichtig: den aggregatorischen Score

#     target_index = None
#     for i, meta in enumerate(metadatas):
#         if meta.get("source", "") == target_doc_base:
#             target_index = i
#             break

#     if target_index is not None:
#         target_similarity = sims[target_index]
#         sorted_indices = np.argsort(sims)[::-1]
#         rank = np.where(sorted_indices == target_index)[0][0] + 1

#         if rank == 1:
#             color = "#6ff261"
#         elif rank <= 3:
#             color = "#f5e551"
#         else:
#             color = "#f26161"

#         st.markdown(
#             f"<span style='color:{color};'>**Ziel-Datei Rang**: {rank}, Similarity: {target_similarity:.4f}</span>",
#             unsafe_allow_html=True
#         )
#         print(f"[DEBUG] => Rang: {rank}, Similarity: {target_similarity:.4f}")
#     else:
#         st.write("Keine passende Zieldatei (Suffix .json) im Embedding-Dataset gefunden.")
#         print("[DEBUG] => Keine passende Datei gefunden.")
#         return

#     if len(top3_docs) == 0:
#         st.write("Keine Dokumente gefunden (Top-3 leer).")
#         print("[DEBUG] => Top3 empty.")
#         return

#     # -----------------------------------------------
#     # Erzeuge Heatmaps: TestCode vs. Ziel, TestCode vs. Top1
#     # -----------------------------------------------

#     # Berechne die Embeddings für den Test-Code
#     test_lines = test_code.splitlines()
#     test_line_embs = []
#     for line in test_lines:
#         emb_list = embeddings.embed_documents([line])
#         emb_arr = np.array(emb_list[0], dtype=np.float32)
#         norm_q = np.linalg.norm(emb_arr)
#         if norm_q != 0:
#             emb_arr = emb_arr / norm_q
#         test_line_embs.append(emb_arr)

#     # Ziel-Dokument auswählen und anzeigen
#     target_doc = documents[target_index]
#     target_doc_content = target_doc.page_content
#     st.markdown(f"**Ziel-Dokument**: {target_doc.metadata.get('source','?')} (Rang={rank}, CosSim={target_similarity:.4f})")

#     # Embeddings für die Zeilen des Ziel-Dokuments berechnen
#     target_doc_lines = target_doc_content.splitlines()
#     target_doc_line_embs = []
#     for line in target_doc_lines:
#         emb_list = embeddings.embed_documents([line])
#         emb_arr = np.array(emb_list[0], dtype=np.float32)
#         norm_val = np.linalg.norm(emb_arr)
#         if norm_val != 0:
#             emb_arr = emb_arr / norm_val
#         target_doc_line_embs.append(emb_arr)

#     # Ähnlichkeitsmatrix (Cosine-Sim) zwischen Test-Code und Ziel-Dokument berechnen
#     sim_matrix_target = np.zeros((len(test_lines), len(target_doc_lines)), dtype=np.float32)
#     for i in range(len(test_lines)):
#         for j in range(len(target_doc_lines)):
#             sim_matrix_target[i, j] = test_line_embs[i].dot(target_doc_line_embs[j])

#     # Verzeichnis für die Erklärplots anlegen
#     os.makedirs("explainfolder", exist_ok=True)
#     explainfolder_path_target = os.path.join("explainfolder", "explain_decision_linesim_target.png")

#     # Heatmap für Test-Code vs. Ziel-Dokument erstellen (Skalierung 0 bis 1)
#     fig2, ax2 = plt.subplots(figsize=(12, 8))
#     sns.heatmap(sim_matrix_target, cmap="coolwarm", annot=False, vmin=0.0, vmax=1.0, ax=ax2)

#     ax2.set_xticks(np.arange(sim_matrix_target.shape[1]) + 0.5)
#     ax2.set_yticks(np.arange(sim_matrix_target.shape[0]) + 0.5)
#     # Nur jede zweite Zeile / Spalte beschriften (Layout)
#     all_x = np.arange(1, sim_matrix_target.shape[1] + 1)
#     all_y = np.arange(1, sim_matrix_target.shape[0] + 1)
#     labels_x = [str(x) if x % 2 == 1 else "" for x in all_x]
#     labels_y = [str(y) if y % 2 == 1 else "" for y in all_y]

#     ax2.set_xticklabels(labels_x)
#     ax2.set_yticklabels(labels_y)

#     ax2.set_title(f"Zeilenbasierte Cosine-Sim\n"
#                 f"(TestCode vs. {target_doc.metadata.get('source','?')})\n"
#                 f"Rang={rank}, CosSim={target_similarity:.4f}, Embedding={chosen_emb}")
#     ax2.set_xlabel("Ziel-Dokument (Zeilen)")
#     ax2.set_ylabel("TestCode (Zeilen)")
#     plt.tight_layout()
#     plt.savefig(explainfolder_path_target)
#     plt.close(fig2)

#     st.markdown("**Heatmap (TestCode vs. Eigentliches Ziel-Dokument)**:")
#     st.image(explainfolder_path_target)
#     print(f"[DEBUG] => Heatmap (Ziel-Dokument) gespeichert nach {explainfolder_path_target}")
#     st.markdown("---")

#     # (B) Heatmap: TestCode vs. Top-1
#     top1_doc = top3_docs[0]
#     st.markdown(f"**Top-1 Dokument** laut Retrieval: {top1_doc.metadata.get('source','?')} "
#                 f"(CosSim={top3_sims[0]:.4f})")

#     top1_content = top1_doc.page_content
#     top1_lines = top1_content.splitlines()

#     doc_line_embs = []
#     for line in top1_lines:
#         emb_list = embeddings.embed_documents([line])
#         emb_arr = np.array(emb_list[0], dtype=np.float32)
#         norm_d = np.linalg.norm(emb_arr)
#         if norm_d != 0:
#             emb_arr = emb_arr / norm_d
#         doc_line_embs.append(emb_arr)

#     sim_matrix_top1 = np.zeros((len(test_lines), len(top1_lines)), dtype=np.float32)
#     for i in range(len(test_lines)):
#         for j in range(len(top1_lines)):
#             sim_matrix_top1[i, j] = test_line_embs[i].dot(doc_line_embs[j])

#     explain_plot_path_top1 = os.path.join("explainfolder", "explain_decision_linesim_top1.png")
#     fig, ax = plt.subplots(figsize=(12, 8))
#     sns.heatmap(sim_matrix_top1, cmap="coolwarm", annot=False, vmin=0.0, vmax=1.0, ax=ax)

#     ax.set_xticks(np.arange(sim_matrix_top1.shape[1]) + 0.5)
#     ax.set_yticks(np.arange(sim_matrix_top1.shape[0]) + 0.5)
#     all_x_top1 = np.arange(1, sim_matrix_top1.shape[1] + 1)
#     all_y_top1 = np.arange(1, sim_matrix_top1.shape[0] + 1)
#     labels_x_top1 = [str(x) if x % 2 == 1 else "" for x in all_x_top1]
#     labels_y_top1 = [str(y) if y % 2 == 1 else "" for y in all_y_top1]

#     ax.set_xticklabels(labels_x_top1)
#     ax.set_yticklabels(labels_y_top1)

#     ax.set_title(f"Zeilenbasierte Cosine-Sim\n"
#                 f"(TestCode vs. [Top1] {top1_doc.metadata.get('source','?')})\n"
#                 f"CosSim={top3_sims[0]:.4f}, Embedding={chosen_emb}")
#     ax.set_xlabel("Top-1 Doc (Zeilen)")
#     ax.set_ylabel("TestCode (Zeilen)")
#     plt.tight_layout()
#     plt.savefig(explain_plot_path_top1)
#     plt.close()

#     st.image(explain_plot_path_top1)
#     st.markdown("**Hinweis**: Rot = hohe Cosine-Sim ~ 1.0, Blau = niedrige Cosine-Sim ~ 0.0")

#     print(f"[DEBUG] => Heatmap gespeichert nach {explain_plot_path_top1}")



# # ------------------ Haupt-Streamlit-Funktion ------------------
# def main():
#     """
#     Streamlit-Einstiegspunkt:
#       - Chat-Eingabebereich
#       - Buttons für Tests
#       - Explain Decision
#     """
#     global documents, document_embeddings, metadatas, embeddings

#     st.title("Forschungsprojekt RAG-LLM")

#     if "messages" not in st.session_state:
#         st.session_state.messages = []
#     if "file_stats" not in st.session_state:
#         st.session_state.file_stats = {}

#     prompt = st.chat_input("Frage hier eingeben:")
#     if prompt:
#         st.session_state.messages.append({"role": "user", "content": prompt})

#         emb_model = embedding_models[0]  # Default: Erstes Embedding
#         clear_gpu_memory()
#         if not os.path.exists("data.pkl"):
#             print("[DEBUG] data.pkl nicht vorhanden, erstelle neu mit:", emb_model)
#             data_dir = os.path.join("data", "data_v1", "only_code")
#             docs = load_data_from_directory(data_dir)
#             for d in docs:
#                 d.metadata["source_folder"] = data_dir

#             documents = docs
#             metadatas = [d.metadata for d in documents]
#             texts = [d.page_content for d in docs]
#             embeddings = HuggingFaceEmbeddings(model_name=emb_model)
#             doc_emb = embeddings.embed_documents(texts)
#             doc_emb = np.array(doc_emb, dtype=np.float32)
#             doc_emb = normalize(doc_emb, axis=1)
#             document_embeddings = doc_emb
#             with open("data.pkl", "wb") as f:
#                 pickle.dump((documents, document_embeddings, metadatas), f)
#         else:
#             print("[DEBUG] data.pkl existiert bereits, lade es.")
#             with open("data.pkl", "rb") as f:
#                 documents, document_embeddings, metadatas = pickle.load(f)

#         llama_response, top_doc, similarities = hybrid_response(prompt, selectedLLM, emb_model)
#         st.session_state.messages.append({"role": "assistant", "content": llama_response})

#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"], unsafe_allow_html=True)

#     st.write("---")

#     # Buttons für Standard-Tests
#     if "results_overall" not in st.session_state:
#         st.session_state.results_overall = []

#     if st.button("Run Tests"):
#         testdata_dir = os.path.join(os.path.dirname(__file__), "TestCode", "TestCode_0_Changes")
#         base_output_dir = os.path.join(os.path.dirname(__file__), "TestResults", "EmbeddingModels")
#         llm_output_base = os.path.join(os.path.dirname(__file__), "TestResults", "LLMs")

#         st.session_state.results_overall = []

#         for emb_model in embedding_models:
#             st.markdown(
#                 f"<span style='color:white; font-size:20px;'>**Starte Tests für Embedding Modell:** </span>"
#                 f"<span style='color:#d99a69; font-size:20px;'>{emb_model}</span>",
#                 unsafe_allow_html=True
#             )
#             st.write("---")
#             run_tests_for_embedding_model(emb_model, testdata_dir, base_output_dir)

#         st.write("**Alle Tests für alle Embedding-Modelle abgeschlossen.**")

#         df = pd.DataFrame(st.session_state.results_overall)
#         if not df.empty:
#             df.sort_values(by=["top3_ratio", "top1_ratio", "avg_similarity"],
#                            ascending=[False, False, False],
#                            inplace=True)
#             df["Top-1 (Prozent)"] = (df["top1_ratio"] * 100).round(1).astype(str) + "%"
#             df["Top-3 (Prozent)"] = (df["top3_ratio"] * 100).round(1).astype(str) + "%"

#             st.write("**Zusammenfassung aller Embedding-Modelle:**")
#             st.write(df[["embedding_model", "avg_similarity", "Top-1 (Prozent)", "Top-3 (Prozent)"]])

#             results_summary_file = os.path.join(base_output_dir, "results_summary_embedding_models.csv")
#             df.to_csv(results_summary_file, index=False)
#             st.write(f"Ergebnisse gespeichert")

#             st.write("---")
#             st.write("**Kategorien (0_20), (0_40), (0_60), (0_80), (0_100)** – Übersichts-Tabelle je Modell:")
#             show_file_stats_tables()

#             best_emb_model = df.iloc[0]["embedding_model"]
#             st.markdown(
#                 f"<span style='color:white;'>**Bestes Embedding Modell (nach Top-3-Ratio):** </span>"
#                 f"<span style='color:#d99a69;'>{best_emb_model}</span>",
#                 unsafe_allow_html=True
#             )

#             clear_gpu_memory()
#             if os.path.exists("data.pkl"):
#                 os.remove("data.pkl")

#             embeddings = HuggingFaceEmbeddings(model_name=best_emb_model)
#             data_dir = os.path.join("data", "data_v1", "only_code")
#             docs = load_data_from_directory(data_dir)
#             for d in docs:
#                 d.metadata["source_folder"] = data_dir

#             documents = docs
#             metadatas = [d.metadata for d in documents]
#             texts = [d.page_content for d in docs]
#             doc_emb = embeddings.embed_documents(texts)
#             doc_emb = np.array(doc_emb, dtype=np.float32)
#             doc_emb = normalize(doc_emb, axis=1)
#             document_embeddings = doc_emb
#             with open("data.pkl", "wb") as f:
#                 pickle.dump((documents, document_embeddings, metadatas), f)

#             st.markdown(
#                 f"<span style='color:yellow;'>**data.pkl** </span>"
#                 f"<span style='color:white;'>**wurde neu erstellt mit** </span>"
#                 f"<span style='color:#d99a69;'>{best_emb_model}</span>",
#                 unsafe_allow_html=True
#             )

#             st.write("---")
#             st.markdown(
#                 f"<span style='color:white; font-size:20px;'>**Zweiter Durchlauf für LLMs** </span>"
#                 f"<span style='color:#d99a69; font-size:20px;'>{best_emb_model}</span>",
#                 unsafe_allow_html=True
#             )
#             st.write("---")

#             for llm_model in llm_models:
#                 st.markdown(
#                     f"<span style='color:white; font-size:18px;'>**Starte Tests für LLM:** </span>"
#                     f"<span style='color:#d99a69; font-size:18px;'>{llm_model}</span> "
#                     f"<span style='color:white;'>(Embedding: {best_emb_model})</span>",
#                     unsafe_allow_html=True
#                 )
#                 st.write("---")
#                 run_tests_for_llm(best_emb_model, llm_model, testdata_dir, llm_output_base)

#             st.write("**Alle Tests abgeschlossen.**\n")
#             st.write("**Getestete LLM-Modelle:**\n")
#             for llm_model in llm_models:
#                 st.markdown(
#                     f"<span style='color:white; font-size:18px;'>- </span>"
#                     f"<span style='color:#d99a69; font-size:18px;'>{llm_model}</span>",
#                     unsafe_allow_html=True
#                 )

#             st.write(df)
#         else:
#             st.write("Keine Ergebnisse verfügbar.")

#     if st.button("(Test Up to 500 Lines)"):
#         run_tests_for_lines()

#     if st.button("Test Changed Codes"):
#         run_tests_for_changed_codes()

#     if st.button("Test Dimensions"):
#         run_tests_for_dimensions()

#     # NEU: Eingabefeld + Button, um Erklärungen zu generieren
#     st.write("---")
#     st.subheader("Explain Decision für eine bestimmte Test-Code-Datei")

#     chosen_embedding = st.selectbox("Wähle ein Embedding-Modell für die Explain-Analyse:", embedding_models)
#     explain_file_input = st.text_input("TestCode-Datei-Pfad eingeben:", "")

#     if st.button("Explain Decision"):
#         if explain_file_input.strip():
#             print("[DEBUG] Explain Decision button geklickt.")
#             explain_decision(explain_file_input.strip(), chosen_embedding)
#         else:
#             st.write("Bitte einen gültigen Dateipfad eingeben.")
#             print("[DEBUG] Kein Dateipfad eingegeben.")


# if __name__ == '__main__':
#     main()


















































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

# NEU: Hier kommt unser Zeilen-Cache dazu.
# Die Idee: Für jedes Dokument in 'documents' halten wir einmal die Zeilen-Embeddings vor.
# So müssen wir pro Test/Abfrage nicht erneut die Dokument-Zeilen einbetten.
document_line_embeddings_cache = []

import torch

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# Zu testende Embedding Modelle.
# embedding_models = [
#     #"voyageai/voyage-code-3",
#     #"Shuu12121/CodeSearch-ModernBERT-Crow-Plus",
#     #"microsoft/codebert-base",
#     #"Salesforce/codet5-small",
#     "intfloat/e5-large-v2",  # <------------------------------Best Model so far
#     #"intfloat/multilingual-e5-large",
#     #"llmrails/ember-v1",
#     #"ibm-granite/granite-embedding-125m-english"
#     # "WhereIsAI/UAE-Large-V1",
#     # "BAAI/bge-large-en-v1.5",
#     # "microsoft/graphcodebert-base"
# ]

embedding_models = [
    "voyageai/voyage-code-3",
    "Shuu12121/CodeSearch-ModernBERT-Crow-Plus",
    "microsoft/codebert-base",
    "Salesforce/codet5-small",
    "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
    "Alibaba-NLP/gte-large-en-v1.5",
    "Alibaba-NLP/gte-multilingual-base",
    "avsolatorio/GIST-Embedding-v0",
    "avsolatorio/GIST-large-Embedding-v0",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "BASF-AI/nomic-embed-text-v1",
    "BASF-AI/nomic-embed-text-v1.5",
    "dunzhang/stella_en_1.5B_v5",
    "dunzhang/stella_en_400M_v5",
    "HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1",
    "HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1.5",
    "HIT-TMG/KaLM-embedding-multilingual-mini-v1",
    "ibm-granite/granite-embedding-125m-english",
    "infgrad/jasper_en_vision_language_v1", 
    "intfloat/e5-large-v2",     # <------------------------------Best Model so far
    "intfloat/multilingual-e5-large",
    "intfloat/multilingual-e5-large-instruct",
    "jinaai/jina-embeddings-v3",
    "jxm/cde-small-v1",
    "Labib11/MUG-B-1.6",
    "llmrails/ember-v1",
    "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-supervised",
    "mixedbread-ai/mxbai-embed-2d-large-v1",
    "mixedbread-ai/mxbai-embed-large-v1",
    "nomic-ai/modernbert-embed-base",
    "nomic-ai/nomic-embed-text-v1",
    "nomic-ai/nomic-embed-text-v1.5",
    "nomic-ai/nomic-embed-text-v1-ablated",
    "nvidia/NV-Embed-v2",
    "PaDaS-Lab/arctic-l-bge-small",
    "pingkeest/learning2_model",
    "sam-babayev/sf_model_e5",
    "sentence-transformers/all-MiniLM-L6-v2",
    "Snowflake/snowflake-arctic-embed-l"
    "microsoft/graphcodebert-base"
    "Snowflake/snowflake-arctic-embed-l-v2.0",
    "Snowflake/snowflake-arctic-embed-m-v1.5",
    "Snowflake/snowflake-arctic-embed-m-v2.0",
    "thenlper/gte-large",
    "tsirif/BinGSE-Meta-Llama-3-8B-Instruct",
    "voyageai/voyage-3-m-exp",
    "voyageai/voyage-lite-02-instruct",
    "w601sxs/b1ade-embed",
    "WhereIsAI/UAE-Large-V1"
]

# Zu testende LLMs.
llm_models = [
    # "deepseek-r1:14b"
]


def clear_gpu_memory():
    """
    GPU-Speicher bereinigen.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("[INFO] GPU-Speicher wurde bereinigt.")
        print("[INFO] ---------.")


# Einmal direkt beim Skript-Start ausführen, um GPU-Speicher aufzuräumen.
clear_gpu_memory()


# ------------------ LLM-Klasse (Ollama) ------------------
class OllamaLLM(LLM):
    """
    Eine angepasste LLM-Klasse zur Anbindung an den Ollama-Server.
    """

    def __init__(self, llm_name):
        super().__init__()
        self._llm_name = llm_name

    def _call(self, prompt: str, stop=None):
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": self._llm_name,
            "prompt": prompt,
            "temperature": 0,
            "top_p": 0.85,
            "top_k": 3
        }
        headers = {"Content-Type": "application/json"}

        print(f"[DEBUG] _call() => Modellname: {self._llm_name}, Prompt length: {len(prompt)} chars")

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
            return f"Fehler bei der Verarbeitung des JSON: {e}"

    def generate(self, prompts, stop=None, callbacks=None, **kwargs):
        generations = []
        for prompt in prompts:
            result = self._call(prompt, stop)
            generations.append([Generation(text=result)])
        return LLMResult(generations=generations)

    @property
    def _llm_type(self):
        return "ollama_llm"

    @property
    def _identifying_params(self):
        return {"name_of_llm": self._llm_name}


# ------------------ Hilfsfunktionen ------------------
def load_data_from_directory(directory):
    global selectedFileFormat
    docs = []
    print(f"[INFO] load_data_from_directory() => Scanne Ordner: {directory}")
    if not os.path.exists(directory):
        print(f"Fehler: Das Verzeichnis '{directory}' existiert nicht.")
        return docs

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        try:
            with open(filepath, 'rb') as raw_file:
                raw_data = raw_file.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']
        except Exception:
            st.write("---")
            continue

        if filename.endswith(".json"):
            try:
                with open(filepath, 'r', encoding=encoding, errors='ignore') as json_file:
                    content = json_file.read()
                parsed_data = json.loads(content)
                docs.append(Document(page_content=json.dumps(parsed_data), metadata={"source": filename}))
                if selectedFileFormat is None:
                    selectedFileFormat = "json"
            except json.JSONDecodeError:
                # Fallback: als TXT interpretieren
                try:
                    with open(filepath, 'r', encoding=encoding, errors='ignore') as fallback_file:
                        fallback_content = fallback_file.read()
                    docs.append(Document(page_content=fallback_content, metadata={"source": filename}))
                    if selectedFileFormat is None:
                        selectedFileFormat = "txt"
                except Exception:
                    continue

        elif filename.endswith(".txt"):
            try:
                with open(filepath, 'r', encoding=encoding, errors='ignore') as text_file:
                    text_content = text_file.read()
                    docs.append(Document(page_content=text_content, metadata={"source": filename}))
                    if selectedFileFormat is None:
                        selectedFileFormat = "txt"
            except Exception:
                st.write("---")
        else:
            pass

    print(f"[INFO] load_data_from_directory() => Fertig, {len(docs)} Dateien geladen.")
    return docs


def normalize_test_filename(test_file: str) -> str:
    filename_noext, _ = os.path.splitext(test_file)
    normalized = re.sub(r"_\(\d+_\d+\)$", "", filename_noext)
    return normalized


def plot_file_specific(similarities, prompt, title_prefix=""):
    """
    Zeichnet einen Scatterplot und einen Barplot für die 'similarities',
    die aus manual_retrieval_with_top3() kommen.
    """
    if len(similarities) > 0 and prompt:
        os.makedirs("GlobalPlots", exist_ok=True)

        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        threshold = 0.9
        outlier_indices = np.where(similarities > threshold)[0]

        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(similarities)), similarities, color='red', edgecolor='black')
        plt.scatter(outlier_indices, similarities[outlier_indices], color='green', edgecolor='black', zorder=5)
        plt.ylim(0, 1)
        plt.title(f"{title_prefix}Cosinus-Ähnlichkeit (einzelne Anfrage)")
        plt.xlabel("Dokument Index")
        plt.ylabel("Cosinus-Ähnlichkeit")
        plt.tight_layout()
        plt.savefig("GlobalPlots/query_to_documents_cosine_similarity_scatter_outliers.png")
        plt.close()

        top_indices = np.argsort(similarities)[::-1][:20]
        top_sims = similarities[top_indices]
        from matplotlib.colors import LinearSegmentedColormap
        colors = [(1, 0, 0), (0, 1, 0)]
        cm = LinearSegmentedColormap.from_list('RedGreenGradient', colors, N=256)
        norm = plt.Normalize(vmin=0.5, vmax=1.0)
        bar_colors = [cm(norm(val)) for val in top_sims]
        top_docs = [documents[i].metadata.get("source", f"Doc_{i}") for i in top_indices]

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
    Zeichnet eine globale Cosine Similarity Matrix & Distanzmatrix
    für die Embeddings, die mit doc-level-Berechnung (document_embeddings)
    entstanden sind. (Reines Logging.)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if len(document_embeddings) > 1:
        sim_matrix_cosine = cosine_similarity(document_embeddings)
        dist_matrix_euclidean = euclidean_distances(document_embeddings)

        plt.figure(figsize=(10, 8))
        sns.heatmap(sim_matrix_cosine, cmap="coolwarm", annot=False, vmin=0, vmax=1)
        plt.title(f"Cosinus-Ähnlichkeitsmatrix\nEmbedding-Modell: {emb_model}")
        plt.xlabel("Dok Index")
        plt.ylabel("Dok Index")
        plt.savefig("GlobalPlots/cosine_similarity_matrix.png")
        plt.close()
        
        euc_max = np.max(dist_matrix_euclidean)
        plt.figure(figsize=(10, 8))
        sns.heatmap(dist_matrix_euclidean, cmap="coolwarm", annot=False, vmin=0, vmax=euc_max)
        plt.title(f"Euklidische Distanzmatrix {title_suffix}")
        plt.xlabel("Dok Index")
        plt.ylabel("Dok Index")
        plt.savefig("GlobalPlots/euclidean_distance_matrix.png")
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.hist(sim_matrix_cosine.flatten(), bins=50, color='skyblue', edgecolor='black',
                 range=(0, 1), density=True)
        plt.title(f"Verteilung der Cosine-Sim-Werte {title_suffix}")
        plt.xlabel("Ähnlichkeitswert")
        plt.ylabel("Prozentuale Häufigkeit")
        plt.xlim(0, 1)
        plt.savefig("GlobalPlots/cosine_similarity_histogram.png")
        plt.close()


def hybrid_response(prompt, llm_name, emb_model=""):
    """
    Wird für Live-Prompts genutzt – holt sich Top3 aus 'manual_retrieval_with_top3()'.
    """
    print(f"[DEBUG] hybrid_response() => LLM: {llm_name}, PromptLen={len(prompt)}")
    top3_docs, top3_sims, similarities, _ = manual_retrieval_with_top3(prompt)

    doc_content_top1 = top3_docs[0].page_content if len(top3_docs) > 0 else ""
    cve_id1 = "N/A"
    try:
        j1 = json.loads(doc_content_top1)
        cve_id1 = j1.get("cve_id", "N/A")
    except:
        pass

    llama_prompt = f"""You are a security analyst specializing in identifying code vulnerabilities (CVEs).

User question: '{prompt}'

Retrieved context with CVE ID {cve_id1}:
{doc_content_top1}

Begin your response with “Vulnerability Found” if a relevant CVE is identified or “Secure” if not.
"""

    current_llm = OllamaLLM(llm_name)
    try:
        llama_response = current_llm._call(llama_prompt)
    except Exception as e:
        llama_response = f"Fehler LLM: {e}"

    if not llama_response.strip():
        llama_response = "Keine Antwort vom LLM."

    plot_file_specific(similarities, prompt, title_prefix=f"LLM {llm_name}: ")

    return llama_response, (top3_docs[0] if len(top3_docs) > 0 else None), similarities


def get_file_stats():
    if "file_stats" not in st.session_state:
        st.session_state.file_stats = {}
    return st.session_state.file_stats


def update_file_stats(embedding_model, filename, is_top1, is_top3, similarity=0.0):
    """
    Erweitert um 'similarity', damit wir später einen 
    Durchschnitt pro Kategorie ausgeben können.
    """
    file_stats_global = get_file_stats()
    if embedding_model not in file_stats_global:
        file_stats_global[embedding_model] = {}
    if filename not in file_stats_global[embedding_model]:
        file_stats_global[embedding_model][filename] = {
            "top1_count": 0,
            "top3_count": 0,
            "tests_done": 0,
            "sum_sims": 0.0
        }
    file_stats_global[embedding_model][filename]["tests_done"] += 1
    if is_top1:
        file_stats_global[embedding_model][filename]["top1_count"] += 1
    if is_top3:
        file_stats_global[embedding_model][filename]["top3_count"] += 1

    file_stats_global[embedding_model][filename]["sum_sims"] += similarity


def show_file_stats_tables():
    if "file_stats" not in st.session_state:
        st.write("Keine Datei-Statistiken vorhanden.")
        return

    categories = ["_(0_20)", "_(0_40)", "_(1_40)", "_(2_40)", "_(0_60)", "_(0_80)", "_(0_100)", "_(0_200)", "_(0_300)", "_(0_400)", "_(0_500)"]
    suffix_regex = re.compile(r"_\(\d+_\d+\)$")

    file_stats_global = st.session_state.file_stats

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
        from collections import defaultdict
        cat_stats = defaultdict(lambda: [0, 0, 0, 0.0])

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
                cat_stats[bracket][3] += stt.get("sum_sims", 0.0)

        row_list = []
        for cat in categories:
            sum_t1, sum_t3, sum_tests, sum_sims = cat_stats[cat]
            if sum_tests == 0:
                continue

            top1_pct = sum_t1 / sum_tests * 100
            top3_pct = sum_t3 / sum_tests * 100
            avg_sim = sum_sims / sum_tests

            row_list.append({
                "Kategorie": cat,
                "Top-1": f"{top1_pct:.1f}%",
                "Top-3": f"{top3_pct:.1f}%",
                "AvgSim": f"{avg_sim:.4f}"
            })

        if not row_list:
            st.write(f"(Keine Dateien in den bekannten Kategorien für {emb_model})")
            return

        df = pd.DataFrame(row_list, columns=["Kategorie", "Top-1", "Top-3", "AvgSim"])
        st.write(f"**Modell: {emb_model}**")
        st.write(df)
        st.write("-----")

    for emb_model, files_map in file_stats_global.items():
        produce_category_table(emb_model, files_map)


# ------------------ WICHTIG: Funktion zum Vorbereiten der Zeilen-Embeddings ------------------
def prepare_line_embeddings():
    """
    Diese Funktion initialisiert den Cache `document_line_embeddings_cache` neu
    und speichert für jedes Dokument die Zeilen-Embeddings genau einmal.
    So sparen wir uns das ständige Neuberechnen bei jeder Anfrage.
    """
    global document_line_embeddings_cache, documents, embeddings

    # Erst Cache leeren
    document_line_embeddings_cache = [None]*len(documents)

    # Jetzt jedes Dokument einmal zeilenweise einbetten
    for i, doc in enumerate(documents):
        lines = doc.page_content.splitlines()
        if len(lines) == 0:
            document_line_embeddings_cache[i] = np.zeros((0, 0), dtype=np.float32)
            continue

        line_embs = embeddings.embed_documents(lines)
        line_embs = normalize(line_embs, axis=1)
        document_line_embeddings_cache[i] = line_embs



# ------------------ Top-K Retrieval-Funktion ------------------
def manual_retrieval_with_top3(prompt):
    # """
    # Verschiedene Ansätze auf Dokument-Ebene oder Zeilen-Ebene.
    # In dieser Version ist der Modus #5 (Zeilenbasiert mit "Top-K-Mean") entkommentiert.
    # Dazu haben wir ein globales Caching für Dokumentzeilen implementiert.
    # """

    # print("[DEBUG] manual_retrieval_with_top3() => Starte Einbettung des Prompt...")

    # # Falls du mal doc-level statt Zeilen-Level probieren willst, 
    # # kommentiere bitte den Code #5 aus und nutze die anderen.
    # #
    # # Aktuell aktiv: "ANSATZ #5: Zeilenbasiert (Top-K-Mean)"
    # # Dazu haben wir den globalen Cache 'document_line_embeddings_cache'.

    # # 1) Promptzeilen einbetten
    # prompt_lines = prompt.splitlines()
    # if not prompt_lines:
    #     print("   -> Prompt ist leer, gebe 0-Scores zurück.")
    #     doc_scores = np.zeros(len(documents), dtype=np.float32)
    #     sorted_indices = np.argsort(doc_scores)[::-1]
    #     top3_indices = sorted_indices[:3]
    #     top3_docs = [documents[idx] for idx in top3_indices]
    #     top3_sims = [doc_scores[idx] for idx in top3_indices]
    #     return top3_docs, top3_sims, doc_scores, None

    # prompt_line_embs = embeddings.embed_documents(prompt_lines)
    # prompt_line_embs = normalize(prompt_line_embs, axis=1)

    # # WICHTIG: Einmal den Cache vorbereiten, falls noch nicht geschehen
    # # (z.B. nach dem Laden neuer Dokumente oder nach dem Wechseln in ein anderes Verzeichnis)
    # if not document_line_embeddings_cache or len(document_line_embeddings_cache) != len(documents):
    #     prepare_line_embeddings()

    # doc_scores = []

    # for i, doc in enumerate(documents):
    #     lines = doc.page_content.splitlines()
    #     print(f"\n[DEBUG] Dokument-Index {i} => Quelle: {doc.metadata.get('source','?')}, Zeilen: {len(lines)}")

    #     if not lines:
    #         doc_scores.append(0.0)
    #         print("   -> Dokument ist leer, Score=0.0")
    #         continue

    #     # Hier holen wir uns die Zeilen-Embeddings aus dem Cache!
    #     doc_line_embs = document_line_embeddings_cache[i]
    #     # doc_line_embs.shape = (Anzahl Zeilen, Embedding-Dimension)

    #     # 3) Für jede Doc-Zeile: max Similarity zu einer Prompt-Zeile
    #     line_max_sims = []
    #     for j, doc_line_emb in enumerate(doc_line_embs):
    #         sims_to_prompt = doc_line_emb @ prompt_line_embs.T
    #         best_line_sim = np.max(sims_to_prompt)
    #         line_max_sims.append(best_line_sim)

    #     # 4) Mittelwert aller Zeilen-Maxima = doc_score
    #     doc_score = np.mean(line_max_sims)
    #     doc_scores.append(doc_score)

    #     # Debug-Ausgabe
    #     print(f"   -> line_max_sims[:5] (Beispiele): {line_max_sims[:7]} ...")
    #     print(f"   -> doc_score (Avg of line-wise maxima): {doc_score:.4f}")

    # doc_scores = np.array(doc_scores)
    # # Ranking
    # sorted_indices = np.argsort(doc_scores)[::-1]
    # top3_indices = sorted_indices[:3]
    # top3_docs = [documents[idx] for idx in top3_indices]
    # top3_sims = [doc_scores[idx] for idx in top3_indices]

    # return top3_docs, top3_sims, doc_scores, None

#     # =========================================================================
#     # ========== ANSATZ #1: Doc-Level Single-Vektor pro Dokument  =============
#     # =========================================================================
#     # => Dies ist der Standard (doc_embeddings)
#     #
    query_embedding = embeddings.embed_query(prompt)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    similarities = document_embeddings @ query_embedding
    sorted_indices = np.argsort(similarities)[::-1]
    top3_indices = sorted_indices[:3]
    top3_docs = [documents[i] for i in top3_indices]
    top3_sims = [similarities[i] for i in top3_indices]
    
    return top3_docs, top3_sims, similarities, query_embedding


# ------------------ TEST-Funktionen (Embedding+LLM) ------------------
def run_tests_for_embedding_model(embedding_model_name, testdata_dir, base_output_dir):
    """
    Führt automatische Tests für ein bestimmtes Embedding-Modell durch.
    Nutzt 'manual_retrieval_with_top3()' (Zeilenbasiert).
    """
    global embeddings, documents, document_embeddings, metadatas, selectedFileFormat
    try:
        clear_gpu_memory()
        if os.path.exists("data.pkl"):
            os.remove("data.pkl")

        selectedFileFormat = None

        data_dir = os.path.join("data", "data_v1", "only_code")
        print(f"[INFO] Lade Dokumente aus => {data_dir}")
        docs = load_data_from_directory(data_dir)

        for d in docs:
            d.metadata["source_folder"] = data_dir

        documents = docs
        metadatas = [d.metadata for d in documents]
        texts = [d.page_content for d in docs]

        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

        # Doc-Level-Embeddings (nur zum Plotten):
        doc_emb = embeddings.embed_documents(texts)
        doc_emb = np.array(doc_emb, dtype=np.float32)
        doc_emb = normalize(doc_emb, axis=1)
        global document_embeddings
        document_embeddings = doc_emb

        # data.pkl zur Sicherheit abspeichern
        with open("data.pkl", "wb") as f:
            pickle.dump((documents, document_embeddings, metadatas), f)

        # Wichtig: den Zeilen-Cache einmal vorbereiten
        prepare_line_embeddings()

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
                f"<span style='color:#7bd1ed;'>**Verarbeite Testdatei:** </span>"
                f"<span style='color:#7bd1ed;'>{test_file}</span>",
                unsafe_allow_html=True
            )

            top3_docs, top3_sims, doc_scores, _ = manual_retrieval_with_top3(test_code)
            plot_file_specific(doc_scores, test_code, title_prefix=f"Emb {embedding_model_name}: ")

            output_subdir = os.path.join(model_output_dir, test_file)
            os.makedirs(output_subdir, exist_ok=True)

            plots_dir = "GlobalPlots"
            for p in ["top_20_similar_documents_gradient.png", "query_to_documents_cosine_similarity_scatter_outliers.png"]:
                src = os.path.join(plots_dir, p)
                if os.path.exists(src):
                    dst = os.path.join(output_subdir, p)
                    shutil.copyfile(src, dst)

            normalized_name = normalize_test_filename(test_file)
            target_doc_base = normalized_name + ".json"

            target_index = None
            for i, meta in enumerate(metadatas):
                doc_src = meta.get("source", "")
                if doc_src == target_doc_base:
                    target_index = i
                    break

            if target_index is not None:
                target_similarity = doc_scores[target_index]
                sorted_indices = np.argsort(doc_scores)[::-1]
                rank = np.where(sorted_indices == target_index)[0][0] + 1

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

                relevant_similarities.append(target_similarity)

                st.markdown(
                    f"<span style='color:{color};'>**Ziel-Datei Rang**: {rank}, Similarity: {target_similarity:.4f}</span>",
                    unsafe_allow_html=True
                )

                sim_path = os.path.join(output_subdir, "similarity.txt")
                with open(sim_path, "w", encoding="utf-8") as sf:
                    sf.write(f"Rank: {rank}\nSimilarity: {target_similarity:.4f}\n")

                update_file_stats(embedding_model_name, test_file, is_top1, is_top3, target_similarity)
            else:
                st.write(f"**Keine passende .json-Datei für {normalized_name} gefunden**")

            st.write("---")

        # Nur zum Logging: Plot der doc-level Embeddings
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
                f"<span style='color:yellow;'>**Trefferquote (Top-3)**: {top3_str}</span>",
                unsafe_allow_html=True
            )
            st.write("---")

            avg_file = os.path.join(model_output_dir, "average_similarity.txt")
            with open(avg_file, "w", encoding="utf-8") as af:
                af.write(f"AvgSimilarity: {avg_similarity:.4f}\nTop-1: {top1_str}\nTop-3: {top3_str}\n")

            if "results_overall" not in st.session_state:
                st.session_state.results_overall = []
            st.session_state.results_overall.append({
                "embedding_model": embedding_model_name,
                "avg_similarity": avg_similarity,
                "top1_ratio": top1_count / total_tests if total_tests else 0.0,
                "top3_ratio": top3_count / total_tests if total_tests else 0.0
            })
        else:
            st.write("Keine relevanten Similarities berechnet.")

    except Exception as e:
        st.write(f"Fehler in run_tests_for_embedding_model mit Modell {embedding_model_name}: {e}")


def run_tests_for_llm(best_embedding_model, llm_name, testdata_dir, llm_output_base):
    """
    LLM-Test mit bereits erstelltem Embedding (data.pkl).
    Nutzt 'manual_retrieval_with_top3()' (Zeilenbasiert) und den globalen Cache.
    """
    global embeddings, documents, document_embeddings, metadatas
    try:
        clear_gpu_memory()

        if not os.path.exists("data.pkl"):
            st.write("data.pkl nicht gefunden – Abbruch.")
            return

        with open("data.pkl", "rb") as f:
            documents, document_embeddings, metadatas = pickle.load(f)

        embeddings = HuggingFaceEmbeddings(model_name=best_embedding_model)

        # Wichtig: Zeilen-Cache bauen
        prepare_line_embeddings()

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

            # Erneut manual_retrieval_with_top3()
            try:
                llama_response, top_doc, doc_scores = hybrid_response(test_code, llm_name, best_embedding_model)
            except Exception as e:
                st.markdown(f"<span style='color:red;'>Fehler beim LLM-Aufruf: {e}</span>", unsafe_allow_html=True)
                llama_response = f"LLM-Fehler: {e}"

            st.write(llama_response)

            output_subdir = os.path.join(llm_output_dir, test_file)
            os.makedirs(output_subdir, exist_ok=True)

            response_file = os.path.join(output_subdir, f"{safe_llm_model}_response.txt")
            with open(response_file, "w", encoding="utf-8") as rf:
                rf.write(llama_response)

            plot_file_specific(doc_scores, test_code, title_prefix=f"LLM {llm_name}: ")
            for p in ["top_20_similar_documents_gradient.png", "query_to_documents_cosine_similarity_scatter_outliers.png"]:
                src = os.path.join("GlobalPlots", p)
                if os.path.exists(src):
                    dst = os.path.join(output_subdir, p)
                    shutil.copyfile(src, dst)

            normalized_name = normalize_test_filename(test_file)
            target_doc_base = normalized_name + ".json"

            target_index = None
            for i, meta in enumerate(metadatas):
                doc_src = meta.get("source", "")
                if doc_src == target_doc_base:
                    target_index = i
                    break

            if target_index is not None:
                target_similarity = doc_scores[target_index]
                sorted_indices = np.argsort(doc_scores)[::-1]
                rank = np.where(sorted_indices == target_index)[0][0] + 1

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

                relevant_similarities.append(target_similarity)

                st.markdown(
                    f"<span style='color:{color};'>Rang: {rank}, Similarity: {target_similarity:.4f}</span>",
                    unsafe_allow_html=True
                )
            else:
                st.write(f"**Keine passende .json-Datei für {normalized_name} gefunden**")

            st.write("---")

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


def run_tests_for_dimensions():
    """
    Testet das erste Embedding-Modell mit dem Ordner in "data/Dimensions".
    Jetzt so verändert, dass wir nur einmal pro Ordner die Dokumente
    und deren Zeilen-Embeddings berechnen.
    """
    global embeddings, documents, document_embeddings, metadatas
    try:
        if "dimensions_results" not in st.session_state:
            st.session_state.dimensions_results = []
        else:
            st.session_state.dimensions_results = []

        embedding_model_name = embedding_models[0]
        st.write(f"Starte Dimension-Tests für Embedding-Modell: {embedding_model_name}")

        dimension_dir = "/beegfs/scratch/workspace/es_dihoit00-RAG/data/Dimensions"
        testdata_dir = "/beegfs/scratch/workspace/es_dihoit00-RAG/TestCode/TestCode_Dimensions"
        base_output_dir = os.path.join("/beegfs/scratch/workspace/es_dihoit00-RAG/TestResults/Dimensions")
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

            embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

            current_data_dir = os.path.join(dimension_dir, dim_folder, "only_code")
            st.write(f"[Dimension Info] Embedding-Ordner: {current_data_dir}")
            docs = load_data_from_directory(current_data_dir)
            for d in docs:
                d.metadata["source_folder"] = current_data_dir

            documents = docs
            metadatas = [d.metadata for d in documents]
            texts = [d.page_content for d in docs]

            if len(texts) == 0:
                st.write(f"Keine Dateien im Ordner {dim_folder}, fahre fort.")
                continue

            doc_emb = embeddings.embed_documents(texts)
            doc_emb = np.array(doc_emb, dtype=np.float32)
            doc_emb = normalize(doc_emb, axis=1)
            document_embeddings = doc_emb

            with open("data.pkl", "wb") as f:
                pickle.dump((documents, document_embeddings, metadatas), f)

            # Jetzt Zeilen-Cache aufbauen
            prepare_line_embeddings()

            test_files = [f for f in os.listdir(testdata_dir) if os.path.isfile(os.path.join(testdata_dir, f))]

            total_tests = len(test_files)
            top1_count = 0
            top3_count = 0
            relevant_sims = []

            for test_file in test_files:
                st.markdown(
                    f"<span style='color:#7bd1ed;'>**Verarbeite Datei:** </span>"
                    f"<span style='color:#7bd1ed;'>{test_file}</span>",
                    unsafe_allow_html=True
                )
                test_file_path = os.path.join(testdata_dir, test_file)
                with open(test_file_path, "r", encoding="utf-8", errors="ignore") as f:
                    test_code = f.read()

                top3_docs, top3_sims, doc_scores, _ = manual_retrieval_with_top3(test_code)

                normalized_name = normalize_test_filename(test_file)
                target_doc_base = normalized_name + ".json"

                target_index = None
                for i, meta in enumerate(metadatas):
                    doc_src = meta.get("source", "")
                    if doc_src == target_doc_base:
                        target_index = i
                        break

                if target_index is not None:
                    target_similarity = doc_scores[target_index]
                    sorted_indices = np.argsort(doc_scores)[::-1]
                    rank = np.where(sorted_indices == target_index)[0][0] + 1

                    if rank == 1:
                        color = "#6ff261"
                        top1_count += 1
                    elif rank <= 3:
                        color = "#f5e551"
                        top3_count += 1
                    else:
                        color = "#f26161"

                    relevant_sims.append(target_similarity)

                    st.markdown(
                        f"<span style='color:{color};'>**Ziel-Datei Rang**: {rank}, Similarity: {target_similarity:.4f}</span>",
                        unsafe_allow_html=True
                    )
                else:
                    st.write(f"**Keine passende .json-Datei für {normalized_name} gefunden**")

                st.write("---")

            top1_ratio = top1_count / total_tests if total_tests else 0
            top3_ratio = top3_count / total_tests if total_tests else 0
            top1plus3_ratio = (top1_count + top3_count) / total_tests if total_tests else 0

            if relevant_sims:
                avg_similarity = sum(relevant_sims) / len(relevant_sims)
            else:
                avg_similarity = 0.0

            st.session_state.dimensions_results.append({
                "embedding_model": embedding_model_name,
                "folder": dim_folder,
                "top1_ratio": top1_ratio,
                "top3_ratio": top3_ratio,
                "top1plus3_ratio": top1plus3_ratio,
                "avg_similarity": avg_similarity
            })

            df_single = pd.DataFrame([{
                "Embedding-Modell": embedding_model_name,
                "Ordner": dim_folder,
                "Top-1 (Prozent)": f"{top1_ratio*100:.1f}%",
                "Top-3 (Prozent)": f"{top3_ratio*100:.1f}%",
                "Top-1+Top-3 (Prozent)": f"{top1plus3_ratio*100:.1f}%",
                "AvgSimilarity": f"{avg_similarity:.4f}"
            }])
            st.write("**Ergebnis-Tabelle für diesen Ordner**:")
            st.write(df_single)

            single_csv_path = os.path.join(base_output_dir, f"results_{dim_folder}.csv")
            df_single.to_csv(single_csv_path, index=False)
            st.write(f"Tabelle für {dim_folder} gespeichert als {single_csv_path}")

        st.write("---")
        st.write("**Gesamtergebnisse aller getesteten Ordner:**")
        if len(st.session_state.dimensions_results) > 0:
            df_all = pd.DataFrame(st.session_state.dimensions_results)
            df_all["Top-1 (Prozent)"] = (df_all["top1_ratio"] * 100).round(1)
            df_all["Top-3 (Prozent)"] = (df_all["top3_ratio"] * 100).round(1)
            df_all["Top-1+Top-3 (Prozent)"] = (df_all["top1plus3_ratio"] * 100).round(1)
            df_all["AvgSimilarity"] = df_all["avg_similarity"].round(4)

            df_all = df_all[["embedding_model", "folder", "Top-1 (Prozent)", "Top-3 (Prozent)", "Top-1+Top-3 (Prozent)", "AvgSimilarity"]]
            st.write(df_all)

            all_csv_path = os.path.join(base_output_dir, "all_dimensions_results.csv")
            df_all.to_csv(all_csv_path, index=False)
            st.write(f"Gesamttabelle gespeichert als {all_csv_path}")
        else:
            st.write("Keine Ergebnisse in st.session_state.dimensions_results vorhanden.")
    except Exception as e:
        print(f"Fehler: {e}")


def run_tests_for_lines():
    """
    Führt für JEDES Embedding-Modell Tests mit Unterordnern in "TestCode/TestCode_lines".
    Ebenfalls zeilenbasiertes Retrieval + Cache.
    """
    global embeddings, documents, document_embeddings, metadatas, selectedFileFormat
    try:
        if "lines_results" not in st.session_state:
            st.session_state.lines_results = []
        else:
            st.session_state.lines_results = []

        base_test_folder = "TestCode/TestCode_lines"
        if not os.path.exists(base_test_folder):
            st.write(f"Ordner '{base_test_folder}' existiert nicht – Abbruch.")
            return

        line_folders = [
            d for d in os.listdir(base_test_folder)
            if os.path.isdir(os.path.join(base_test_folder, d))
        ]

        def extract_number(folder_name):
            return int(re.sub("[^0-9]", "", folder_name)) if re.search(r"\d+", folder_name) else 0

        line_folders.sort(key=extract_number)

        for emb_model in embedding_models:
            st.markdown(
                f"<span style='color:white; font-size:20px;'>**Starte Up-to-500-Lines-Test für Embedding:** </span>"
                f"<span style='color:#d99a69; font-size:20px;'>{emb_model}</span>",
                unsafe_allow_html=True
            )
            st.write("---")

            clear_gpu_memory()
            if os.path.exists("data.pkl"):
                os.remove("data.pkl")

            selectedFileFormat = None
            embeddings = HuggingFaceEmbeddings(model_name=emb_model)

            data_dir = os.path.join("data", "data_v1", "only_code")
            st.write(f"[Lines Info] Embedding-Ordner: {data_dir}")
            docs = load_data_from_directory(data_dir)
            for d in docs:
                d.metadata["source_folder"] = data_dir

            documents = docs
            metadatas = [d.metadata for d in documents]
            texts = [d.page_content for d in docs]
            doc_emb = embeddings.embed_documents(texts)
            doc_emb = np.array(doc_emb, dtype=np.float32)
            doc_emb = normalize(doc_emb, axis=1)
            document_embeddings = doc_emb

            with open("data.pkl", "wb") as f:
                pickle.dump((documents, document_embeddings, metadatas), f)

            # Wichtig: Zeilen-Embeddings einmal vorbereiten
            prepare_line_embeddings()

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
                relevant_sims = []

                for test_file in test_files:
                    st.markdown(
                        f"<span style='color:#7bd1ed;'>**Verarbeite Datei:** </span>"
                        f"<span style='color:#7bd1ed;'>{test_file}</span>",
                        unsafe_allow_html=True
                    )
                    file_path = os.path.join(folder_path, test_file)
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        test_code = f.read()

                    top3_docs, top3_sims, doc_scores, _ = manual_retrieval_with_top3(test_code)

                    normalized_name = normalize_test_filename(test_file)
                    target_doc_base = normalized_name + ".json"

                    target_index = None
                    for i, meta in enumerate(metadatas):
                        doc_src = meta.get("source", "")
                        if doc_src == target_doc_base:
                            target_index = i
                            break

                    if target_index is not None:
                        target_similarity = doc_scores[target_index]
                        relevant_sims.append(target_similarity)
                        sorted_indices = np.argsort(doc_scores)[::-1]
                        rank = np.where(sorted_indices == target_index)[0][0] + 1

                        is_top1 = (rank == 1)
                        is_top3 = (rank <= 3)

                        if rank == 1:
                            color = "#6ff261"
                            top1_count += 1
                        elif rank <= 3:
                            color = "#f5e551"
                            top3_count += 1
                        else:
                            color = "#f26161"

                        st.markdown(
                            f"<span style='color:{color};'>**Ziel-Datei Rang**: {rank}, Similarity: {target_similarity:.4f}</span>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.write(f"**Keine passende .json-Datei für {normalized_name} gefunden**")

                    st.write("---")

                top1_ratio = top1_count / total_tests if total_tests else 0
                top3_ratio = top3_count / total_tests if total_tests else 0
                top1plus3_ratio = (top1_count + top3_count) / total_tests if total_tests else 0

                if relevant_sims:
                    avg_similarity = sum(relevant_sims) / len(relevant_sims)
                else:
                    avg_similarity = 0.0

                st.session_state.lines_results.append({
                    "embedding_model": emb_model,
                    "lines_folder": folder_name,
                    "top1_ratio": top1_ratio,
                    "top3_ratio": top3_ratio,
                    "avg_similarity": avg_similarity
                })

                df_single = pd.DataFrame([{
                    "Embedding-Modell": emb_model,
                    "Folder": folder_name,
                    "Top-1 (Prozent)": f"{top1_ratio*100:.1f}%",
                    "Top-3 (Prozent)": f"{top3_ratio*100:.1f}%",
                    "AvgSimilarity": f"{avg_similarity:.4f}"
                }])
                st.write("**Ergebnis-Tabelle für diesen Ordner**:")
                st.write(df_single)

        st.write("---")
        st.write("**Gesamtergebnisse (Up-to-500-Lines) aller Embedding-Modelle:**")
        if len(st.session_state.lines_results) > 0:
            df_all = pd.DataFrame(st.session_state.lines_results)
            df_all["Top-1 (%)"] = (df_all["top1_ratio"]*100).round(1)
            df_all["Top-3 (%)"] = (df_all["top3_ratio"]*100).round(1)
            df_all["AvgSimilarity"] = df_all["avg_similarity"].round(4)

            df_all = df_all[["embedding_model", "lines_folder", "Top-1 (%)", "Top-3 (%)", "AvgSimilarity"]]
            st.write(df_all)

            base_output_dir = os.path.join("TestResults", "EmbeddingModels")
            lines_csv_path = os.path.join(base_output_dir, "results_lines_100_500.csv")
            df_all.to_csv(lines_csv_path, index=False)
            st.write(f"Gesamttabelle (Up to 500 Lines) gespeichert als {lines_csv_path}")
        else:
            st.write("Keine Ergebnisse in st.session_state.lines_results vorhanden.")

    except Exception as e:
        st.write(f"Fehler in run_tests_for_lines: {e}")


def run_tests_for_changed_codes():
    """
    Führt Tests mit dem Ordner "TestCode/TestCode_Changed" durch.
    """
    global documents, document_embeddings, metadatas
    try:
        st.session_state.results_overall = []
    except:
        pass

    testdata_dir = os.path.join(os.path.dirname(__file__), "TestCode", "TestCode_Changed")
    base_output_dir = os.path.join(os.path.dirname(__file__), "TestResults", "EmbeddingModels")
    llm_output_base = os.path.join(os.path.dirname(__file__), "TestResults", "LLMs")

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

    st.write("---")
    st.write("**Kategorien (0_20), (0_40), (0_60), (0_80), (0_100), (0_200), (0_300), (0_400), (0_500)** – Übersicht je Modell:")
    show_file_stats_tables()

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
    data_dir = os.path.join("data", "data_v1", "only_code")
    docs = load_data_from_directory(data_dir)
    for d in docs:
        d.metadata["source_folder"] = data_dir

    documents = docs
    metadatas = [d.metadata for d in documents]
    texts = [d.page_content for d in docs]
    doc_emb = embeddings.embed_documents(texts)
    doc_emb = np.array(doc_emb, dtype=np.float32)
    doc_emb = normalize(doc_emb, axis=1)
    document_embeddings = doc_emb

    with open("data.pkl", "wb") as f:
        pickle.dump((documents, document_embeddings, metadatas), f)

    # Zeilen-Cache neu
    prepare_line_embeddings()

    st.markdown(
        f"<span style='color:yellow;'>**data.pkl** </span>"
        f"<span style='color:white;'>**wurde neu erstellt mit** </span>"
        f"<span style='color:#d99a69;'>{best_emb_model}</span>",
        unsafe_allow_html=True
    )

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
        
        run_tests_for_llm(best_emb_model, llm_model, testdata_dir, os.path.join(os.path.dirname(__file__), "TestResults", "LLMs"))

    st.write("**Alle Tests (Changed Codes) abgeschlossen.**\n")
    st.write("**Getestete LLM-Modelle:**\n")
    for llm_model in llm_models:
        st.markdown(
            f"<span style='color:white; font-size:18px;'>- </span>"
            f"<span style='color:#d99a69; font-size:18px;'>{llm_model}</span>",
            unsafe_allow_html=True
        )

    st.write(df)


def explain_decision(test_filename: str, chosen_emb: str):
    """
    Demonstrationsfunktion: einmal die data.pkl neu mit chosen_emb aufbauen,
    Top-3 Ranking ausgeben und Heatmaps TestCode vs. Zieldokument + Top1-Dokument.
    """
    print("[DEBUG] explain_decision() aufgerufen.")
    print(f"[DEBUG] Eingabe => test_filename: {test_filename}, chosen_emb: {chosen_emb}")

    clear_gpu_memory()
    print("[DEBUG] Neuberechnung der data.pkl mit Embedding:", chosen_emb)

    data_dir = os.path.join("data", "data_v1", "only_code")
    new_docs = load_data_from_directory(data_dir)
    for d in new_docs:
        d.metadata["source_folder"] = data_dir

    global documents, document_embeddings, metadatas, embeddings
    documents = new_docs
    metadatas = [d.metadata for d in documents]
    texts = [d.page_content for d in new_docs]
    embeddings = HuggingFaceEmbeddings(model_name=chosen_emb)

    doc_emb = embeddings.embed_documents(texts)
    doc_emb = np.array(doc_emb, dtype=np.float32)
    doc_emb = normalize(doc_emb, axis=1)
    document_embeddings = doc_emb

    with open("data.pkl", "wb") as f:
        pickle.dump((documents, document_embeddings, metadatas), f)
    print("[DEBUG] data.pkl erstellt.")

    # Cache initialisieren
    prepare_line_embeddings()

    if not os.path.exists(test_filename):
        st.write(f"Datei {test_filename} wurde nicht gefunden. Bitte vollständigen Pfad angeben.")
        print("[DEBUG] test_filename nicht gefunden:", test_filename)
        return

    with open(test_filename, "r", encoding="utf-8", errors="ignore") as f:
        test_code = f.read()

    print("[DEBUG] Starte manual_retrieval_with_top3() für diese Testdatei.")
    top3_docs, top3_sims, similarities, query_emb = manual_retrieval_with_top3(test_code)

    plot_file_specific(similarities, test_code, title_prefix=f"Explain Decision - {chosen_emb}: ")

    normalized_name = normalize_test_filename(os.path.basename(test_filename))
    target_doc_base = normalized_name + ".json"
    sims = similarities

    target_index = None
    for i, meta in enumerate(metadatas):
        if meta.get("source", "") == target_doc_base:
            target_index = i
            break

    if target_index is not None:
        target_similarity = sims[target_index]
        sorted_indices = np.argsort(sims)[::-1]
        rank = np.where(sorted_indices == target_index)[0][0] + 1

        if rank == 1:
            color = "#6ff261"
        elif rank <= 3:
            color = "#f5e551"
        else:
            color = "#f26161"

        st.markdown(
            f"<span style='color:{color};'>**Ziel-Datei Rang**: {rank}, Similarity: {target_similarity:.4f}</span>",
            unsafe_allow_html=True
        )
        print(f"[DEBUG] => Rang: {rank}, Similarity: {target_similarity:.4f}")
    else:
        st.write("Keine passende Zieldatei (Suffix .json) im Embedding-Dataset gefunden.")
        print("[DEBUG] => Keine passende Datei gefunden.")
        return

    if len(top3_docs) == 0:
        st.write("Keine Dokumente gefunden (Top-3 leer).")
        print("[DEBUG] => Top3 empty.")
        return

    # -----------------------------------------------
    # Heatmap: TestCode vs. Zieldokument
    # -----------------------------------------------
    test_lines = test_code.splitlines()
    test_line_embs = []
    for line in test_lines:
        emb_list = embeddings.embed_documents([line])
        emb_arr = np.array(emb_list[0], dtype=np.float32)
        norm_q = np.linalg.norm(emb_arr)
        if norm_q != 0:
            emb_arr = emb_arr / norm_q
        test_line_embs.append(emb_arr)

    target_doc = documents[target_index]
    target_doc_content = target_doc.page_content
    st.markdown(f"**Ziel-Dokument**: {target_doc.metadata.get('source','?')} (Rang={rank}, CosSim={target_similarity:.4f})")

    target_doc_lines = target_doc_content.splitlines()
    target_doc_line_embs = []
    for line in target_doc_lines:
        emb_list = embeddings.embed_documents([line])
        emb_arr = np.array(emb_list[0], dtype=np.float32)
        norm_val = np.linalg.norm(emb_arr)
        if norm_val != 0:
            emb_arr = emb_arr / norm_val
        target_doc_line_embs.append(emb_arr)

    sim_matrix_target = np.zeros((len(test_lines), len(target_doc_lines)), dtype=np.float32)
    for i in range(len(test_lines)):
        for j in range(len(target_doc_lines)):
            sim_matrix_target[i, j] = test_line_embs[i].dot(target_doc_line_embs[j])

    os.makedirs("explainfolder", exist_ok=True)
    explainfolder_path_target = os.path.join("explainfolder", "explain_decision_linesim_target.png")

    fig2, ax2 = plt.subplots(figsize=(12, 8))
    sns.heatmap(sim_matrix_target, cmap="coolwarm", annot=False, vmin=0.0, vmax=1.0, ax=ax2)

    ax2.set_xticks(np.arange(sim_matrix_target.shape[1]) + 0.5)
    ax2.set_yticks(np.arange(sim_matrix_target.shape[0]) + 0.5)
    all_x = np.arange(1, sim_matrix_target.shape[1] + 1)
    all_y = np.arange(1, sim_matrix_target.shape[0] + 1)
    labels_x = [str(x) if x % 2 == 1 else "" for x in all_x]
    labels_y = [str(y) if y % 2 == 1 else "" for y in all_y]

    ax2.set_xticklabels(labels_x)
    ax2.set_yticklabels(labels_y)

    ax2.set_title(f"Zeilenbasierte Cosine-Sim\n"
                  f"(TestCode vs. {target_doc.metadata.get('source','?')})\n"
                  f"Rang={rank}, CosSim={target_similarity:.4f}, Embedding={chosen_emb}")
    ax2.set_xlabel("Ziel-Dokument (Zeilen)")
    ax2.set_ylabel("TestCode (Zeilen)")
    plt.tight_layout()
    plt.savefig(explainfolder_path_target)
    plt.close(fig2)

    st.markdown("**Heatmap (TestCode vs. Eigentliches Ziel-Dokument)**:")
    st.image(explainfolder_path_target)
    print(f"[DEBUG] => Heatmap (Ziel-Dokument) gespeichert nach {explainfolder_path_target}")
    st.markdown("---")

    # -----------------------------------------------
    # Heatmap: TestCode vs. Top-1
    # -----------------------------------------------
    top1_doc = top3_docs[0]
    st.markdown(f"**Top-1 Dokument** laut Retrieval: {top1_doc.metadata.get('source','?')} "
                f"(CosSim={top3_sims[0]:.4f})")

    top1_content = top1_doc.page_content
    top1_lines = top1_content.splitlines()

    doc_line_embs = []
    for line in top1_lines:
        emb_list = embeddings.embed_documents([line])
        emb_arr = np.array(emb_list[0], dtype=np.float32)
        norm_d = np.linalg.norm(emb_arr)
        if norm_d != 0:
            emb_arr = emb_arr / norm_d
        doc_line_embs.append(emb_arr)

    sim_matrix_top1 = np.zeros((len(test_lines), len(top1_lines)), dtype=np.float32)
    for i in range(len(test_lines)):
        for j in range(len(top1_lines)):
            sim_matrix_top1[i, j] = test_line_embs[i].dot(doc_line_embs[j])

    explain_plot_path_top1 = os.path.join("explainfolder", "explain_decision_linesim_top1.png")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(sim_matrix_top1, cmap="coolwarm", annot=False, vmin=0.0, vmax=1.0, ax=ax)

    ax.set_xticks(np.arange(sim_matrix_top1.shape[1]) + 0.5)
    ax.set_yticks(np.arange(sim_matrix_top1.shape[0]) + 0.5)
    all_x_top1 = np.arange(1, sim_matrix_top1.shape[1] + 1)
    all_y_top1 = np.arange(1, sim_matrix_top1.shape[0] + 1)
    labels_x_top1 = [str(x) if x % 2 == 1 else "" for x in all_x_top1]
    labels_y_top1 = [str(y) if y % 2 == 1 else "" for y in all_y_top1]

    ax.set_xticklabels(labels_x_top1)
    ax.set_yticklabels(labels_y_top1)

    ax.set_title(f"Zeilenbasierte Cosine-Sim\n"
                 f"(TestCode vs. [Top1] {top1_doc.metadata.get('source','?')})\n"
                 f"CosSim={top3_sims[0]:.4f}, Embedding={chosen_emb}")
    ax.set_xlabel("Top-1 Doc (Zeilen)")
    ax.set_ylabel("TestCode (Zeilen)")
    plt.tight_layout()
    plt.savefig(explain_plot_path_top1)
    plt.close()

    st.image(explain_plot_path_top1)
    st.markdown("**Hinweis**: Rot = hohe Cosine-Sim ~ 1.0, Blau = niedrige Cosine-Sim ~ 0.0")

    print(f"[DEBUG] => Heatmap gespeichert nach {explain_plot_path_top1}")


# ------------------ Haupt-Streamlit-Funktion ------------------
def main():
    """
    Streamlit-Einstiegspunkt:
      - Chat-Eingabebereich
      - Buttons für Tests
      - Explain Decision
    """
    global documents, document_embeddings, metadatas, embeddings

    st.title("Forschungsprojekt RAG-LLM")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "file_stats" not in st.session_state:
        st.session_state.file_stats = {}

    prompt = st.chat_input("Frage hier eingeben:")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        emb_model = embedding_models[0]  # Default: Erstes Embedding
        clear_gpu_memory()
        if not os.path.exists("data.pkl"):
            print("[DEBUG] data.pkl nicht vorhanden, erstelle neu mit:", emb_model)
            data_dir = os.path.join("data", "data_v1", "only_code")
            docs = load_data_from_directory(data_dir)
            for d in docs:
                d.metadata["source_folder"] = data_dir

            documents = docs
            metadatas = [d.metadata for d in documents]
            texts = [d.page_content for d in docs]
            embeddings = HuggingFaceEmbeddings(model_name=emb_model)
            doc_emb = embeddings.embed_documents(texts)
            doc_emb = np.array(doc_emb, dtype=np.float32)
            doc_emb = normalize(doc_emb, axis=1)
            document_embeddings = doc_emb
            with open("data.pkl", "wb") as f:
                pickle.dump((documents, document_embeddings, metadatas), f)

            # Cache anlegen
            prepare_line_embeddings()
        else:
            print("[DEBUG] data.pkl existiert bereits, lade es.")
            with open("data.pkl", "rb") as f:
                documents, document_embeddings, metadatas = pickle.load(f)
            embeddings = HuggingFaceEmbeddings(model_name=emb_model)
            # Cache anlegen
            prepare_line_embeddings()

        llama_response, top_doc, similarities = hybrid_response(prompt, selectedLLM, emb_model)
        st.session_state.messages.append({"role": "assistant", "content": llama_response})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)

    st.write("---")

    if "results_overall" not in st.session_state:
        st.session_state.results_overall = []

    if st.button("Run Tests"):
        testdata_dir = os.path.join(os.path.dirname(__file__), "TestCode", "TestCode_Dimensions")
        base_output_dir = os.path.join(os.path.dirname(__file__), "TestResults", "EmbeddingModels")
        llm_output_base = os.path.join(os.path.dirname(__file__), "TestResults", "LLMs")

        st.session_state.results_overall = []

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
            df.sort_values(by=["top3_ratio", "top1_ratio", "avg_similarity"],
                           ascending=[False, False, False],
                           inplace=True)
            df["Top-1 (Prozent)"] = (df["top1_ratio"] * 100).round(1).astype(str) + "%"
            df["Top-3 (Prozent)"] = (df["top3_ratio"] * 100).round(1).astype(str) + "%"

            st.write("**Zusammenfassung aller Embedding-Modelle:**")
            st.write(df[["embedding_model", "avg_similarity", "Top-1 (Prozent)", "Top-3 (Prozent)"]])

            results_summary_file = os.path.join(base_output_dir, "results_summary_embedding_models.csv")
            df.to_csv(results_summary_file, index=False)
            st.write(f"Ergebnisse gespeichert")

            st.write("---")
            st.write("**Kategorien (0_20), (0_40), (0_60), (0_80), (0_100)** – Übersichts-Tabelle je Modell:")
            show_file_stats_tables()

            best_emb_model = df.iloc[0]["embedding_model"]
            st.markdown(
                f"<span style='color:white;'>**Bestes Embedding Modell (nach Top-3-Ratio):** </span>"
                f"<span style='color:#d99a69;'>{best_emb_model}</span>",
                unsafe_allow_html=True
            )

            clear_gpu_memory()
            if os.path.exists("data.pkl"):
                os.remove("data.pkl")

            embeddings = HuggingFaceEmbeddings(model_name=best_emb_model)
            data_dir = os.path.join("data", "data_v1", "only_code")
            docs = load_data_from_directory(data_dir)
            for d in docs:
                d.metadata["source_folder"] = data_dir

            documents = docs
            metadatas = [d.metadata for d in documents]
            texts = [d.page_content for d in docs]
            doc_emb = embeddings.embed_documents(texts)
            doc_emb = np.array(doc_emb, dtype=np.float32)
            doc_emb = normalize(doc_emb, axis=1)
            document_embeddings = doc_emb
            with open("data.pkl", "wb") as f:
                pickle.dump((documents, document_embeddings, metadatas), f)

            # Zeilen-Cache
            prepare_line_embeddings()

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
            st.write("Keine Ergebnisse verfügbar.")

    if st.button("(Test Up to 500 Lines)"):
        run_tests_for_lines()

    if st.button("Test Changed Codes"):
        run_tests_for_changed_codes()

    if st.button("Test Dimensions"):
        run_tests_for_dimensions()

    st.write("---")
    st.subheader("Explain Decision für eine bestimmte Test-Code-Datei")

    chosen_embedding = st.selectbox("Wähle ein Embedding-Modell für die Explain-Analyse:", embedding_models)
    explain_file_input = st.text_input("TestCode-Datei-Pfad eingeben:", "")

    if st.button("Explain Decision"):
        if explain_file_input.strip():
            print("[DEBUG] Explain Decision button geklickt.")
            explain_decision(explain_file_input.strip(), chosen_embedding)
        else:
            st.write("Bitte einen gültigen Dateipfad eingeben.")
            print("[DEBUG] Kein Dateipfad eingegeben.")


if __name__ == '__main__':
    main()

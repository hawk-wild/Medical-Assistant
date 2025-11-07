import json
import re
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class HybridDiseaseRAG:
    def __init__(self, json_file_path, model_name='all-MiniLM-L6-v2'):
        """
        Hybrid Disease Diagnosis System:
        Combines graph reasoning (disease–symptom structure) and vector similarity.
        """
        self.model = SentenceTransformer(model_name)
        self.graph = nx.Graph()
        self.diseases_data = self._load_data(json_file_path)
        self._build_graph()
        self.symptom_embeddings = self._precompute_symptom_embeddings()

    # ---------- Data & Graph ----------
    def _load_data(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def _build_graph(self):
        """Build a disease–symptom–precaution knowledge graph."""
        for entry in self.diseases_data:
            disease = entry['disease']
            self.graph.add_node(disease, type='disease')

            for symptom in entry.get('symptoms', []):
                self.graph.add_node(symptom, type='symptom')
                self.graph.add_edge(disease, symptom, relation='has_symptom')

            for precaution in entry.get('precautions', []):
                self.graph.add_node(precaution, type='precaution')
                self.graph.add_edge(disease, precaution, relation='has_precaution')

    def _precompute_symptom_embeddings(self):
        """Encode all symptom nodes for semantic comparison."""
        symptoms = [n for n, d in self.graph.nodes(data=True) if d['type'] == 'symptom']
        embeddings = self.model.encode(symptoms)
        return dict(zip(symptoms, embeddings))

    # ---------- Symptom Extraction ----------
    def extract_symptoms_from_query(self, query, threshold=0.45):
        """Find similar symptoms in the graph to user query sentences."""
        sentences = re.split(r'[.!?]+', query)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            sentences = [query]

        query_embeddings = self.model.encode(sentences)
        matched_symptoms = {}

        for sent_emb in query_embeddings:
            for symptom, sym_emb in self.symptom_embeddings.items():
                sim = cosine_similarity([sent_emb], [sym_emb])[0][0]
                if sim >= threshold:
                    matched_symptoms[symptom] = max(sim, matched_symptoms.get(symptom, 0))

        return sorted(matched_symptoms.items(), key=lambda x: x[1], reverse=True)

    # ---------- Graph Scoring ----------
    def rank_diseases_by_graph(self, matched_symptoms):
        """Compute graph-based disease relevance based on matched symptom connectivity."""
        disease_scores = {}
        disease_symptom_matches = {}

        for symptom, sim in matched_symptoms:
            if symptom in self.graph:
                connected_diseases = [
                    n for n in self.graph.neighbors(symptom)
                    if self.graph.nodes[n]['type'] == 'disease'
                ]
                for disease in connected_diseases:
                    disease_scores[disease] = disease_scores.get(disease, 0) + sim
                    disease_symptom_matches.setdefault(disease, []).append((symptom, sim))

        # Normalize graph scores by total disease symptoms
        for disease, score in disease_scores.items():
            total = len([
                n for n in self.graph.neighbors(disease)
                if self.graph.nodes[n]['type'] == 'symptom'
            ])
            if total > 0:
                disease_scores[disease] = score / total

        return disease_scores, disease_symptom_matches

    # ---------- Vector Scoring ----------
    def compute_vector_similarity(self, query, diseases):
        """Compare query embedding with all symptoms of each disease (semantic similarity)."""
        query_emb = self.model.encode([query])[0]
        disease_sim = {}

        for disease in diseases:
            symptoms = [
                n for n in self.graph.neighbors(disease)
                if self.graph.nodes[n]['type'] == 'symptom'
            ]
            if not symptoms:
                continue
            sym_embs = [self.symptom_embeddings[s] for s in symptoms if s in self.symptom_embeddings]
            if not sym_embs:
                continue
            avg_emb = np.mean(sym_embs, axis=0)
            sim = cosine_similarity([query_emb], [avg_emb])[0][0]
            disease_sim[disease] = sim

        return disease_sim

    # ---------- Diagnosis ----------
    def diagnose(self, query, top_k=3, similarity_threshold=0.45, alpha=0.6):
        """
        Diagnose diseases from user query.
        alpha = weight for graph vs vector (0.6 = 60% graph, 40% vector)
        """
        matched_symptoms = self.extract_symptoms_from_query(query, similarity_threshold)
        if not matched_symptoms:
            return {"status": "no_match", "message": "No matching symptoms found."}

        graph_scores, disease_symptom_matches = self.rank_diseases_by_graph(matched_symptoms)
        diseases = list(graph_scores.keys())
        vector_scores = self.compute_vector_similarity(query, diseases)

        # Combine scores (hybrid fusion)
        hybrid_scores = {}
        for disease in diseases:
            g_score = graph_scores.get(disease, 0)
            v_score = vector_scores.get(disease, 0)
            hybrid_scores[disease] = alpha * g_score + (1 - alpha) * v_score

        # Rank by hybrid score
        ranked_diseases = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        top_diseases = ranked_diseases[:top_k]

        # Build detailed result
        results = []
        for disease, score in top_diseases:
            total_symptoms = len([
                n for n in self.graph.neighbors(disease)
                if self.graph.nodes[n]['type'] == 'symptom'
            ])
            matches = disease_symptom_matches.get(disease, [])
            precautions = [
                n for n in self.graph.neighbors(disease)
                if self.graph.nodes[n]['type'] == 'precaution'
            ]
            results.append({
                "disease": disease,
                "hybrid_score": score,
                "graph_score": graph_scores.get(disease, 0),
                "vector_score": vector_scores.get(disease, 0),
                "matched_symptoms": [m[0] for m in matches],
                "similarities": [round(m[1], 3) for m in matches],
                "num_matches": len(matches),
                "total_symptoms": total_symptoms,
                "precautions": precautions
            })

        return {
            "status": "success",
            "query": query,
            "matched_symptoms": [s for s, _ in matched_symptoms],
            "top_diseases": results,
            "best_match": results[0] if results else None
        }

    # ---------- Response ----------
    def generate_response(self, diagnosis_result):
        if diagnosis_result["status"] == "no_match":
            return diagnosis_result["message"]

        best = diagnosis_result["best_match"]
        response = f" Based on your symptoms, you may have **{best['disease']}**.\n\n"
        response += (
            f"**Matched symptoms ({best['num_matches']}/{best['total_symptoms']}):**\n"
        )
        for symptom, sim in zip(best['matched_symptoms'], best['similarities']):
            response += f"- {symptom} (similarity: {sim:.2f})\n"

        response += "\n**Precautions:**\n"
        for p in best['precautions']:
            response += f"- {p}\n"

        response += (
            f"\n**Scores:** Graph={best['graph_score']:.3f}, "
            f"Vector={best['vector_score']:.3f}, Hybrid={best['hybrid_score']:.3f}\n"
        )

        if len(diagnosis_result["top_diseases"]) > 1:
            response += "\n**Other possible conditions:**\n"
            for d in diagnosis_result["top_diseases"][1:]:
                response += (
                    f"- {d['disease']} (hybrid={d['hybrid_score']:.3f}, "
                    f"matched {d['num_matches']}/{d['total_symptoms']})\n"
                )

        response += "\n*Note: This is an automated assessment. Please consult a healthcare professional for proper diagnosis and treatment.*"
        return response

# ---------- Example usage ----------
if __name__ == "__main__":
    print("\n" + "="*50)
    print("Initializing Hybrid Disease RAG System...")
    print("="*50)
    rag = HybridDiseaseRAG("medical_dataset.json")

    query = "I have rashes on my shoulder which itches a lot. There are also dark patches on my neck region."
    print("\nPlease describe your issue to the fullest extent:")
    print(f"\nQuery: {query}\n")

    result = rag.diagnose(query, top_k=3, alpha=0.6)  # 60% graph weight, 40% vector weight
    print(rag.generate_response(result))

    print("\n" + "="*50)
    print("DETAILED RESULTS")
    print("="*50)
    print(f"\nMatched symptoms from query: {result['matched_symptoms']}")
    print(f"\nTop diseases:")
    for d in result['top_diseases']:
        print(f"\n{d['disease']}: Hybrid={d['hybrid_score']:.3f}, Graph={d['graph_score']:.3f}, Vector={d['vector_score']:.3f}")
        print(f"Matched Symptoms ({d['num_matches']}/{d['total_symptoms']}): {d['matched_symptoms']}")
        print(f"Precautions: {d['precautions']}")

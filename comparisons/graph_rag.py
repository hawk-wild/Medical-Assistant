import json
import networkx as nx
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class DiseaseGraphRAG:
    def __init__(self, json_file_path, model_name='all-MiniLM-L6-v2'):
        """Graph-based disease diagnosis system using knowledge graph reasoning."""
        self.model = SentenceTransformer(model_name)
        self.graph = nx.Graph()
        self.diseases_data = self._load_data(json_file_path)
        self._build_graph()
        self.symptom_embeddings = self._precompute_symptom_embeddings()

    def _load_data(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def _build_graph(self):
        """Build a graph linking diseases, symptoms, and precautions."""
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
        """Compute embeddings for all unique symptoms."""
        symptoms = [n for n, d in self.graph.nodes(data=True) if d['type'] == 'symptom']
        embeddings = self.model.encode(symptoms)
        return dict(zip(symptoms, embeddings))

    def extract_symptoms_from_query(self, query, threshold=0.45):
        """Find the most similar known symptoms to a user's query using embeddings."""
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
                    if symptom not in matched_symptoms or matched_symptoms[symptom] < sim:
                        matched_symptoms[symptom] = sim

        return sorted(matched_symptoms.items(), key=lambda x: x[1], reverse=True)

    def rank_diseases_by_graph(self, matched_symptoms):
        """Rank diseases by how many matched symptoms connect to them."""
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
                    if disease not in disease_symptom_matches:
                        disease_symptom_matches[disease] = []
                    disease_symptom_matches[disease].append((symptom, sim))

        # Normalize by total symptom count for fairness
        for disease in list(disease_scores.keys()):
            total_symptoms = len([
                n for n in self.graph.neighbors(disease)
                if self.graph.nodes[n]['type'] == 'symptom'
            ])
            if total_symptoms > 0:
                disease_scores[disease] /= total_symptoms

        ranked = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked, disease_symptom_matches

    def get_precautions(self, disease):
        """Return all precautions connected to a disease."""
        if disease not in self.graph:
            return []
        return [
            n for n in self.graph.neighbors(disease)
            if self.graph.nodes[n]['type'] == 'precaution'
        ]

    def get_total_symptoms(self, disease):
        """Return total number of known symptoms for a disease."""
        return len([
            n for n in self.graph.neighbors(disease)
            if self.graph.nodes[n]['type'] == 'symptom'
        ])

    def diagnose(self, query, top_k=3, similarity_threshold=0.45):
        """Perform full diagnosis using graph-based reasoning."""
        matched_symptoms = self.extract_symptoms_from_query(query, similarity_threshold)
        if not matched_symptoms:
            return {"status": "no_match", "message": "No matching symptoms found."}

        ranked_diseases, disease_symptom_matches = self.rank_diseases_by_graph(matched_symptoms)
        top_diseases = ranked_diseases[:top_k]

        results = []
        for disease, score in top_diseases:
            precautions = self.get_precautions(disease)
            matches = disease_symptom_matches.get(disease, [])
            total_symptoms = self.get_total_symptoms(disease)

            results.append({
                "disease": disease,
                "score": score,
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

    def generate_response(self, diagnosis_result):
        """Generate a user-friendly response."""
        if diagnosis_result["status"] == "no_match":
            return diagnosis_result["message"]

        best = diagnosis_result["best_match"]
        response = f"Based on your symptoms, you may have **{best['disease']}**.\n\n"

        # show matched symptom fraction
        response += f"**Matched symptoms ({best['num_matches']}/{best['total_symptoms']}):**\n"
        for symptom, sim in zip(best['matched_symptoms'], best['similarities']):
            response += f"- {symptom} (similarity: {sim:.2f})\n"

        response += f"\n**Precautions:**\n"
        for p in best['precautions']:
            response += f"- {p}\n"

        if len(diagnosis_result["top_diseases"]) > 1:
            response += f"\n**Other possible conditions:**\n"
            for d in diagnosis_result["top_diseases"][1:]:
                response += (
                    f"- {d['disease']} (score: {d['score']:.3f}, "
                    f"matched {d['num_matches']}/{d['total_symptoms']})\n"
                )

        response += "\n*Note: This is an automated assessment. Please consult a healthcare professional for proper diagnosis and treatment.*"
        return response

        
# Example usage
if __name__ == "__main__":
    print("\n" + "="*50)
    print("Initializing Graph-based Disease RAG System...")
    print("="*50)
    rag = DiseaseGraphRAG("medical_dataset.json")

    query = "I have rashes on my shoulder which itches a lot. There are also dark patches on my neck region."
    print("\nPlease describe your issue to the fullest extent:")
    print(f"\nQuery: {query}\n")

    result = rag.diagnose(query)
    print(rag.generate_response(result))

    print("\n" + "="*50)
    print("DETAILED RESULTS")
    print("="*50)
    for d in result['top_diseases']:
        print(f"\n{d['disease']}: {d['score']:.4f}")
        print(f"Matched Symptoms ({d['num_matches']}/{d['total_symptoms']}): {d['matched_symptoms']}")
        print(f"Precautions: {d['precautions']}")


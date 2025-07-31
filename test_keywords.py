#!/usr/bin/env python3
"""
Script de test pour l'API de recommandation par mots-clés
Version simplifiée pour exécution dans Docker
"""

import requests
import json
import sys

# Configuration
API_BASE_URL = "http://localhost:8000"
KEYWORDS_ENDPOINT = f"{API_BASE_URL}/ai/recommendations/keywords"

def test_keywords_recommendation():
    """Test de l'API de recommandation par mots-clés"""
    
    # Données de test
    test_data = {
        "keywords": ["Python", "Java", "C"],
        "per_keyword_limit": 5,
        "final_limit": 5
    }
    
    print("🧪 Test de l'API de recommandation par mots-clés")
    print(f"📡 Endpoint: {KEYWORDS_ENDPOINT}")
    print(f"📝 Données de test: {json.dumps(test_data, indent=2)}")
    print("💡 Note: Les mots-clés sont temporairement stockés dans Qdrant pour la recherche")
    print("-" * 50)
    
    try:
        # Appel de l'API
        response = requests.post(
            KEYWORDS_ENDPOINT,
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            recommendations = response.json()
            print(f"✅ Succès! {len(recommendations)} recommandations trouvées")
            print("\n📋 Recommandations:")
            
            for i, rec in enumerate(recommendations, 1):
                print(f"\n{i}. {rec.get('filename', 'N/A')}")
                print(f"   Type: {rec.get('type', 'N/A')}")
                print(f"   Mot-clé: {rec.get('keyword', 'N/A')}")
                print(f"   Score de similarité: {rec.get('similarity_score', 'N/A'):.4f}")
                print(f"   Doc ID: {rec.get('doc_id', 'N/A')}")
                
        else:
            print(f"❌ Erreur: {response.status_code}")
            print(f"📄 Réponse: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Erreur de connexion: Impossible de se connecter à l'API")
        print("💡 Assurez-vous que le serveur FastAPI est en cours d'exécution")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print("❌ Timeout: L'API a pris trop de temps à répondre")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("🚀 Démarrage des tests de l'API de recommandation par mots-clés")
    print("=" * 60)
    
    # Test principal
    test_keywords_recommendation()
    
    print("\n✨ Tests terminés!")
    print("💡 Les points temporaires ont été automatiquement supprimés de Qdrant") 
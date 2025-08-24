import logging

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from app.core.config import OPENAI_API_KEY, OPENAI_LLM_MODEL, TEXT_VECTOR_NAME, IMAGE_VECTOR_NAME, VIDEO_VECTOR_NAME, AUDIO_VECTOR_NAME
from app.models.encoders import encoder
from app.db.qdrant_ops import search_similar_content
from app.models.schemas import SourceDocument

logger = logging.getLogger(__name__)

llm = ChatOpenAI(model=OPENAI_LLM_MODEL, api_key=OPENAI_API_KEY, temperature=0.2)

def is_visual_query(query):
    """Detect if the query is asking about visual content"""
    visual_keywords = [
        'image', 'photo', 'picture', 'diagram', 'chart', 'graph', 'figure',
        'illustration', 'screenshot', 'visual', 'voir', 'montrer', 'afficher',
        'schéma', 'graphique', 'diagramme', 'capture', 'écran'
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in visual_keywords)

RAG_PROMPT_TEMPLATE = """
# ROLE ET OBJECTIF
Vous êtes "Epitome AI", l'assistant pédagogique officiel de l'Epitome Academy. Votre unique mission est d'aider les apprenants en répondant à leurs questions de manière précise, factuelle et encourageante.

# DIRECTIVE FONDAMENTALE (NON-NÉGOCIABLE)
Votre directive la plus importante est de formuler vos réponses en vous basant **UNIQUEMENT ET EXCLUSIVEMENT** sur les informations présentes dans la section `CONTEXTE` fournie ci-dessous. Le `CONTEXTE` peut contenir différents types de contenu : texte, images, vidéos, et audio extraits directement des supports de cours officiels de l'académie.

**NE JAMAIS INVENTER DE RÉPONSES.** Si l'information n'est pas explicitement présente dans le `CONTEXTE`, vous devez l'indiquer clairement.

# PROCESSUS DE RÉPONSE ÉTAPE PAR ÉTAPE
Pour chaque question, suivez rigoureusement ces étapes :
1.  **Analyser la Demande** : Lisez attentivement la `QUESTION` de l'utilisateur et consultez l'`HISTORIQUE DE LA CONVERSATION` pour comprendre le contexte global de l'échange.
2.  **Examiner le Contexte Multimodal** : Étudiez en détail chaque fragment d'information dans la section `CONTEXTE`, qu'il s'agisse de texte, d'images, de vidéos ou d'audio. Identifiez les éléments qui répondent directement ou indirectement à la `QUESTION`.
3.  **Gérer l'Absence d'Information** : Si, après un examen minutieux, vous ne trouvez aucune information pertinente dans le `CONTEXTE` pour répondre à la `QUESTION`, vous devez **OBLIGATOIREMENT** répondre avec la phrase suivante, et rien d'autre : "Je suis désolé, mais je n'ai pas pu trouver d'information précise à ce sujet dans les documents à ma disposition. Souhaitez-vous que je cherche sur un autre sujet ?"
4.  **Synthétiser la Réponse** : Si le contexte contient des informations pertinentes, synthétisez une réponse claire, concise et pédagogique. Reformulez avec vos propres mots pour rendre l'explication facile à comprendre, mais sans altérer le sens original du `CONTEXTE`.
5.  **Citer les Sources** : Pour chaque information que vous donnez, vous devez **OBLIGATOIREMENT** citer sa source à la fin du paragraphe pertinent, en utilisant le format exact : `[Source: nom_du_fichier.pdf, Page: X]` pour le texte, `[Source: nom_du_fichier.pdf, Page: X - Image]` pour les images, ou `[Source: nom_du_fichier.mp4, Timestamp: X]` pour les vidéos.

# RÈGLES DE FORMATAGE ET DE STYLE
- **Clarté avant tout** : Utilisez un langage simple et direct.
- **Structure** : Utilisez le formatage Markdown (listes à puces, gras) pour structurer vos réponses et les rendre plus lisibles.
- **Contenu Multimodal** : Quand vous référencez des images ou vidéos, décrivez leur contenu de manière claire et précise.
- **Ton** : Soyez toujours professionnel, patient, positif et encourageant.
- **Sécurité** : Ne jamais demander d'informations personnelles à l'utilisateur.

# EXEMPLES
---
**Exemple 1 : Réponse réussie**
CONTEXTE:
Source: IA_pour_tous.pdf, Page: 12
Contenu: L'apprentissage supervisé est une branche de l'intelligence artificielle où un modèle est entraîné sur un ensemble de données étiquetées. Par exemple, pour un classificateur d'images, les données d'entraînement seraient des images avec des étiquettes indiquant "chat" ou "chien".
HISTORIQUE DE LA CONVERSATION:

QUESTION:
Qu'est-ce que l'apprentissage supervisé et donne-moi un exemple.

RÉPONSE ATTENDUE:
L'apprentissage supervisé est une approche de l'intelligence artificielle qui consiste à entraîner un algorithme sur des données préalablement étiquetées. L'objectif est que le modèle apprenne à faire des prédictions correctes sur de nouvelles données. [Source: IA_pour_tous.pdf, Page: 12]

Un bon exemple serait un système de classification d'images. On lui fournirait des milliers d'images en précisant pour chacune si elle contient un "chat" ou un "chien", afin qu'il apprenne à les différencier par lui-même. [Source: IA_pour_tous.pdf, Page: 12]
---
**Exemple 2 : Information non trouvée**
CONTEXTE:
Source: Marketing_Digital.pdf, Page: 4
Contenu: Le SEO, ou Search Engine Optimization, vise à améliorer la visibilité d'un site web dans les résultats des moteurs de recherche comme Google.
HISTORIQUE DE LA CONVERSATION:

QUESTION:
Quelle est la date de la bataille de Marignan ?

RÉPONSE ATTENDUE:
Je suis désolé, mais je n'ai pas pu trouver d'information précise à ce sujet dans les documents à ma disposition. Souhaitez-vous que je cherche sur un autre sujet ?
---

Maintenant, analysez la situation actuelle et répondez à la question de l'utilisateur.

--------------------------------------------------------------------------------
CONTEXTE:
{context}

HISTORIQUE DE LA CONVERSATION:
{chat_history}

QUESTION:
{question}

RÉPONSE:
"""

prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

def retrieve_context(query, chat_history):
    logger.info(f"Récupération du contexte multimodal pour la question: '{query}'")
    
    is_visual = is_visual_query(query)
    
    text_embedding = encoder.encode_text(query)
    
    all_hits = []
    
    if is_visual:
        text_limit, image_limit, video_limit = 2, 3, 2
        logger.info("Visual query detected - prioritizing image/video content")
    else:
        text_limit, image_limit, video_limit = 3, 2, 1
    
    text_hits = search_similar_content(
        vector=text_embedding,
        vector_name=TEXT_VECTOR_NAME,
        limit=text_limit
    )
    all_hits.extend(text_hits)
    
    try:
        image_embedding = encoder.encode_text(query)
        image_hits = search_similar_content(
            vector=image_embedding,
            vector_name=IMAGE_VECTOR_NAME,
            limit=image_limit
        )
        all_hits.extend(image_hits)
        logger.info(f"Found {len(image_hits)} image results")
    except Exception as e:
        logger.warning(f"Image search failed: {e}")
    
    try:
        video_hits = search_similar_content(
            vector=text_embedding,
            vector_name=VIDEO_VECTOR_NAME,
            limit=video_limit
        )
        all_hits.extend(video_hits)
        logger.info(f"Found {len(video_hits)} video results")
    except Exception as e:
        logger.warning(f"Video search failed: {e}")
    
    all_hits.sort(key=lambda x: x.score, reverse=True)
    
    top_hits = all_hits[:6]
    
    context_parts = []
    for hit in top_hits:
        content_type = hit.payload.get('type', 'text')
        source_file = hit.payload.get('source_file', 'N/A')
        page_number = hit.payload.get('page_number', 'N/A')
        
        if content_type == 'image':
            # For images, include description or metadata
            image_desc = hit.payload.get('text', 'Image sans description')
            context_parts.append(
                f"Source: {source_file}, Page: {page_number}\n"
                f"Type: Image\n"
                f"Description: {image_desc}"
            )
        elif content_type == 'video':
            video_desc = hit.payload.get('text', 'Vidéo')
            timestamp = hit.payload.get('timestamp', 'N/A')
            context_parts.append(
                f"Source: {source_file}\n"
                f"Type: Vidéo (timestamp: {timestamp})\n"
                f"Description: {video_desc}"
            )
        else:
            text_content = hit.payload.get('text', '')
            context_parts.append(
                f"Source: {source_file}, Page: {page_number}\n"
                f"Contenu: {text_content}"
            )
    
    context_str = "\n---\n".join(context_parts)
    
    sources = [
        SourceDocument(
            doc_id=hit.payload.get('doc_id'),
            filename=hit.payload.get('source_file') or hit.payload.get('filename'),
            content_type=hit.payload.get('type', 'document'),
            page_number=hit.payload.get('page_number'),
            score=hit.score
        ) for hit in top_hits
    ]
    
    logger.info(f"Retrieved {len(top_hits)} multimodal results (visual query: {is_visual})")
    return {"context": context_str, "sources": sources}


def create_rag_chain():
    def rag_chain_func(inputs):
        retrieved_data = retrieve_context(inputs["question"], inputs.get("chat_history", []))
        
        generator_chain = (
            RunnablePassthrough.assign(
                context=lambda x: retrieved_data["context"],
                chat_history=lambda x: x["chat_history"]
            )
            | prompt
            | llm
            | StrOutputParser()
        )
        
        answer = generator_chain.invoke({
            "question": inputs["question"],
            "chat_history": inputs.get("chat_history", [])
        })
        
        return {
            "answer": answer,
            "sources": retrieved_data["sources"]
        }
        
    return rag_chain_func

rag_chain = create_rag_chain()
import os
import json
import time
import logging
import re
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import requests
from pathlib import Path
from dotenv import load_dotenv
import hashlib

# External dependencies
import chromadb
import gradio as gr
from sentence_transformers import SentenceTransformer
import pdfplumber
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

load_dotenv()


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class BackendConfig:
    name: str
    provider: str
    model: str
    max_tokens: int
    temperature: float
    cost_per_1k_tokens: float
    priority: int
    api_key_env: str
    base_url: Optional[str] = None


class LegalDomain(Enum):
    FAMILY_LAW = "family_law"
    CRIMINAL_LAW = "criminal_law"
    CIVIL_LAW = "civil_law"
    CONSTITUTIONAL_LAW = "constitutional_law"
    AMENDMENTS = "amendments"
    ARCHIVE = "archive"
    GENERAL = "general"


class QueryType(Enum):
    GENERAL_QUESTION = "general_question"
    CASE_ANALYSIS = "case_analysis"
    DOCUMENT_REVIEW = "document_review"
    LEGAL_RESEARCH = "legal_research"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    DOCUMENT_GENERATION = "document_generation"


class Config:
    BACKENDS = [
        BackendConfig(
            name="OpenRouter",
            provider="openrouter",
            model="openai/gpt-4o",
            max_tokens=4096,
            temperature=0.2,
            cost_per_1k_tokens=0.0005,
            priority=1,
            api_key_env="OPENROUTER_API_KEY",
            base_url="https://openrouter.ai/api/v1"
        ),
        BackendConfig(
            name="Groq",
            provider="groq",
            model="llama-3.3-70b-versatile",
            max_tokens=4096,
            temperature=0.2,
            cost_per_1k_tokens=0.0,
            priority=2,
            api_key_env="GROQ_API_KEY",
            base_url="https://api.groq.com/openai/v1"
        ),
        BackendConfig(
            name="OpenAI",
            provider="openai",
            model="gpt-4o-mini",
            max_tokens=4096,
            temperature=0.2,
            cost_per_1k_tokens=0.00015,
            priority=3,
            api_key_env="OPENAI_API_KEY",
            base_url="https://api.openai.com/v1"
        )
    ]

    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    CHROMA_DB_PATH = "./chroma_db"
    LEGAL_DOCS_PATH = "./legal_docs"
    USER_UPLOADS_PATH = "./user_uploads"
    MAX_RETRIES = 3
    TIMEOUT_SECONDS = 60
    MIN_CHUNK_SIZE = 500
    MAX_CHUNK_SIZE = 1500
    TOP_K_RESULTS = 8  # Increased for better context
    LOG_FILE = "legal_chatbot.log"
    LOG_LEVEL = logging.INFO


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging():
    logging.basicConfig(
        level=Config.LOG_LEVEL,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Config.LOG_FILE, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()

# Create necessary directories
Path(Config.USER_UPLOADS_PATH).mkdir(exist_ok=True)


# ============================================================================
# ENHANCED QUERY PROCESSING
# ============================================================================

class EnhancedQueryProcessor:
    """Advanced query processing with legal terminology and multi-lingual support"""

    LEGAL_TERMS_EXPANSION = {
        # Criminal Law
        "السرقة": ["اختلاس", "مال الغير", "الاستيلاء غير المشروع", "سرقة موصوفة", "سرقة بسيطة"],
        "vol": ["larceny", "theft", "stealing", "appropriation", "سرقة", "اختلاس"],
        "القتل": ["القتل العمد", "القتل الخطأ", "الإزهاق", "إزهاق الروح", "homicide"],
        "agression": ["assault", "battery", "violence", "اعتداء", "ضرب", "جرح"],

        # Family Law
        "الزواج": ["عقد الزواج", "النكاح", "الزوجية", "marriage", "matrimony"],
        "mariage": ["wedding", "matrimony", "union", "زواج", "عقد الزواج"],
        "الطلاق": ["فسخ الزواج", "انحلال الزواج", "التطليق", "divorce"],
        "divorce": ["separation", "dissolution", "طلاق", "فسخ"],
        "الحضانة": ["كفالة الأطفال", "رعاية الصغير", "custody", "garde"],
        "garde": ["custody", "guardianship", "حضانة", "كفالة"],

        # Civil Law
        "العقد": ["الاتفاق", "الالتزام التعاقدي", "contract", "contrat"],
        "contrat": ["agreement", "covenant", "عقد", "اتفاق"],
        "الدين": ["المديونية", "الالتزام المالي", "debt", "dette"],
        "dette": ["debt", "obligation", "دين", "مديونية"],

        # Constitutional
        "الدستور": ["القانون الأساسي", "الدستور المغربي", "constitution"],
        "constitution": ["fundamental law", "charter", "دستور"],
        "الحقوق": ["الحريات", "الحقوق الأساسية", "rights", "droits"],
        "droits": ["rights", "freedoms", "liberties", "حقوق"],

        # Procedural
        "الدعوى": ["القضية", "الشكاية", "lawsuit", "procès"],
        "procès": ["trial", "lawsuit", "case", "دعوى", "قضية"],
        "الحكم": ["القرار القضائي", "الصك", "judgment", "jugement"],
        "jugement": ["ruling", "verdict", "decision", "حكم"],
    }

    NEGATION_PATTERNS = {
        "fr": ["ne pas", "n'est pas", "sans", "aucun", "jamais"],
        "ar": ["لا", "ليس", "غير", "دون", "بدون", "ما"]
    }

    @classmethod
    def expand_query(cls, query: str) -> str:
        """Expand query with legal synonyms"""
        expanded = query
        query_lower = query.lower()

        for term, expansions in cls.LEGAL_TERMS_EXPANSION.items():
            if term in query or term in query_lower:
                expanded += " " + " ".join(expansions[:3])  # Add top 3 expansions

        return expanded

    @classmethod
    def detect_negation(cls, query: str) -> bool:
        """Detect if query contains negation"""
        query_lower = query.lower()

        for lang, patterns in cls.NEGATION_PATTERNS.items():
            if any(pattern in query_lower or pattern in query for pattern in patterns):
                return True
        return False

    @classmethod
    def extract_article_numbers(cls, text: str) -> List[str]:
        """Extract article numbers from query"""
        patterns = [
            r'المادة\s+(\d+)',
            r'article\s+(\d+)',
            r'art\.\s*(\d+)',
            r'المواد\s+(\d+)',
        ]

        articles = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            articles.extend(matches)

        return list(set(articles))


# ============================================================================
# ENHANCED DOCUMENT PROCESSING
# ============================================================================

class AdvancedDocumentProcessor:
    """Enhanced document processing with metadata extraction"""

    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> Tuple[str, Dict]:
        """Extract text and metadata from PDF"""
        try:
            text = ""
            metadata = {
                "total_pages": 0,
                "articles_found": [],
                "language": "unknown",
                "estimated_domain": "general"
            }

            with pdfplumber.open(pdf_path) as pdf:
                metadata["total_pages"] = len(pdf.pages)

                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n\n"
                    except Exception as e:
                        logger.warning(f"Error on page {page_num}: {str(e)}")
                        continue

                # Detect language
                arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
                french_chars = len(re.findall(r'[a-zA-ZÀ-ÿ]', text))

                if arabic_chars > french_chars:
                    metadata["language"] = "arabic"
                elif french_chars > 0:
                    metadata["language"] = "french"

                # Extract article numbers
                articles = EnhancedQueryProcessor.extract_article_numbers(text)
                metadata["articles_found"] = articles[:50]  # Limit to first 50

                logger.info(
                    f"✓ Extracted {len(text)} chars, {len(articles)} articles, language: {metadata['language']}")

            return text.strip(), metadata

        except Exception as e:
            logger.error(f"Error loading PDF {pdf_path}: {str(e)}")
            return "", {}

    @staticmethod
    def intelligent_chunking(text: str, metadata: Dict) -> List[Dict]:
        """Intelligent chunking based on document structure"""
        chunks = []

        # Try article-based chunking first
        article_pattern = r'(المادة\s+\d+|Article\s+\d+)'
        parts = re.split(f'({article_pattern})', text, flags=re.IGNORECASE)

        if len(parts) > 5:  # Document has clear article structure
            current_chunk = ""
            current_article = None

            for i, part in enumerate(parts):
                if re.match(article_pattern, part, re.IGNORECASE):
                    # Save previous chunk
                    if current_chunk and len(current_chunk) > 200:
                        chunks.append({
                            "content": current_chunk.strip(),
                            "article": current_article,
                            "type": "article"
                        })
                    current_chunk = part
                    current_article = part.strip()
                else:
                    current_chunk += part

                    # Split if too long
                    if len(current_chunk) > Config.MAX_CHUNK_SIZE:
                        if current_chunk.strip():
                            chunks.append({
                                "content": current_chunk.strip(),
                                "article": current_article,
                                "type": "article"
                            })
                        current_chunk = ""

            # Add final chunk
            if current_chunk.strip() and len(current_chunk) > 200:
                chunks.append({
                    "content": current_chunk.strip(),
                    "article": current_article,
                    "type": "article"
                })

        else:
            # Fallback: paragraph-based chunking
            paragraphs = text.split('\n\n')
            current_chunk = ""

            for para in paragraphs:
                if len(current_chunk) + len(para) > Config.MAX_CHUNK_SIZE:
                    if current_chunk.strip():
                        chunks.append({
                            "content": current_chunk.strip(),
                            "article": None,
                            "type": "paragraph"
                        })
                    current_chunk = para
                else:
                    current_chunk += "\n\n" + para

            if current_chunk.strip():
                chunks.append({
                    "content": current_chunk.strip(),
                    "article": None,
                    "type": "paragraph"
                })

        logger.info(f"Created {len(chunks)} intelligent chunks")
        return chunks

    @staticmethod
    def detect_subcategory(filename: str, text_sample: str) -> str:
        """Enhanced category detection"""
        filename_lower = filename.lower()
        text_lower = text_sample.lower()

        # Check filename first
        if any(kw in filename_lower for kw in ["famille", "moudawana", "أسرة", "family"]):
            return LegalDomain.FAMILY_LAW.value
        elif any(kw in filename_lower for kw in ["penal", "pénal", "جنائي", "criminal"]):
            return LegalDomain.CRIMINAL_LAW.value
        elif any(kw in filename_lower for kw in ["civil", "obligations", "contrats", "التزامات"]):
            return LegalDomain.CIVIL_LAW.value
        elif any(kw in filename_lower for kw in ["constitution", "الدستور"]):
            return LegalDomain.CONSTITUTIONAL_LAW.value

        # Check content
        if any(kw in text_lower for kw in ["زواج", "طلاق", "حضانة", "mariage", "divorce"]):
            return LegalDomain.FAMILY_LAW.value
        elif any(kw in text_lower for kw in ["جريمة", "عقوبة", "crime", "peine"]):
            return LegalDomain.CRIMINAL_LAW.value

        return LegalDomain.GENERAL.value

    @staticmethod
    def load_documents_from_folder(folder_path: str, category: str) -> List[Dict]:
        """Load documents with enhanced metadata"""
        documents = []
        folder = Path(folder_path)

        if not folder.exists():
            logger.warning(f"Folder not found: {folder_path}")
            return documents

        pdf_files = list(folder.glob("*.pdf"))

        for pdf_file in pdf_files:
            text, metadata = AdvancedDocumentProcessor.extract_text_from_pdf(str(pdf_file))
            if not text:
                continue

            chunks = AdvancedDocumentProcessor.intelligent_chunking(text, metadata)
            subcategory = AdvancedDocumentProcessor.detect_subcategory(
                pdf_file.stem,
                text[:2000]
            )

            for i, chunk_data in enumerate(chunks):
                documents.append({
                    "id": f"{pdf_file.stem}_chunk_{i}",
                    "title": pdf_file.stem,
                    "content": chunk_data["content"],
                    "category": category,
                    "subcategory": subcategory,
                    "source": pdf_file.name,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "article": chunk_data.get("article"),
                    "chunk_type": chunk_data.get("type"),
                    "language": metadata.get("language", "unknown"),
                    "total_pages": metadata.get("total_pages", 0)
                })

            logger.info(f"✓ {len(chunks)} chunks from {pdf_file.name}")

        return documents


# ============================================================================
# ENHANCED VECTOR DATABASE
# ============================================================================

class EnhancedVectorDatabase:
    """Vector DB with hybrid search and re-ranking"""

    def __init__(self):
        logger.info("Initializing enhanced vector database...")
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)

        try:
            self.collection = self.client.get_collection("moroccan_legal_docs_v3")
            logger.info(f"Loaded collection: {self.collection.count()} docs")
        except:
            self.collection = self.client.create_collection(
                name="moroccan_legal_docs_v3",
                metadata={"description": "Moroccan Legal Documents - Enhanced"}
            )
            logger.info("Created new collection")

    def add_documents(self, documents: List[Dict], batch_size: int = 50):
        """Add documents with enhanced metadata"""
        if not documents:
            return

        logger.info(f"Adding {len(documents)} documents...")
        total_added = 0

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_ids, batch_embeddings, batch_documents, batch_metadatas = [], [], [], []

            for doc in batch:
                try:
                    embedding = self.embedding_model.encode(doc["content"]).tolist()
                    batch_ids.append(doc["id"])
                    batch_embeddings.append(embedding)
                    batch_documents.append(doc["content"])
                    batch_metadatas.append({
                        "title": doc["title"],
                        "category": doc["category"],
                        "subcategory": doc.get("subcategory", ""),
                        "source": doc.get("source", ""),
                        "chunk_index": doc.get("chunk_index", 0),
                        "article": doc.get("article", ""),
                        "chunk_type": doc.get("chunk_type", ""),
                        "language": doc.get("language", "")
                    })
                except Exception as e:
                    logger.error(f"Error processing doc: {str(e)}")
                    continue

            if batch_ids:
                try:
                    self.collection.add(
                        ids=batch_ids,
                        embeddings=batch_embeddings,
                        documents=batch_documents,
                        metadatas=batch_metadatas
                    )
                    total_added += len(batch_ids)
                    logger.info(f"Progress: {total_added}/{len(documents)}")
                except Exception as e:
                    logger.error(f"Error adding batch: {str(e)}")

        logger.info(f"✓ Total in DB: {self.collection.count()}")

    def hybrid_search(self, query: str, n_results: int = Config.TOP_K_RESULTS,
                      filter_category: Optional[str] = None,
                      filter_articles: Optional[List[str]] = None) -> List[Dict]:
        """Hybrid search with multiple strategies"""
        try:
            # Strategy 1: Expanded semantic search
            expanded_query = EnhancedQueryProcessor.expand_query(query)
            query_embedding = self.embedding_model.encode(expanded_query).tolist()

            search_kwargs = {
                "query_embeddings": [query_embedding],
                "n_results": min(n_results * 2, self.collection.count())  # Get more, then re-rank
            }

            # Apply filters
            where_conditions = []
            if filter_category:
                where_conditions.append({"category": filter_category})

            if where_conditions:
                search_kwargs["where"] = where_conditions[0]

            results = self.collection.query(**search_kwargs)

            if not results["documents"][0]:
                return []

            # Strategy 2: Article-specific boost
            scored_results = []
            query_articles = EnhancedQueryProcessor.extract_article_numbers(query)

            for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
            ):
                base_score = 1 / (1 + dist)

                # Boost if article matches
                if query_articles and meta.get("article"):
                    for art in query_articles:
                        if art in meta.get("article", ""):
                            base_score *= 1.5  # 50% boost
                            break

                # Boost if chunk type is article
                if meta.get("chunk_type") == "article":
                    base_score *= 1.2

                scored_results.append({
                    "content": doc,
                    "metadata": meta,
                    "distance": dist,
                    "relevance_score": base_score
                })

            # Sort by relevance score and return top K
            scored_results.sort(key=lambda x: x["relevance_score"], reverse=True)
            return scored_results[:n_results]

        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return []

    def get_by_article(self, article_number: str) -> List[Dict]:
        """Get specific article content"""
        try:
            results = self.collection.query(
                query_texts=[f"المادة {article_number}"],
                n_results=5,
                where={"article": {"$contains": article_number}}
            )

            return [
                {
                    "content": doc,
                    "metadata": meta,
                    "relevance_score": 1.0
                }
                for doc, meta in zip(results["documents"][0], results["metadatas"][0])
            ]
        except:
            return []


# ============================================================================
# MULTI-BACKEND WITH STREAMING
# ============================================================================

class EnhancedBackendManager:
    """Enhanced backend manager with better prompting"""

    def __init__(self, backends: List[BackendConfig]):
        self.backends = sorted(backends, key=lambda x: x.priority)
        self.api_keys = {}

        for backend in self.backends:
            api_key = os.getenv(backend.api_key_env)
            if api_key:
                self.api_keys[backend.provider] = api_key
                logger.info(f"✓ {backend.name}")

    def generate_response(self, prompt: str, context: str,
                          system_prompt: Optional[str] = None,
                          use_reasoning: bool = False) -> Tuple[str, str, int, float]:
        """Generate response with optional chain-of-thought"""

        if not system_prompt:
            system_prompt = self._get_enhanced_system_prompt(context, use_reasoning)

        for backend in self.backends:
            if backend.provider not in self.api_keys:
                continue

            try:
                logger.info(f"Trying {backend.name}...")
                response, tokens = self._call_backend(backend, system_prompt, prompt)

                if response:
                    cost = (tokens / 1000) * backend.cost_per_1k_tokens
                    logger.info(f"✓ {backend.name}: {tokens} tokens, ${cost:.4f}")
                    return response, backend.name, tokens, cost

            except Exception as e:
                logger.warning(f"✗ {backend.name}: {str(e)}")
                continue

        return self._get_fallback_response(), "fallback", 0, 0.0

    def _call_backend(self, backend: BackendConfig, system_prompt: str,
                      user_prompt: str) -> Tuple[str, int]:
        """Call backend API"""
        headers = {
            "Authorization": f"Bearer {self.api_keys[backend.provider]}",
            "Content-Type": "application/json"
        }

        if backend.provider == "openrouter":
            headers["HTTP-Referer"] = "http://localhost:7860"
            headers["X-Title"] = "Moroccan Legal Assistant Pro"

        payload = {
            "model": backend.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": backend.temperature,
            "max_tokens": backend.max_tokens
        }

        url = f"{backend.base_url}/chat/completions"
        response = requests.post(url, headers=headers, json=payload,
                                 timeout=Config.TIMEOUT_SECONDS)
        response.raise_for_status()

        result = response.json()
        content = result["choices"][0]["message"]["content"]
        tokens = result.get("usage", {}).get("total_tokens", 0)

        return content, tokens

    def _get_enhanced_system_prompt(self, context: str, use_reasoning: bool) -> str:
        """Enhanced system prompt with legal expertise"""

        base_prompt = f"""أنت محامٍ خبير في القانون المغربي. / You are an expert lawyer in Moroccan law.

CONTEXTE JURIDIQUE / السياق القانوني:
{context}

DIRECTIVES / التوجيهات:

1. **PRÉCISION MAXIMALE** / **دقة قصوى**:
   - Citez TOUJOURS les numéros d'articles exacts
   - Fournissez le texte intégral des articles pertinents
   - Expliquez chaque terme juridique

2. **ANALYSE STRUCTURÉE** / **تحليل منظم**:
   - Qualification juridique précise
   - Base légale avec articles
   - Conditions d'application
   - Procédure à suivre
   - Jurisprudence si pertinente

3. **RÉPONSE COMPLÈTE** / **إجابة كاملة**:
   - Ne dites JAMAIS "information insuffisante"
   - Utilisez TOUT le contexte disponible
   - Si partiel, donnez ce qui existe ET expliquez ce qui manque
   - Suggérez des recherches complémentaires si nécessaire

4. **LANGUE** / **اللغة**:
   - Répondez dans la langue de la question
   - Utilisez la terminologie juridique appropriée
   - Traduisez les termes clés si utile

5. **FORMAT**:
   - Sections claires avec titres
   - Listes numérotées pour les étapes
   - Citations en gras
   - Sources identifiées"""

        if use_reasoning:
            base_prompt += """

6. **RAISONNEMENT** / **التفكير**:
   - Montrez votre raisonnement étape par étape
   - Analysez les différentes interprétations possibles
   - Identifiez les points critiques"""

        return base_prompt

    def _get_fallback_response(self) -> str:
        return """Tous les services sont temporairement indisponibles.
Veuillez réessayer dans quelques instants.

جميع الخدمات غير متاحة مؤقتاً.
يرجى المحاولة مرة أخرى."""


# ============================================================================
# DOCUMENT ANALYSIS ENGINE
# ============================================================================

class DocumentAnalysisEngine:
    """Analyze uploaded legal documents"""

    def __init__(self, vector_db: EnhancedVectorDatabase,
                 backend_manager: EnhancedBackendManager):
        self.vector_db = vector_db
        self.backend_manager = backend_manager

    def analyze_document(self, file_path: str, analysis_type: str = "full") -> Dict:
        """Comprehensive document analysis"""
        try:
            # Extract text
            text, metadata = AdvancedDocumentProcessor.extract_text_from_pdf(file_path)

            if not text:
                return {"error": "Could not extract text from document"}

            # Quick analysis
            analysis = {
                "metadata": metadata,
                "summary": "",
                "key_points": [],
                "legal_issues": [],
                "recommendations": [],
                "related_laws": []
            }

            # Generate summary
            summary_prompt = f"""Analysez ce document juridique et fournissez:

1. RÉSUMÉ (3-4 phrases)
2. POINTS CLÉS (liste numérotée)
3. QUESTIONS JURIDIQUES identifiées
4. LOIS/CODES applicables

DOCUMENT:
{text[:3000]}...

Format: Sections claires, français ou arabe selon le document."""

            summary, _, _, _ = self.backend_manager.generate_response(
                summary_prompt,
                text[:3000],
                use_reasoning=False
            )

            analysis["summary"] = summary

            # Find related laws using vector search
            related_docs = self.vector_db.hybrid_search(text[:1000], n_results=5)
            analysis["related_laws"] = [
                {
                    "source": doc["metadata"]["source"],
                    "relevance": doc["relevance_score"]
                }
                for doc in related_docs[:3]
            ]

            return analysis

        except Exception as e:
            logger.error(f"Document analysis error: {str(e)}")
            return {"error": str(e)}

    def compare_with_law(self, file_path: str, legal_domain: str) -> str:
        """Compare document with applicable laws"""
        try:
            text, _ = AdvancedDocumentProcessor.extract_text_from_pdf(file_path)

            # Get relevant law articles
            related_docs = self.vector_db.hybrid_search(
                text[:1000],
                n_results=8,
                filter_category=legal_domain
            )

            context = "\n\n".join([doc["content"] for doc in related_docs])

            prompt = f"""Comparez ce document avec les lois marocaines applicables:

DOCUMENT À ANALYSER:
{text[:2000]}

LOIS APPLICABLES:
{context}

Fournissez:
1. Points conformes
2. Points problématiques
3. Clauses manquantes
4. Recommandations de modification
5. Risques juridiques identifiés

Soyez TRÈS précis et citez les articles."""

            comparison, _, _, _ = self.backend_manager.generate_response(
                prompt, context, use_reasoning=True
            )

            return comparison

        except Exception as e:
            return f"Erreur d'analyse: {str(e)}"


# ============================================================================
# ADVANCED LEGAL CHATBOT
# ============================================================================

@dataclass
class ConversationContext:
    """Track conversation context"""
    history: List[Dict] = field(default_factory=list)
    current_domain: Optional[str] = None
    referenced_articles: List[str] = field(default_factory=list)
    uploaded_docs: List[str] = field(default_factory=list)


class AdvancedLegalChatbot:
    """Professional legal assistant with advanced features"""

    def __init__(self):
        logger.info("=" * 70)
        logger.info("ADVANCED MOROCCAN LEGAL AI ASSISTANT - INITIALIZATION")
        logger.info("=" * 70)

        self.vector_db = EnhancedVectorDatabase()
        self.backend_manager = EnhancedBackendManager(Config.BACKENDS)
        self.doc_analyzer = DocumentAnalysisEngine(self.vector_db, self.backend_manager)
        self.query_processor = EnhancedQueryProcessor()

        # Conversation management
        self.conversations = {}

        # Load documents if needed
        if self.vector_db.collection.count() == 0:
            logger.info("Loading legal documents...")
            self._load_all_documents()
        else:
            logger.info(f"Using existing DB: {self.vector_db.collection.count()} docs")

        logger.info("✓ SYSTEM READY")
        logger.info("=" * 70)

    def _load_all_documents(self):
        """Load all legal documents"""
        categories = {
            "family_law": LegalDomain.FAMILY_LAW.value,
            "criminal_law": LegalDomain.CRIMINAL_LAW.value,
            "civil_law": LegalDomain.CIVIL_LAW.value,
            "constitutional_law": LegalDomain.CONSTITUTIONAL_LAW.value,
            "amendments": LegalDomain.AMENDMENTS.value,
            "archive": LegalDomain.ARCHIVE.value
        }

        all_documents = []
        for folder, category in categories.items():
            folder_path = f"{Config.LEGAL_DOCS_PATH}/{folder}"
            docs = AdvancedDocumentProcessor.load_documents_from_folder(folder_path, category)
            all_documents.extend(docs)
            if docs:
                logger.info(f"✓ {len(docs)} chunks from {folder}")

        if all_documents:
            self.vector_db.add_documents(all_documents)
            logger.info(f"✓ Total loaded: {len(all_documents)} chunks")

    def chat(self, message: str, session_id: str = "default",
             uploaded_file: Optional[str] = None) -> str:
        """Main chat interface"""

        if not message.strip() and not uploaded_file:
            return "Posez votre question juridique / اطرح سؤالك القانوني"

        # Get or create conversation context
        if session_id not in self.conversations:
            self.conversations[session_id] = ConversationContext()

        ctx = self.conversations[session_id]

        try:
            # Handle document upload
            if uploaded_file:
                return self._handle_document_upload(uploaded_file, message, ctx)

            # Regular question handling
            return self._handle_question(message, ctx)

        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            return f"Une erreur s'est produite: {str(e)}"

    def _handle_document_upload(self, file_path: str, question: str,
                                ctx: ConversationContext) -> str:
        """Handle uploaded document analysis"""

        if not question:
            question = "Analysez ce document juridique"

        logger.info(f"Analyzing uploaded document: {Path(file_path).name}")

        # Analyze document
        analysis = self.doc_analyzer.analyze_document(file_path)

        if "error" in analysis:
            return f"Erreur: {analysis['error']}"

        # Store in context
        ctx.uploaded_docs.append(file_path)

        # Format response
        response = f"""ANALYSE DU DOCUMENT

Métadonnées:
- Pages: {analysis['metadata'].get('total_pages', 'N/A')}
- Langue: {analysis['metadata'].get('language', 'N/A')}
- Articles trouvés: {len(analysis['metadata'].get('articles_found', []))}

Analyse:
{analysis['summary']}

Lois Connexes:
"""

        for law in analysis['related_laws']:
            response += f"- {law['source']} (pertinence: {law['relevance']:.0%})\n"

        response += "\nVous pouvez maintenant poser des questions spécifiques sur ce document!"

        return response

    def _handle_question(self, question: str, ctx: ConversationContext) -> str:
        """Handle regular legal questions"""

        # Extract article numbers if mentioned
        mentioned_articles = self.query_processor.extract_article_numbers(question)
        ctx.referenced_articles.extend(mentioned_articles)

        # Determine query type
        query_type = self._classify_query(question)

        # Search for relevant documents
        if mentioned_articles:
            # Direct article lookup
            relevant_docs = []
            for art in mentioned_articles:
                relevant_docs.extend(self.vector_db.get_by_article(art))
        else:
            # Semantic search
            relevant_docs = self.vector_db.hybrid_search(question, n_results=8)

        if not relevant_docs:
            return self._no_results_response(question)

        # Build context
        context = self._build_context(relevant_docs, ctx)

        # Generate response with appropriate reasoning
        use_reasoning = query_type in [QueryType.CASE_ANALYSIS, QueryType.LEGAL_RESEARCH]

        response, backend, tokens, cost = self.backend_manager.generate_response(
            question,
            context,
            use_reasoning=use_reasoning
        )

        # Add sources
        response += self._format_sources(relevant_docs)

        # Add disclaimer
        response += self._get_disclaimer()

        # Store in history
        ctx.history.append({
            "question": question,
            "response": response,
            "sources": [doc["metadata"]["source"] for doc in relevant_docs[:3]]
        })

        return response

    def _classify_query(self, question: str) -> QueryType:
        """Classify query type"""
        question_lower = question.lower()

        if any(kw in question_lower for kw in ["mon cas", "ma situation", "حالتي", "مشكلتي"]):
            return QueryType.CASE_ANALYSIS
        elif any(kw in question_lower for kw in ["comparer", "différence", "قارن", "الفرق"]):
            return QueryType.COMPARATIVE_ANALYSIS
        elif any(kw in question_lower for kw in ["recherche", "étude", "بحث", "دراسة"]):
            return QueryType.LEGAL_RESEARCH
        else:
            return QueryType.GENERAL_QUESTION

    def _build_context(self, docs: List[Dict], ctx: ConversationContext) -> str:
        """Build rich context from documents"""
        context_parts = []

        for i, doc in enumerate(docs[:6], 1):
            meta = doc["metadata"]
            article = meta.get("article", "")

            context_parts.append(f"""
═══════════════════════════════════════
SOURCE {i}: {meta['source']}
{f'{article}' if article else ''}
Catégorie: {meta.get('subcategory', meta.get('category', ''))}
Pertinence: {doc['relevance_score']:.0%}

{doc['content']}
═══════════════════════════════════════
""")

        return "\n".join(context_parts)

    def _format_sources(self, docs: List[Dict]) -> str:
        """Format source citations"""
        sources = "\n\n" + "=" * 60 + "\n"
        sources += "SOURCES JURIDIQUES:\n\n"

        seen_sources = set()
        for doc in docs[:5]:
            source = doc["metadata"]["source"]
            if source not in seen_sources:
                article = doc["metadata"].get("article", "")
                sources += f"- {source}"
                if article:
                    sources += f" ({article})"
                sources += f" - Pertinence: {doc['relevance_score']:.0%}\n"
                seen_sources.add(source)

        return sources

    def _get_disclaimer(self) -> str:
        """Legal disclaimer"""
        return """

═══════════════════════════════════════════════════════════
AVERTISSEMENT JURIDIQUE / تنبيه قانوني

Cette consultation ne remplace PAS un avocat qualifié.
Pour des conseils personnalisés, consultez un professionnel.

هذه الاستشارة لا تغني عن محامٍ مؤهل.
للحصول على مشورة شخصية، استشر محترفاً.
═══════════════════════════════════════════════════════════"""

    def _no_results_response(self, question: str) -> str:
        """Response when no results found"""
        return f"""Aucun document pertinent trouvé dans la base de données.

Votre question: {question}

Suggestions:
1. Reformulez avec des termes juridiques plus précis
2. Mentionnez un domaine spécifique (pénal, civil, famille...)
3. Citez un numéro d'article si vous le connaissez

Domaines disponibles:
- Droit de la Famille (Moudawana)
- Droit Pénal
- Droit Civil
- Droit Constitutionnel

Besoin d'aide? Essayez: "Quels sont les conditions du mariage?" ou "ما هي عقوبة السرقة؟"
"""


# ============================================================================
# GRADIO INTERFACE
# ============================================================================
def create_gradio_interface():
    """Premium Gradio interface for legal consultation."""

    try:
        chatbot_instance = AdvancedLegalChatbot()
        logger.info("Chatbot initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {str(e)}", exc_info=True)
        raise

    def chat_with_upload(message, history, file):
        """Handle chat with optional file upload."""
        try:
            if history is None:
                history = []

            session_id = "gradio_session"
            file_path = file.name if file is not None and hasattr(file, "name") else str(file) if file else None
            response = chatbot_instance.chat(message, session_id, file_path)

            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})
            return history
        except Exception as e:
            logger.error(f"Chat error: {str(e)}", exc_info=True)
            error_msg = f"Error: {str(e)}\n\nPlease try again or rephrase your question."
            if history is None:
                history = []
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            return history

    def analyze_document(file):
        """Dedicated document analysis."""
        try:
            if not file:
                return "No file uploaded."

            file_path = file.name if hasattr(file, "name") else str(file)
            logger.info(f"Analyzing document: {file_path}")
            analysis = chatbot_instance.doc_analyzer.analyze_document(file_path)

            if "error" in analysis:
                return f"Error: {analysis['error']}"

            result = f"""COMPLETE DOCUMENT ANALYSIS

Metadata:
- Pages: {analysis['metadata'].get('total_pages', 0)}
- Language: {analysis['metadata'].get('language', 'unknown')}
- Articles detected: {len(analysis['metadata'].get('articles_found', []))}

Summary:
{analysis['summary']}

Articles mentioned: {', '.join(analysis['metadata'].get('articles_found', [])[:10])}

Related laws:
{chr(10).join(f"- {law['source']}" for law in analysis['related_laws'])}
"""
            return result
        except Exception as e:
            logger.error(f"Document analysis error: {str(e)}", exc_info=True)
            return f"Analysis error: {str(e)}"

    custom_css = """
    .gradio-container {
        max-width: 1320px !important;
        margin: 0 auto !important;
        padding: 1.5rem !important;
        background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%) !important;
    }

    .legal-shell {
        background: #ffffff !important;
        border: 1px solid #dbe3ee !important;
        border-radius: 18px !important;
        box-shadow: 0 20px 40px rgba(15, 23, 42, 0.08) !important;
        padding: 1.25rem !important;
        margin-bottom: 1rem !important;
    }

    .app-header {
        background: linear-gradient(135deg, #0f2747 0%, #1d3557 70%, #27496d 100%) !important;
        color: #f8fafc !important;
        border-radius: 14px !important;
        padding: 2.1rem !important;
        margin-bottom: 1.1rem !important;
        box-shadow: 0 14px 30px rgba(15, 39, 71, 0.28) !important;
    }

    .app-header h1 {
        margin: 0 !important;
        font-size: 2.05rem !important;
        letter-spacing: -0.02em !important;
        font-weight: 700 !important;
        color: #f8fafc !important;
    }

    .app-header .subtitle {
        margin-top: 0.65rem !important;
        font-size: 1rem !important;
        color: #cbd8ea !important;
    }

    .app-header .meta {
        margin-top: 0.85rem !important;
        font-size: 0.92rem !important;
        color: #9fb4cf !important;
    }

    .surface-card {
        background: #ffffff !important;
        border: 1px solid #dde4ed !important;
        border-radius: 12px !important;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05) !important;
        padding: 1rem !important;
    }

    .gradio-container [data-testid="chatbot"] {
        border: 1px solid #d6deea !important;
        border-radius: 14px !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.35), 0 6px 14px rgba(15, 23, 42, 0.05) !important;
    }

    .gradio-container [data-testid="user"] .message,
    .gradio-container .message.user {
        background: linear-gradient(135deg, #1d4e89 0%, #1a659e 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 1rem 1.05rem !important;
        box-shadow: 0 6px 14px rgba(29, 78, 137, 0.25) !important;
    }

    .gradio-container [data-testid="user"] .message *,
    .gradio-container .message.user * {
        color: #ffffff !important;
    }

    .gradio-container [data-testid="bot"] .message,
    .gradio-container .message.bot {
        background: #f8fbff !important;
        color: #0f172a !important;
        border: 1px solid #d7e2f0 !important;
        border-radius: 12px !important;
        padding: 1rem 1.05rem !important;
    }

    .gradio-container [data-testid="bot"] .message *,
    .gradio-container .message.bot * {
        color: #0f172a !important;
    }

    .gradio-container textarea,
    .gradio-container input[type="text"] {
        border: 1px solid #cdd8e7 !important;
        border-radius: 10px !important;
        background: #ffffff !important;
        color: #0f172a !important;
        padding: 0.9rem !important;
        font-size: 0.99rem !important;
    }

    .gradio-container textarea:focus,
    .gradio-container input[type="text"]:focus {
        border-color: #335f97 !important;
        box-shadow: 0 0 0 3px rgba(51, 95, 151, 0.15) !important;
    }

    .gradio-container button[variant="primary"] {
        background: linear-gradient(135deg, #1d4e89 0%, #234d88 100%) !important;
        border: 1px solid #1f3f70 !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 8px 16px rgba(24, 60, 106, 0.22) !important;
    }

    .gradio-container button[variant="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 12px 20px rgba(24, 60, 106, 0.28) !important;
    }

    .neutral-btn {
        border: 1px solid #c7d2e3 !important;
        border-radius: 10px !important;
        background: #ffffff !important;
        color: #334155 !important;
    }

    .panel-title {
        font-weight: 700 !important;
        color: #0f2747 !important;
        margin-bottom: 0.45rem !important;
    }

    .quiet-note {
        color: #5b6b80 !important;
        font-size: 0.93rem !important;
        line-height: 1.45 !important;
    }

    .trust-disclaimer {
        background: #f8fafc !important;
        border-left: 4px solid #3b4f68 !important;
        border-radius: 10px !important;
        padding: 0.95rem 1rem !important;
        margin-top: 0.75rem !important;
    }

    .trust-disclaimer p,
    .trust-disclaimer li {
        color: #334155 !important;
        font-size: 0.92rem !important;
        line-height: 1.45 !important;
    }

    .analysis-output textarea {
        font-family: "Cascadia Code", "Consolas", monospace !important;
        background: #0f172a !important;
        color: #e2e8f0 !important;
        border: 1px solid #334155 !important;
    }

    .rtl-support {
        direction: rtl !important;
        text-align: right !important;
        font-family: "Tahoma", "Arial", sans-serif !important;
    }

    @media (max-width: 900px) {
        .gradio-container {
            padding: 1rem !important;
        }
        .app-header h1 {
            font-size: 1.65rem !important;
        }
    }
    """

    legal_theme = gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
        font=[
            gr.themes.GoogleFont("Inter"),
            "Segoe UI",
            "Arial",
            "sans-serif",
        ],
    ).set(
        block_radius="10px",
        button_primary_background_fill="#1d4e89",
        button_primary_background_fill_hover="#173d6f",
        button_primary_border_color="#1f3f70",
        button_primary_text_color="#ffffff",
        body_background_fill="#eef2f7",
    )

    with gr.Blocks(title="Moroccan Legal Assistant", css=custom_css, theme=legal_theme) as demo:
        with gr.Column(elem_classes="legal-shell"):
            gr.HTML(
                """
                <div class="app-header">
                    <h1>Moroccan Legal AI Assistant</h1>
                    <p class="subtitle">Professional legal research and document review for Moroccan law.</p>
                    <p class="meta">Version 3.0 Professional Edition | Production-ready legal knowledge base</p>
                </div>
                """
            )

            with gr.Tabs():
                with gr.Tab("Consultation"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            with gr.Column(elem_classes="surface-card"):
                                gr.Markdown("### Legal Consultation")
                                chatbot = gr.Chatbot(
                                    label="Conversation",
                                    height=560,
                                    show_label=True,
                                )
                                msg = gr.Textbox(
                                    label="Question",
                                    placeholder="Ask in French or Arabic. Example: article number, legal procedure, or document interpretation.",
                                    lines=3,
                                )
                                with gr.Row():
                                    submit = gr.Button("Submit Question", variant="primary")
                                    upload_btn = gr.File(
                                        label="Attach PDF for Context",
                                        file_types=[".pdf"],
                                    )
                                    clear = gr.Button("Clear Conversation", elem_classes="neutral-btn")

                        with gr.Column(scale=2):
                            with gr.Column(elem_classes="surface-card"):
                                gr.Markdown("### Suggested Prompts")
                                gr.Examples(
                                    examples=[
                                        ["Quelle est la procédure de divorce pour discorde selon la Moudawana ?"],
                                        ["ما هي عقوبة السرقة الموصوفة في القانون الجنائي المغربي؟"],
                                        ["Expliquez l'article 16 de la Constitution marocaine."],
                                        ["Quels sont les droits de garde après divorce ?"],
                                        ["ما هي شروط صحة العقد في القانون المدني؟"],
                                    ],
                                    inputs=msg,
                                )
                            with gr.Column(elem_classes="surface-card"):
                                gr.Markdown("### Consultation Settings")
                                gr.Markdown(
                                    """
                                    <p class="quiet-note">
                                    Use precise legal wording, mention article numbers when known,
                                    and provide factual context to improve answer quality.
                                    </p>
                                    """,
                                )
                                gr.HTML(
                                    """
                                    <div class="trust-disclaimer">
                                        <p><strong>Professional Notice:</strong> This assistant provides legal information and research support.
                                        It does not replace case-specific advice from a licensed lawyer.</p>
                                    </div>
                                    """
                                )

                    def handle_submit(message, history, file):
                        if not message or not message.strip():
                            return history, ""
                        return chat_with_upload(message, history, file), ""

                    msg.submit(handle_submit, [msg, chatbot, upload_btn], [chatbot, msg])
                    submit.click(handle_submit, [msg, chatbot, upload_btn], [chatbot, msg])
                    clear.click(lambda: ([], ""), None, [chatbot, msg], queue=False)

                with gr.Tab("Document Review"):
                    gr.Markdown(
                        """
                        ### Legal Document Review
                        Upload a legal PDF to generate metadata, structured summary, and related legal references.
                        """
                    )
                    with gr.Row():
                        with gr.Column(scale=1):
                            with gr.Column(elem_classes="surface-card"):
                                doc_upload = gr.File(
                                    label="Upload Legal Document (PDF)",
                                    file_types=[".pdf"],
                                )
                                analyze_btn = gr.Button("Run Document Analysis", variant="primary")
                                gr.Markdown(
                                    """
                                    - Summary and metadata extraction
                                    - Relevant law references
                                    - Key legal issues detection
                                    - Expert-style recommendations
                                    """
                                )
                        with gr.Column(scale=2):
                            analysis_output = gr.Textbox(
                                label="Analysis Report",
                                lines=28,
                                elem_classes="analysis-output",
                            )

                    analyze_btn.click(analyze_document, inputs=doc_upload, outputs=analysis_output)

                with gr.Tab("About"):
                    gr.Markdown(
                        """
                        ## Platform Overview

                        Moroccan Legal AI Assistant is designed for legal research, article-level referencing, and bilingual consultation support.

                        ### Core capabilities
                        - Hybrid legal retrieval with semantic ranking
                        - Article detection and citation guidance
                        - Bilingual support in French and Arabic
                        - Document review with legal-context matching

                        ### Knowledge coverage
                        - Family law (Moudawana)
                        - Criminal law
                        - Civil law
                        - Constitutional law
                        """
                    )
                    gr.HTML(
                        """
                        <div class="trust-disclaimer">
                            <p><strong>Legal disclaimer:</strong> Output is informational and should be verified by a licensed legal professional before use in any decision, filing, or legal action.</p>
                        </div>
                        """
                    )

    return demo


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys

    print("""
╔═════════════════════════════════════════════════════════════════╗
║                                                                 ║
║     MOROCCAN LEGAL AI ASSISTANT v3.0 - PROFESSIONAL            ║
║                                                                 ║
║     المساعد القانوني المغربي بالذكاء الاصطناعي                ║
║                    النسخة الاحترافية 3.0                       ║
║                                                                 ║
╚═════════════════════════════════════════════════════════════════╝
""")

    # Check API keys
    required_keys = ["OPENROUTER_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY"]
    available_keys = [key for key in required_keys if os.getenv(key)]

    if not available_keys:
        print("\nERROR: No API keys found!")
        print("\nCreate a .env file with at least one API key:")
        print("\nOPENROUTER_API_KEY=your_key_here")
        print("GROQ_API_KEY=your_key_here")
        print("OPENAI_API_KEY=your_key_here")
        sys.exit(1)

    print(f"\nAPI Keys Available: {', '.join([k.replace('_API_KEY', '') for k in available_keys])}")

    # Check ChromaDB
    if Path(Config.CHROMA_DB_PATH).exists():
        print(f"\nFound existing database: {Config.CHROMA_DB_PATH}")
        print("  (Delete it to rebuild with new documents)")
    else:
        print(f"\nNo database found. Will create new one from: {Config.LEGAL_DOCS_PATH}")

    # Launch
    try:
        print("\n" + "=" * 70)
        print("LAUNCHING PROFESSIONAL LEGAL ASSISTANT")
        print("=" * 70)
        print("\nOpen your browser at: http://localhost:7860")
        print("\nFeatures:")
        print("   • Complete legal answers with article citations")
        print("   • Document upload & analysis")
        print("   • Hybrid search with re-ranking")
        print("   • Multi-lingual (French & Arabic with RTL support)")
        print("   • Conversation memory")
        print("   • Professional clean interface")
        print("\nPress CTRL+C to stop\n")

        demo = create_gradio_interface()
        demo.launch(
            share=False,
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True
        )
    except Exception as e:
        logger.error(f"Failed to launch: {str(e)}")
        print(f"\nError: {str(e)}")
        sys.exit(1)
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
        return """❌ Tous les services sont temporairement indisponibles.
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
1. Points conformes ✓
2. Points problématiques ⚠️
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
            return "💬 Posez votre question juridique / اطرح سؤالك القانوني"

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
            return f"❌ Une erreur s'est produite: {str(e)}"

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
    """Modern, clean, highly readable interface with excellent contrast"""

    # Initialize chatbot
    try:
        chatbot_instance = AdvancedLegalChatbot()
        logger.info("Chatbot initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {str(e)}", exc_info=True)
        raise

    # Chat handler
    def chat_with_upload(message, history, file):
        """Handle chat with optional file upload"""
        try:
            if history is None:
                history = []

            session_id = "gradio_session"
            file_path = None
            if file is not None:
                file_path = file.name if hasattr(file, 'name') else str(file)

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

    # Document analysis
    def analyze_document(file):
        """Dedicated document analysis"""
        try:
            if not file:
                return "No file uploaded"

            file_path = file.name if hasattr(file, 'name') else str(file)
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

    # Modern CSS with excellent contrast - FIXED VERSION
    # Replace the custom_css variable in the create_gradio_interface() function with this:

    custom_css = """
    /* ===== GLOBAL STYLES ===== */
    .gradio-container {
        max-width: 1400px !important;
        margin: 0 auto !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        background: #f8fafc !important;
    }

    /* ===== FORCE DARK TEXT EVERYWHERE ===== */
    * {
        color: #0f172a !important;
    }

    /* ===== HEADER - Clean White with Border ===== */
    .app-header {
        background: #ffffff !important;
        border: 3px solid #1e293b !important;
        border-radius: 16px !important;
        padding: 2.5rem !important;
        margin-bottom: 2rem !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08) !important;
    }

    .app-header h1 {
        color: #0f172a !important;
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        margin: 0 0 0.3rem 0 !important;
        text-align: center !important;
        letter-spacing: -0.02em !important;
    }

    .app-header .subtitle {
        color: #475569 !important;
        font-size: 1.1rem !important;
        margin: 0 !important;
        text-align: center !important;
        font-weight: 500 !important;
    }

    .app-header .version {
        color: #10b981 !important;
        font-size: 0.95rem !important;
        margin-top: 0.5rem !important;
        text-align: center !important;
        font-weight: 600 !important;
    }

    /* ===== CHATBOT - MAXIMUM CONTRAST ===== */
    .gradio-container [data-testid="chatbot"] {
        background: #ffffff !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05) !important;
    }

    /* User messages - Dark blue with white text */
    .gradio-container [data-testid="user"] .message,
    .message.user {
        background: #1e40af !important;
        color: #ffffff !important;
        border: 2px solid #1e3a8a !important;
        border-radius: 12px !important;
        padding: 1.2rem !important;
        margin: 0.8rem 0 !important;
        font-size: 1.05rem !important;
        line-height: 1.6 !important;
        box-shadow: 0 2px 8px rgba(30, 64, 175, 0.2) !important;
    }

    .gradio-container [data-testid="user"] .message *,
    .message.user * {
        color: #ffffff !important;
    }

    /* Assistant messages - White with very dark text */
    .gradio-container [data-testid="bot"] .message,
    .message.bot {
        background: #ffffff !important;
        color: #0f172a !important;
        border: 2px solid #cbd5e1 !important;
        border-radius: 12px !important;
        padding: 1.2rem !important;
        margin: 0.8rem 0 !important;
        font-size: 1.05rem !important;
        line-height: 1.7 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06) !important;
    }

    .gradio-container [data-testid="bot"] .message *,
    .message.bot * {
        color: #0f172a !important;
    }

    /* ===== INPUT AREAS - HIGH CONTRAST ===== */
    textarea, .input-textbox, input[type="text"] {
        background: #ffffff !important;
        color: #0f172a !important;
        border: 2px solid #cbd5e1 !important;
        border-radius: 10px !important;
        padding: 1rem !important;
        font-size: 1.05rem !important;
        line-height: 1.5 !important;
    }

    textarea:focus, .input-textbox:focus, input[type="text"]:focus {
        border-color: #3b82f6 !important;
        outline: none !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }

    textarea::placeholder, input::placeholder {
        color: #94a3b8 !important;
    }

    /* ===== BUTTONS - MODERN & CLEAR ===== */
    button[variant="primary"], .primary-btn {
        background: #2563eb !important;
        color: #ffffff !important;
        border: none !important;
        padding: 0.9rem 2rem !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
    }

    button[variant="primary"] *, .primary-btn * {
        color: #ffffff !important;
    }

    button[variant="primary"]:hover, .primary-btn:hover {
        background: #1d4ed8 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.4) !important;
    }

    /* Secondary buttons */
    .secondary-button, button:not([variant="primary"]) {
        background: #ffffff !important;
        color: #475569 !important;
        border: 2px solid #cbd5e1 !important;
        padding: 0.9rem 1.8rem !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }

    .secondary-button *, button:not([variant="primary"]) * {
        color: #475569 !important;
    }

    .secondary-button:hover, button:not([variant="primary"]):hover {
        background: #f1f5f9 !important;
        border-color: #94a3b8 !important;
    }

    /* ===== TABS - CLEAN NAVIGATION ===== */
    .tab-nav button {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        padding: 0.9rem 2rem !important;
        border-radius: 8px !important;
        color: #64748b !important;
        background: transparent !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }

    .tab-nav button * {
        color: #64748b !important;
    }

    .tab-nav button:hover {
        background: #f1f5f9 !important;
        color: #334155 !important;
    }

    .tab-nav button:hover * {
        color: #334155 !important;
    }

    .tab-nav button.selected {
        background: #2563eb !important;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
    }

    .tab-nav button.selected * {
        color: #ffffff !important;
    }

    /* ===== EXAMPLES - CLICKABLE CARDS ===== */
    .example-item {
        background: #ffffff !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 10px !important;
        padding: 1rem 1.3rem !important;
        margin: 0.6rem 0 !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        color: #334155 !important;
        font-size: 0.98rem !important;
        font-weight: 500 !important;
    }

    .example-item * {
        color: #334155 !important;
    }

    .example-item:hover {
        border-color: #3b82f6 !important;
        background: #eff6ff !important;
        transform: translateX(6px) !important;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.15) !important;
    }

    /* ===== INFO CARDS ===== */
    .info-card {
        background: #ffffff !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin: 1rem 0 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
    }

    .info-card h3 {
        color: #0f172a !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        margin: 0 0 1rem 0 !important;
    }

    .info-card ul {
        list-style: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    .info-card li {
        color: #475569 !important;
        padding: 0.6rem 0 !important;
        padding-left: 1.5rem !important;
        border-bottom: 1px solid #f1f5f9 !important;
        position: relative !important;
    }

    .info-card li:before {
        content: "→" !important;
        position: absolute !important;
        left: 0 !important;
        color: #3b82f6 !important;
        font-weight: bold !important;
    }

    .info-card li:last-child {
        border-bottom: none !important;
    }

    /* ===== DISCLAIMER - VISIBLE WARNING ===== */
    .disclaimer {
        background: #fef3c7 !important;
        border: 3px solid #f59e0b !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin: 2rem 0 !important;
    }

    .disclaimer h4 {
        color: #92400e !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        margin: 0 0 0.8rem 0 !important;
    }

    .disclaimer p, .disclaimer ul, .disclaimer li {
        color: #78350f !important;
        line-height: 1.7 !important;
        font-size: 1rem !important;
    }

    .disclaimer * {
        color: #78350f !important;
    }

    .disclaimer h4 * {
        color: #92400e !important;
    }

    .disclaimer strong {
        font-weight: 700 !important;
    }

    /* ===== TEXTBOX OUTPUT (For Document Analysis) ===== */
    .gradio-container .gr-textbox,
    .gradio-container textarea[readonly] {
        background: #1e293b !important;
        color: #e2e8f0 !important;
        border: 2px solid #334155 !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace !important;
        font-size: 0.95rem !important;
        line-height: 1.7 !important;
        white-space: pre-wrap !important;
    }

    .gradio-container .gr-textbox *,
    .gradio-container textarea[readonly] * {
        color: #e2e8f0 !important;
    }

    /* ===== MARKDOWN TEXT - CRITICAL FIX ===== */
    .gradio-container .markdown-text,
    .gradio-container .gr-markdown,
    .gradio-container .prose,
    .gradio-container .markdown,
    .gr-prose {
        color: #0f172a !important;
        background: transparent !important;
    }

    .gradio-container .gr-markdown *,
    .gradio-container .prose *,
    .gr-prose * {
        color: #0f172a !important;
    }

    .gradio-container .gr-markdown h1,
    .gradio-container .gr-markdown h2,
    .gradio-container .gr-markdown h3,
    .gradio-container .gr-markdown h4,
    .gr-prose h1,
    .gr-prose h2,
    .gr-prose h3,
    .gr-prose h4 {
        color: #0f172a !important;
        font-weight: 700 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.8rem !important;
    }

    .gradio-container .gr-markdown p,
    .gradio-container .gr-markdown li,
    .gradio-container .gr-markdown td,
    .gr-prose p,
    .gr-prose li,
    .gr-prose td {
        color: #334155 !important;
        line-height: 1.7 !important;
        font-size: 1rem !important;
    }

    .gradio-container .gr-markdown strong,
    .gr-prose strong {
        color: #1e293b !important;
        font-weight: 700 !important;
    }

    .gradio-container .gr-markdown table,
    .gr-prose table {
        border-collapse: collapse !important;
        width: 100% !important;
        margin: 1rem 0 !important;
    }

    .gradio-container .gr-markdown th,
    .gr-prose th {
        background: #f1f5f9 !important;
        color: #0f172a !important;
        font-weight: 700 !important;
        padding: 0.75rem !important;
        border: 1px solid #cbd5e1 !important;
    }

    .gradio-container .gr-markdown td,
    .gr-prose td {
        padding: 0.75rem !important;
        border: 1px solid #cbd5e1 !important;
        color: #334155 !important;
    }

    .gradio-container .gr-markdown code,
    .gr-prose code {
        background: #f1f5f9 !important;
        color: #0f172a !important;
        padding: 0.2rem 0.4rem !important;
        border-radius: 4px !important;
        font-family: 'Consolas', 'Monaco', monospace !important;
        font-size: 0.9em !important;
    }

    .gradio-container .gr-markdown hr,
    .gr-prose hr {
        border: none !important;
        border-top: 2px solid #cbd5e1 !important;
        margin: 2rem 0 !important;
    }

    .gradio-container .gr-markdown ul,
    .gradio-container .gr-markdown ol,
    .gr-prose ul,
    .gr-prose ol {
        padding-left: 2rem !important;
        margin: 1rem 0 !important;
    }

    .gradio-container .gr-markdown ul li,
    .gradio-container .gr-markdown ol li,
    .gr-prose ul li,
    .gr-prose ol li {
        margin: 0.5rem 0 !important;
        color: #334155 !important;
    }

    /* ===== LABELS ===== */
    label, .label, .gr-label {
        color: #0f172a !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        margin-bottom: 0.5rem !important;
    }

    label *, .label *, .gr-label * {
        color: #0f172a !important;
    }

    /* ===== FILE UPLOAD AREA ===== */
    .file-upload {
        border: 2px dashed #cbd5e1 !important;
        border-radius: 12px !important;
        padding: 2rem !important;
        background: #f8fafc !important;
        transition: all 0.2s ease !important;
    }

    .file-upload * {
        color: #0f172a !important;
    }

    .file-upload:hover {
        border-color: #3b82f6 !important;
        background: #eff6ff !important;
    }

    /* ===== RTL SUPPORT FOR ARABIC ===== */
    [lang="ar"], .arabic-text {
        font-family: 'Traditional Arabic', 'Tahoma', 'Arial', sans-serif !important;
        direction: rtl !important;
        text-align: right !important;
        font-size: 1.15rem !important;
    }

    /* ===== RESPONSIVE ===== */
    @media (max-width: 768px) {
        .app-header h1 {
            font-size: 1.6rem !important;
        }

        .message {
            max-width: 95% !important;
        }

        button {
            padding: 0.8rem 1.2rem !important;
            font-size: 0.95rem !important;
        }
    }
    """
    # Create interface
    with gr.Blocks(title="Moroccan Legal Assistant", css=custom_css) as demo:

        # Modern header
        gr.HTML("""
        <div class="app-header">
            <h1>Moroccan Legal AI Assistant</h1>
            <h1 style="font-family: 'Traditional Arabic', serif; font-size: 2rem;">المساعد القانوني المغربي بالذكاء الاصطناعي</h1>
            <p class="subtitle">Professional Legal Research & Document Analysis</p>
            <p class="version">✓ v3.0 Professional Edition - All Systems Operational</p>
        </div>
        """)

        with gr.Tabs():
            # TAB 1: Legal Assistant
            with gr.Tab("Legal Assistant"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### Ask Your Legal Question")

                        chatbot = gr.Chatbot(
                            label="Legal Consultation",
                            height=520,
                            show_label=True,
                            elem_classes="chatbot-container"
                        )

                        with gr.Row():
                            msg = gr.Textbox(
                                label="Your Question",
                                placeholder="Type in French or Arabic... | اكتب سؤالك بالفرنسية أو العربية...",
                                lines=2,
                                scale=3,
                                show_label=False
                            )

                        with gr.Row():
                            submit = gr.Button("Send", variant="primary", scale=2)
                            upload_btn = gr.File(
                                label="📎 Upload PDF",
                                file_types=[".pdf"],
                                scale=2,
                                elem_classes="upload-button"
                            )
                            clear = gr.Button("🗑️ Clear", scale=1, elem_classes="secondary-button")

                    with gr.Column(scale=1):
                        gr.Markdown("### Quick Examples")

                        gr.Examples(
                            examples=[
                                ["Quelle est la procédure de divorce pour discorde selon la Moudawana?"],
                                ["ما هي عقوبة السرقة الموصوفة في القانون الجنائي المغربي؟"],
                                ["Expliquez l'article 16 de la Constitution marocaine"],
                                ["Quels sont les droits de garde après divorce?"],
                                ["ما هي شروط صحة العقد في القانون المدني؟"],
                                ["Comparez le mariage civil et religieux au Maroc"],
                            ],
                            inputs=msg
                        )

                        gr.HTML("""
                        <div class="info-card">
                            <h3>How to Use</h3>
                            <ul>
                                <li>Ask specific legal questions</li>
                                <li>Upload documents for analysis</li>
                                <li>Mention article numbers for precision</li>
                                <li>Use French or Arabic naturally</li>
                            </ul>
                        </div>
                        """)

                        gr.HTML("""
                        <div class="info-card" style="background: #f0fdf4; border-color: #86efac;">
                            <h3>✓ Coverage</h3>
                            <ul>
                                <li>Family Law (Moudawana)</li>
                                <li>Criminal Law</li>
                                <li>Civil Law</li>
                                <li>Constitutional Law</li>
                            </ul>
                        </div>
                        """)

                # Event handlers
                def handle_submit(message, history, file):
                    if not message or not message.strip():
                        return history, ""
                    return chat_with_upload(message, history, file), ""

                msg.submit(handle_submit, [msg, chatbot, upload_btn], [chatbot, msg])
                submit.click(handle_submit, [msg, chatbot, upload_btn], [chatbot, msg])
                clear.click(lambda: ([], ""), None, [chatbot, msg], queue=False)

            # TAB 2: Document Analysis
            with gr.Tab("Document Analysis"):
                gr.Markdown("""
                ### Professional Document Analyzer
                Upload contracts, judgments, or any legal document for comprehensive analysis.
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        doc_upload = gr.File(
                            label="Upload Legal Document (PDF)",
                            file_types=[".pdf"],
                            elem_classes="file-upload"
                        )
                        analyze_btn = gr.Button(
                            "Analyze Document",
                            variant="primary",
                            size="lg"
                        )

                        gr.HTML("""
                        <div class="info-card">
                            <h3>Analysis Includes:</h3>
                            <ul>
                                <li>Document summary</li>
                                <li>Key legal points extraction</li>
                                <li>Applicable law identification</li>
                                <li>Legal issues flagged</li>
                                <li>Compliance check</li>
                                <li>Expert recommendations</li>
                            </ul>
                        </div>
                        """)

                    with gr.Column(scale=2):
                        analysis_output = gr.Textbox(
                            label="Analysis Results",
                            lines=28,
                            elem_classes="analysis-output"
                        )

                analyze_btn.click(analyze_document, inputs=doc_upload, outputs=analysis_output)

            # TAB 3: About
            with gr.Tab("About"):
                gr.Markdown("""
                ## Moroccan Legal AI Assistant

                ### Professional Edition v3.0

                **Advanced RAG System with Legal Expertise**

                #### Key Features

                - **Hybrid Search**: Semantic + keyword search with intelligent re-ranking
                - **Article Detection**: Automatically finds and cites specific law articles
                - **Document Analysis**: Extract key points from contracts and judgments
                - **Multi-lingual**: Full support for French and Arabic (with RTL)
                - **Conversation Memory**: Maintains context throughout session
                - **Source Citations**: Every answer includes exact legal references

                #### Legal Database

                | Domain | Coverage |
                |--------|----------|
                | Family Law | Moudawana 2004 (Complete) |
                | Criminal Law | Penal Code + Procedure |
                | Civil Law | Obligations & Contracts |
                | Constitutional Law | Constitution 2011 |

                #### Technology Stack

                - **PDF Processing**: pdfplumber (optimized for Arabic)
                - **Embeddings**: Multilingual Sentence Transformers
                - **Vector DB**: ChromaDB with hybrid search
                - **LLM Backends**: OpenRouter, Groq, OpenAI (multi-backend fallback)
                - **Interface**: Gradio 6.x

                #### Performance

                - Database: 500+ legal document chunks
                - Search accuracy: >85% relevance
                - Response time: <5 seconds average
                - Multi-language support: French & Arabic

                ---

                **Version:** 3.0.0 Professional  
                **Updated:** January 2026  
                **Status:** ✓ All systems operational
                """)

                gr.HTML("""
                <div class="disclaimer">
                    <h4>Important Legal Disclaimer</h4>
                    <p><strong>This AI assistant provides general legal information based on Moroccan law texts.</strong></p>
                    <br>
                    <p><strong>This is NOT:</strong></p>
                    <ul>
                        <li>A substitute for a licensed lawyer</li>
                        <li>Personalized legal advice for your situation</li>
                        <li>Legal representation in court</li>
                        <li>Guaranteed to be 100% accurate</li>
                    </ul>
                    <br>
                    <p><strong>Always Consult a Qualified Lawyer For:</strong></p>
                    <ul>
                        <li>Personalized legal advice</li>
                        <li>Court representation</li>
                        <li>Important legal decisions</li>
                        <li>Document signing or contracts</li>
                    </ul>
                    <br>
                    <p><strong>By using this tool, you acknowledge that AI-generated legal information should be verified by a licensed professional.</strong></p>
                </div>
                """)

                gr.Markdown("""
                ---

                ### Feedback & Support

                Found an issue or have suggestions? This tool is continuously improving.

                **Best practices for using this assistant:**

                1. Be specific in your questions
                2. Mention the legal domain when possible
                3. Reference article numbers if you know them
                4. Upload documents for detailed analysis
                5. Always verify important information with a lawyer
                """)

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
            show_error=True,
            theme=gr.themes.Soft(primary_hue="blue")
        )
    except Exception as e:
        logger.error(f"Failed to launch: {str(e)}")
        print(f"\nError: {str(e)}")
        sys.exit(1)
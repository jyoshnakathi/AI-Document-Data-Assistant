# ============================================================
# AI DOCUMENT & DATA ASSISTANT
# HIGH-RELIABILITY RAG + DATA ANALYTICS
# ============================================================

import io
import re
import json
import hashlib

from collections import defaultdict
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.documents import Document


# ============================================================
# OPTIONAL OCR IMPORTS
# ============================================================

try:
    import fitz
    from PIL import Image
    import pytesseract

    OCR_AVAILABLE = True

except Exception:

    OCR_AVAILABLE = False


# ============================================================
# OPTIONAL BM25 IMPORT
# ============================================================

try:
    from rank_bm25 import BM25Okapi

    BM25_AVAILABLE = True

except Exception:

    BM25_AVAILABLE = False


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Document & Data Assistant",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# APP HEADER
# ============================================================

st.title("🤖 AI Document & Data Assistant")

st.caption(
    "Chat with PDFs • Analyze CSV/Excel • "
    "Evidence-grounded answers • Source-page verification"
)


# ============================================================
# IMPORTANT CONSTANTS
# ============================================================

PDF_FALLBACK = (
    "I couldn't verify this answer from the uploaded document."
)

OCR_UNCERTAIN = (
    "I found potentially relevant content, but the scanned text "
    "is not clear enough to provide a reliable answer."
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "current_file" not in st.session_state:

    st.session_state.current_file = None


if "pdf_documents" not in st.session_state:

    st.session_state.pdf_documents = []


if "pdf_page_count" not in st.session_state:

    st.session_state.pdf_page_count = 0


if "pdf_extraction_report" not in st.session_state:

    st.session_state.pdf_extraction_report = {}

# ============================================================
# PART 2
# LLM + CORE HELPER FUNCTIONS
# ============================================================


# ============================================================
# LOAD LOCAL OLLAMA LLM
# ============================================================

@st.cache_resource
def load_llm():

    return ChatOllama(
        model="llama3.2:3b",
        temperature=0
    )


llm = load_llm()


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embeddings():

    return OllamaEmbeddings(
        model="nomic-embed-text"
    )


embeddings = load_embeddings()


# ============================================================
# CLEAN EXTRACTED PDF TEXT
# ============================================================

def clean_text(text):

    if not text:
        return ""

    # Remove null characters
    text = text.replace(
        "\x00",
        " "
    )

    # Normalize spaces
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Clean spaces after new lines
    text = re.sub(
        r"\n[ \t]+",
        "\n",
        text
    )

    # Remove excessive blank lines
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# NORMALIZE TEXT FOR MATCHING
# ============================================================

def normalize_text(value):

    return re.sub(
        r"[^a-z0-9]+",
        " ",
        str(value).lower()
    ).strip()


# ============================================================
# TOKENIZER
# Used later by BM25 / keyword retrieval
# ============================================================

def tokenize(text):

    if not text:
        return []

    return re.findall(
        r"[a-zA-Z0-9][a-zA-Z0-9_\-./%]*",
        str(text).lower()
    )


# ============================================================
# SAFE JSON PARSER
# ============================================================

def safe_json(text):

    if not text:
        return None

    # First try direct JSON
    try:

        return json.loads(
            text
        )

    except Exception:

        pass

    # Sometimes LLM returns:
    #
    # ```json
    # {...}
    # ```
    #
    # So find the JSON object manually.

    match = re.search(
        r"\{.*\}",
        text,
        flags=re.DOTALL
    )

    if not match:
        return None

    try:

        return json.loads(
            match.group(0)
        )

    except Exception:

        return None


# ============================================================
# DOCUMENT UNIQUE KEY
# ============================================================

def document_key(document):

    page = document.metadata.get(
        "page",
        0
    )

    chunk = document.metadata.get(
        "chunk",
        0
    )

    content = document.page_content.strip()

    return (
        page,
        chunk,
        content[:200]
    )


# ============================================================
# REMOVE DUPLICATE DOCUMENT CHUNKS
# ============================================================

def deduplicate_documents(documents):

    seen = set()

    unique_documents = []

    for document in documents:

        key = document_key(
            document
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        unique_documents.append(
            document
        )

    return unique_documents


# ============================================================
# GET SOURCE PAGES
# ============================================================

def get_source_pages(documents):

    pages = set()

    for document in documents:

        page = document.metadata.get(
            "page"
        )

        if isinstance(page, int):

            pages.add(
                page
            )

    return sorted(
        pages
    )


# ============================================================
# SIMPLE KEYWORD OVERLAP SCORE
# ============================================================

def keyword_overlap_score(
    question,
    document
):

    question_tokens = set(
        tokenize(question)
    )

    document_tokens = set(
        tokenize(
            document.page_content
        )
    )

    if not question_tokens:
        return 0.0

    common_tokens = (
        question_tokens
        & document_tokens
    )

    return (
        len(common_tokens)
        / len(question_tokens)
    )


# ============================================================
# FORMAT DATA VALUES
# Used later for CSV / Excel analytics
# ============================================================

def format_result(value):

    if pd.isna(value):

        return "N/A"

    if isinstance(
        value,
        (int, np.integer)
    ):

        return f"{int(value):,}"

    if isinstance(
        value,
        (float, np.floating)
    ):

        value = float(value)

        if value.is_integer():

            return f"{int(value):,}"

        return (
            f"{value:,.3f}"
            .rstrip("0")
            .rstrip(".")
        )

    return str(value)


# ============================================================
# FILE HASH
# Prevent different PDFs sharing the same Chroma collection
# ============================================================

def create_file_hash(
    file_bytes
):

    return hashlib.sha256(
        file_bytes
    ).hexdigest()


# ============================================================
# CHECK WHETHER TEXT IS USEFUL
# ============================================================

def has_useful_text(
    text,
    minimum_characters=80
):

    if not text:
        return False

    compact_text = re.sub(
        r"\s+",
        "",
        text
    )

    return (
        len(compact_text)
        >= minimum_characters
    )


# ============================================================
# CHECK WHETHER OCR TEXT LOOKS USABLE
# ============================================================

def is_ocr_text_usable(text):

    if not text:
        return False

    compact = re.sub(
        r"\s+",
        "",
        text
    )

    if len(compact) < 30:
        return False

    # Count alphabetic/numeric characters
    useful_characters = sum(
        character.isalnum()
        for character in compact
    )

    ratio = (
        useful_characters
        / max(
            1,
            len(compact)
        )
    )

    return ratio >= 0.55


# ============================================================
# SAFE LLM INVOCATION
# ============================================================

def invoke_llm(
    prompt,
    fallback=""
):

    try:

        response = llm.invoke(
            prompt
        )

        content = (
            response.content
            if hasattr(
                response,
                "content"
            )
            else str(response)
        )

        content = str(
            content
        ).strip()

        if content:

            return content

        return fallback

    except Exception as error:

        print(
            "LLM ERROR:",
            error
        )

        return fallback
    # ============================================================
# PART 3
# PDF EXTRACTION + AUTOMATIC OCR FALLBACK
# ============================================================


# ============================================================
# OCR ONE PDF PAGE
# ============================================================

def ocr_pdf_page(
    file_bytes,
    page_index
):

    # page_index is zero-based

    if not OCR_AVAILABLE:

        return ""

    pdf_document = None

    try:

        # Open PDF using PyMuPDF
        pdf_document = fitz.open(
            stream=file_bytes,
            filetype="pdf"
        )

        page = pdf_document.load_page(
            page_index
        )

        # Render at higher resolution for OCR
        matrix = fitz.Matrix(
            2.5,
            2.5
        )

        pixmap = page.get_pixmap(
            matrix=matrix,
            alpha=False
        )

        # Convert rendered page to PIL image
        image = Image.frombytes(
            "RGB",
            [
                pixmap.width,
                pixmap.height
            ],
            pixmap.samples
        )

        # Run OCR
        extracted_text = (
            pytesseract.image_to_string(
                image,
                lang="eng",
                config="--psm 3"
            )
        )

        extracted_text = clean_text(
            extracted_text
        )

        return extracted_text

    except Exception as error:

        print(
            f"OCR ERROR PAGE {page_index + 1}:",
            error
        )

        return ""

    finally:

        if pdf_document is not None:

            try:
                pdf_document.close()

            except Exception:
                pass


# ============================================================
# EXTRACT NATIVE PDF TEXT
# ============================================================

def extract_native_page_text(
    page
):

    try:

        text = page.extract_text()

        if not text:

            return ""

        return clean_text(
            text
        )

    except Exception as error:

        print(
            "PDF TEXT EXTRACTION ERROR:",
            error
        )

        return ""


# ============================================================
# CHOOSE BEST TEXT
# Native text is preferred.
# OCR is used when native extraction is too weak.
# ============================================================

def extract_best_page_text(
    file_bytes,
    page,
    page_index
):

    native_text = (
        extract_native_page_text(
            page
        )
    )

    # ----------------------------------------
    # Normal text PDF
    # ----------------------------------------

    if has_useful_text(
        native_text,
        minimum_characters=80
    ):

        return (
            native_text,
            "native"
        )

    # ----------------------------------------
    # Possible scanned/image PDF
    # ----------------------------------------

    ocr_text = ocr_pdf_page(
        file_bytes,
        page_index
    )

    # Use OCR only when it looks usable
    if (
        is_ocr_text_usable(
            ocr_text
        )
        and
        len(ocr_text)
        > len(native_text)
    ):

        return (
            ocr_text,
            "ocr"
        )

    # ----------------------------------------
    # Native extraction may contain a little
    # useful information even when <80 chars.
    # ----------------------------------------

    if len(
        re.sub(
            r"\s+",
            "",
            native_text
        )
    ) >= 20:

        return (
            native_text,
            "native_low_text"
        )

    # ----------------------------------------
    # OCR might be short but still readable
    # ----------------------------------------

    if is_ocr_text_usable(
        ocr_text
    ):

        return (
            ocr_text,
            "ocr"
        )

    # ----------------------------------------
    # Page cannot be reliably extracted
    # ----------------------------------------

    return (
        "",
        "unreadable"
    )


# ============================================================
# PDF CHUNKING
# ============================================================

def create_pdf_text_splitter():

    return RecursiveCharacterTextSplitter(

        chunk_size=1200,

        chunk_overlap=250,

        separators=[
            "\n\n",
            "\n",
            ". ",
            "; ",
            ": ",
            " ",
            ""
        ]
    )


# ============================================================
# EXTRACT + CHUNK COMPLETE PDF
# ============================================================

@st.cache_data(
    show_spinner=False
)
def extract_and_chunk_pdf(
    file_bytes
):

    reader = PdfReader(
        io.BytesIO(
            file_bytes
        )
    )

    page_count = len(
        reader.pages
    )

    splitter = (
        create_pdf_text_splitter()
    )

    documents = []

    native_pages = []

    ocr_pages = []

    low_text_pages = []

    unreadable_pages = []

    page_text_map = {}


    # ========================================================
    # PROCESS EVERY PAGE
    # ========================================================

    for page_index, page in enumerate(
        reader.pages
    ):

        page_number = (
            page_index + 1
        )

        page_text, extraction_method = (
            extract_best_page_text(
                file_bytes=file_bytes,
                page=page,
                page_index=page_index
            )
        )


        # ====================================================
        # PAGE EXTRACTION REPORT
        # ====================================================

        if extraction_method == "native":

            native_pages.append(
                page_number
            )

        elif extraction_method == "ocr":

            ocr_pages.append(
                page_number
            )

        elif extraction_method == "native_low_text":

            low_text_pages.append(
                page_number
            )

        else:

            unreadable_pages.append(
                page_number
            )


        # ====================================================
        # SKIP UNREADABLE PAGE
        # ====================================================

        if not page_text:

            continue


        # Keep complete page text.
        #
        # This will become useful later for:
        # - exact searches
        # - structural retrieval
        # - page summarization
        # - list recovery
        # - second-pass retrieval

        page_text_map[
            page_number
        ] = page_text


        # ====================================================
        # SPLIT PAGE INTO CHUNKS
        # ====================================================

        chunks = splitter.split_text(
            page_text
        )


        # ====================================================
        # CREATE LANGCHAIN DOCUMENTS
        # ====================================================

        for chunk_number, chunk in enumerate(
            chunks,
            start=1
        ):

            chunk = clean_text(
                chunk
            )

            if not chunk:

                continue


            document = Document(

                page_content=chunk,

                metadata={

                    "page":
                        page_number,

                    "chunk":
                        chunk_number,

                    "extraction":
                        extraction_method,

                    "source":
                        "uploaded_pdf"
                }
            )


            documents.append(
                document
            )


    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    documents = (
        deduplicate_documents(
            documents
        )
    )


    # ========================================================
    # EXTRACTION REPORT
    # ========================================================

    extraction_report = {

        "total_pages":
            page_count,

        "native_pages":
            native_pages,

        "ocr_pages":
            ocr_pages,

        "low_text_pages":
            low_text_pages,

        "unreadable_pages":
            unreadable_pages,

        "total_chunks":
            len(documents),

        "readable_pages":
            len(page_text_map)
    }


    return (
        documents,
        page_count,
        extraction_report,
        page_text_map
    )
# ============================================================
# PART 4
# CHROMA VECTOR SEARCH + BM25 KEYWORD SEARCH
# ============================================================


# ============================================================
# CREATE CHROMA VECTOR STORE
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def create_vector_store(
    file_hash,
    _documents
):

    if not _documents:

        return None


    # Unique collection for each uploaded PDF
    collection_name = (
        "pdf_"
        + file_hash[:24]
    )


    vector_store = Chroma.from_documents(

        documents=list(
            _documents
        ),

        embedding=embeddings,

        collection_name=collection_name
    )


    return vector_store


# ============================================================
# BM25 / KEYWORD RETRIEVER
# ============================================================

class BM25DocumentRetriever:

    def __init__(
        self,
        documents
    ):

        self.documents = list(
            documents
        )

        self.tokenized_corpus = []

        self.bm25 = None


        # ----------------------------------------
        # Tokenize every document chunk
        # ----------------------------------------

        for document in self.documents:

            tokens = tokenize(
                document.page_content
            )

            self.tokenized_corpus.append(
                tokens
            )


        # ----------------------------------------
        # Build BM25 index
        # ----------------------------------------

        if (
            BM25_AVAILABLE
            and
            self.tokenized_corpus
        ):

            try:

                self.bm25 = BM25Okapi(
                    self.tokenized_corpus
                )

            except Exception as error:

                print(
                    "BM25 INDEX ERROR:",
                    error
                )

                self.bm25 = None


    # ========================================================
    # BM25 SEARCH
    # ========================================================

    def search(
        self,
        query,
        k=15
    ):

        if not self.documents:

            return []


        query_tokens = tokenize(
            query
        )


        if not query_tokens:

            return []


        # ====================================================
        # REAL BM25 SEARCH
        # ====================================================

        if self.bm25 is not None:

            try:

                scores = (
                    self.bm25
                    .get_scores(
                        query_tokens
                    )
                )


                ranked_indices = (
                    np.argsort(
                        scores
                    )[::-1]
                )


                results = []


                for index in ranked_indices:

                    score = float(
                        scores[index]
                    )


                    # Ignore completely unrelated chunks
                    if score <= 0:

                        continue


                    results.append(
                        (
                            self.documents[
                                int(index)
                            ],
                            score
                        )
                    )


                    if len(results) >= k:

                        break


                return results


            except Exception as error:

                print(
                    "BM25 SEARCH ERROR:",
                    error
                )


        # ====================================================
        # FALLBACK KEYWORD SEARCH
        # ====================================================
        #
        # If rank-bm25 is unavailable,
        # the application still performs
        # simple lexical matching.
        # ====================================================

        query_token_set = set(
            query_tokens
        )

        scored_documents = []


        for document in self.documents:

            document_tokens = set(
                tokenize(
                    document.page_content
                )
            )


            common_tokens = (
                query_token_set
                & document_tokens
            )


            if not common_tokens:

                continue


            score = (
                len(common_tokens)
                / max(
                    1,
                    len(query_token_set)
                )
            )


            scored_documents.append(
                (
                    document,
                    score
                )
            )


        scored_documents.sort(
            key=lambda item: item[1],
            reverse=True
        )


        return scored_documents[:k]


# ============================================================
# SAFE VECTOR SEARCH
# ============================================================

def vector_search(
    vector_store,
    query,
    k=15
):

    if vector_store is None:

        return []


    try:

        documents = (
            vector_store
            .similarity_search(
                query,
                k=k
            )
        )


        return documents


    except Exception as error:

        print(
            "VECTOR SEARCH ERROR:",
            error
        )


        return []


# ============================================================
# VECTOR SEARCH WITH SCORES
# ============================================================

def vector_search_with_scores(
    vector_store,
    query,
    k=15
):

    if vector_store is None:

        return []


    try:

        results = (
            vector_store
            .similarity_search_with_score(
                query,
                k=k
            )
        )


        return results


    except Exception as error:

        print(
            "VECTOR SCORE SEARCH ERROR:",
            error
        )


        return []


# ============================================================
# EXACT PHRASE SEARCH ACROSS ALL CHUNKS
# ============================================================

def exact_phrase_search(
    documents,
    phrase,
    limit=20
):

    if not phrase:

        return []


    normalized_phrase = (
        normalize_text(
            phrase
        )
    )


    if not normalized_phrase:

        return []


    matches = []


    for document in documents:

        normalized_document = (
            normalize_text(
                document.page_content
            )
        )


        if (
            normalized_phrase
            in normalized_document
        ):

            matches.append(
                document
            )


            if len(matches) >= limit:

                break


    return matches


# ============================================================
# SEARCH COMPLETE PAGE TEXT
# ============================================================

def search_full_pages(
    page_text_map,
    query,
    limit=15
):

    query_tokens = set(
        tokenize(
            query
        )
    )


    if not query_tokens:

        return []


    results = []


    for page_number, page_text in (
        page_text_map.items()
    ):

        page_tokens = set(
            tokenize(
                page_text
            )
        )


        common_tokens = (
            query_tokens
            & page_tokens
        )


        if not common_tokens:

            continue


        score = (
            len(common_tokens)
            / max(
                1,
                len(query_tokens)
            )
        )


        results.append(
            (
                page_number,
                score,
                page_text
            )
        )


    results.sort(
        key=lambda item: item[1],
        reverse=True
    )


    return results[:limit]


# ============================================================
# CONVERT FULL PAGE RESULT INTO DOCUMENT
# ============================================================

def page_result_to_document(
    page_number,
    page_text
):

    return Document(

        page_content=page_text,

        metadata={

            "page":
                page_number,

            "chunk":
                0,

            "extraction":
                "full_page",

            "source":
                "full_page_recovery"
        }
    )


# ============================================================
# GET DOCUMENT ID
# Used later by RRF fusion
# ============================================================

def get_document_id(
    document
):

    page = document.metadata.get(
        "page",
        0
    )

    chunk = document.metadata.get(
        "chunk",
        0
    )


    content_hash = hashlib.md5(
        document.page_content
        .encode(
            "utf-8",
            errors="ignore"
        )
    ).hexdigest()[:12]


    return (
        f"{page}_"
        f"{chunk}_"
        f"{content_hash}"
    )# ============================================================
# PART 5
# QUESTION UNDERSTANDING + MULTI-QUERY GENERATION
# ============================================================


# ============================================================
# NUMBER WORD MAPPING
# ============================================================

NUMBER_WORDS = {

    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20
}


# ============================================================
# LIST-TYPE WORDS
# ============================================================
#
# These are generic structural concepts.
#
# They are NOT document-specific.
#
# Examples:
#
# four pillars
# five findings
# three recommendations
# six components
# seven steps
#
# ============================================================

LIST_CONCEPTS = [

    "pillar",
    "pillars",

    "finding",
    "findings",

    "recommendation",
    "recommendations",

    "objective",
    "objectives",

    "goal",
    "goals",

    "aim",
    "aims",

    "step",
    "steps",

    "stage",
    "stages",

    "phase",
    "phases",

    "component",
    "components",

    "principle",
    "principles",

    "strategy",
    "strategies",

    "factor",
    "factors",

    "feature",
    "features",

    "benefit",
    "benefits",

    "challenge",
    "challenges",

    "limitation",
    "limitations",

    "reason",
    "reasons",

    "method",
    "methods",

    "approach",
    "approaches",

    "conclusion",
    "conclusions",

    "theme",
    "themes",

    "point",
    "points",

    "category",
    "categories",

    "requirement",
    "requirements",

    "criterion",
    "criteria",

    "element",
    "elements",

    "area",
    "areas"
]


# ============================================================
# DETECT EXPECTED ITEM COUNT
# ============================================================

def detect_expected_count(
    question
):

    question_lower = (
        question.lower()
    )


    # ========================================================
    # NUMERIC COUNT
    #
    # Examples:
    #
    # 4 pillars
    # 5 findings
    # 10 recommendations
    # ========================================================

    concepts_pattern = "|".join(
        re.escape(concept)
        for concept in LIST_CONCEPTS
    )


    numeric_pattern = (

        rf"\b(\d{{1,2}})\s+"

        rf"(?:key\s+|main\s+|major\s+|important\s+)?"

        rf"(?:{concepts_pattern})\b"
    )


    match = re.search(
        numeric_pattern,
        question_lower
    )


    if match:

        try:

            count = int(
                match.group(1)
            )

            if 1 <= count <= 50:

                return count

        except Exception:

            pass


    # ========================================================
    # WORD COUNT
    #
    # Examples:
    #
    # four pillars
    # five findings
    # three recommendations
    # ========================================================

    for word, number in (
        NUMBER_WORDS.items()
    ):

        word_pattern = (

            rf"\b{word}\s+"

            rf"(?:key\s+|main\s+|major\s+|important\s+)?"

            rf"(?:{concepts_pattern})\b"
        )


        if re.search(
            word_pattern,
            question_lower
        ):

            return number


    return None


# ============================================================
# DETECT LIST CONCEPT
# ============================================================

def detect_list_concept(
    question
):

    question_lower = (
        question.lower()
    )


    # Longer words first
    concepts = sorted(
        LIST_CONCEPTS,
        key=len,
        reverse=True
    )


    for concept in concepts:

        if re.search(
            rf"\b{re.escape(concept)}\b",
            question_lower
        ):

            # Convert plural to simple singular
            singular_map = {

                "pillars": "pillar",
                "findings": "finding",
                "recommendations": "recommendation",
                "objectives": "objective",
                "goals": "goal",
                "aims": "aim",
                "steps": "step",
                "stages": "stage",
                "phases": "phase",
                "components": "component",
                "principles": "principle",
                "strategies": "strategy",
                "factors": "factor",
                "features": "feature",
                "benefits": "benefit",
                "challenges": "challenge",
                "limitations": "limitation",
                "reasons": "reason",
                "methods": "method",
                "approaches": "approach",
                "conclusions": "conclusion",
                "themes": "theme",
                "points": "point",
                "categories": "category",
                "requirements": "requirement",
                "criteria": "criterion",
                "elements": "element",
                "areas": "area"
            }


            return singular_map.get(
                concept,
                concept
            )


    return None


# ============================================================
# QUESTION TYPE DETECTION
# ============================================================

def detect_question_type(
    question
):

    q = question.lower()


    # ----------------------------------------
    # Summary
    # ----------------------------------------

    if any(
        phrase in q
        for phrase in [

            "summarize",
            "summary",
            "overview",
            "briefly explain",
            "main idea"
        ]
    ):

        return "summary"


    # ----------------------------------------
    # Methodology
    # ----------------------------------------

    if any(
        word in q
        for word in [

            "methodology",
            "method used",
            "research method",
            "study design",
            "approach used"
        ]
    ):

        return "methodology"


    # ----------------------------------------
    # Findings
    # ----------------------------------------

    if any(
        word in q
        for word in [

            "finding",
            "findings",
            "result",
            "results"
        ]
    ):

        return "findings"


    # ----------------------------------------
    # Recommendations
    # ----------------------------------------

    if any(
        word in q
        for word in [

            "recommendation",
            "recommendations",
            "suggested",
            "suggestions"
        ]
    ):

        return "recommendations"


    # ----------------------------------------
    # Limitations
    # ----------------------------------------

    if any(
        word in q
        for word in [

            "limitation",
            "limitations",
            "weakness",
            "weaknesses"
        ]
    ):

        return "limitations"


    # ----------------------------------------
    # Conclusion
    # ----------------------------------------

    if any(
        word in q
        for word in [

            "conclusion",
            "conclusions",
            "conclude",
            "concluded"
        ]
    ):

        return "conclusion"


    # ----------------------------------------
    # Objectives
    # ----------------------------------------

    if any(
        word in q
        for word in [

            "objective",
            "objectives",
            "aim",
            "aims",
            "purpose"
        ]
    ):

        return "objectives"


    # ----------------------------------------
    # Dataset
    # ----------------------------------------

    if any(
        phrase in q
        for phrase in [

            "dataset",
            "sample size",
            "participants",
            "respondents",
            "data source"
        ]
    ):

        return "dataset"


    # ----------------------------------------
    # Comparison
    # ----------------------------------------

    if any(
        phrase in q
        for phrase in [

            "compare",
            "comparison",
            "difference between",
            "differences between",
            "versus",
            " vs "
        ]
    ):

        return "comparison"


    # ----------------------------------------
    # Numeric
    # ----------------------------------------

    if any(
        phrase in q
        for phrase in [

            "how many",
            "percentage",
            "percent",
            "rate",
            "amount",
            "number of",
            "total",
            "average",
            "mean"
        ]
    ):

        return "numeric"


    # ----------------------------------------
    # List question
    # ----------------------------------------

    if (
        detect_expected_count(
            question
        )
        is not None
        or
        detect_list_concept(
            question
        )
        is not None
    ):

        return "list"


    return "general"


# ============================================================
# GENERATE MULTIPLE SEARCH QUERIES
# ============================================================

def generate_search_queries(
    question
):

    question_type = (
        detect_question_type(
            question
        )
    )


    expected_count = (
        detect_expected_count(
            question
        )
    )


    list_concept = (
        detect_list_concept(
            question
        )
    )


    prompt = f"""
You are a retrieval-query generator for a document question-answering system.

The user uploaded ONE document.

Your job is NOT to answer the question.

Your job is only to generate alternative search queries that can locate
the evidence needed to answer the question.

USER QUESTION:

{question}

QUESTION TYPE:

{question_type}

EXPECTED LIST COUNT FROM QUESTION WORDING:

{expected_count}

LIST CONCEPT:

{list_concept}

IMPORTANT RULES:

1. Do not answer the user's question.

2. Do not invent facts.

3. Do not assume information exists merely because the user asked for it.

4. Preserve important:
   - names
   - organizations
   - acronyms
   - dates
   - technical terms
   - numbers

5. Generate queries using different wording.

6. For list questions, generate queries that could find:
   - the complete list
   - section headings
   - numbered items
   - individual list members
   - continuation of the list on another page

7. For methodology questions, search for:
   methodology
   methods
   study design
   data collection
   analysis

8. For findings questions, search for:
   findings
   results
   outcomes
   key results

9. For recommendations, search for:
   recommendations
   proposed actions
   future actions
   suggestions

10. Keep each query short.

Return JSON only.

Required format:

{{
    "queries": [
        "query 1",
        "query 2",
        "query 3",
        "query 4",
        "query 5"
    ]
}}
"""


    response = invoke_llm(
        prompt,
        fallback=""
    )


    parsed = safe_json(
        response
    )


    generated_queries = []


    if isinstance(
        parsed,
        dict
    ):

        queries = parsed.get(
            "queries",
            []
        )


        if isinstance(
            queries,
            list
        ):

            for query in queries:

                query = str(
                    query
                ).strip()


                if query:

                    generated_queries.append(
                        query
                    )


    # ========================================================
    # ORIGINAL QUESTION MUST ALWAYS BE SEARCHED
    # ========================================================

    final_queries = [
        question.strip()
    ]


    # ========================================================
    # ADD GENERATED QUERIES
    # ========================================================

    for query in generated_queries:

        existing = {
            item.lower()
            for item in final_queries
        }


        if query.lower() not in existing:

            final_queries.append(
                query
            )


    # ========================================================
    # ADD GENERIC STRUCTURAL SEARCH QUERIES
    # ========================================================

    if list_concept:

        structural_queries = [

            f"{list_concept}",

            f"{list_concept} 1",

            f"{list_concept} 2",

            f"key {list_concept}",

            f"{list_concept} section",

            f"{list_concept} list"
        ]


        # If user asks for N items,
        # search each possible numbered heading.
        #
        # Example:
        #
        # 4 pillars
        #
        # searches:
        #
        # pillar 1
        # pillar 2
        # pillar 3
        # pillar 4
        #
        # This is generic.
        # It does NOT know what the pillars are.

        if expected_count:

            for number in range(
                1,
                expected_count + 1
            ):

                structural_queries.append(
                    f"{list_concept} {number}"
                )


        for query in structural_queries:

            existing = {
                item.lower()
                for item in final_queries
            }


            if query.lower() not in existing:

                final_queries.append(
                    query
                )


    # Limit excessive LLM-generated searches
    return final_queries[:15]
# ============================================================
# PART 6
# HYBRID RETRIEVAL + RECIPROCAL RANK FUSION (RRF)
# ============================================================


# ============================================================
# RRF SCORE
# ============================================================

def rrf_score(
    rank,
    constant=60
):

    return 1.0 / (
        constant + rank
    )


# ============================================================
# HYBRID RETRIEVAL
# ============================================================

def hybrid_retrieve(
    question,
    search_queries,
    vector_store,
    bm25_retriever,
    page_count,
    vector_k=15,
    bm25_k=15,
    final_k=30
):

    # Stores combined RRF score
    combined_scores = defaultdict(
        float
    )

    # Stores actual Document object
    document_store = {}

    # Track how many independent searches found a chunk
    search_hits = defaultdict(
        set
    )


    # ========================================================
    # SEARCH EVERY QUERY
    # ========================================================

    for query_index, query in enumerate(
        search_queries
    ):

        # ====================================================
        # 1. VECTOR SEARCH
        # ====================================================

        vector_results = vector_search(
            vector_store=vector_store,
            query=query,
            k=vector_k
        )


        for rank, document in enumerate(
            vector_results,
            start=1
        ):

            page = document.metadata.get(
                "page"
            )


            # Ignore invalid metadata
            if not isinstance(
                page,
                int
            ):

                continue


            if not (
                1 <= page <= page_count
            ):

                continue


            doc_id = get_document_id(
                document
            )


            document_store[
                doc_id
            ] = document


            combined_scores[
                doc_id
            ] += rrf_score(
                rank
            )


            search_hits[
                doc_id
            ].add(
                (
                    "vector",
                    query_index
                )
            )


        # ====================================================
        # 2. BM25 SEARCH
        # ====================================================

        bm25_results = (
            bm25_retriever.search(
                query=query,
                k=bm25_k
            )
        )


        for rank, result in enumerate(
            bm25_results,
            start=1
        ):

            document = result[0]

            page = document.metadata.get(
                "page"
            )


            if not isinstance(
                page,
                int
            ):

                continue


            if not (
                1 <= page <= page_count
            ):

                continue


            doc_id = get_document_id(
                document
            )


            document_store[
                doc_id
            ] = document


            # Slightly higher weight for lexical matches.
            #
            # Exact headings, acronyms and numbered items
            # can be extremely important in reports.

            combined_scores[
                doc_id
            ] += (
                1.15
                * rrf_score(
                    rank
                )
            )


            search_hits[
                doc_id
            ].add(
                (
                    "bm25",
                    query_index
                )
            )


    # ========================================================
    # MULTIPLE-SEARCH BONUS
    # ========================================================
    #
    # If the same chunk was independently found by several
    # query variants / retrieval methods, give it a small bonus.
    # ========================================================

    for doc_id in combined_scores:

        number_of_hits = len(
            search_hits[
                doc_id
            ]
        )


        combined_scores[
            doc_id
        ] += (
            0.002
            * number_of_hits
        )


    # ========================================================
    # SORT BY FINAL RRF SCORE
    # ========================================================

    ranked_ids = sorted(

        combined_scores.keys(),

        key=lambda doc_id:
            combined_scores[
                doc_id
            ],

        reverse=True
    )


    # ========================================================
    # BUILD FINAL RESULT
    # ========================================================

    ranked_documents = []


    for doc_id in ranked_ids[
        :final_k
    ]:

        document = (
            document_store[
                doc_id
            ]
        )


        # Add retrieval metadata
        document.metadata[
            "rrf_score"
        ] = combined_scores[
            doc_id
        ]


        document.metadata[
            "retrieval_hits"
        ] = len(
            search_hits[
                doc_id
            ]
        )


        ranked_documents.append(
            document
        )


    return ranked_documents


# ============================================================
# WHOLE-PAGE RECOVERY SEARCH
# ============================================================

def retrieve_full_page_matches(
    page_text_map,
    search_queries,
    limit=10
):

    page_scores = defaultdict(
        float
    )


    # ========================================================
    # SEARCH EVERY QUERY AGAINST FULL PAGES
    # ========================================================

    for query in search_queries:

        results = search_full_pages(
            page_text_map=page_text_map,
            query=query,
            limit=limit
        )


        for rank, result in enumerate(
            results,
            start=1
        ):

            page_number = result[0]

            score = result[1]


            page_scores[
                page_number
            ] += (
                score
                * rrf_score(
                    rank
                )
            )


    # ========================================================
    # RANK FULL PAGES
    # ========================================================

    ranked_pages = sorted(

        page_scores.keys(),

        key=lambda page:
            page_scores[
                page
            ],

        reverse=True
    )


    documents = []


    for page_number in ranked_pages[
        :limit
    ]:

        page_text = (
            page_text_map.get(
                page_number,
                ""
            )
        )


        if not page_text:

            continue


        document = (
            page_result_to_document(
                page_number,
                page_text
            )
        )


        document.metadata[
            "full_page_score"
        ] = page_scores[
            page_number
        ]


        documents.append(
            document
        )


    return documents


# ============================================================
# COMBINE CHUNK + FULL-PAGE RESULTS
# ============================================================

def combine_retrieval_results(
    chunk_documents,
    page_documents
):

    combined = []


    # Chunk results first
    for document in chunk_documents:

        combined.append(
            document
        )


    # Full page recovery results
    for document in page_documents:

        combined.append(
            document
        )


    return deduplicate_documents(
        combined
    )


# ============================================================
# COMPLETE FIRST-PASS RETRIEVAL
# ============================================================

def first_pass_retrieval(
    question,
    vector_store,
    all_documents,
    page_text_map,
    page_count
):

    # ========================================================
    # GENERATE MULTIPLE SEARCH QUERIES
    # ========================================================

    search_queries = (
        generate_search_queries(
            question
        )
    )


    # ========================================================
    # BUILD BM25 RETRIEVER
    # ========================================================

    bm25_retriever = (
        BM25DocumentRetriever(
            all_documents
        )
    )


    # ========================================================
    # HYBRID CHUNK SEARCH
    # ========================================================

    chunk_results = (
        hybrid_retrieve(

            question=question,

            search_queries=
                search_queries,

            vector_store=
                vector_store,

            bm25_retriever=
                bm25_retriever,

            page_count=
                page_count,

            vector_k=15,

            bm25_k=15,

            final_k=30
        )
    )


    # ========================================================
    # FULL-PAGE RECOVERY SEARCH
    # ========================================================

    page_results = (
        retrieve_full_page_matches(

            page_text_map=
                page_text_map,

            search_queries=
                search_queries,

            limit=8
        )
    )


    # ========================================================
    # COMBINE BOTH RETRIEVAL CHANNELS
    # ========================================================

    combined_results = (
        combine_retrieval_results(

            chunk_documents=
                chunk_results,

            page_documents=
                page_results
        )
    )


    return (
        combined_results,
        search_queries,
        bm25_retriever
    )
# ============================================================
# PART 7
# STRUCTURAL RETRIEVAL + NEIGHBOR CHUNK RECOVERY
# ============================================================


# ============================================================
# FIND A DOCUMENT BY PAGE + CHUNK
# ============================================================

def build_chunk_lookup(
    all_documents
):

    lookup = {}


    for document in all_documents:

        page = document.metadata.get(
            "page"
        )

        chunk = document.metadata.get(
            "chunk"
        )


        if (
            isinstance(page, int)
            and
            isinstance(chunk, int)
        ):

            lookup[
                (
                    page,
                    chunk
                )
            ] = document


    return lookup


# ============================================================
# EXPAND NEIGHBORING CHUNKS
# ============================================================
#
# Example:
#
# Retrieved:
# Page 10 - Chunk 3
#
# Also retrieve:
# Page 10 - Chunk 2
# Page 10 - Chunk 4
#
# This is useful when:
#
# heading is in one chunk
# explanation is in next chunk
#
# or
#
# list starts in one chunk
# and continues in another.
#
# ============================================================

def expand_neighbor_chunks(
    retrieved_documents,
    all_documents,
    neighbor_distance=1,
    max_documents=45
):

    lookup = build_chunk_lookup(
        all_documents
    )


    expanded = []


    for document in retrieved_documents:

        page = document.metadata.get(
            "page"
        )

        chunk = document.metadata.get(
            "chunk"
        )


        # Full-page recovery document
        if chunk == 0:

            expanded.append(
                document
            )

            continue


        if not (
            isinstance(page, int)
            and
            isinstance(chunk, int)
        ):

            expanded.append(
                document
            )

            continue


        # ----------------------------------------
        # Add original chunk first
        # ----------------------------------------

        expanded.append(
            document
        )


        # ----------------------------------------
        # Add neighboring chunks
        # ----------------------------------------

        for distance in range(
            1,
            neighbor_distance + 1
        ):

            previous_key = (
                page,
                chunk - distance
            )

            next_key = (
                page,
                chunk + distance
            )


            if previous_key in lookup:

                expanded.append(
                    lookup[
                        previous_key
                    ]
                )


            if next_key in lookup:

                expanded.append(
                    lookup[
                        next_key
                    ]
                )


    expanded = (
        deduplicate_documents(
            expanded
        )
    )


    return expanded[
        :max_documents
    ]


# ============================================================
# GENERIC STRUCTURAL PATTERN BUILDER
# ============================================================

def build_structural_patterns(
    concept,
    expected_count=None
):

    if not concept:

        return []


    concept = re.escape(
        concept
    )


    patterns = []


    # ========================================================
    # GENERAL HEADING PATTERNS
    #
    # Examples:
    #
    # Pillar 1
    # Pillar: 1
    # Pillar - 1
    # Step 3
    # Finding 5
    # Recommendation 2
    # ========================================================

    patterns.extend([

        rf"\b{concept}\s*[-:#.]?\s*\d+\b",

        rf"\b{concept}\s+(?:one|two|three|four|five|six|seven|eight|nine|ten)\b",

        rf"\b(?:key|main|major)\s+{concept}\b"
    ])


    # ========================================================
    # SPECIFIC ITEM NUMBERS
    # ========================================================

    if expected_count:

        for number in range(
            1,
            expected_count + 1
        ):

            patterns.extend([

                rf"\b{concept}\s*[-:#.]?\s*{number}\b",

                rf"\b{number}\s*[-.)]\s*"
            ])


    return patterns


# ============================================================
# STRUCTURAL SEARCH THROUGH ALL DOCUMENT CHUNKS
# ============================================================

def structural_chunk_search(
    question,
    all_documents,
    max_results=30
):

    concept = (
        detect_list_concept(
            question
        )
    )


    expected_count = (
        detect_expected_count(
            question
        )
    )


    if not concept:

        return []


    patterns = (
        build_structural_patterns(
            concept=concept,
            expected_count=expected_count
        )
    )


    matches = []


    for document in all_documents:

        text = (
            document.page_content
        )


        text_lower = (
            text.lower()
        )


        score = 0


        # ====================================================
        # CONCEPT APPEARS IN CHUNK
        # ====================================================

        if concept.lower() in text_lower:

            score += 2


        # ====================================================
        # STRUCTURAL REGEX MATCH
        # ====================================================

        for pattern in patterns:

            try:

                if re.search(
                    pattern,
                    text,
                    flags=re.IGNORECASE
                ):

                    score += 4

            except Exception:

                continue


        # ====================================================
        # QUESTION KEYWORD OVERLAP
        # ====================================================

        overlap = (
            keyword_overlap_score(
                question,
                document
            )
        )


        score += (
            overlap * 3
        )


        if score > 0:

            matches.append(
                (
                    score,
                    document
                )
            )


    # ========================================================
    # SORT STRUCTURAL MATCHES
    # ========================================================

    matches.sort(
        key=lambda item:
            item[0],
        reverse=True
    )


    results = [
        item[1]
        for item in matches[
            :max_results
        ]
    ]


    return (
        deduplicate_documents(
            results
        )
    )


# ============================================================
# SEARCH SPECIFIC NUMBERED ITEMS
# ============================================================
#
# Example:
#
# Question:
#
# "What are the four pillars?"
#
# This searches the complete extracted document for:
#
# pillar 1
# pillar 2
# pillar 3
# pillar 4
#
# IMPORTANT:
#
# This does NOT assume that all four exist.
#
# It only searches for evidence.
#
# ============================================================

def retrieve_numbered_items(
    question,
    all_documents
):

    concept = (
        detect_list_concept(
            question
        )
    )


    expected_count = (
        detect_expected_count(
            question
        )
    )


    if (
        not concept
        or
        not expected_count
    ):

        return []


    results = []


    for number in range(
        1,
        expected_count + 1
    ):

        # ----------------------------------------
        # Different heading formats
        # ----------------------------------------

        patterns = [

            rf"\b{re.escape(concept)}\s*{number}\b",

            rf"\b{re.escape(concept)}\s*[-:#.]\s*{number}\b",

            rf"\b{re.escape(concept)}\s+{number}\s*[-:#.]"
        ]


        item_matches = []


        for document in all_documents:

            text = (
                document.page_content
            )


            for pattern in patterns:

                if re.search(
                    pattern,
                    text,
                    flags=re.IGNORECASE
                ):

                    item_matches.append(
                        document
                    )

                    break


        # ----------------------------------------
        # Keep matches for this item
        # ----------------------------------------

        results.extend(
            item_matches
        )


    return (
        deduplicate_documents(
            results
        )
    )


# ============================================================
# STRUCTURAL SEARCH THROUGH FULL PAGES
# ============================================================

def structural_page_search(
    question,
    page_text_map,
    max_pages=15
):

    concept = (
        detect_list_concept(
            question
        )
    )


    expected_count = (
        detect_expected_count(
            question
        )
    )


    if not concept:

        return []


    patterns = (
        build_structural_patterns(
            concept,
            expected_count
        )
    )


    page_matches = []


    for page_number, page_text in (
        page_text_map.items()
    ):

        score = 0


        # ----------------------------------------
        # Concept match
        # ----------------------------------------

        if (
            concept.lower()
            in page_text.lower()
        ):

            score += 2


        # ----------------------------------------
        # Structural matches
        # ----------------------------------------

        for pattern in patterns:

            try:

                if re.search(
                    pattern,
                    page_text,
                    flags=re.IGNORECASE
                ):

                    score += 4

            except Exception:

                continue


        if score > 0:

            page_matches.append(
                (
                    score,
                    page_number,
                    page_text
                )
            )


    page_matches.sort(
        key=lambda item:
            item[0],
        reverse=True
    )


    documents = []


    for (
        score,
        page_number,
        page_text
    ) in page_matches[
        :max_pages
    ]:

        document = (
            page_result_to_document(
                page_number,
                page_text
            )
        )


        document.metadata[
            "structural_score"
        ] = score


        document.metadata[
            "structural_match"
        ] = True


        documents.append(
            document
        )


    return documents


# ============================================================
# PROTECT STRUCTURAL EVIDENCE
# ============================================================
#
# Structural matches should not disappear just because
# semantic ranking places them lower.
#
# ============================================================

def mark_structural_documents(
    documents
):

    for document in documents:

        document.metadata[
            "protected_evidence"
        ] = True


    return documents


# ============================================================
# COMPLETE STRUCTURAL RETRIEVAL
# ============================================================

def retrieve_structural_evidence(
    question,
    all_documents,
    page_text_map
):

    structural_documents = []


    # ========================================================
    # 1. GENERAL STRUCTURAL CHUNK SEARCH
    # ========================================================

    chunk_matches = (
        structural_chunk_search(
            question=question,
            all_documents=all_documents,
            max_results=30
        )
    )


    structural_documents.extend(
        chunk_matches
    )


    # ========================================================
    # 2. NUMBERED ITEM SEARCH
    # ========================================================

    numbered_matches = (
        retrieve_numbered_items(
            question=question,
            all_documents=all_documents
        )
    )


    structural_documents.extend(
        numbered_matches
    )


    # ========================================================
    # 3. FULL-PAGE STRUCTURAL SEARCH
    # ========================================================

    page_matches = (
        structural_page_search(
            question=question,
            page_text_map=page_text_map,
            max_pages=15
        )
    )


    structural_documents.extend(
        page_matches
    )


    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    structural_documents = (
        deduplicate_documents(
            structural_documents
        )
    )


    # ========================================================
    # PROTECT THESE DOCUMENTS
    # ========================================================

    structural_documents = (
        mark_structural_documents(
            structural_documents
        )
    )


    return structural_documents


# ============================================================
# MERGE NORMAL + STRUCTURAL EVIDENCE
# ============================================================

def merge_retrieved_evidence(
    normal_documents,
    structural_documents,
    all_documents,
    max_documents=50
):

    # ========================================================
    # STRUCTURAL EVIDENCE FIRST
    # ========================================================

    combined = []


    for document in structural_documents:

        combined.append(
            document
        )


    # ========================================================
    # THEN NORMAL RRF RESULTS
    # ========================================================

    for document in normal_documents:

        combined.append(
            document
        )


    combined = (
        deduplicate_documents(
            combined
        )
    )


    # ========================================================
    # NEIGHBOR RECOVERY
    # ========================================================

    combined = (
        expand_neighbor_chunks(

            retrieved_documents=
                combined,

            all_documents=
                all_documents,

            neighbor_distance=1,

            max_documents=
                max_documents
        )
    )


    return combined
# ============================================================
# PART 8
# EVIDENCE RERANKING + CONTEXT CONSTRUCTION
# ============================================================


# ============================================================
# CALCULATE EVIDENCE RELEVANCE SCORE
# ============================================================

def calculate_evidence_score(
    question,
    document
):

    score = 0.0


    # ========================================================
    # 1. QUESTION / DOCUMENT KEYWORD OVERLAP
    # ========================================================

    overlap_score = (
        keyword_overlap_score(
            question,
            document
        )
    )

    score += (
        overlap_score * 5.0
    )


    # ========================================================
    # 2. EXISTING RRF SCORE
    # ========================================================

    rrf_value = (
        document.metadata.get(
            "rrf_score",
            0
        )
    )

    if isinstance(
        rrf_value,
        (int, float)
    ):

        score += (
            float(rrf_value) * 20
        )


    # ========================================================
    # 3. MULTIPLE RETRIEVAL HITS
    # ========================================================

    retrieval_hits = (
        document.metadata.get(
            "retrieval_hits",
            0
        )
    )

    if isinstance(
        retrieval_hits,
        (int, float)
    ):

        score += (
            min(
                float(retrieval_hits),
                10
            )
            * 0.25
        )


    # ========================================================
    # 4. STRUCTURAL MATCH BONUS
    # ========================================================

    if document.metadata.get(
        "structural_match",
        False
    ):

        score += 4.0


    # ========================================================
    # 5. PROTECTED EVIDENCE BONUS
    # ========================================================

    if document.metadata.get(
        "protected_evidence",
        False
    ):

        score += 5.0


    # ========================================================
    # 6. FULL-PAGE RECOVERY
    # ========================================================
    #
    # Full pages are useful for recovery but should not
    # automatically dominate precise chunks.
    # ========================================================

    if (
        document.metadata.get(
            "source"
        )
        == "full_page_recovery"
    ):

        score += 0.5


    # ========================================================
    # 7. OCR HANDLING
    # ========================================================
    #
    # OCR evidence is allowed.
    #
    # We do NOT remove it simply because it came from OCR.
    # But native text receives a tiny preference when all
    # other relevance signals are equal.
    # ========================================================

    extraction_method = (
        document.metadata.get(
            "extraction",
            ""
        )
    )


    if extraction_method == "native":

        score += 0.25


    elif extraction_method == "native_low_text":

        score += 0.10


    return score


# ============================================================
# RERANK EVIDENCE
# ============================================================

def rerank_evidence(
    question,
    documents,
    max_documents=35
):

    if not documents:

        return []


    documents = (
        deduplicate_documents(
            documents
        )
    )


    protected_documents = []

    normal_documents = []


    # ========================================================
    # SEPARATE PROTECTED STRUCTURAL EVIDENCE
    # ========================================================

    for document in documents:

        if document.metadata.get(
            "protected_evidence",
            False
        ):

            protected_documents.append(
                document
            )

        else:

            normal_documents.append(
                document
            )


    # ========================================================
    # SCORE PROTECTED DOCUMENTS
    # ========================================================

    protected_scored = []


    for index, document in enumerate(
        protected_documents
    ):

        score = calculate_evidence_score(
            question,
            document
        )


        protected_scored.append(
            (
                score,
                index,
                document
            )
        )


    # ========================================================
    # SCORE NORMAL DOCUMENTS
    # ========================================================

    normal_scored = []


    for index, document in enumerate(
        normal_documents
    ):

        score = calculate_evidence_score(
            question,
            document
        )


        normal_scored.append(
            (
                score,
                index,
                document
            )
        )


    # ========================================================
    # SORT BOTH GROUPS
    # ========================================================

    protected_scored.sort(

        key=lambda item: (
            -item[0],
            item[1]
        )
    )


    normal_scored.sort(

        key=lambda item: (
            -item[0],
            item[1]
        )
    )


    # ========================================================
    # KEEP PROTECTED EVIDENCE FIRST
    # ========================================================
    #
    # This is important.
    #
    # Example:
    #
    # Pillar 3 was found by structural search.
    #
    # Even if its semantic score is lower,
    # it should not disappear from the final evidence.
    # ========================================================

    final_documents = []


    for score, index, document in (
        protected_scored
    ):

        document.metadata[
            "final_evidence_score"
        ] = score


        final_documents.append(
            document
        )


    # ========================================================
    # FILL REMAINING SPACE WITH NORMAL EVIDENCE
    # ========================================================

    remaining_slots = max(
        0,
        max_documents
        - len(final_documents)
    )


    for score, index, document in (
        normal_scored[
            :remaining_slots
        ]
    ):

        document.metadata[
            "final_evidence_score"
        ] = score


        final_documents.append(
            document
        )


    # ========================================================
    # FINAL DEDUPLICATION
    # ========================================================

    final_documents = (
        deduplicate_documents(
            final_documents
        )
    )


    return final_documents[
        :max_documents
    ]


# ============================================================
# LIMIT VERY LARGE FULL-PAGE EVIDENCE
# ============================================================
#
# A full page can contain a lot of text.
#
# We do not want one page to consume the entire LLM context.
# ============================================================

def trim_evidence_text(
    text,
    max_characters=4500
):

    text = clean_text(
        text
    )


    if len(text) <= max_characters:

        return text


    return (
        text[:max_characters]
        + "\n\n[Evidence text truncated]"
    )


# ============================================================
# BUILD EVIDENCE CONTEXT
# ============================================================

def build_evidence_context(
    documents
):

    if not documents:

        return ""


    context_parts = []


    for evidence_number, document in enumerate(
        documents,
        start=1
    ):

        page = document.metadata.get(
            "page",
            "Unknown"
        )


        chunk = document.metadata.get(
            "chunk",
            "Unknown"
        )


        extraction_method = (
            document.metadata.get(
                "extraction",
                "unknown"
            )
        )


        source = (
            document.metadata.get(
                "source",
                "uploaded_pdf"
            )
        )


        protected = (
            document.metadata.get(
                "protected_evidence",
                False
            )
        )


        text = trim_evidence_text(
            document.page_content
        )


        # ====================================================
        # EVIDENCE HEADER
        # ====================================================

        header = (

            f"[EVIDENCE {evidence_number}]\n"

            f"PAGE: {page}\n"

            f"CHUNK: {chunk}\n"

            f"EXTRACTION: {extraction_method}\n"

            f"SOURCE: {source}\n"

            f"STRUCTURAL_EVIDENCE: {protected}\n"
        )


        context_part = (

            header
            + "\n"
            + text
        )


        context_parts.append(
            context_part
        )


    return "\n\n".join(
        context_parts
    )


# ============================================================
# GET EVIDENCE SOURCE PAGES
# ============================================================

def get_evidence_pages(
    documents
):

    pages = []


    for document in documents:

        page = document.metadata.get(
            "page"
        )


        if (
            isinstance(page, int)
            and
            page not in pages
        ):

            pages.append(
                page
            )


    return sorted(
        pages
    )


# ============================================================
# DETECT WHETHER EVIDENCE CONTAINS OCR
# ============================================================

def evidence_contains_ocr(
    documents
):

    for document in documents:

        extraction_method = (
            document.metadata.get(
                "extraction",
                ""
            )
        )


        if extraction_method == "ocr":

            return True


    return False


# ============================================================
# CHECK EVIDENCE SIZE
# ============================================================

def calculate_context_size(
    documents
):

    total_characters = 0


    for document in documents:

        total_characters += len(
            document.page_content
        )


    return total_characters


# ============================================================
# PREPARE FINAL EVIDENCE
# ============================================================

def prepare_final_evidence(
    question,
    normal_retrieval_documents,
    structural_documents,
    all_documents
):

    # ========================================================
    # MERGE NORMAL + STRUCTURAL RESULTS
    # ========================================================

    combined_documents = (
        merge_retrieved_evidence(

            normal_documents=
                normal_retrieval_documents,

            structural_documents=
                structural_documents,

            all_documents=
                all_documents,

            max_documents=50
        )
    )


    # ========================================================
    # RERANK
    # ========================================================

    final_evidence = (
        rerank_evidence(

            question=
                question,

            documents=
                combined_documents,

            max_documents=35
        )
    )


    return final_evidence
# ============================================================
# PART 9
# STRICT GROUNDED ANSWER GENERATION
# ============================================================


# ============================================================
# GROUNDING RULES
# ============================================================

GROUNDING_RULES = """
You are a high-precision evidence-grounded document assistant.

CRITICAL RULES:

1. Answer ONLY from the supplied DOCUMENT EVIDENCE.

2. Never use outside knowledge, memory, assumptions,
   common knowledge, or prior knowledge.

3. The user's question is NOT evidence.

4. Never invent:
   - names
   - numbers
   - dates
   - percentages
   - findings
   - recommendations
   - conclusions
   - list items
   - organizations
   - acronyms
   - statistics
   - quotations

5. Every factual claim in your answer must be directly
   supported by the supplied evidence.

6. Evidence may come from multiple pages.
   Combine it carefully when necessary.

7. Preserve important document wording including:
   - official names
   - acronyms
   - numbers
   - dates
   - percentages
   - units
   - terminology

8. If evidence conflicts, explicitly state that the
   document contains conflicting information.

9. OCR text may contain errors.
   If a critical word, name, number, date, or value is
   unclear or corrupted, do not guess what it should be.

10. Never create an item simply to satisfy a requested count.

11. If the user asks for N items but fewer than N items
    are explicitly supported by the evidence:
       - return only the supported items
       - state that the remaining item(s) could not be
         verified from the supplied document evidence

12. Never use unrelated sections to fill a missing list item.

13. Never convert a general initiative, heading, paragraph,
    or topic into a requested list item unless the document
    explicitly supports that interpretation.

14. Add page citations after factual claims using:

    [Page X]

15. Use ONLY page numbers present in the supplied evidence.

16. If there is not enough evidence to answer the question,
    respond exactly:

    I couldn't verify this answer from the uploaded document.

Accuracy is more important than answering.
"""


# ============================================================
# BUILD COUNT-SENSITIVE INSTRUCTIONS
# ============================================================

def build_count_instruction(
    question
):

    expected_count = (
        detect_expected_count(
            question
        )
    )


    concept = (
        detect_list_concept(
            question
        )
    )


    if not expected_count:

        return ""


    concept_text = (
        concept
        if concept
        else "items"
    )


    return f"""
The user's question appears to request {expected_count}
{concept_text}(s).

IMPORTANT:

The number {expected_count} comes from the QUESTION.

It is NOT evidence that the uploaded document actually
contains {expected_count} supported {concept_text}(s).

You must independently verify each item from DOCUMENT EVIDENCE.

Do not invent or infer missing items.

If only M items are explicitly supported where M is less
than {expected_count}, return those M verified items only.

Then clearly state:

"I could verify only M of the requested {expected_count}
{concept_text}(s) from the uploaded document evidence."

Do not fill the remaining positions using related topics.
"""


# ============================================================
# BUILD OCR INSTRUCTION
# ============================================================

def build_ocr_instruction(
    evidence_documents
):

    if not evidence_contains_ocr(
        evidence_documents
    ):

        return ""


    return """
Some supplied evidence was extracted using OCR.

OCR can contain character recognition errors.

For OCR evidence:

- Do not silently correct uncertain text.
- Do not guess unclear names.
- Do not guess unclear numbers.
- Do not guess unclear dates.
- Do not guess unclear percentages.
- Do not guess unclear acronyms.

If the answer depends on text that appears corrupted or
ambiguous, say that the relevant scanned text could not
be reliably verified.
"""


# ============================================================
# GENERATE GROUNDED ANSWER
# ============================================================

def generate_grounded_answer(
    question,
    evidence_documents
):

    if not evidence_documents:

        return PDF_FALLBACK


    # ========================================================
    # BUILD EVIDENCE CONTEXT
    # ========================================================

    evidence_context = (
        build_evidence_context(
            evidence_documents
        )
    )


    if not evidence_context.strip():

        return PDF_FALLBACK


    # ========================================================
    # QUESTION TYPE
    # ========================================================

    question_type = (
        detect_question_type(
            question
        )
    )


    # ========================================================
    # COUNT RULE
    # ========================================================

    count_instruction = (
        build_count_instruction(
            question
        )
    )


    # ========================================================
    # OCR RULE
    # ========================================================

    ocr_instruction = (
        build_ocr_instruction(
            evidence_documents
        )
    )


    # ========================================================
    # FINAL PROMPT
    # ========================================================

    prompt = f"""
{GROUNDING_RULES}

QUESTION TYPE:

{question_type}


USER QUESTION:

{question}


COUNT-SENSITIVE INSTRUCTIONS:

{count_instruction}


OCR-SENSITIVE INSTRUCTIONS:

{ocr_instruction}


DOCUMENT EVIDENCE:

{evidence_context}


ANSWERING INSTRUCTIONS:

Read ALL supplied evidence before answering.

Do not answer from only the first evidence block.

Look for evidence across different pages.

If the answer is a list, include only distinct list items
that are explicitly supported.

If multiple evidence blocks describe the same item,
combine them rather than counting them as separate items.

If an item has an official name followed by an acronym,
preserve both when supported.

Do not mention retrieval scores, chunk numbers,
RRF, embeddings, BM25, or internal processing.

Do not say "according to my knowledge."

Do not say "probably."

Do not guess.

Return the final user-facing answer only.
"""


    # ========================================================
    # CALL LLM
    # ========================================================

    answer = invoke_llm(
        prompt,
        fallback=PDF_FALLBACK
    )


    answer = answer.strip()


    if not answer:

        return PDF_FALLBACK


    return answer


# ============================================================
# EXTRACT PAGE CITATIONS FROM GENERATED ANSWER
# ============================================================

def extract_answer_page_citations(
    answer
):

    if not answer:

        return []


    matches = re.findall(
        r"\[Page\s+(\d+)\]",
        answer,
        flags=re.IGNORECASE
    )


    pages = []


    for match in matches:

        try:

            page = int(
                match
            )


            if page not in pages:

                pages.append(
                    page
                )


        except Exception:

            continue


    return sorted(
        pages
    )


# ============================================================
# VALIDATE PAGE CITATIONS
# ============================================================

def validate_answer_page_citations(
    answer,
    evidence_documents
):

    answer_pages = (
        extract_answer_page_citations(
            answer
        )
    )


    evidence_pages = set(
        get_evidence_pages(
            evidence_documents
        )
    )


    # No citations generated
    if not answer_pages:

        return False


    # Check every generated page
    for page in answer_pages:

        if page not in evidence_pages:

            return False


    return True


# ============================================================
# BASIC UNSAFE LANGUAGE CHECK
# ============================================================

def contains_guessing_language(
    answer
):

    if not answer:

        return False


    answer_lower = (
        answer.lower()
    )


    suspicious_phrases = [

        "i assume",

        "i think",

        "probably",

        "likely",

        "might be",

        "could be",

        "seems to be",

        "perhaps",

        "possibly",

        "based on general knowledge",

        "from my knowledge",

        "not mentioned but",

        "no specific",
    ]


    for phrase in suspicious_phrases:

        if phrase in answer_lower:

            return True


    return False
# ============================================================
# PART 10
# ANSWER VERIFICATION + COMPLETENESS CHECK
# + SECOND RETRIEVAL PASS
# ============================================================


# ============================================================
# VERIFY DRAFT ANSWER AGAINST EVIDENCE
# ============================================================

def verify_answer_against_evidence(
    question,
    draft_answer,
    evidence_documents
):

    if not draft_answer:
        return {
            "supported": False,
            "complete": False,
            "supported_item_count": 0,
            "missing_information": [],
            "unsupported_claims": [],
            "needs_second_pass": True
        }


    evidence_context = build_evidence_context(
        evidence_documents
    )


    expected_count = detect_expected_count(
        question
    )


    list_concept = detect_list_concept(
        question
    )


    prompt = f"""
You are a strict answer verifier.

You are NOT answering the user's question.

You are checking whether a DRAFT ANSWER is fully supported
by DOCUMENT EVIDENCE.

USER QUESTION:

{question}


EXPECTED COUNT FROM QUESTION:

{expected_count}


LIST CONCEPT:

{list_concept}


DRAFT ANSWER:

{draft_answer}


DOCUMENT EVIDENCE:

{evidence_context}


VERIFICATION RULES:

1. The document evidence is the only source of truth.

2. The user's requested count is NOT evidence.

3. Check every factual claim in the draft answer.

4. A claim is supported only when the supplied evidence
   directly supports it.

5. Do not use outside knowledge.

6. Do not repair unsupported claims yourself.

7. For list questions, count only distinct supported items.

8. If the user requests N items and fewer than N can be
   verified, mark complete as false.

9. If the draft invents an item to reach the requested
   count, place that claim in unsupported_claims.

10. If the answer appears incomplete but relevant evidence
    may exist elsewhere in the document, describe what
    information should be searched for again.

11. If OCR text is ambiguous, treat the uncertain claim
    as unsupported.

Return JSON only.

Required format:

{{
    "supported": true,
    "complete": true,
    "supported_item_count": 0,
    "missing_information": [],
    "unsupported_claims": [],
    "needs_second_pass": false
}}
"""


    response = invoke_llm(
        prompt,
        fallback=""
    )


    result = safe_json(
        response
    )


    if not isinstance(
        result,
        dict
    ):
        return {
            "supported": False,
            "complete": False,
            "supported_item_count": 0,
            "missing_information": [
                "Verification could not be completed reliably."
            ],
            "unsupported_claims": [],
            "needs_second_pass": True
        }


    supported = bool(
        result.get(
            "supported",
            False
        )
    )


    complete = bool(
        result.get(
            "complete",
            False
        )
    )


    try:
        supported_item_count = int(
            result.get(
                "supported_item_count",
                0
            )
        )
    except Exception:
        supported_item_count = 0


    missing_information = result.get(
        "missing_information",
        []
    )


    unsupported_claims = result.get(
        "unsupported_claims",
        []
    )


    if not isinstance(
        missing_information,
        list
    ):
        missing_information = []


    if not isinstance(
        unsupported_claims,
        list
    ):
        unsupported_claims = []


    needs_second_pass = bool(
        result.get(
            "needs_second_pass",
            False
        )
    )


    # ========================================================
    # DETERMINISTIC COUNT SAFETY
    # ========================================================

    if (
        expected_count is not None
        and
        supported_item_count < expected_count
    ):
        complete = False
        needs_second_pass = True


    # Unsupported claims always require another attempt
    if unsupported_claims:
        supported = False
        needs_second_pass = True


    return {
        "supported": supported,
        "complete": complete,
        "supported_item_count": supported_item_count,
        "missing_information": missing_information,
        "unsupported_claims": unsupported_claims,
        "needs_second_pass": needs_second_pass
    }


# ============================================================
# BUILD SECOND-PASS SEARCH QUERIES
# ============================================================

def build_second_pass_queries(
    question,
    verification_result
):

    queries = [
        question
    ]


    expected_count = detect_expected_count(
        question
    )


    concept = detect_list_concept(
        question
    )


    # ========================================================
    # SEARCH FOR MISSING INFORMATION IDENTIFIED BY VERIFIER
    # ========================================================

    missing_information = (
        verification_result.get(
            "missing_information",
            []
        )
    )


    for missing in missing_information:

        missing = str(
            missing
        ).strip()


        if missing:
            queries.append(
                missing
            )


    # ========================================================
    # SEARCH UNSUPPORTED CLAIM TERMS
    # ========================================================

    unsupported_claims = (
        verification_result.get(
            "unsupported_claims",
            []
        )
    )


    for claim in unsupported_claims:

        claim = str(
            claim
        ).strip()


        if claim:
            queries.append(
                claim
            )


    # ========================================================
    # GENERIC NUMBERED SEARCH
    # ========================================================

    if (
        concept
        and
        expected_count
    ):

        for number in range(
            1,
            expected_count + 1
        ):

            queries.extend([

                f"{concept} {number}",

                f"{concept}: {number}",

                f"{concept} - {number}",

                f"{concept} number {number}"
            ])


    # ========================================================
    # ASK LLM FOR TARGETED RECOVERY QUERIES
    # ========================================================

    prompt = f"""
Generate targeted document-search queries for information
that may have been missed during the first retrieval pass.

Do NOT answer the user's question.

USER QUESTION:

{question}

VERIFICATION RESULT:

{json.dumps(
    verification_result,
    ensure_ascii=False
)}

Generate short search queries designed to locate missing
evidence elsewhere in the SAME document.

Focus on:

- missing list members
- headings
- numbered sections
- acronyms
- alternate wording
- continuation pages
- tables or sections containing the missing information

Do not invent factual answers.

Return JSON only:

{{
    "queries": [
        "query 1",
        "query 2",
        "query 3",
        "query 4"
    ]
}}
"""


    response = invoke_llm(
        prompt,
        fallback=""
    )


    parsed = safe_json(
        response
    )


    if isinstance(
        parsed,
        dict
    ):

        generated = parsed.get(
            "queries",
            []
        )


        if isinstance(
            generated,
            list
        ):

            for query in generated:

                query = str(
                    query
                ).strip()


                if query:
                    queries.append(
                        query
                    )


    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    unique_queries = []


    for query in queries:

        query = str(
            query
        ).strip()


        if not query:
            continue


        if query.lower() not in {
            item.lower()
            for item in unique_queries
        }:

            unique_queries.append(
                query
            )


    return unique_queries[:20]


# ============================================================
# SECOND RETRIEVAL PASS
# ============================================================

def second_pass_retrieval(
    question,
    verification_result,
    vector_store,
    bm25_retriever,
    all_documents,
    page_text_map,
    page_count
):

    search_queries = (
        build_second_pass_queries(
            question,
            verification_result
        )
    )


    # ========================================================
    # HYBRID RETRIEVAL AGAIN
    # ========================================================

    chunk_results = hybrid_retrieve(

        question=question,

        search_queries=search_queries,

        vector_store=vector_store,

        bm25_retriever=bm25_retriever,

        page_count=page_count,

        vector_k=20,

        bm25_k=20,

        final_k=40
    )


    # ========================================================
    # FULL PAGE RECOVERY AGAIN
    # ========================================================

    page_results = (
        retrieve_full_page_matches(

            page_text_map=
                page_text_map,

            search_queries=
                search_queries,

            limit=12
        )
    )


    # ========================================================
    # STRUCTURAL SEARCH AGAIN
    # ========================================================

    structural_results = (
        retrieve_structural_evidence(

            question=
                question,

            all_documents=
                all_documents,

            page_text_map=
                page_text_map
        )
    )


    # ========================================================
    # MERGE EVERYTHING
    # ========================================================

    combined = []


    combined.extend(
        structural_results
    )


    combined.extend(
        chunk_results
    )


    combined.extend(
        page_results
    )


    combined = (
        deduplicate_documents(
            combined
        )
    )


    # ========================================================
    # NEIGHBOR RECOVERY
    # ========================================================

    combined = (
        expand_neighbor_chunks(

            retrieved_documents=
                combined,

            all_documents=
                all_documents,

            neighbor_distance=2,

            max_documents=60
        )
    )


    # ========================================================
    # FINAL RERANK
    # ========================================================

    combined = (
        rerank_evidence(

            question=
                question,

            documents=
                combined,

            max_documents=45
        )
    )


    return (
        combined,
        search_queries
    )


# ============================================================
# FINAL ANSWER SAFETY CHECK
# ============================================================

def final_answer_safety_check(
    question,
    answer,
    evidence_documents
):

    if not answer:
        return False


    if answer.strip() == PDF_FALLBACK:
        return True


    # ========================================================
    # CHECK FOR OBVIOUS GUESSING LANGUAGE
    # ========================================================

    if contains_guessing_language(
        answer
    ):
        return False


    # ========================================================
    # CHECK PAGE CITATIONS
    # ========================================================

    if not validate_answer_page_citations(
        answer,
        evidence_documents
    ):
        return False


    # ========================================================
    # STRICT FINAL VERIFICATION
    # ========================================================

    verification = (
        verify_answer_against_evidence(

            question=
                question,

            draft_answer=
                answer,

            evidence_documents=
                evidence_documents
        )
    )


    if not verification.get(
        "supported",
        False
    ):
        return False


    return True


# ============================================================
# COMPLETE RELIABLE PDF QUESTION ANSWERING PIPELINE
# ============================================================

def answer_pdf_question(
    question,
    vector_store,
    all_documents,
    page_text_map,
    page_count
):

    if not question.strip():

        return {
            "answer": PDF_FALLBACK,
            "source_pages": [],
            "used_second_pass": False
        }


    if not all_documents:

        return {
            "answer": PDF_FALLBACK,
            "source_pages": [],
            "used_second_pass": False
        }


    # ========================================================
    # PASS 1
    # ========================================================

    (
        first_results,
        search_queries,
        bm25_retriever
    ) = first_pass_retrieval(

        question=question,

        vector_store=vector_store,

        all_documents=all_documents,

        page_text_map=page_text_map,

        page_count=page_count
    )


    # ========================================================
    # STRUCTURAL EVIDENCE
    # ========================================================

    structural_results = (
        retrieve_structural_evidence(

            question=
                question,

            all_documents=
                all_documents,

            page_text_map=
                page_text_map
        )
    )


    # ========================================================
    # PREPARE EVIDENCE
    # ========================================================

    evidence = prepare_final_evidence(

        question=
            question,

        normal_retrieval_documents=
            first_results,

        structural_documents=
            structural_results,

        all_documents=
            all_documents
    )


    # ========================================================
    # FIRST DRAFT
    # ========================================================

    first_answer = (
        generate_grounded_answer(

            question=
                question,

            evidence_documents=
                evidence
        )
    )


    # ========================================================
    # VERIFY FIRST DRAFT
    # ========================================================

    verification = (
        verify_answer_against_evidence(

            question=
                question,

            draft_answer=
                first_answer,

            evidence_documents=
                evidence
        )
    )


    # ========================================================
    # ACCEPT FIRST PASS WHEN SAFE
    # ========================================================

    if (
        verification.get(
            "supported",
            False
        )
        and
        verification.get(
            "complete",
            False
        )
        and
        not verification.get(
            "needs_second_pass",
            False
        )
    ):

        if final_answer_safety_check(
            question,
            first_answer,
            evidence
        ):

            return {
                "answer":
                    first_answer,

                "source_pages":
                    get_evidence_pages(
                        evidence
                    ),

                "used_second_pass":
                    False
            }


    # ========================================================
    # PASS 2
    # ========================================================

    (
        second_results,
        second_queries
    ) = second_pass_retrieval(

        question=
            question,

        verification_result=
            verification,

        vector_store=
            vector_store,

        bm25_retriever=
            bm25_retriever,

        all_documents=
            all_documents,

        page_text_map=
            page_text_map,

        page_count=
            page_count
    )


    # ========================================================
    # MERGE PASS 1 + PASS 2 EVIDENCE
    # ========================================================

    combined_evidence = (
        deduplicate_documents(

            evidence
            +
            second_results
        )
    )


    combined_evidence = (
        rerank_evidence(

            question=
                question,

            documents=
                combined_evidence,

            max_documents=45
        )
    )


    # ========================================================
    # GENERATE SECOND ANSWER
    # ========================================================

    second_answer = (
        generate_grounded_answer(

            question=
                question,

            evidence_documents=
                combined_evidence
        )
    )


    # ========================================================
    # VERIFY SECOND ANSWER
    # ========================================================

    second_verification = (
        verify_answer_against_evidence(

            question=
                question,

            draft_answer=
                second_answer,

            evidence_documents=
                combined_evidence
        )
    )


    # ========================================================
    # RETURN VERIFIED SECOND ANSWER
    # ========================================================

    if second_verification.get(
        "supported",
        False
    ):

        if final_answer_safety_check(
            question,
            second_answer,
            combined_evidence
        ):

            return {
                "answer":
                    second_answer,

                "source_pages":
                    get_evidence_pages(
                        combined_evidence
                    ),

                "used_second_pass":
                    True
            }


    # ========================================================
    # SAFE FAILURE
    # ========================================================

    return {
        "answer":
            PDF_FALLBACK,

        "source_pages":
            get_evidence_pages(
                combined_evidence
            ),

        "used_second_pass":
            True
    }
# ============================================================
# PART 11 - UPDATED
# UNIFIED FILE UPLOAD + PDF PROCESSING + ROUTING SETUP
# ============================================================


# ============================================================
# SESSION STATE
# ============================================================

if "pdf_vector_store" not in st.session_state:
    st.session_state.pdf_vector_store = None

if "pdf_page_text_map" not in st.session_state:
    st.session_state.pdf_page_text_map = {}

if "pdf_file_hash" not in st.session_state:
    st.session_state.pdf_file_hash = None

if "pdf_ready" not in st.session_state:
    st.session_state.pdf_ready = False

if "active_file_type" not in st.session_state:
    st.session_state.active_file_type = None


# ============================================================
# FILE TYPE HELPER
# ============================================================

def get_uploaded_file_extension(filename):

    if not filename:
        return ""

    filename = filename.lower().strip()

    if "." not in filename:
        return ""

    return filename.rsplit(".", 1)[-1]


def detect_uploaded_file_type(filename):

    extension = get_uploaded_file_extension(
        filename
    )

    if extension == "pdf":
        return "pdf"

    if extension == "csv":
        return "csv"

    if extension in ["xlsx", "xls"]:
        return "excel"

    return "unsupported"


# ============================================================
# SIDEBAR - ONE UPLOADER FOR EVERYTHING
# ============================================================

with st.sidebar:

    st.header("📂 Upload File")

    uploaded_file = st.file_uploader(
        "Upload PDF, CSV, or Excel",
        type=[
            "pdf",
            "csv",
            "xlsx",
            "xls"
        ],
        key="main_file_uploader"
    )

    st.caption(
        "PDF: document Q&A • "
        "CSV/Excel: full-dataset analytics"
    )


# ============================================================
# DETECT CURRENT FILE TYPE
# ============================================================

uploaded_type = None


if uploaded_file is not None:

    uploaded_type = (
        detect_uploaded_file_type(
            uploaded_file.name
        )
    )


# ============================================================
# PDF PROCESSING
# ============================================================

if (
    uploaded_file is not None
    and
    uploaded_type == "pdf"
):

    st.session_state.active_file_type = "pdf"

    file_bytes = uploaded_file.getvalue()

    current_hash = create_file_hash(
        file_bytes
    )


    # ========================================================
    # PROCESS ONLY WHEN PDF CHANGES
    # ========================================================

    if (
        st.session_state.pdf_file_hash
        != current_hash
    ):

        st.session_state.pdf_ready = False

        st.session_state.messages = []

        st.session_state.current_file = (
            uploaded_file.name
        )


        with st.spinner(
            "Reading, indexing and preparing your PDF..."
        ):

            try:

                (
                    documents,
                    page_count,
                    extraction_report,
                    page_text_map
                ) = extract_and_chunk_pdf(
                    file_bytes
                )


                # =================================================
                # PDF COULD NOT BE READ
                # =================================================

                if not documents:

                    st.session_state.pdf_documents = []

                    st.session_state.pdf_page_count = (
                        page_count
                    )

                    st.session_state.pdf_extraction_report = (
                        extraction_report
                    )

                    st.session_state.pdf_page_text_map = {}

                    st.session_state.pdf_vector_store = None

                    st.session_state.pdf_file_hash = (
                        current_hash
                    )

                    st.session_state.pdf_ready = False


                    st.error(
                        "I couldn't extract enough reliable "
                        "text from this PDF."
                    )


                # =================================================
                # PDF READY
                # =================================================

                else:

                    vector_store = (
                        create_vector_store(
                            current_hash,
                            documents
                        )
                    )


                    st.session_state.pdf_documents = (
                        documents
                    )

                    st.session_state.pdf_page_count = (
                        page_count
                    )

                    st.session_state.pdf_extraction_report = (
                        extraction_report
                    )

                    st.session_state.pdf_page_text_map = (
                        page_text_map
                    )

                    st.session_state.pdf_vector_store = (
                        vector_store
                    )

                    st.session_state.pdf_file_hash = (
                        current_hash
                    )

                    st.session_state.pdf_ready = True


            except Exception as error:

                st.session_state.pdf_ready = False

                st.session_state.pdf_vector_store = None


                st.error(
                    "The PDF could not be processed."
                )


                with st.expander(
                    "Technical details"
                ):

                    st.code(
                        str(error)
                    )


# ============================================================
# UNSUPPORTED FILE
# ============================================================

elif (
    uploaded_file is not None
    and
    uploaded_type == "unsupported"
):

    st.session_state.active_file_type = None

    st.error(
        "Unsupported file type. "
        "Please upload PDF, CSV, XLSX, or XLS."
    )


# ============================================================
# PDF SIDEBAR STATUS
# ============================================================

if (
    st.session_state.active_file_type == "pdf"
    and
    st.session_state.pdf_ready
):

    with st.sidebar:

        report = (
            st.session_state
            .pdf_extraction_report
        )


        st.success(
            "✅ PDF ready"
        )


        st.write(
            f"**File:** "
            f"{st.session_state.current_file}"
        )


        st.write(
            f"**Pages:** "
            f"{report.get('total_pages', 0)}"
        )


        st.write(
            f"**Readable pages:** "
            f"{report.get('readable_pages', 0)}"
        )


        st.write(
            f"**Chunks:** "
            f"{report.get('total_chunks', 0)}"
        )


        # ====================================================
        # OCR STATUS
        # ====================================================

        ocr_pages = report.get(
            "ocr_pages",
            []
        )


        if ocr_pages:

            st.info(
                f"🔎 OCR used on "
                f"{len(ocr_pages)} page(s)"
            )


        # ====================================================
        # UNREADABLE PAGES
        # ====================================================

        unreadable_pages = report.get(
            "unreadable_pages",
            []
        )


        if unreadable_pages:

            pages_text = ", ".join(
                str(page)
                for page in unreadable_pages
            )


            st.warning(
                f"Unreadable pages: "
                f"{pages_text}"
            )


        # ====================================================
        # CLEAR CHAT
        # ====================================================

        if st.button(
            "🗑️ Clear Chat",
            key="clear_pdf_chat",
            use_container_width=True
        ):

            st.session_state.messages = []

            st.rerun()


# ============================================================
# PDF MAIN INTERFACE
# ============================================================

if (
    st.session_state.active_file_type == "pdf"
):

    if not st.session_state.pdf_ready:

        st.info(
            "Processing the uploaded PDF..."
        )


    else:

        st.success(
            f"📄 Ready to chat with "
            f"**{st.session_state.current_file}**"
        )


        st.caption(
            "Answers are grounded in retrieved document "
            "evidence and verified before being returned."
        )


        # ====================================================
        # CHAT HISTORY
        # ====================================================

        for message in (
            st.session_state.messages
        ):

            role = message.get(
                "role",
                "assistant"
            )


            with st.chat_message(
                role
            ):

                st.markdown(
                    message.get(
                        "content",
                        ""
                    )
                )


                if role == "assistant":

                    source_pages = (
                        message.get(
                            "source_pages",
                            []
                        )
                    )


                    if source_pages:

                        pages_text = ", ".join(
                            str(page)
                            for page in source_pages
                        )


                        st.caption(
                            f"📚 Evidence pages: "
                            f"{pages_text}"
                        )


                    if message.get(
                        "used_second_pass",
                        False
                    ):

                        st.caption(
                            "🔎 Additional evidence search "
                            "was performed."
                        )


        # ====================================================
        # PDF CHAT INPUT
        # ====================================================

        user_question = st.chat_input(
            "Ask anything about this PDF...",
            key="pdf_chat_input"
        )


        if user_question:

            # ================================================
            # USER MESSAGE
            # ================================================

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": user_question
                }
            )


            with st.chat_message(
                "user"
            ):

                st.markdown(
                    user_question
                )


            # ================================================
            # ASSISTANT
            # ================================================

            with st.chat_message(
                "assistant"
            ):

                with st.spinner(
                    "Searching and verifying "
                    "document evidence..."
                ):

                    try:

                        result = (
                            answer_pdf_question(

                                question=
                                    user_question,

                                vector_store=
                                    st.session_state
                                    .pdf_vector_store,

                                all_documents=
                                    st.session_state
                                    .pdf_documents,

                                page_text_map=
                                    st.session_state
                                    .pdf_page_text_map,

                                page_count=
                                    st.session_state
                                    .pdf_page_count
                            )
                        )


                        answer = result.get(
                            "answer",
                            PDF_FALLBACK
                        )


                        source_pages = result.get(
                            "source_pages",
                            []
                        )


                        used_second_pass = result.get(
                            "used_second_pass",
                            False
                        )


                        # ====================================
                        # PREFER PAGES ACTUALLY CITED
                        # IN FINAL ANSWER
                        # ====================================

                        cited_pages = (
                            extract_answer_page_citations(
                                answer
                            )
                        )


                        if cited_pages:

                            source_pages = (
                                cited_pages
                            )


                        # ====================================
                        # DISPLAY ANSWER
                        # ====================================

                        st.markdown(
                            answer
                        )


                        # ====================================
                        # DISPLAY EVIDENCE PAGES
                        # ====================================

                        if source_pages:

                            pages_text = ", ".join(
                                str(page)
                                for page in source_pages
                            )


                            st.caption(
                                f"📚 Evidence pages: "
                                f"{pages_text}"
                            )


                        # ====================================
                        # SECOND SEARCH STATUS
                        # ====================================

                        if used_second_pass:

                            st.caption(
                                "🔎 Additional evidence "
                                "search was performed."
                            )


                        # ====================================
                        # SAVE MESSAGE
                        # ====================================

                        st.session_state.messages.append(
                            {
                                "role":
                                    "assistant",

                                "content":
                                    answer,

                                "source_pages":
                                    source_pages,

                                "used_second_pass":
                                    used_second_pass
                            }
                        )


                    except Exception as error:

                        error_message = (
                            "I couldn't reliably process "
                            "this question."
                        )


                        st.error(
                            error_message
                        )


                        with st.expander(
                            "Technical details"
                        ):

                            st.code(
                                str(error)
                            )


                        st.session_state.messages.append(
                            {
                                "role":
                                    "assistant",

                                "content":
                                    error_message,

                                "source_pages":
                                    [],

                                "used_second_pass":
                                    False
                            }
                        )
                        # ============================================================
# PART 12 - UPDATED
# CSV / EXCEL PROCESSING + DATA MODE
# ============================================================


# ============================================================
# DATA SESSION STATE
# ============================================================

if "dataframe" not in st.session_state:
    st.session_state.dataframe = None

if "data_file_name" not in st.session_state:
    st.session_state.data_file_name = None

if "data_file_hash" not in st.session_state:
    st.session_state.data_file_hash = None

if "data_ready" not in st.session_state:
    st.session_state.data_ready = False

if "excel_sheet_names" not in st.session_state:
    st.session_state.excel_sheet_names = []

if "selected_excel_sheet" not in st.session_state:
    st.session_state.selected_excel_sheet = None

if "data_file_bytes" not in st.session_state:
    st.session_state.data_file_bytes = None


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

def clean_column_names(
    dataframe
):

    dataframe = dataframe.copy()

    cleaned_columns = []


    for index, column in enumerate(
        dataframe.columns
    ):

        column = str(
            column
        ).strip()


        column = re.sub(
            r"\s+",
            " ",
            column
        )


        if (
            not column
            or
            column.lower().startswith(
                "unnamed:"
            )
        ):

            column = (
                f"Column_{index + 1}"
            )


        cleaned_columns.append(
            column
        )


    # ========================================================
    # MAKE DUPLICATE COLUMN NAMES UNIQUE
    # ========================================================

    counts = defaultdict(
        int
    )

    unique_columns = []


    for column in cleaned_columns:

        counts[
            column
        ] += 1


        if counts[column] == 1:

            unique_columns.append(
                column
            )

        else:

            unique_columns.append(
                f"{column}_{counts[column]}"
            )


    dataframe.columns = (
        unique_columns
    )


    return dataframe


# ============================================================
# CLEAN DATAFRAME
# ============================================================

def clean_dataframe(
    dataframe
):

    dataframe = dataframe.copy()


    # Remove completely empty rows
    dataframe = dataframe.dropna(
        axis=0,
        how="all"
    )


    # Remove completely empty columns
    dataframe = dataframe.dropna(
        axis=1,
        how="all"
    )


    dataframe = clean_column_names(
        dataframe
    )


    dataframe = dataframe.reset_index(
        drop=True
    )


    return dataframe


# ============================================================
# LOAD CSV
# ============================================================

def load_csv_file(
    file_bytes
):

    encodings = [

        "utf-8",
        "utf-8-sig",
        "cp1252",
        "latin1"
    ]


    last_error = None


    for encoding in encodings:

        try:

            dataframe = pd.read_csv(

                io.BytesIO(
                    file_bytes
                ),

                encoding=encoding
            )


            dataframe = clean_dataframe(
                dataframe
            )


            if dataframe.empty:

                raise ValueError(
                    "The CSV does not contain "
                    "usable data."
                )


            return dataframe


        except Exception as error:

            last_error = error


    raise ValueError(
        f"Unable to read CSV file: "
        f"{last_error}"
    )


# ============================================================
# EXCEL HEADER DETECTION
# ============================================================

def detect_excel_header_row(
    file_bytes,
    sheet_name=0,
    max_rows_to_check=20
):

    preview = pd.read_excel(

        io.BytesIO(
            file_bytes
        ),

        sheet_name=sheet_name,

        header=None,

        nrows=max_rows_to_check
    )


    if preview.empty:

        return 0


    best_row = 0

    best_score = -1


    for row_index in range(
        len(preview)
    ):

        row = preview.iloc[
            row_index
        ]


        non_empty = int(
            row.notna().sum()
        )


        text_cells = 0

        unique_cells = set()


        for value in row:

            if pd.isna(value):

                continue


            text = str(
                value
            ).strip()


            if not text:

                continue


            unique_cells.add(
                text.lower()
            )


            if isinstance(
                value,
                str
            ):

                text_cells += 1


        # ====================================================
        # HEADER SCORE
        # ====================================================

        score = (

            non_empty * 2

            +

            text_cells

            +

            len(unique_cells)
            * 0.25
        )


        if score > best_score:

            best_score = score

            best_row = row_index


    return best_row


# ============================================================
# GET EXCEL SHEETS
# ============================================================

def get_excel_sheet_names(
    file_bytes
):

    excel_file = pd.ExcelFile(

        io.BytesIO(
            file_bytes
        )
    )


    return (
        excel_file.sheet_names
    )


# ============================================================
# LOAD ONE EXCEL SHEET
# ============================================================

def load_excel_sheet(
    file_bytes,
    sheet_name
):

    header_row = (
        detect_excel_header_row(

            file_bytes,

            sheet_name=
                sheet_name
        )
    )


    dataframe = pd.read_excel(

        io.BytesIO(
            file_bytes
        ),

        sheet_name=
            sheet_name,

        header=
            header_row
    )


    dataframe = (
        clean_dataframe(
            dataframe
        )
    )


    if dataframe.empty:

        raise ValueError(
            "This worksheet does not "
            "contain usable data."
        )


    return dataframe


# ============================================================
# DATASET PROFILE
# ============================================================

def get_dataframe_profile(
    dataframe
):

    if dataframe is None:

        return {}


    numeric_columns = list(

        dataframe
        .select_dtypes(
            include=np.number
        )
        .columns
    )


    non_numeric_columns = [

        column

        for column
        in dataframe.columns

        if column
        not in numeric_columns
    ]


    return {

        "rows":
            int(
                len(dataframe)
            ),

        "columns":
            int(
                len(
                    dataframe.columns
                )
            ),

        "missing_values":
            int(
                dataframe
                .isna()
                .sum()
                .sum()
            ),

        "duplicate_rows":
            int(
                dataframe
                .duplicated()
                .sum()
            ),

        "numeric_columns":
            numeric_columns,

        "non_numeric_columns":
            non_numeric_columns
    }


# ============================================================
# DATASET SCHEMA
# ============================================================

def build_dataframe_schema(
    dataframe
):

    schema_lines = []


    for column in dataframe.columns:

        series = dataframe[
            column
        ]


        dtype = str(
            series.dtype
        )


        non_null = int(
            series
            .notna()
            .sum()
        )


        unique_values = int(
            series
            .nunique(
                dropna=True
            )
        )


        missing = int(
            series
            .isna()
            .sum()
        )


        schema_lines.append(

            f"- {column} | "
            f"type={dtype} | "
            f"non_null={non_null} | "
            f"missing={missing} | "
            f"unique={unique_values}"
        )


    return "\n".join(
        schema_lines
    )


# ============================================================
# PROCESS CSV
# ============================================================

def process_csv_file(
    uploaded_file
):

    file_bytes = (
        uploaded_file.getvalue()
    )


    dataframe = (
        load_csv_file(
            file_bytes
        )
    )


    return {
        "dataframe":
            dataframe,

        "file_hash":
            create_file_hash(
                file_bytes
            ),

        "file_bytes":
            file_bytes
    }


# ============================================================
# CSV ROUTING
# ============================================================

if (
    uploaded_file is not None
    and
    uploaded_type == "csv"
):

    st.session_state.active_file_type = (
        "data"
    )


    file_bytes = (
        uploaded_file.getvalue()
    )


    current_hash = (
        create_file_hash(
            file_bytes
        )
    )


    if (
        st.session_state.data_file_hash
        != current_hash
    ):

        with st.spinner(
            "Reading your CSV dataset..."
        ):

            try:

                result = (
                    process_csv_file(
                        uploaded_file
                    )
                )


                st.session_state.dataframe = (
                    result[
                        "dataframe"
                    ]
                )


                st.session_state.data_file_hash = (
                    result[
                        "file_hash"
                    ]
                )


                st.session_state.data_file_bytes = (
                    result[
                        "file_bytes"
                    ]
                )


                st.session_state.data_file_name = (
                    uploaded_file.name
                )


                st.session_state.data_ready = True


                st.session_state.excel_sheet_names = []

                st.session_state.selected_excel_sheet = None


                st.session_state.messages = []


            except Exception as error:

                st.session_state.data_ready = False


                st.error(
                    "I couldn't read this CSV file."
                )


                with st.expander(
                    "Technical details"
                ):

                    st.code(
                        str(error)
                    )


# ============================================================
# EXCEL ROUTING
# ============================================================

elif (
    uploaded_file is not None
    and
    uploaded_type == "excel"
):

    st.session_state.active_file_type = (
        "data"
    )


    file_bytes = (
        uploaded_file.getvalue()
    )


    current_hash = (
        create_file_hash(
            file_bytes
        )
    )


    # ========================================================
    # NEW EXCEL FILE
    # ========================================================

    if (
        st.session_state.data_file_hash
        != current_hash
    ):

        try:

            sheet_names = (
                get_excel_sheet_names(
                    file_bytes
                )
            )


            if not sheet_names:

                raise ValueError(
                    "No worksheets found."
                )


            st.session_state.data_file_hash = (
                current_hash
            )


            st.session_state.data_file_bytes = (
                file_bytes
            )


            st.session_state.data_file_name = (
                uploaded_file.name
            )


            st.session_state.excel_sheet_names = (
                sheet_names
            )


            st.session_state.selected_excel_sheet = (
                sheet_names[0]
            )


            st.session_state.messages = []


        except Exception as error:

            st.session_state.data_ready = False


            st.error(
                "I couldn't read this Excel file."
            )


            with st.expander(
                "Technical details"
            ):

                st.code(
                    str(error)
                )


    # ========================================================
    # SHEET SELECTOR
    # ========================================================

    if (
        st.session_state
        .excel_sheet_names
    ):

        with st.sidebar:

            selected_sheet = (
                st.selectbox(

                    "Worksheet",

                    st.session_state
                    .excel_sheet_names,

                    index=
                        st.session_state
                        .excel_sheet_names
                        .index(
                            st.session_state
                            .selected_excel_sheet
                        )
                    if
                        st.session_state
                        .selected_excel_sheet
                        in
                        st.session_state
                        .excel_sheet_names
                    else 0,

                    key=
                        "excel_sheet_selector"
                )
            )


        # ====================================================
        # LOAD SELECTED SHEET
        # ====================================================

        if (
            selected_sheet
            !=
            st.session_state
            .selected_excel_sheet
            or
            st.session_state.dataframe
            is None
        ):

            with st.spinner(
                f"Reading worksheet: "
                f"{selected_sheet}..."
            ):

                try:

                    dataframe = (
                        load_excel_sheet(

                            st.session_state
                            .data_file_bytes,

                            selected_sheet
                        )
                    )


                    st.session_state.dataframe = (
                        dataframe
                    )


                    st.session_state.selected_excel_sheet = (
                        selected_sheet
                    )


                    st.session_state.data_ready = True


                    st.session_state.messages = []


                except Exception as error:

                    st.session_state.data_ready = False


                    st.error(
                        "I couldn't read the "
                        "selected worksheet."
                    )


                    with st.expander(
                        "Technical details"
                    ):

                        st.code(
                            str(error)
                        )


# ============================================================
# DATA MODE SIDEBAR
# ============================================================

if (
    st.session_state.active_file_type
    == "data"
    and
    st.session_state.data_ready
    and
    st.session_state.dataframe
    is not None
):

    dataframe = (
        st.session_state.dataframe
    )


    profile = (
        get_dataframe_profile(
            dataframe
        )
    )


    with st.sidebar:

        st.success(
            "✅ Dataset ready"
        )


        st.write(
            f"**File:** "
            f"{st.session_state.data_file_name}"
        )


        if (
            st.session_state
            .selected_excel_sheet
        ):

            st.write(
                f"**Sheet:** "
                f"{st.session_state.selected_excel_sheet}"
            )


        st.write(
            f"**Rows:** "
            f"{profile['rows']:,}"
        )


        st.write(
            f"**Columns:** "
            f"{profile['columns']:,}"
        )


        if st.button(
            "🗑️ Clear Data Chat",
            key="clear_data_chat",
            use_container_width=True
        ):

            st.session_state.messages = []

            st.rerun()


# ============================================================
# DATASET MAIN SCREEN
# ============================================================

if (
    st.session_state.active_file_type
    == "data"
    and
    st.session_state.data_ready
    and
    st.session_state.dataframe
    is not None
):

    dataframe = (
        st.session_state.dataframe
    )


    profile = (
        get_dataframe_profile(
            dataframe
        )
    )


    st.success(
        f"📊 Dataset ready: "
        f"**{st.session_state.data_file_name}**"
    )


    # ========================================================
    # METRICS
    # ========================================================

    col1, col2, col3, col4 = (
        st.columns(4)
    )


    col1.metric(
        "Rows",
        f"{profile['rows']:,}"
    )


    col2.metric(
        "Columns",
        f"{profile['columns']:,}"
    )


    col3.metric(
        "Missing Values",
        f"{profile['missing_values']:,}"
    )


    col4.metric(
        "Duplicate Rows",
        f"{profile['duplicate_rows']:,}"
    )


    # ========================================================
    # DATA PREVIEW
    # ========================================================

    with st.expander(
        "📋 Dataset Preview",
        expanded=True
    ):

        st.dataframe(
            dataframe.head(
                100
            ),
            use_container_width=True
        )


    # ========================================================
    # SCHEMA
    # ========================================================

    with st.expander(
        "🧩 Dataset Schema"
    ):

        st.code(
            build_dataframe_schema(
                dataframe
            )
        )


# ============================================================
# NOTHING UPLOADED
# ============================================================

if uploaded_file is None:

    st.info(
        "👈 Upload a PDF, CSV, XLSX, or XLS file to begin."
    )
    # ============================================================
# PART 13
# FULL-DATASET ANALYTICS ENGINE
# LLM PLANS -> PANDAS CALCULATES
# ============================================================


# ============================================================
# RESOLVE USER COLUMN NAME
# ============================================================

def resolve_column_name(
    requested_column,
    dataframe
):

    if not requested_column:
        return None

    requested = normalize_text(
        requested_column
    )

    # Exact normalized match
    for column in dataframe.columns:

        if normalize_text(column) == requested:
            return column

    # Partial match
    matches = []

    for column in dataframe.columns:

        normalized_column = normalize_text(
            column
        )

        if (
            requested in normalized_column
            or
            normalized_column in requested
        ):
            matches.append(column)

    if len(matches) == 1:
        return matches[0]

    return None


# ============================================================
# DETECT NUMERIC-LIKE COLUMNS
# ============================================================

def detect_numeric_columns(
    dataframe
):

    numeric_columns = []


    for column in dataframe.columns:

        series = dataframe[column]


        # Already numeric
        if pd.api.types.is_numeric_dtype(
            series
        ):

            numeric_columns.append(
                column
            )

            continue


        # Try conversion
        converted = pd.to_numeric(
            series.astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("₹", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.strip(),
            errors="coerce"
        )


        original_non_null = int(
            series.notna().sum()
        )


        converted_non_null = int(
            converted.notna().sum()
        )


        if original_non_null == 0:
            continue


        conversion_ratio = (
            converted_non_null
            /
            original_non_null
        )


        if conversion_ratio >= 0.80:

            numeric_columns.append(
                column
            )


    return numeric_columns


# ============================================================
# DETECT DATE-LIKE COLUMNS
# ============================================================

def detect_date_columns(
    dataframe
):

    date_columns = []


    date_name_hints = [
        "date",
        "time",
        "day",
        "month",
        "year",
        "created",
        "updated",
        "order date",
        "delivery date",
        "ship date"
    ]


    for column in dataframe.columns:

        series = dataframe[column]


        if pd.api.types.is_datetime64_any_dtype(
            series
        ):

            date_columns.append(
                column
            )

            continue


        normalized_name = normalize_text(
            column
        )


        has_date_hint = any(
            hint in normalized_name
            for hint in date_name_hints
        )


        # Avoid trying to convert every random text column
        if not has_date_hint:
            continue


        converted = pd.to_datetime(
            series,
            errors="coerce"
        )


        original_non_null = int(
            series.notna().sum()
        )


        converted_non_null = int(
            converted.notna().sum()
        )


        if original_non_null == 0:
            continue


        ratio = (
            converted_non_null
            /
            original_non_null
        )


        if ratio >= 0.60:

            date_columns.append(
                column
            )


    return date_columns


# ============================================================
# BUILD DATASET CONTEXT FOR PLANNER
# ============================================================

def build_dataset_context(
    dataframe
):

    numeric_columns = (
        detect_numeric_columns(
            dataframe
        )
    )


    date_columns = (
        detect_date_columns(
            dataframe
        )
    )


    schema = (
        build_dataframe_schema(
            dataframe
        )
    )


    context = f"""
DATASET ROWS:
{len(dataframe)}

DATASET COLUMNS:
{len(dataframe.columns)}

COLUMN NAMES:
{json.dumps(
    list(dataframe.columns),
    ensure_ascii=False
)}

NUMERIC-LIKE COLUMNS:
{json.dumps(
    numeric_columns,
    ensure_ascii=False
)}

DATE-LIKE COLUMNS:
{json.dumps(
    date_columns,
    ensure_ascii=False
)}

SCHEMA:
{schema}
"""


    return context


# ============================================================
# CREATE ANALYSIS PLAN
# ============================================================

def create_analysis_plan(
    question,
    dataframe
):

    dataset_context = (
        build_dataset_context(
            dataframe
        )
    )


    prompt = f"""
You are a data-analysis planner.

The user uploaded a CSV or Excel dataset.

Your job is NOT to calculate the answer.

Your job is ONLY to translate the user's question into a
safe structured analysis plan.

Pandas will execute the plan on the COMPLETE dataset.

USER QUESTION:

{question}


DATASET INFORMATION:

{dataset_context}


SUPPORTED OPERATIONS:

count
sum
average
min
max
median
nunique
groupby
top_n
bottom_n
value_counts
filter
describe


SUPPORTED FILTER OPERATORS:

equals
not_equals
greater_than
greater_or_equal
less_than
less_or_equal
contains
not_contains
in
between


IMPORTANT RULES:

1. Use ONLY columns that exist in the supplied schema.

2. Never invent column names.

3. Do not calculate results yourself.

4. Do not guess values.

5. If the question asks "how many", usually use count.

6. If the question asks total sales/revenue/amount,
   usually use sum.

7. If the question asks average/mean, use average.

8. For "highest", "top", "best", use top_n when appropriate.

9. For "lowest", "bottom", use bottom_n when appropriate.

10. If grouping is requested, set group_by.

11. Filters must be represented in the filters list.

12. Use null for fields that are not required.

13. If the question cannot be safely represented using
    the supported operations, set operation to "unsupported".

Return JSON only.

FORMAT:

{{
    "operation": "count",
    "target_column": null,
    "group_by": null,
    "filters": [],
    "sort_column": null,
    "n": 10
}}

FILTER FORMAT:

{{
    "column": "Column Name",
    "operator": "equals",
    "value": "some value"
}}
"""


    response = invoke_llm(
        prompt,
        fallback=""
    )


    plan = safe_json(
        response
    )


    if not isinstance(
        plan,
        dict
    ):

        return {
            "operation": "unsupported",
            "target_column": None,
            "group_by": None,
            "filters": [],
            "sort_column": None,
            "n": 10
        }


    return plan


# ============================================================
# CONVERT COLUMN TO NUMERIC SAFELY
# ============================================================

def get_numeric_series(
    dataframe,
    column
):

    series = dataframe[
        column
    ]


    if pd.api.types.is_numeric_dtype(
        series
    ):

        return pd.to_numeric(
            series,
            errors="coerce"
        )


    cleaned = (
        series
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
    )


    return pd.to_numeric(
        cleaned,
        errors="coerce"
    )


# ============================================================
# APPLY ONE FILTER
# ============================================================

def apply_single_filter(
    dataframe,
    filter_rule
):

    column = resolve_column_name(
        filter_rule.get(
            "column"
        ),
        dataframe
    )


    if column is None:

        raise ValueError(
            "A requested filter column "
            "could not be matched."
        )


    operator = str(
        filter_rule.get(
            "operator",
            ""
        )
    ).lower().strip()


    value = filter_rule.get(
        "value"
    )


    series = dataframe[
        column
    ]


    # ========================================================
    # EQUALS
    # ========================================================

    if operator == "equals":

        mask = (
            series.astype(str)
            .str.strip()
            .str.lower()
            ==
            str(value)
            .strip()
            .lower()
        )


    # ========================================================
    # NOT EQUALS
    # ========================================================

    elif operator == "not_equals":

        mask = (
            series.astype(str)
            .str.strip()
            .str.lower()
            !=
            str(value)
            .strip()
            .lower()
        )


    # ========================================================
    # CONTAINS
    # ========================================================

    elif operator == "contains":

        mask = (
            series.astype(str)
            .str.contains(
                str(value),
                case=False,
                na=False,
                regex=False
            )
        )


    # ========================================================
    # NOT CONTAINS
    # ========================================================

    elif operator == "not_contains":

        mask = ~(
            series.astype(str)
            .str.contains(
                str(value),
                case=False,
                na=False,
                regex=False
            )
        )


    # ========================================================
    # NUMERIC OPERATORS
    # ========================================================

    elif operator in [
        "greater_than",
        "greater_or_equal",
        "less_than",
        "less_or_equal"
    ]:

        numeric_series = (
            get_numeric_series(
                dataframe,
                column
            )
        )


        numeric_value = float(
            value
        )


        if operator == "greater_than":

            mask = (
                numeric_series
                >
                numeric_value
            )


        elif operator == "greater_or_equal":

            mask = (
                numeric_series
                >=
                numeric_value
            )


        elif operator == "less_than":

            mask = (
                numeric_series
                <
                numeric_value
            )


        else:

            mask = (
                numeric_series
                <=
                numeric_value
            )


    # ========================================================
    # IN
    # ========================================================

    elif operator == "in":

        if not isinstance(
            value,
            list
        ):

            value = [value]


        allowed_values = {
            str(item)
            .strip()
            .lower()

            for item in value
        }


        mask = (
            series.astype(str)
            .str.strip()
            .str.lower()
            .isin(
                allowed_values
            )
        )


    # ========================================================
    # BETWEEN
    # ========================================================

    elif operator == "between":

        if (
            not isinstance(
                value,
                list
            )
            or
            len(value) != 2
        ):

            raise ValueError(
                "Between filter requires "
                "two values."
            )


        numeric_series = (
            get_numeric_series(
                dataframe,
                column
            )
        )


        lower = float(
            value[0]
        )


        upper = float(
            value[1]
        )


        mask = (
            numeric_series.between(
                lower,
                upper,
                inclusive="both"
            )
        )


    else:

        raise ValueError(
            f"Unsupported filter operator: "
            f"{operator}"
        )


    return dataframe.loc[
        mask
    ].copy()


# ============================================================
# APPLY ALL FILTERS
# ============================================================

def apply_filters(
    dataframe,
    filters
):

    filtered = (
        dataframe.copy()
    )


    if not isinstance(
        filters,
        list
    ):

        return filtered


    for filter_rule in filters:

        if not isinstance(
            filter_rule,
            dict
        ):

            continue


        filtered = (
            apply_single_filter(
                filtered,
                filter_rule
            )
        )


    return filtered


# ============================================================
# EXECUTE ANALYSIS PLAN
# ============================================================

def execute_analysis_plan(
    dataframe,
    plan
):

    operation = str(
        plan.get(
            "operation",
            ""
        )
    ).lower().strip()


    supported_operations = {

        "count",
        "sum",
        "average",
        "min",
        "max",
        "median",
        "nunique",
        "groupby",
        "top_n",
        "bottom_n",
        "value_counts",
        "filter",
        "describe"
    }


    if operation not in supported_operations:

        raise ValueError(
            "This question requires an "
            "analysis operation that is "
            "not supported yet."
        )


    # ========================================================
    # FILTER COMPLETE DATASET
    # ========================================================

    working_df = (
        apply_filters(
            dataframe,
            plan.get(
                "filters",
                []
            )
        )
    )


    # ========================================================
    # RESOLVE COLUMNS
    # ========================================================

    target_column = (
        resolve_column_name(
            plan.get(
                "target_column"
            ),
            working_df
        )
        if plan.get(
            "target_column"
        )
        else None
    )


    group_by = (
        resolve_column_name(
            plan.get(
                "group_by"
            ),
            working_df
        )
        if plan.get(
            "group_by"
        )
        else None
    )


    sort_column = (
        resolve_column_name(
            plan.get(
                "sort_column"
            ),
            working_df
        )
        if plan.get(
            "sort_column"
        )
        else None
    )


    try:

        n = int(
            plan.get(
                "n",
                10
            )
        )

    except Exception:

        n = 10


    n = max(
        1,
        min(
            n,
            100
        )
    )


    # ========================================================
    # COUNT
    # ========================================================

    if operation == "count":

        if group_by:

            result = (
                working_df
                .groupby(
                    group_by,
                    dropna=False
                )
                .size()
                .reset_index(
                    name="Count"
                )
                .sort_values(
                    "Count",
                    ascending=False
                )
            )

            return {
                "type": "table",
                "result": result,
                "rows_used": len(working_df)
            }


        if target_column:

            result = int(
                working_df[
                    target_column
                ]
                .notna()
                .sum()
            )

        else:

            result = int(
                len(
                    working_df
                )
            )


        return {
            "type": "scalar",
            "result": result,
            "rows_used": len(working_df)
        }


    # ========================================================
    # NUMERIC AGGREGATIONS
    # ========================================================

    if operation in [
        "sum",
        "average",
        "min",
        "max",
        "median"
    ]:

        if not target_column:

            raise ValueError(
                "A numeric target column "
                "is required."
            )


        numeric_series = (
            get_numeric_series(
                working_df,
                target_column
            )
        )


        if numeric_series.notna().sum() == 0:

            raise ValueError(
                f"'{target_column}' does not "
                "contain usable numeric data."
            )


        # ====================================================
        # GROUPED NUMERIC AGGREGATION
        # ====================================================

        if group_by:

            temp_df = (
                working_df.copy()
            )


            temp_df[
                "__numeric_target__"
            ] = numeric_series


            grouped = (
                temp_df
                .groupby(
                    group_by,
                    dropna=False
                )[
                    "__numeric_target__"
                ]
            )


            if operation == "sum":

                result = (
                    grouped.sum()
                    .reset_index(
                        name=f"Sum of {target_column}"
                    )
                )


            elif operation == "average":

                result = (
                    grouped.mean()
                    .reset_index(
                        name=f"Average {target_column}"
                    )
                )


            elif operation == "min":

                result = (
                    grouped.min()
                    .reset_index(
                        name=f"Minimum {target_column}"
                    )
                )


            elif operation == "max":

                result = (
                    grouped.max()
                    .reset_index(
                        name=f"Maximum {target_column}"
                    )
                )


            else:

                result = (
                    grouped.median()
                    .reset_index(
                        name=f"Median {target_column}"
                    )
                )


            return {
                "type": "table",
                "result": result,
                "rows_used": len(working_df)
            }


        # ====================================================
        # SINGLE VALUE AGGREGATION
        # ====================================================

        if operation == "sum":

            result = (
                numeric_series.sum()
            )


        elif operation == "average":

            result = (
                numeric_series.mean()
            )


        elif operation == "min":

            result = (
                numeric_series.min()
            )


        elif operation == "max":

            result = (
                numeric_series.max()
            )


        else:

            result = (
                numeric_series.median()
            )


        return {
            "type": "scalar",
            "result": result,
            "rows_used": len(working_df)
        }


    # ========================================================
    # UNIQUE COUNT
    # ========================================================

    if operation == "nunique":

        if not target_column:

            raise ValueError(
                "A target column is required."
            )


        result = int(
            working_df[
                target_column
            ]
            .nunique(
                dropna=True
            )
        )


        return {
            "type": "scalar",
            "result": result,
            "rows_used": len(working_df)
        }


    # ========================================================
    # VALUE COUNTS
    # ========================================================

    if operation == "value_counts":

        if not target_column:

            raise ValueError(
                "A target column is required."
            )


        result = (
            working_df[
                target_column
            ]
            .value_counts(
                dropna=False
            )
            .head(n)
            .rename_axis(
                target_column
            )
            .reset_index(
                name="Count"
            )
        )


        return {
            "type": "table",
            "result": result,
            "rows_used": len(working_df)
        }


    # ========================================================
    # GROUP BY
    # ========================================================

    if operation == "groupby":

        if not group_by:

            raise ValueError(
                "A group-by column is required."
            )


        if target_column:

            numeric_series = (
                get_numeric_series(
                    working_df,
                    target_column
                )
            )


            temp_df = (
                working_df.copy()
            )


            temp_df[
                "__numeric_target__"
            ] = numeric_series


            result = (
                temp_df
                .groupby(
                    group_by,
                    dropna=False
                )[
                    "__numeric_target__"
                ]
                .sum()
                .reset_index(
                    name=f"Total {target_column}"
                )
                .sort_values(
                    f"Total {target_column}",
                    ascending=False
                )
            )


        else:

            result = (
                working_df
                .groupby(
                    group_by,
                    dropna=False
                )
                .size()
                .reset_index(
                    name="Count"
                )
                .sort_values(
                    "Count",
                    ascending=False
                )
            )


        return {
            "type": "table",
            "result": result,
            "rows_used": len(working_df)
        }


    # ========================================================
    # TOP / BOTTOM N
    # ========================================================

    if operation in [
        "top_n",
        "bottom_n"
    ]:

        ranking_column = (
            sort_column
            or
            target_column
        )


        if not ranking_column:

            raise ValueError(
                "A ranking column is required."
            )


        numeric_series = (
            get_numeric_series(
                working_df,
                ranking_column
            )
        )


        temp_df = (
            working_df.copy()
        )


        temp_df[
            "__sort_value__"
        ] = numeric_series


        temp_df = temp_df.dropna(
            subset=[
                "__sort_value__"
            ]
        )


        ascending = (
            operation
            ==
            "bottom_n"
        )


        result = (
            temp_df
            .sort_values(
                "__sort_value__",
                ascending=ascending
            )
            .head(n)
            .drop(
                columns=[
                    "__sort_value__"
                ]
            )
        )


        return {
            "type": "table",
            "result": result,
            "rows_used": len(working_df)
        }


    # ========================================================
    # FILTER
    # ========================================================

    if operation == "filter":

        return {
            "type": "table",
            "result": working_df,
            "rows_used": len(working_df)
        }


    # ========================================================
    # DESCRIBE
    # ========================================================

    if operation == "describe":

        if target_column:

            result = (
                working_df[
                    target_column
                ]
                .describe()
                .reset_index()
            )


            result.columns = [
                "Statistic",
                "Value"
            ]


        else:

            result = (
                working_df
                .describe(
                    include="all"
                )
                .transpose()
                .reset_index()
                .rename(
                    columns={
                        "index":
                            "Column"
                    }
                )
            )


        return {
            "type": "table",
            "result": result,
            "rows_used": len(working_df)
        }


    raise ValueError(
        "Unable to execute analysis."
    )


# ============================================================
# FORMAT ANALYTICS ANSWER
# ============================================================

def format_analytics_answer(
    question,
    plan,
    execution_result
):

    result_type = (
        execution_result.get(
            "type"
        )
    )


    result = (
        execution_result.get(
            "result"
        )
    )


    rows_used = (
        execution_result.get(
            "rows_used",
            0
        )
    )


    operation = plan.get(
        "operation",
        ""
    )


    # ========================================================
    # SCALAR RESULT
    # ========================================================

    if result_type == "scalar":

        formatted_value = (
            format_result(
                result
            )
        )


        return (
            f"**Result:** {formatted_value}\n\n"
            f"Calculated using the uploaded dataset "
            f"({rows_used:,} matching rows)."
        )


    # ========================================================
    # TABLE RESULT
    # ========================================================

    if result_type == "table":

        return (
            f"Analysis completed using the uploaded "
            f"dataset ({rows_used:,} matching rows)."
        )


    return (
        "The analysis was completed."
    )


# ============================================================
# COMPLETE DATASET QUESTION ENGINE
# ============================================================

def analyze_complete_dataset(
    question,
    dataframe
):

    if dataframe is None:

        return {
            "answer":
                "No dataset is currently loaded.",

            "result_type":
                "error",

            "table":
                None,

            "plan":
                None
        }


    if not question.strip():

        return {
            "answer":
                "Please enter a question about the dataset.",

            "result_type":
                "error",

            "table":
                None,

            "plan":
                None
        }


    # ========================================================
    # 1. CREATE PLAN
    # ========================================================

    plan = (
        create_analysis_plan(
            question,
            dataframe
        )
    )


    if (
        plan.get(
            "operation"
        )
        ==
        "unsupported"
    ):

        return {
            "answer":
                "I couldn't safely translate that question "
                "into a supported dataset calculation.",

            "result_type":
                "unsupported",

            "table":
                None,

            "plan":
                plan
        }


    # ========================================================
    # 2. EXECUTE WITH PANDAS
    # ========================================================

    try:

        execution_result = (
            execute_analysis_plan(
                dataframe,
                plan
            )
        )


    except Exception as error:

        return {
            "answer":
                f"I couldn't safely calculate that result. "
                f"{str(error)}",

            "result_type":
                "error",

            "table":
                None,

            "plan":
                plan
        }


    # ========================================================
    # 3. FORMAT RESULT
    # ========================================================

    answer = (
        format_analytics_answer(
            question,
            plan,
            execution_result
        )
    )


    result_type = (
        execution_result.get(
            "type"
        )
    )


    table = None


    if result_type == "table":

        table = (
            execution_result.get(
                "result"
            )
        )


    return {
        "answer":
            answer,

        "result_type":
            result_type,

        "table":
            table,

        "raw_result":
            execution_result.get(
                "result"
            ),

        "plan":
            plan
    }
# ============================================================
# DATASET CHAT INTERFACE
# ============================================================

if (
    st.session_state.active_file_type == "data"
    and
    st.session_state.data_ready
    and
    st.session_state.dataframe is not None
):

    st.divider()

    st.subheader(
        "💬 Ask Your Dataset"
    )


    st.caption(
        "Calculations are performed on the complete "
        "uploaded dataset using Pandas."
    )


    # ========================================================
    # SHOW DATA CHAT HISTORY
    # ========================================================

    for message in st.session_state.messages:

        role = message.get(
            "role",
            "assistant"
        )


        with st.chat_message(
            role
        ):

            st.markdown(
                message.get(
                    "content",
                    ""
                )
            )


            # =================================================
            # PREVIOUS TABLE RESULT
            # =================================================

            table = message.get(
                "table"
            )


            if isinstance(
                table,
                pd.DataFrame
            ):

                st.dataframe(
                    table,
                    use_container_width=True
                )


    # ========================================================
    # DATA CHAT INPUT
    # ========================================================

    data_question = st.chat_input(
        "Ask a question about your dataset...",
        key="data_chat_input"
    )


    if data_question:

        # ====================================================
        # USER MESSAGE
        # ====================================================

        st.session_state.messages.append(
            {
                "role":
                    "user",

                "content":
                    data_question
            }
        )


        with st.chat_message(
            "user"
        ):

            st.markdown(
                data_question
            )


        # ====================================================
        # ANALYZE COMPLETE DATASET
        # ====================================================

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Analyzing the complete dataset..."
            ):

                result = (
                    analyze_complete_dataset(

                        question=
                            data_question,

                        dataframe=
                            st.session_state
                            .dataframe
                    )
                )


            answer = result.get(
                "answer",
                "Unable to calculate the result."
            )


            table = result.get(
                "table"
            )


            st.markdown(
                answer
            )


            # =================================================
            # TABLE RESULT
            # =================================================

            if isinstance(
                table,
                pd.DataFrame
            ):

                st.dataframe(
                    table,
                    use_container_width=True
                )


            # =================================================
            # OPTIONAL ANALYSIS DETAILS
            # =================================================

            plan = result.get(
                "plan"
            )


            if isinstance(
                plan,
                dict
            ):

                with st.expander(
                    "🔍 Analysis details"
                ):

                    st.json(
                        plan
                    )


            # =================================================
            # SAVE ASSISTANT MESSAGE
            # =================================================

            st.session_state.messages.append(
                {
                    "role":
                        "assistant",

                    "content":
                        answer,

                    "table":
                        table
                }
            )
            # ============================================================
# PART 14
# ADVANCED DATE / TIME ANALYTICS
# MONTHLY TRENDS + DELIVERY / SHIPPING DURATION
# ============================================================


# ============================================================
# DATE COLUMN NAME HINTS
# ============================================================

DATE_COLUMN_HINTS = [
    "date",
    "time",
    "created",
    "updated",
    "ordered",
    "order date",
    "ship date",
    "shipping date",
    "shipped date",
    "delivery date",
    "delivered date",
    "purchase date",
    "invoice date"
]


# ============================================================
# DATE PARSING
# ============================================================

def convert_to_datetime_safe(series):

    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(
            series,
            errors="coerce"
        )

    # First normal parsing
    parsed = pd.to_datetime(
        series,
        errors="coerce"
    )

    original_non_null = int(
        series.notna().sum()
    )

    parsed_non_null = int(
        parsed.notna().sum()
    )

    if original_non_null == 0:
        return parsed

    ratio = (
        parsed_non_null
        /
        original_non_null
    )

    # Try day-first format if first attempt was weak
    if ratio < 0.60:

        parsed_day_first = pd.to_datetime(
            series,
            errors="coerce",
            dayfirst=True
        )

        if (
            parsed_day_first
            .notna()
            .sum()
            >
            parsed
            .notna()
            .sum()
        ):
            parsed = parsed_day_first

    return parsed


# ============================================================
# IMPROVED DATE COLUMN DETECTION
# ============================================================

def detect_date_columns_advanced(
    dataframe
):

    date_columns = []

    for column in dataframe.columns:

        series = dataframe[column]

        # Already datetime
        if pd.api.types.is_datetime64_any_dtype(
            series
        ):
            date_columns.append(column)
            continue

        normalized_name = normalize_text(
            column
        )

        name_has_date_hint = any(
            normalize_text(hint)
            in normalized_name
            for hint in DATE_COLUMN_HINTS
        )

        # Avoid converting arbitrary numeric IDs
        if (
            pd.api.types.is_numeric_dtype(series)
            and
            not name_has_date_hint
        ):
            continue

        # Only aggressively test likely date columns
        if not name_has_date_hint:
            continue

        parsed = convert_to_datetime_safe(
            series
        )

        original_count = int(
            series.notna().sum()
        )

        parsed_count = int(
            parsed.notna().sum()
        )

        if original_count == 0:
            continue

        ratio = (
            parsed_count
            /
            original_count
        )

        if ratio >= 0.60:
            date_columns.append(
                column
            )

    return date_columns


# ============================================================
# RESOLVE DATE COLUMN
# ============================================================

def resolve_date_column(
    requested_column,
    dataframe
):

    if requested_column:

        resolved = resolve_column_name(
            requested_column,
            dataframe
        )

        if resolved:
            return resolved

    date_columns = (
        detect_date_columns_advanced(
            dataframe
        )
    )

    if len(date_columns) == 1:
        return date_columns[0]

    return None


# ============================================================
# MONTH NAME MAP
# ============================================================

MONTH_MAP = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12
}


# ============================================================
# EXTRACT MONTH FROM QUESTION
# ============================================================

def extract_month_from_question(
    question
):

    normalized = normalize_text(
        question
    )

    tokens = normalized.split()

    for token in tokens:

        if token in MONTH_MAP:

            return MONTH_MAP[token]

    return None


# ============================================================
# EXTRACT YEAR FROM QUESTION
# ============================================================

def extract_year_from_question(
    question
):

    matches = re.findall(
        r"\b(19\d{2}|20\d{2}|21\d{2})\b",
        question
    )

    if not matches:
        return None

    try:
        return int(matches[0])

    except Exception:
        return None


# ============================================================
# DATE RANGE FILTER
# ============================================================

def apply_date_filter(
    dataframe,
    date_column,
    start_date=None,
    end_date=None,
    month=None,
    year=None
):

    if date_column not in dataframe.columns:

        raise ValueError(
            f"Date column '{date_column}' "
            "does not exist."
        )

    result = dataframe.copy()

    parsed_dates = (
        convert_to_datetime_safe(
            result[date_column]
        )
    )

    result[
        "__parsed_date__"
    ] = parsed_dates

    result = result[
        result[
            "__parsed_date__"
        ].notna()
    ].copy()


    # ========================================================
    # START DATE
    # ========================================================

    if start_date is not None:

        start = pd.to_datetime(
            start_date,
            errors="coerce"
        )

        if pd.isna(start):

            raise ValueError(
                "Invalid start date."
            )

        result = result[
            result[
                "__parsed_date__"
            ] >= start
        ]


    # ========================================================
    # END DATE
    # ========================================================

    if end_date is not None:

        end = pd.to_datetime(
            end_date,
            errors="coerce"
        )

        if pd.isna(end):

            raise ValueError(
                "Invalid end date."
            )

        result = result[
            result[
                "__parsed_date__"
            ] <= end
        ]


    # ========================================================
    # YEAR
    # ========================================================

    if year is not None:

        result = result[
            result[
                "__parsed_date__"
            ].dt.year == int(year)
        ]


    # ========================================================
    # MONTH
    # ========================================================

    if month is not None:

        result = result[
            result[
                "__parsed_date__"
            ].dt.month == int(month)
        ]


    return result


# ============================================================
# MONTHLY TREND
# ============================================================

def calculate_monthly_trend(
    dataframe,
    date_column,
    target_column=None,
    aggregation="count"
):

    working = dataframe.copy()

    working[
        "__date__"
    ] = convert_to_datetime_safe(
        working[
            date_column
        ]
    )

    working = working[
        working[
            "__date__"
        ].notna()
    ].copy()

    working[
        "Month"
    ] = (
        working[
            "__date__"
        ]
        .dt.to_period("M")
        .astype(str)
    )


    # ========================================================
    # COUNT
    # ========================================================

    if aggregation == "count":

        result = (
            working
            .groupby(
                "Month"
            )
            .size()
            .reset_index(
                name="Count"
            )
        )

        return result


    # ========================================================
    # NUMERIC TARGET REQUIRED
    # ========================================================

    if not target_column:

        raise ValueError(
            "A target column is required "
            "for this monthly aggregation."
        )


    working[
        "__numeric__"
    ] = get_numeric_series(
        working,
        target_column
    )


    grouped = (
        working
        .groupby(
            "Month"
        )[
            "__numeric__"
        ]
    )


    if aggregation == "sum":

        result = (
            grouped.sum()
            .reset_index(
                name=f"Total {target_column}"
            )
        )


    elif aggregation == "average":

        result = (
            grouped.mean()
            .reset_index(
                name=f"Average {target_column}"
            )
        )


    elif aggregation == "min":

        result = (
            grouped.min()
            .reset_index(
                name=f"Minimum {target_column}"
            )
        )


    elif aggregation == "max":

        result = (
            grouped.max()
            .reset_index(
                name=f"Maximum {target_column}"
            )
        )


    else:

        raise ValueError(
            f"Unsupported monthly "
            f"aggregation: {aggregation}"
        )


    return result


# ============================================================
# YEARLY TREND
# ============================================================

def calculate_yearly_trend(
    dataframe,
    date_column,
    target_column=None,
    aggregation="count"
):

    working = dataframe.copy()

    working[
        "__date__"
    ] = convert_to_datetime_safe(
        working[
            date_column
        ]
    )

    working = working[
        working[
            "__date__"
        ].notna()
    ].copy()

    working[
        "Year"
    ] = (
        working[
            "__date__"
        ].dt.year
    )


    if aggregation == "count":

        return (
            working
            .groupby("Year")
            .size()
            .reset_index(
                name="Count"
            )
        )


    if not target_column:

        raise ValueError(
            "A target column is required."
        )


    working[
        "__numeric__"
    ] = get_numeric_series(
        working,
        target_column
    )


    grouped = (
        working
        .groupby("Year")[
            "__numeric__"
        ]
    )


    if aggregation == "sum":

        return (
            grouped.sum()
            .reset_index(
                name=f"Total {target_column}"
            )
        )


    if aggregation == "average":

        return (
            grouped.mean()
            .reset_index(
                name=f"Average {target_column}"
            )
        )


    if aggregation == "min":

        return (
            grouped.min()
            .reset_index(
                name=f"Minimum {target_column}"
            )
        )


    if aggregation == "max":

        return (
            grouped.max()
            .reset_index(
                name=f"Maximum {target_column}"
            )
        )


    raise ValueError(
        "Unsupported yearly aggregation."
    )


# ============================================================
# DELIVERY / SHIPPING COLUMN DETECTION
# ============================================================

def find_delivery_date_columns(
    dataframe
):

    start_column = None
    end_column = None


    start_hints = [
        "order date",
        "ordered date",
        "purchase date",
        "shipping date",
        "ship date",
        "shipped date"
    ]


    end_hints = [
        "delivery date",
        "delivered date",
        "arrival date",
        "received date"
    ]


    for column in dataframe.columns:

        normalized = normalize_text(
            column
        )


        if start_column is None:

            for hint in start_hints:

                if normalize_text(
                    hint
                ) in normalized:

                    start_column = (
                        column
                    )

                    break


        if end_column is None:

            for hint in end_hints:

                if normalize_text(
                    hint
                ) in normalized:

                    end_column = (
                        column
                    )

                    break


    return (
        start_column,
        end_column
    )


# ============================================================
# CALCULATE DELIVERY DURATION
# ============================================================

def calculate_delivery_duration(
    dataframe,
    start_column,
    end_column
):

    working = dataframe.copy()


    start_dates = (
        convert_to_datetime_safe(
            working[
                start_column
            ]
        )
    )


    end_dates = (
        convert_to_datetime_safe(
            working[
                end_column
            ]
        )
    )


    working[
        "Delivery Duration (Days)"
    ] = (
        end_dates
        -
        start_dates
    ).dt.total_seconds() / 86400


    # Remove invalid / missing durations
    working = working[
        working[
            "Delivery Duration (Days)"
        ].notna()
    ].copy()


    # Negative duration is invalid
    working = working[
        working[
            "Delivery Duration (Days)"
        ] >= 0
    ].copy()


    return working


# ============================================================
# DELIVERY QUESTION DETECTOR
# ============================================================

def is_delivery_duration_question(
    question
):

    normalized = normalize_text(
        question
    )


    duration_terms = [
        "delivery duration",
        "delivery time",
        "shipping duration",
        "shipping time",
        "days to deliver",
        "days for delivery",
        "how long to deliver",
        "average delivery",
        "average shipping"
    ]


    return any(
        term in normalized
        for term in duration_terms
    )


# ============================================================
# MONTHLY TREND QUESTION DETECTOR
# ============================================================

def is_monthly_trend_question(
    question
):

    normalized = normalize_text(
        question
    )


    terms = [
        "monthly",
        "month wise",
        "monthwise",
        "by month",
        "each month",
        "monthly trend",
        "month trend"
    ]


    return any(
        term in normalized
        for term in terms
    )


# ============================================================
# YEARLY TREND QUESTION DETECTOR
# ============================================================

def is_yearly_trend_question(
    question
):

    normalized = normalize_text(
        question
    )


    terms = [
        "yearly",
        "year wise",
        "yearwise",
        "by year",
        "each year",
        "annual trend",
        "yearly trend"
    ]


    return any(
        term in normalized
        for term in terms
    )


# ============================================================
# DETECT AGGREGATION FROM QUESTION
# ============================================================

def detect_requested_aggregation(
    question
):

    normalized = normalize_text(
        question
    )


    if any(
        term in normalized
        for term in [
            "average",
            "avg",
            "mean"
        ]
    ):
        return "average"


    if any(
        term in normalized
        for term in [
            "total",
            "sum"
        ]
    ):
        return "sum"


    if any(
        term in normalized
        for term in [
            "maximum",
            "highest",
            "max"
        ]
    ):
        return "max"


    if any(
        term in normalized
        for term in [
            "minimum",
            "lowest",
            "min"
        ]
    ):
        return "min"


    return "count"


# ============================================================
# FIND COLUMN MENTIONED IN QUESTION
# ============================================================

def find_column_in_question(
    question,
    dataframe,
    exclude_columns=None
):

    if exclude_columns is None:
        exclude_columns = []


    normalized_question = (
        normalize_text(
            question
        )
    )


    candidates = []


    for column in dataframe.columns:

        if column in exclude_columns:
            continue


        normalized_column = (
            normalize_text(
                column
            )
        )


        if not normalized_column:
            continue


        if normalized_column in normalized_question:

            candidates.append(
                (
                    len(
                        normalized_column
                    ),
                    column
                )
            )


    if not candidates:
        return None


    candidates.sort(
        reverse=True
    )


    return candidates[0][1]


# ============================================================
# DETERMINISTIC DATE ANALYTICS
# ============================================================

def analyze_date_question(
    question,
    dataframe
):

    # ========================================================
    # DELIVERY / SHIPPING DURATION
    # ========================================================

    if is_delivery_duration_question(
        question
    ):

        (
            start_column,
            end_column
        ) = find_delivery_date_columns(
            dataframe
        )


        if (
            not start_column
            or
            not end_column
        ):

            return {
                "handled": True,

                "answer":
                    "I found a delivery/shipping duration "
                    "question, but I couldn't reliably identify "
                    "both the start-date and delivery-date columns.",

                "table": None
            }


        duration_df = (
            calculate_delivery_duration(
                dataframe,
                start_column,
                end_column
            )
        )


        if duration_df.empty:

            return {
                "handled": True,

                "answer":
                    "I couldn't calculate reliable delivery "
                    "durations from the available date values.",

                "table": None
            }


        average_days = (
            duration_df[
                "Delivery Duration (Days)"
            ].mean()
        )


        median_days = (
            duration_df[
                "Delivery Duration (Days)"
            ].median()
        )


        minimum_days = (
            duration_df[
                "Delivery Duration (Days)"
            ].min()
        )


        maximum_days = (
            duration_df[
                "Delivery Duration (Days)"
            ].max()
        )


        answer = (
            f"**Average delivery duration:** "
            f"{format_result(average_days)} days\n\n"
            f"**Median:** "
            f"{format_result(median_days)} days  \n"
            f"**Minimum:** "
            f"{format_result(minimum_days)} days  \n"
            f"**Maximum:** "
            f"{format_result(maximum_days)} days  \n\n"
            f"Calculated from "
            f"{len(duration_df):,} valid records using "
            f"**{start_column} → {end_column}**."
        )


        preview_columns = [
            start_column,
            end_column,
            "Delivery Duration (Days)"
        ]


        return {
            "handled": True,
            "answer": answer,
            "table":
                duration_df[
                    preview_columns
                ].head(100)
        }


    # ========================================================
    # MONTHLY TREND
    # ========================================================

    if is_monthly_trend_question(
        question
    ):

        date_column = (
            find_column_in_question(
                question,
                dataframe
            )
        )


        if (
            date_column
            not in
            detect_date_columns_advanced(
                dataframe
            )
        ):

            date_column = (
                resolve_date_column(
                    None,
                    dataframe
                )
            )


        if not date_column:

            return {
                "handled": True,

                "answer":
                    "I couldn't determine which date column "
                    "should be used for the monthly analysis.",

                "table": None
            }


        aggregation = (
            detect_requested_aggregation(
                question
            )
        )


        target_column = (
            find_column_in_question(
                question,
                dataframe,
                exclude_columns=[
                    date_column
                ]
            )
        )


        # Count does not need numeric target
        if aggregation == "count":

            target_column = None


        elif (
            target_column
            not in
            detect_numeric_columns(
                dataframe
            )
        ):

            return {
                "handled": True,

                "answer":
                    "I identified the monthly trend request, "
                    "but I couldn't reliably determine the "
                    "numeric column to aggregate.",

                "table": None
            }


        result = (
            calculate_monthly_trend(
                dataframe,
                date_column,
                target_column,
                aggregation
            )
        )


        return {
            "handled": True,

            "answer":
                f"Monthly analysis completed using "
                f"**{date_column}** across the complete dataset.",

            "table":
                result
        }


    # ========================================================
    # YEARLY TREND
    # ========================================================

    if is_yearly_trend_question(
        question
    ):

        date_column = (
            resolve_date_column(
                None,
                dataframe
            )
        )


        if not date_column:

            return {
                "handled": True,

                "answer":
                    "I couldn't determine which date column "
                    "should be used for the yearly analysis.",

                "table": None
            }


        aggregation = (
            detect_requested_aggregation(
                question
            )
        )


        target_column = (
            find_column_in_question(
                question,
                dataframe,
                exclude_columns=[
                    date_column
                ]
            )
        )


        if aggregation == "count":

            target_column = None


        elif (
            target_column
            not in
            detect_numeric_columns(
                dataframe
            )
        ):

            return {
                "handled": True,

                "answer":
                    "I couldn't reliably determine the "
                    "numeric column for the yearly analysis.",

                "table": None
            }


        result = (
            calculate_yearly_trend(
                dataframe,
                date_column,
                target_column,
                aggregation
            )
        )


        return {
            "handled": True,

            "answer":
                f"Yearly analysis completed using "
                f"**{date_column}** across the complete dataset.",

            "table":
                result
        }


    return {
        "handled": False,
        "answer": None,
        "table": None
    }


# ============================================================
# DATE-AWARE COMPLETE DATASET ENGINE
# ============================================================

def analyze_dataset_safely(
    question,
    dataframe
):

    # ========================================================
    # FIRST TRY DETERMINISTIC DATE ANALYTICS
    # ========================================================

    date_result = (
        analyze_date_question(
            question,
            dataframe
        )
    )


    if date_result.get(
        "handled",
        False
    ):

        return {
            "answer":
                date_result.get(
                    "answer"
                ),

            "result_type":
                "table"
                if isinstance(
                    date_result.get(
                        "table"
                    ),
                    pd.DataFrame
                )
                else "scalar",

            "table":
                date_result.get(
                    "table"
                ),

            "raw_result":
                date_result.get(
                    "table"
                ),

            "plan": {
                "engine":
                    "deterministic_date_analytics"
            }
        }


    # ========================================================
    # OTHERWISE USE PART 13 ENGINE
    # ========================================================

    return analyze_complete_dataset(
        question,
        dataframe
    )
# ============================================================
# PART 15
# ADVANCED NATURAL-LANGUAGE FILTERS
# DATE + MONTH + YEAR + MULTIPLE CONDITIONS
# ============================================================


# ============================================================
# DATE FILTER INTENT DETECTION
# ============================================================

def has_date_filter_intent(question):

    normalized = normalize_text(question)

    month_found = (
        extract_month_from_question(question)
        is not None
    )

    year_found = (
        extract_year_from_question(question)
        is not None
    )

    date_terms = [
        "before",
        "after",
        "between",
        "from",
        "until",
        "since",
        "during",
        "on date",
        "in month",
        "in year"
    ]

    term_found = any(
        term in normalized
        for term in date_terms
    )

    return (
        month_found
        or
        year_found
        or
        term_found
    )


# ============================================================
# EXTRACT EXPLICIT DATE STRINGS
# ============================================================

def extract_dates_from_question(question):

    dates = []

    # ========================================================
    # YYYY-MM-DD
    # ========================================================

    patterns = [
        r"\b\d{4}-\d{1,2}-\d{1,2}\b",

        # DD/MM/YYYY
        r"\b\d{1,2}/\d{1,2}/\d{4}\b",

        # DD-MM-YYYY
        r"\b\d{1,2}-\d{1,2}-\d{4}\b",

        # 1 January 2026
        r"\b\d{1,2}\s+"
        r"(?:January|February|March|April|May|June|"
        r"July|August|September|October|November|December)"
        r"\s+\d{4}\b",

        # January 1 2026
        r"\b"
        r"(?:January|February|March|April|May|June|"
        r"July|August|September|October|November|December)"
        r"\s+\d{1,2},?\s+\d{4}\b"
    ]


    for pattern in patterns:

        matches = re.findall(
            pattern,
            question,
            flags=re.IGNORECASE
        )

        for match in matches:

            if isinstance(match, tuple):
                continue

            parsed = pd.to_datetime(
                match,
                errors="coerce",
                dayfirst=True
            )

            if pd.notna(parsed):

                dates.append(
                    parsed
                )


    # Remove duplicate dates
    unique_dates = []

    seen = set()


    for date in dates:

        key = str(
            date.date()
        )

        if key not in seen:

            seen.add(key)

            unique_dates.append(
                date
            )


    return unique_dates


# ============================================================
# FIND BEST DATE COLUMN FOR QUESTION
# ============================================================

def find_best_date_column(
    question,
    dataframe
):

    date_columns = (
        detect_date_columns_advanced(
            dataframe
        )
    )


    if not date_columns:

        return None


    normalized_question = (
        normalize_text(
            question
        )
    )


    # ========================================================
    # EXACT COLUMN MENTION
    # ========================================================

    candidates = []


    for column in date_columns:

        normalized_column = (
            normalize_text(
                column
            )
        )


        if (
            normalized_column
            and
            normalized_column
            in normalized_question
        ):

            candidates.append(
                (
                    len(
                        normalized_column
                    ),
                    column
                )
            )


    if candidates:

        candidates.sort(
            reverse=True
        )

        return candidates[0][1]


    # ========================================================
    # INTENT-BASED DATE COLUMN
    # ========================================================

    order_terms = [
        "order",
        "ordered",
        "purchase",
        "purchased"
    ]


    delivery_terms = [
        "delivery",
        "delivered",
        "arrival",
        "received"
    ]


    shipping_terms = [
        "shipping",
        "shipped",
        "ship"
    ]


    # Order date
    if any(
        term in normalized_question
        for term in order_terms
    ):

        for column in date_columns:

            normalized = normalize_text(
                column
            )

            if (
                "order" in normalized
                or
                "purchase" in normalized
            ):

                return column


    # Delivery date
    if any(
        term in normalized_question
        for term in delivery_terms
    ):

        for column in date_columns:

            normalized = normalize_text(
                column
            )

            if (
                "delivery" in normalized
                or
                "delivered" in normalized
                or
                "arrival" in normalized
            ):

                return column


    # Shipping date
    if any(
        term in normalized_question
        for term in shipping_terms
    ):

        for column in date_columns:

            normalized = normalize_text(
                column
            )

            if (
                "ship" in normalized
                or
                "shipping" in normalized
            ):

                return column


    # Only one possible date column
    if len(date_columns) == 1:

        return date_columns[0]


    return None


# ============================================================
# BUILD DATE FILTER SPECIFICATION
# ============================================================

def build_date_filter_spec(
    question,
    dataframe
):

    if not has_date_filter_intent(
        question
    ):

        return None


    date_column = (
        find_best_date_column(
            question,
            dataframe
        )
    )


    if not date_column:

        return None


    normalized = normalize_text(
        question
    )


    month = (
        extract_month_from_question(
            question
        )
    )


    year = (
        extract_year_from_question(
            question
        )
    )


    explicit_dates = (
        extract_dates_from_question(
            question
        )
    )


    specification = {
        "date_column":
            date_column,

        "month":
            month,

        "year":
            year,

        "start_date":
            None,

        "end_date":
            None
    }


    # ========================================================
    # BETWEEN DATE A AND DATE B
    # ========================================================

    if (
        "between" in normalized
        and
        len(explicit_dates) >= 2
    ):

        specification[
            "start_date"
        ] = explicit_dates[0]


        specification[
            "end_date"
        ] = explicit_dates[1]


        return specification


    # ========================================================
    # FROM DATE A TO DATE B
    # ========================================================

    if (
        "from" in normalized
        and
        len(explicit_dates) >= 2
    ):

        specification[
            "start_date"
        ] = explicit_dates[0]


        specification[
            "end_date"
        ] = explicit_dates[1]


        return specification


    # ========================================================
    # AFTER DATE
    # ========================================================

    if (
        "after" in normalized
        and
        explicit_dates
    ):

        specification[
            "start_date"
        ] = explicit_dates[0]


        return specification


    # ========================================================
    # SINCE DATE
    # ========================================================

    if (
        "since" in normalized
        and
        explicit_dates
    ):

        specification[
            "start_date"
        ] = explicit_dates[0]


        return specification


    # ========================================================
    # BEFORE DATE
    # ========================================================

    if (
        "before" in normalized
        and
        explicit_dates
    ):

        specification[
            "end_date"
        ] = explicit_dates[0]


        return specification


    # ========================================================
    # UNTIL DATE
    # ========================================================

    if (
        "until" in normalized
        and
        explicit_dates
    ):

        specification[
            "end_date"
        ] = explicit_dates[0]


        return specification


    return specification


# ============================================================
# APPLY ADVANCED DATE FILTER
# ============================================================

def apply_advanced_date_filter(
    dataframe,
    specification
):

    if not specification:

        return dataframe.copy()


    return apply_date_filter(

        dataframe=

            dataframe,

        date_column=

            specification[
                "date_column"
            ],

        start_date=

            specification.get(
                "start_date"
            ),

        end_date=

            specification.get(
                "end_date"
            ),

        month=

            specification.get(
                "month"
            ),

        year=

            specification.get(
                "year"
            )
    )


# ============================================================
# GET SAMPLE VALUES FOR CATEGORICAL COLUMNS
# ============================================================

def get_categorical_value_context(
    dataframe,
    max_columns=30,
    max_values=20
):

    context = {}


    count = 0


    for column in dataframe.columns:

        if count >= max_columns:
            break


        series = dataframe[
            column
        ]


        # Avoid high-cardinality numeric columns
        if pd.api.types.is_numeric_dtype(
            series
        ):
            continue


        unique_count = int(
            series.nunique(
                dropna=True
            )
        )


        if (
            unique_count == 0
            or
            unique_count > 100
        ):
            continue


        values = (
            series
            .dropna()
            .astype(str)
            .drop_duplicates()
            .head(max_values)
            .tolist()
        )


        context[
            column
        ] = values


        count += 1


    return context


# ============================================================
# CREATE ADVANCED FILTER PLAN
# ============================================================

def create_advanced_filter_plan(
    question,
    dataframe
):

    schema = (
        build_dataframe_schema(
            dataframe
        )
    )


    categorical_values = (
        get_categorical_value_context(
            dataframe
        )
    )


    prompt = f"""
You are a strict dataset-filter planner.

Your task is ONLY to identify non-date filters in the
user's question.

Do not calculate the answer.

Do not invent columns.

Do not invent category values.

USER QUESTION:

{question}


DATASET COLUMNS:

{json.dumps(
    list(dataframe.columns),
    ensure_ascii=False
)}


SCHEMA:

{schema}


KNOWN CATEGORICAL VALUES:

{json.dumps(
    categorical_values,
    ensure_ascii=False,
    default=str
)}


SUPPORTED OPERATORS:

equals
not_equals
greater_than
greater_or_equal
less_than
less_or_equal
contains
not_contains
in
between


RULES:

1. Return only filters explicitly requested by the user.

2. Do NOT create month/year/date filters here.
   Date filtering is handled separately.

3. Use only real dataset column names.

4. For categorical filters, prefer values visible in
   KNOWN CATEGORICAL VALUES.

5. Do not guess a value that is not supported by the
   question or dataset context.

6. Multiple filters represent AND conditions.

7. If there are no non-date filters, return an empty list.

8. Do not calculate totals, averages, counts or rankings.

Return JSON only:

{{
    "filters": [
        {{
            "column": "Region",
            "operator": "equals",
            "value": "South"
        }},
        {{
            "column": "Status",
            "operator": "equals",
            "value": "Delivered"
        }}
    ]
}}
"""


    response = invoke_llm(
        prompt,
        fallback=""
    )


    parsed = safe_json(
        response
    )


    if not isinstance(
        parsed,
        dict
    ):

        return []


    filters = parsed.get(
        "filters",
        []
    )


    if not isinstance(
        filters,
        list
    ):

        return []


    safe_filters = []


    for rule in filters:

        if not isinstance(
            rule,
            dict
        ):
            continue


        column = resolve_column_name(
            rule.get(
                "column"
            ),
            dataframe
        )


        if not column:
            continue


        operator = str(
            rule.get(
                "operator",
                ""
            )
        ).lower().strip()


        if operator not in [
            "equals",
            "not_equals",
            "greater_than",
            "greater_or_equal",
            "less_than",
            "less_or_equal",
            "contains",
            "not_contains",
            "in",
            "between"
        ]:
            continue


        safe_filters.append(
            {
                "column":
                    column,

                "operator":
                    operator,

                "value":
                    rule.get(
                        "value"
                    )
            }
        )


    return safe_filters


# ============================================================
# APPLY ALL NATURAL-LANGUAGE FILTERS
# ============================================================

def apply_natural_language_filters(
    question,
    dataframe
):

    working = dataframe.copy()


    filter_report = {
        "date_filter":
            None,

        "filters":
            [],

        "original_rows":
            len(dataframe),

        "remaining_rows":
            len(dataframe)
    }


    # ========================================================
    # DATE FILTER
    # ========================================================

    date_spec = (
        build_date_filter_spec(
            question,
            working
        )
    )


    if date_spec:

        working = (
            apply_advanced_date_filter(
                working,
                date_spec
            )
        )


        filter_report[
            "date_filter"
        ] = {
            "date_column":
                date_spec.get(
                    "date_column"
                ),

            "month":
                date_spec.get(
                    "month"
                ),

            "year":
                date_spec.get(
                    "year"
                ),

            "start_date":
                str(
                    date_spec.get(
                        "start_date"
                    )
                )
                if date_spec.get(
                    "start_date"
                ) is not None
                else None,

            "end_date":
                str(
                    date_spec.get(
                        "end_date"
                    )
                )
                if date_spec.get(
                    "end_date"
                ) is not None
                else None
        }


    # ========================================================
    # CATEGORY / NUMERIC FILTERS
    # ========================================================

    filters = (
        create_advanced_filter_plan(
            question,
            working
        )
    )


    if filters:

        working = (
            apply_filters(
                working,
                filters
            )
        )


    filter_report[
        "filters"
    ] = filters


    filter_report[
        "remaining_rows"
    ] = len(
        working
    )


    return (
        working,
        filter_report
    )


# ============================================================
# CREATE CALCULATION PLAN AFTER FILTERING
# ============================================================

def create_filtered_calculation_plan(
    question,
    dataframe
):

    dataset_context = (
        build_dataset_context(
            dataframe
        )
    )


    prompt = f"""
You are a strict data calculation planner.

The dataset has ALREADY been filtered according to the
user's requested date/category conditions.

Your task is ONLY to determine what calculation should
now be performed.

Do not create filters.

Do not calculate the result yourself.

USER QUESTION:

{question}


FILTERED DATASET:

{dataset_context}


SUPPORTED OPERATIONS:

count
sum
average
min
max
median
nunique
groupby
top_n
bottom_n
value_counts
filter
describe


RULES:

1. Use only real column names.

2. Never invent columns.

3. Do not create filters.

4. If the user asks "how many", use count.

5. "total sales/revenue/amount" normally means sum.

6. "average" or "mean" means average.

7. "highest" may mean max or top_n depending on context.

8. "lowest" may mean min or bottom_n.

9. Use group_by only when the user explicitly requests
   categories/groups such as "by region" or "by product".

Return JSON only:

{{
    "operation": "sum",
    "target_column": "Sales",
    "group_by": null,
    "filters": [],
    "sort_column": null,
    "n": 10
}}
"""


    response = invoke_llm(
        prompt,
        fallback=""
    )


    plan = safe_json(
        response
    )


    if not isinstance(
        plan,
        dict
    ):

        return {
            "operation":
                "unsupported",

            "target_column":
                None,

            "group_by":
                None,

            "filters":
                [],

            "sort_column":
                None,

            "n":
                10
        }


    # Filters have already been applied.
    plan[
        "filters"
    ] = []


    return plan


# ============================================================
# ADVANCED FILTERED ANALYSIS
# ============================================================

def analyze_with_advanced_filters(
    question,
    dataframe
):

    try:

        (
            filtered_df,
            filter_report
        ) = apply_natural_language_filters(
            question,
            dataframe
        )


    except Exception as error:

        return {
            "handled":
                True,

            "answer":
                f"I couldn't safely apply the requested "
                f"filters. {str(error)}",

            "table":
                None,

            "result_type":
                "error",

            "plan":
                None
        }


    # ========================================================
    # NO MATCHING RECORDS
    # ========================================================

    if filtered_df.empty:

        return {
            "handled":
                True,

            "answer":
                "No rows matched all of the requested "
                "conditions in the uploaded dataset.",

            "table":
                None,

            "result_type":
                "scalar",

            "plan": {
                "engine":
                    "advanced_filters",

                "filter_report":
                    filter_report
            }
        }


    # ========================================================
    # CALCULATION PLAN
    # ========================================================

    plan = (
        create_filtered_calculation_plan(
            question,
            filtered_df
        )
    )


    if (
        plan.get(
            "operation"
        )
        ==
        "unsupported"
    ):

        return {
            "handled":
                True,

            "answer":
                "I applied the requested filters, but "
                "couldn't safely determine the calculation.",

            "table":
                None,

            "result_type":
                "unsupported",

            "plan":
                plan
        }


    # ========================================================
    # PANDAS CALCULATION
    # ========================================================

    try:

        execution_result = (
            execute_analysis_plan(
                filtered_df,
                plan
            )
        )


    except Exception as error:

        return {
            "handled":
                True,

            "answer":
                f"I couldn't safely calculate the result. "
                f"{str(error)}",

            "table":
                None,

            "result_type":
                "error",

            "plan":
                plan
        }


    result_type = (
        execution_result.get(
            "type"
        )
    )


    raw_result = (
        execution_result.get(
            "result"
        )
    )


    # ========================================================
    # SCALAR RESULT
    # ========================================================

    if result_type == "scalar":

        answer = (
            f"**Result:** "
            f"{format_result(raw_result)}\n\n"
            f"Calculated from "
            f"**{len(filtered_df):,} matching rows** "
            f"out of {len(dataframe):,} total rows."
        )


        table = None


    # ========================================================
    # TABLE RESULT
    # ========================================================

    else:

        answer = (
            f"Analysis completed using "
            f"**{len(filtered_df):,} matching rows** "
            f"out of {len(dataframe):,} total rows."
        )


        table = (
            raw_result
            if isinstance(
                raw_result,
                pd.DataFrame
            )
            else None
        )


    return {
        "handled":
            True,

        "answer":
            answer,

        "table":
            table,

        "result_type":
            result_type,

        "raw_result":
            raw_result,

        "plan": {
            "engine":
                "advanced_filtered_analysis",

            "filter_report":
                filter_report,

            "calculation":
                plan
        }
    }


# ============================================================
# SHOULD ADVANCED FILTER ENGINE RUN?
# ============================================================

def should_use_advanced_filters(
    question
):

    normalized = normalize_text(
        question
    )


    if has_date_filter_intent(
        question
    ):

        return True


    filter_terms = [
        "where",
        "with status",
        "status is",
        "region is",
        "city is",
        "category is",
        "greater than",
        "less than",
        "more than",
        "under",
        "above",
        "below",
        "between",
        "contains",
        "only",
        "for hyderabad",
        "for bangalore",
        "for delivered",
        "for cancelled"
    ]


    return any(
        term in normalized
        for term in filter_terms
    )


# ============================================================
# FINAL SMART DATASET ENGINE
# ============================================================

def analyze_dataset_safely(
    question,
    dataframe
):

    # ========================================================
    # 1. SPECIAL DATE ANALYTICS
    # Monthly trends / yearly trends / delivery duration
    # ========================================================

    special_date_result = (
        analyze_date_question(
            question,
            dataframe
        )
    )


    if special_date_result.get(
        "handled",
        False
    ):

        table = (
            special_date_result.get(
                "table"
            )
        )


        return {
            "answer":
                special_date_result.get(
                    "answer"
                ),

            "result_type":
                "table"
                if isinstance(
                    table,
                    pd.DataFrame
                )
                else "scalar",

            "table":
                table,

            "raw_result":
                table,

            "plan": {
                "engine":
                    "deterministic_date_analytics"
            }
        }


    # ========================================================
    # 2. ADVANCED FILTERED QUESTIONS
    # ========================================================

    if should_use_advanced_filters(
        question
    ):

        filtered_result = (
            analyze_with_advanced_filters(
                question,
                dataframe
            )
        )


        if filtered_result.get(
            "handled",
            False
        ):

            return filtered_result


    # ========================================================
    # 3. NORMAL DATASET QUESTIONS
    # ========================================================

    return analyze_complete_dataset(
        question,
        dataframe
    )

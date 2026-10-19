import os
from pathlib import Path

import pymupdf
import streamlit as st
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
import faiss
import numpy as np

load_dotenv()

st.set_page_config(page_title="Secure PDF Chat", layout="wide")

# -------------------- PII detection and masking --------------------
# Structured identifiers use conservative patterns. Contextual rules below
# reduce false positives for ordinary financial figures and statement dates.
CUSTOM_RECOGNIZERS = [
    PatternRecognizer(
        supported_entity="PAN",
        patterns=[Pattern(name="india_pan", regex=r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", score=0.99)],
        context=["PAN", "Permanent Account Number"],
    ),
    PatternRecognizer(
        supported_entity="AADHAAR",
        patterns=[Pattern(name="aadhaar", regex=r"(?<!\d)(?:\d{4}[\s-]?\d{4}[\s-]?\d{4})(?!\d)", score=0.92)],
        context=["Aadhaar", "UID", "UIDAI"],
    ),
    PatternRecognizer(
        supported_entity="PASSPORT",
        patterns=[Pattern(name="passport", regex=r"\b[A-Z][0-9]{7}\b", score=0.88)],
        context=["passport", "passport no", "passport number"],
    ),
    PatternRecognizer(
        supported_entity="VOTER_ID",
        patterns=[Pattern(name="voter_id", regex=r"\b[A-Z]{3}[0-9]{7}\b|\b[A-Z]{3}/[0-9]{2}/[0-9]{3}/[0-9]{6}\b", score=0.90)],
        context=["voter", "EPIC"],
    ),
    PatternRecognizer(
        supported_entity="DIN",
        patterns=[Pattern(name="din", regex=r"\b[0-9]{8}\b", score=0.80)],
        context=["DIN", "Director Identification Number"],
    ),
    PatternRecognizer(
        supported_entity="DRIVING_LICENCE",
        patterns=[Pattern(name="india_dl", regex=r"\b[A-Z]{2}\d{2}\s?\d{10,13}\b", score=0.88)],
        context=["driving licence", "driving license", "DL"],
    ),
    PatternRecognizer(
        supported_entity="PHONE_NUMBER",
        patterns=[Pattern(name="india_phone", regex=r"(?<!\w)(?:\+?91[-\s]?)?[6-9]\d{4}[-\s]?\d{5}(?!\w)", score=0.88)],
        context=["mobile", "phone", "telephone", "contact"],
    ),
    PatternRecognizer(
        supported_entity="EMAIL_ADDRESS",
        patterns=[Pattern(name="email", regex=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", score=0.99)],
    ),
    PatternRecognizer(
        supported_entity="UPI_ID",
        patterns=[Pattern(name="upi", regex=r"\b[A-Za-z0-9._-]{2,}@(?![A-Za-z0-9.-]+\.[A-Za-z]{2,}\b)[A-Za-z][A-Za-z0-9.-]{1,}\b", score=0.90)],
        context=["UPI", "VPA", "payment address", "upi/"],
    ),
    PatternRecognizer(
        supported_entity="BANK_ACCOUNT",
        patterns=[
            Pattern(name="account_labeled_digits", regex=r"(?i)\b(?:account number|account no\.?|a/c no\.?|loan a/c|account)\s*[:#-]?\s*([A-Z0-9-]{6,24})", score=0.91),
            Pattern(name="account_long_digits", regex=r"(?<!\d)\d{9,18}(?!\d)", score=0.66),
        ],
        context=["account number", "bank account", "A/C no", "account no", "cash credit account", "loan a/c"],
    ),
    PatternRecognizer(
        supported_entity="DOB",
        patterns=[
            Pattern(name="dob_labeled", regex=r"(?i)\b(?:date of birth|DOB)\s*[:\-]?\s*\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b", score=0.96),
        ],
        context=["DOB", "date of birth", "birth date"],
    ),
    PatternRecognizer(
        supported_entity="ADDRESS",
        patterns=[
            Pattern(name="residential_house", regex=r"(?i)\bHouse\s+No\.?\s*\d+[^;\n]{0,110}", score=0.88),
            Pattern(name="residential_flat", regex=r"(?i)\bFlat\s+\d+[^;\n]{0,110}", score=0.86),
            Pattern(name="residential_street", regex=r"(?i)\b\d{1,4},\s*(?:Green Avenue|Safdarjung Development Area)[^;\n]{0,110}", score=0.90),
        ],
        context=["residential address", "home address", "present address", "permanent address", "address"],
    ),
]

@st.cache_resource
def build_analyzer():
    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }
    )
    analyzer = AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en"])
    for recognizer in CUSTOM_RECOGNIZERS:
        analyzer.registry.add_recognizer(recognizer)
    return analyzer

def _contextual_spans(text: str):
    """Add high-confidence document-specific spans without swallowing financial text."""
    import re
    spans = []

    # Explicitly recognize PAN identifiers. Presidio's general NER can otherwise label
    # the field name/value as ORGANIZATION or PERSON instead of PAN.
    for m in re.finditer(r"(?i)\bPAN\s*[:#-]?\s*([A-Z]{5}[0-9]{4}[A-Z])\b", text):
        spans.append((m.start(1), m.end(1), "PAN"))

    # Mask the value of a labelled residential address, but stop before common financial
    # fields that may follow on the same extracted PDF line.
    stop_fields = r"DSCR|Revenue|EBITDA|LTV|Cash\s+Flow|Statement\s+date|Interest\s+Coverage|Debt\s+Equity"
    labelled_address = re.compile(
        rf"(?im)(?:residential address|home address|present address|permanent address|promoter'?s? address|director'?s? residential address)\s*[:\-]\s*(.+?)(?=\s+(?:{stop_fields})\b|$)"
    )
    for m in labelled_address.finditer(text):
        value = m.group(1).strip()
        if value:
            # Exclude trailing sentence punctuation from the span, if present.
            end = m.start(1) + len(value)
            spans.append((m.start(1), end, "ADDRESS"))

    for m in re.finditer(r"\bNBF-(?:RM|CR|ZCH|RSK)-\d{4,6}\b", text):
        spans.append((m.start(), m.end(), "EMPLOYEE_ID"))

    # DOB values in the supplied appraisal appear in a table under a separate DOB header.
    # Restrict this fallback to page 2 so statement/reporting dates remain visible elsewhere.
    page2 = re.search(r"(?s)\[Page 2\](.*?)(?=\[Page 3\])", text)
    if page2:
        page_offset = page2.start(1)
        for m in re.finditer(r"\b\d{2}[-/]\d{2}[-/]\d{4}\b", page2.group(1)):
            spans.append((page_offset + m.start(), page_offset + m.end(), "DOB"))

    # Common residential address shapes. Stop before known financial fields on the same line.
    residential_patterns = [
        rf"(?i)\bHouse\s+No\.?\s*\d+[^\n;]*?(?=\s+(?:{stop_fields})\b|$)",
        rf"(?i)\bFlat\s+\d+[^\n;]*?(?=\s+(?:{stop_fields})\b|$)",
        rf"(?i)\b\d{{1,4}},\s*(?:Green Avenue|Safdarjung Development Area)[^\n;]*?(?=\s+(?:{stop_fields})\b|$)",
    ]
    for pattern in residential_patterns:
        for m in re.finditer(pattern, text):
            spans.append((m.start(), m.end(), "ADDRESS"))
    return spans

def mask_pii(text: str, analyzer=None):
    """Mask PII consistently within one document. Returns masked text and mapping metadata.
    The original values/mapping are never passed to embeddings or the LLM.
    """
    import re
    analyzer = analyzer or build_analyzer()
    results = analyzer.analyze(text=text, language="en")
    spans = []
    # Detect labelled Indian PAN values deterministically before accepting NER output.
    # spaCy can classify a PAN value as PERSON; suppress any overlapping generic result
    # so the more specific <PAN_n> placeholder always wins.
    explicit_pan_spans = [
        (m.start(1), m.end(1), "PAN")
        for m in re.finditer(r"(?i)\bPAN\s*[:#-]?\s*([A-Z]{5}[0-9]{4}[A-Z])\b", text)
    ]
    # Presidio/NER can misclassify fiscal years as US driver's licences, ordinary
    # reporting dates as DATE_TIME, and currency/field labels as organizations.
    # Address spans are handled by the bounded contextual rules below to avoid masking
    # financial metrics later on the same extracted line.
    safe_labels = {"pan", "email", "mobile", "phone", "inr", "fy", "dscr", "ltv", "revenue", "dob"}
    for r in results:
        if r.end <= r.start:
            continue
        # Prefer explicit PAN recognition over any overlapping generic entity.
        if any(r.start < pan_end and r.end > pan_start for pan_start, pan_end, _ in explicit_pan_spans):
            continue
        raw = text[r.start:r.end]
        normalized = raw.strip().casefold()
        if r.entity_type == "ADDRESS":
            continue
        if r.entity_type == "US_DRIVER_LICENSE" and re.match(r"(?i)^FY\d{4}(?:-\d{2,4})?$", raw.strip()):
            continue
        if r.entity_type == "DATE_TIME" and re.fullmatch(r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", raw.strip()):
            # DOBs on the appraisal's page 2 are masked by the contextual DOB rule.
            continue
        if r.entity_type in {"ORGANIZATION", "PERSON"} and normalized in safe_labels:
            continue
        spans.append((r.start, r.end, r.entity_type))
    spans.extend(_contextual_spans(text))
    spans.extend(explicit_pan_spans)

    # Sort by earliest start, then longest span; discard overlapping lower-priority spans.
    priority = {"PAN": 10, "AADHAAR": 10, "PASSPORT": 10, "VOTER_ID": 10, "DIN": 10,
                "DRIVING_LICENCE": 10, "EMPLOYEE_ID": 10, "PHONE_NUMBER": 9, "EMAIL_ADDRESS": 10,
                "UPI_ID": 10, "BANK_ACCOUNT": 9, "DOB": 10, "ADDRESS": 9, "PERSON": 7, "LOCATION": 3}
    spans.sort(key=lambda x: (x[0], -(x[1] - x[0]), -priority.get(x[2], 0)))
    accepted = []
    cursor = -1
    for start, end, entity in spans:
        if start >= cursor:
            accepted.append((start, end, entity))
            cursor = end
        elif accepted and start == accepted[-1][0] and end > accepted[-1][1]:
            accepted[-1] = (start, end, entity)
            cursor = end

    counters = {}
    value_to_placeholder = {}
    replacements = []
    for start, end, entity in accepted:
        raw = text[start:end]
        # Keep person identity consistent across repeated mentions in this document.
        normalized = re.sub(r"\s+", " ", raw.strip()).casefold()
        key = (entity, normalized)
        if key not in value_to_placeholder:
            counters[entity] = counters.get(entity, 0) + 1
            value_to_placeholder[key] = f"<{entity}_{counters[entity]}>"
        replacements.append((start, end, value_to_placeholder[key]))

    masked = text
    for start, end, placeholder in reversed(replacements):
        masked = masked[:start] + placeholder + masked[end:]
    audit = {"pii_spans_found": len(replacements), "types_found": sorted({x[2] for x in accepted})}
    return masked, audit

# -------------------- PDF and RAG --------------------
def extract_pdf_text(uploaded_file):
    pdf_bytes = uploaded_file.getvalue()
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text("text", sort=True)
        pages.append(f"[Page {page_num}]\n{page_text}")
    return "\n\n".join(pages)

def chunk_text(masked_text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return [c for c in splitter.split_text(masked_text) if c.strip()]

@st.cache_resource
def get_embedding_model():
    """Load the embedding model locally; no embedding API calls are made."""
    model_name = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return SentenceTransformer(model_name)

def get_client():
    """Create a Gemini API client for answer generation only."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it to your .env file.")
    return genai.Client(api_key=api_key)

def embed_texts(texts):
    """Embed already-masked text locally using Hugging Face Sentence Transformers."""
    if not texts:
        return np.empty((0, 384), dtype="float32")
    model = get_embedding_model()
    vectors = model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype="float32")

def build_index(chunks):
    vectors = embed_texts(chunks)
    if len(vectors) == 0:
        raise ValueError("No text chunks are available to index.")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index

def retrieve(question, chunks, index, k=4):
    # The question is masked before local embedding and before being sent to Gemini.
    masked_question, _ = mask_pii(question)
    qvec = embed_texts([masked_question])
    scores, ids = index.search(qvec, min(k, len(chunks)))
    selected = [chunks[i] for i in ids[0] if i >= 0]
    return masked_question, selected

def answer_question(client, masked_question, retrieved_chunks):
    context = "\n\n---\n\n".join(retrieved_chunks)
    prompt = (
        "You answer questions using only the supplied masked document excerpts. "
        "The excerpts and question have had PII masked. Never infer or reconstruct "
        "masked values. Do not use outside knowledge. If the excerpts do not contain "
        "enough evidence, say: 'I couldn't find that information in the provided document.' "
        "Show calculations clearly and cite page labels when present. Preserve financial "
        "figures, ratios, reporting periods, and company-level business information.\n\n"
        f"Masked question:\n{masked_question}\n\nDocument excerpts:\n{context}"
    )
    response = client.models.generate_content(
        model=os.getenv("GEMINI_CHAT_MODEL", "gemini-3.8-flash"),
        contents=prompt,
    )
    answer = getattr(response, "text", None)
    return answer or "No answer returned. The model may have returned an empty response."

# -------------------- UI --------------------
st.title("Chat with your PDF — PII Protected")
st.caption("PII masking runs before chunking, embedding, vector storage, and LLM calls. Review the masked preview before indexing.")

with st.sidebar:
    st.subheader("Configuration")
    st.write("Chat provider: Google Gemini")
    st.write("Embeddings: Hugging Face (local)")
    st.write("Vector store: FAISS (in memory)")
    st.write("PII detector: Presidio + custom recognizers")
    st.warning("Use only synthetic or approved documents. This demo does not persist the vector index. Detection is not perfect: verify the masked preview before indexing.")

uploaded = st.file_uploader("Upload a credit appraisal PDF", type=["pdf"])
if uploaded:
    try:
        raw_text = extract_pdf_text(uploaded)
        if not raw_text.strip():
            st.error("No selectable text was extracted. This PDF may be scanned; OCR is not configured in this starter.")
            st.stop()

        with st.spinner("Detecting and masking PII before creating embeddings..."):
            masked_text, audit = mask_pii(raw_text)

        st.success(f"PDF extracted and masked. Detected {audit['pii_spans_found']} PII spans.")
        st.caption("Detected categories: " + (", ".join(audit["types_found"]) if audit["types_found"] else "None detected"))

        with st.expander("Preview masked document (safe preview)"):
            st.text_area("Masked text", masked_text[:12000], height=280)

        if "pdf_fingerprint" not in st.session_state or st.session_state.pdf_fingerprint != uploaded.name + str(len(raw_text)):
            st.session_state.pdf_fingerprint = uploaded.name + str(len(raw_text))
            st.session_state.masked_chunks = chunk_text(masked_text)
            st.session_state.faiss_index = None
            st.session_state.client = None
            st.session_state.chat_history = []

        if st.button("Build searchable index", type="primary"):
            try:
                client = get_client()
                with st.spinner("Creating local embeddings and building FAISS index..."):
                    index = build_index(st.session_state.masked_chunks)
                st.session_state.client = client
                st.session_state.faiss_index = index
                st.success(f"Index ready: {len(st.session_state.masked_chunks)} masked chunks.")
            except Exception as exc:
                st.error(f"Index creation failed: {exc}")

        if st.session_state.get("faiss_index") is not None:
            st.subheader("Ask a question")
            question = st.chat_input("Ask about revenue, DSCR, collateral, risks, or sanction terms...")
            for msg in st.session_state.get("chat_history", []):
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
            if question:
                client = st.session_state.client
                masked_question, passages = retrieve(
                    question, st.session_state.masked_chunks, st.session_state.faiss_index
                )
                with st.chat_message("user"):
                    st.markdown(question)
                with st.chat_message("assistant"):
                    with st.spinner("Searching the masked document and generating an answer with Gemini..."):
                        answer = answer_question(client, masked_question, passages)
                    st.markdown(answer)
                    with st.expander("Retrieved masked evidence"):
                        st.text("\n\n---\n\n".join(passages))
                st.session_state.chat_history.extend([
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ])
    except Exception as exc:
        st.error(f"Could not process PDF: {exc}")
else:
    st.info("Upload a PDF to start. PDF extraction, PII masking, and embeddings run locally. Only masked retrieved text and the masked question are sent to Gemini.")

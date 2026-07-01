import re

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.vector_store import get_chroma_db_collection
from app.config import DATA_PATH

CV_SECTION_HEADINGS = {
    "summary",
    "profile",
    "professional summary",
    "career summary",
    "experience",
    "work experience",
    "professional experience",
    "employment history",
    "skills",
    "technical skills",
    "projects",
    "project portfolio",
    "education",
    "academic credentials",
    "certifications",
    "certificates",
    "achievements",
    "awards",
    "languages",
    "interests",
    "contact",
    "personal details",
}


# loading the document
def load_documents():
    file_loader = PyPDFDirectoryLoader(DATA_PATH)
    return file_loader.load()


def _normalize_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_pdf_text(text):
    return re.sub(r"\s+", " ", text.replace("\x00", " ")).strip()


def _slug(value):
    return re.sub(r"[^a-zA-Z0-9]+", "-", str(value)).strip("-").lower()


def _looks_like_section_heading(line):
    clean_line = line.strip().strip(":")
    if len(clean_line) < 3:
        return False

    normalized = re.sub(r"[^a-zA-Z ]", "", clean_line).lower().strip()
    if normalized in CV_SECTION_HEADINGS:
        return True

    word_count = len(clean_line.split())
    return (
        word_count <= 5
        and len(clean_line) <= 60
        and clean_line.upper() == clean_line
        and "," not in clean_line
        and any(char.isalpha() for char in clean_line)
    )


def _split_project_portfolio(document):
    text = _normalize_pdf_text(document.page_content)
    portfolio_match = re.search(r"\bProject\s+Portfolio\b", text, re.IGNORECASE)
    if not portfolio_match:
        return []

    portfolio_text = text[portfolio_match.end():]
    portfolio_text = re.split(
        r"\s+(?:LinkedIn|Linkedin|GitHub|Github)\s*:",
        portfolio_text,
        maxsplit=1,
    )[0].strip()

    project_starts = list(
        re.finditer(
            r"(?:^|(?<=\. ))(?P<title>[A-Z][A-Za-z0-9&().,' ]{4,90}?)\s+-\s+"
            r"(?P<tech>[^●]{5,180})\s+●",
            portfolio_text,
        )
    )

    project_documents = []
    project_summaries = []
    for index, match in enumerate(project_starts):
        start = match.start("title")
        end = (
            project_starts[index + 1].start("title")
            if index + 1 < len(project_starts)
            else len(portfolio_text)
        )
        project_text = f"Project Portfolio: {portfolio_text[start:end].strip()}"
        project_name = match.group("title").strip()
        project_tech = _normalize_pdf_text(match.group("tech"))
        project_summaries.append(f"{project_name} - {project_tech}")

        metadata = dict(document.metadata)
        metadata["section"] = "project portfolio"
        metadata["project_name"] = project_name
        project_documents.append(
            Document(page_content=project_text, metadata=metadata)
        )

    if project_summaries:
        metadata = dict(document.metadata)
        metadata["section"] = "project portfolio"
        metadata["project_name"] = "Portfolio Overview"
        overview = "Project Portfolio Overview: " + "; ".join(project_summaries)
        project_documents.insert(
            0,
            Document(page_content=overview, metadata=metadata),
        )

    return project_documents


def _sectionize_cv_documents(raw_documents):
    section_documents = []

    for document in raw_documents:
        section_documents.extend(_split_project_portfolio(document))

        page_content = re.split(
            r"\bProject\s+Portfolio\b",
            document.page_content,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]

        lines = _normalize_text(page_content).splitlines()
        current_section = "profile"
        current_lines = []

        def flush_section():
            content = _normalize_text("\n".join(current_lines))
            if not content:
                return

            metadata = dict(document.metadata)
            metadata["section"] = current_section
            section_documents.append(
                Document(page_content=content, metadata=metadata)
            )

        for line in lines:
            stripped_line = line.strip()
            if _looks_like_section_heading(stripped_line):
                flush_section()
                current_section = stripped_line.strip(":").lower()
                current_lines = [stripped_line]
                continue

            current_lines.append(stripped_line)

        flush_section()

    return section_documents


# splitting the document
def split_document_to_chunks(raw_documents):
    section_documents = _sectionize_cv_documents(raw_documents)

    text_splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n\n",
            "\n- ",
            "\n● ",
            "\n• ",
            "\n",
            ". ",
            "; ",
            ", ",
            " ",
            "",
        ],
        chunk_size=1200,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(section_documents)
 
    documents = []
    metadata = []
    ids = []  

    for i, chunk in enumerate(chunks):
        chunk_metadata = dict(chunk.metadata)
        chunk_metadata["chunk_index"] = i

        source = _slug(chunk_metadata.get("source", "document"))
        page = chunk_metadata.get("page", "unknown")
        section = _slug(chunk_metadata.get("section", "section"))
        project = _slug(chunk_metadata.get("project_name", ""))
        id_parts = [source, f"page-{page}", section]
        if project:
            id_parts.append(project)
        id_parts.append(f"chunk-{i}")

        documents.append(_normalize_text(chunk.page_content))
        ids.append("-".join(id_parts))
        metadata.append(chunk_metadata)
 

    return {
        "documents": documents,
        "ids": ids,
        "metadata": metadata
    }


def create_collection_chroma_db():
    collection = get_chroma_db_collection()
    raw_documents = load_documents()
    data = split_document_to_chunks(raw_documents)
    existing_ids = set(collection.get().get("ids", []))
 
    collection.upsert(
        documents=data["documents"],
        metadatas=data["metadata"],
        ids=data["ids"]
    )

    stale_ids = list(existing_ids - set(data["ids"]))
    if stale_ids:
        collection.delete(ids=stale_ids)

if __name__ == "__main__":
    create_collection_chroma_db()

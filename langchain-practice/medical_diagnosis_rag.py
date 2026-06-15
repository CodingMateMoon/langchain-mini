# -*- coding: utf-8 -*-
"""
의료 진단 RAG 에이전트 — 근거 기반 답변 + 출처 표시 (CLI 실행판)

LangGraph + ChatUpstage(solar-pro) 패턴을 활용해서,
의료 지식베이스에서 근거 문서를 검색(Retrieval)하고 그 내용만으로 답변을 생성한 뒤
어떤 문서를 근거로 삼았는지 출처(citation)를 명시하는 RAG 파이프라인입니다.

흐름:
    [질문] -> retrieve -> grade --(관련O)--> generate -> cite -> [답변 + 출처]
                                \\--(관련X)--> fallback

실행:
    python medical_diagnosis_rag.py                 # 데모 질문 세트 실행
    python medical_diagnosis_rag.py "증상 질문..."   # 직접 질문
    python medical_diagnosis_rag.py --log-file out.log "증상 질문..."

필요 패키지:
    pip install langchain langchain-core langchain-community langchain-upstage \\
                langgraph faiss-cpu python-dotenv

[!] 의료 면책: 본 프로그램의 답변은 RAG/출처표시 학습용 예제이며 의학적 진단이 아닙니다.
    실제 증상이 있으면 반드시 의료 전문가의 진료를 받으세요.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Annotated, List, Optional, TypedDict

from dotenv import load_dotenv

from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

logger = logging.getLogger("medical_rag")


# ---------------------------------------------------------------------------
# 로깅 설정 — 콘솔 + 로그 파일에 동시 기록
# ---------------------------------------------------------------------------
def setup_logging(log_file: str = "medical_diagnosis_rag.log",
                  level: int = logging.INFO) -> None:
    """콘솔과 파일에 동시에 기록하는 로거를 구성한다."""
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 콘솔 핸들러
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # 파일 핸들러 (UTF-8, 누적 기록)
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.info("로그 파일: %s", log_file)


# ---------------------------------------------------------------------------
# 1. 환경 설정 — LLM & 임베딩 모델
# ---------------------------------------------------------------------------
load_dotenv()  # .env 의 UPSTAGE_API_KEY 로드

# 답변 생성용 LLM (창의성 최소화 -> 근거 기반 답변)
llm = ChatUpstage(model="solar-pro", temperature=0)

# 검색용 임베딩 모델 (질의/문서 임베딩)
embeddings = UpstageEmbeddings(model="embedding-query")


# ---------------------------------------------------------------------------
# 2. 의료 지식베이스 (학습용 요약 문서 + 출처 메타데이터)
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE = [
    Document(
        page_content=(
            "급성 인두염(목감기)은 인두 점막의 염증으로, 인후통, 발열, 연하통(삼킬 때 통증)이 "
            "주요 증상이다. 원인의 약 70~80%는 바이러스이며 세균성(주로 A군 연쇄상구균)은 "
            "소아·청소년에서 비교적 흔하다. 바이러스성은 대증치료(수분 섭취, 휴식, 해열진통제)로 "
            "호전되며 항생제는 권장되지 않는다. 세균성 인두염이 의심되면(고열, 편도 삼출물, 경부 "
            "림프절 종대, 기침 없음 = Centor 기준) 신속항원검사 또는 배양 후 항생제를 고려한다."
        ),
        metadata={"title": "급성 인두염 진료 요약", "source": "대한이비인후과학회 환자 정보",
                  "url": "https://example.org/guideline/pharyngitis", "published": "2023"},
    ),
    Document(
        page_content=(
            "인플루엔자(독감)는 갑작스러운 고열(38도 이상), 근육통, 두통, 심한 피로감, 마른기침이 "
            "특징이며 일반 감기보다 전신 증상이 심하고 급격히 시작된다. 발병 48시간 이내에 항바이러스제"
            "(오셀타미비르 등)를 투여하면 증상 기간을 단축할 수 있다. 고위험군(65세 이상, 만성질환자, "
            "임신부, 영유아)은 합병증(폐렴) 위험이 높아 조기 진료가 권장된다. 예방의 핵심은 매년 백신 접종이다."
        ),
        metadata={"title": "인플루엔자 진단과 치료", "source": "질병관리청 감염병 포털",
                  "url": "https://example.org/guideline/influenza", "published": "2024"},
    ),
    Document(
        page_content=(
            "편두통은 보통 머리 한쪽에서 발생하는 박동성(욱신거리는) 통증으로, 빛(광과민)과 소리(음과민)에 "
            "예민해지고 구역·구토를 동반할 수 있다. 일부 환자는 통증 전 시야 번쩍임 같은 조짐(전조)을 겪는다. "
            "급성기에는 트립탄 계열 또는 NSAIDs를 사용하며, 발작이 잦으면 예방 약물을 고려한다. "
            "갑자기 생긴 벼락두통, 발열·목 경직 동반, 신경학적 이상(마비·언어장애)이 있으면 응급 평가가 필요하다."
        ),
        metadata={"title": "편두통 가이드", "source": "대한신경과학회 두통 지침",
                  "url": "https://example.org/guideline/migraine", "published": "2022"},
    ),
    Document(
        page_content=(
            "위식도역류질환(GERD)은 위산이 식도로 역류하여 가슴 쓰림(흉골 뒤 화끈거림)과 신물 역류를 "
            "유발한다. 식후·눕는 자세에서 악화되며 만성 기침, 쉰 목소리가 나타날 수 있다. 생활요법으로 "
            "과식·취침 전 식사 회피, 체중 감량, 머리 쪽 침상 올리기가 권장되고, 약물로는 양성자펌프억제제(PPI)가 "
            "1차 치료다. 체중 감소, 연하곤란, 토혈·흑색변, 빈혈 같은 경고증상이 있으면 내시경 검사가 필요하다."
        ),
        metadata={"title": "위식도역류질환 진료", "source": "대한소화기학회 진료지침",
                  "url": "https://example.org/guideline/gerd", "published": "2023"},
    ),
    Document(
        page_content=(
            "급성 충수염(맹장염)은 초기에 명치·배꼽 주위의 모호한 통증으로 시작해 수 시간 내 "
            "오른쪽 아랫배(맥버니점)로 이동하는 것이 전형적이다. 식욕부진, 미열, 구역을 동반하며 "
            "걷거나 기침할 때 통증이 심해진다. 진단은 임상 소견과 혈액검사(백혈구 증가), 복부 CT/초음파로 "
            "하며 치료는 충수 절제술이 표준이다. 천공 시 복막염으로 진행할 수 있어 의심되면 즉시 응급실 평가가 필요하다."
        ),
        metadata={"title": "급성 충수염 개요", "source": "대한외과학회 환자 정보",
                  "url": "https://example.org/guideline/appendicitis", "published": "2021"},
    ),
    Document(
        page_content=(
            "제2형 당뇨병의 대표 증상은 다음(多飮), 다뇨(多尿), 다식(多食)과 설명되지 않는 체중 감소, "
            "피로감이다. 초기에는 무증상인 경우가 많아 정기 검진이 중요하다. 진단 기준은 공복혈당 126 mg/dL 이상, "
            "당화혈색소(HbA1c) 6.5% 이상, 또는 무작위 혈당 200 mg/dL 이상과 증상 동반이다. 치료의 기본은 "
            "식사·운동 등 생활습관 교정이며 1차 약물로 메트포르민이 흔히 사용된다. 합병증 예방을 위해 혈압·지질 관리도 병행한다."
        ),
        metadata={"title": "제2형 당뇨병 진단 기준", "source": "대한당뇨병학회 진료지침",
                  "url": "https://example.org/guideline/diabetes", "published": "2024"},
    ),
    Document(
        page_content=(
            "급성 심근경색을 의심해야 하는 흉통은 가슴 가운데를 짓누르는 듯한 압박감으로, 왼팔·턱·어깨로 "
            "뻗치고 식은땀, 호흡곤란, 구역을 동반할 수 있으며 보통 20분 이상 지속된다. 이는 응급 상황으로 "
            "즉시 119에 연락하고 응급실로 가야 한다. 시간이 곧 심근(time is muscle)이므로 빠른 재관류 치료가 예후를 좌우한다. "
            "당뇨·고령 환자는 비전형적(소화불량 같은) 증상으로 나타날 수 있어 주의한다."
        ),
        metadata={"title": "급성 흉통과 심근경색 경고", "source": "대한심장학회 환자 안내",
                  "url": "https://example.org/guideline/acs", "published": "2023"},
    ),
    Document(
        page_content=(
            "외이도염(수영자 귀, swimmer's ear)은 외이도 피부의 염증으로, 귀 통증이 주요 증상이며 "
            "귓바퀴(이개)를 당기거나 귀구슬(이주)을 누르면 통증이 뚜렷하게 심해지는 것이 특징이다. "
            "가려움, 진물(이루), 귀가 먹먹한 느낌, 일시적 청력 저하가 동반될 수 있다. 물놀이·수영, 높은 습도, "
            "면봉이나 귀이개로 과도하게 귀를 후비는 습관이 주요 유발 요인이다. 치료는 외이도를 깨끗하고 건조하게 "
            "유지하고, 항생제·스테로이드 성분의 점이액(귀에 넣는 물약)을 사용하는 것이 기본이다. 통증이 심하거나 "
            "얼굴 부종, 고열, 당뇨 등 면역저하가 있으면 합병증 위험이 있어 이비인후과 진료가 필요하다. "
            "예방을 위해 물놀이 후 귀를 잘 말리고 귀 후비기를 피한다."
        ),
        metadata={"title": "외이도염(수영자 귀) 진료 요약", "source": "대한이비인후과학회 환자 정보",
                  "url": "https://example.org/guideline/otitis-externa", "published": "2023"},
    ),
]


# ---------------------------------------------------------------------------
# 2-b. PubMed 근거 수집 (NCBI E-utilities API)
# ---------------------------------------------------------------------------
# PubMed 웹페이지를 스크래핑하는 대신 NCBI 공식 E-utilities API로 논문 초록을
# 구조화된 XML 형태로 받아온다. (스크래핑보다 안정적이며 NCBI 권장 방식)
#   - efetch: PMID 로 논문 상세(제목/초록/저널/저자) 조회
#   - esearch: 검색어로 관련 PMID 목록 조회
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_TOOL = "medical_diagnosis_rag"
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "")  # .env 의 NCBI_EMAIL (NCBI 권장: 연락 이메일 명시)


def _eutils_get(endpoint: str, params: dict) -> bytes:
    """E-utilities 엔드포인트에 GET 요청을 보내고 원시 응답을 반환."""
    full = {**params, "tool": NCBI_TOOL}
    if NCBI_EMAIL:
        full["email"] = NCBI_EMAIL
    url = f"{EUTILS_BASE}/{endpoint}?{urllib.parse.urlencode(full)}"
    req = urllib.request.Request(url, headers={"User-Agent": NCBI_TOOL})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def parse_pmid(url_or_id: str) -> Optional[str]:
    """PubMed URL 또는 숫자 문자열에서 PMID 를 추출한다."""
    s = url_or_id.strip()
    if s.isdigit():
        return s
    m = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", s)
    if m:
        return m.group(1)
    m = re.search(r"\b(\d{5,9})\b", s)  # 마지막 수단: 5~9자리 숫자
    return m.group(1) if m else None


def _article_to_document(art_elem) -> Optional[Document]:
    """<PubmedArticle> XML 요소를 출처 메타데이터가 붙은 Document 로 변환."""
    medline = art_elem.find(".//MedlineCitation")
    pmid = medline.findtext("PMID") if medline is not None else None
    article = art_elem.find(".//Article")
    if article is None:
        return None

    title = article.findtext("ArticleTitle") or "(제목 없음)"
    journal = article.findtext(".//Journal/Title") or "PubMed"
    year = (article.findtext(".//JournalIssue/PubDate/Year")
            or article.findtext(".//JournalIssue/PubDate/MedlineDate") or "")

    # 초록(여러 라벨 섹션으로 나뉠 수 있음)
    parts = []
    for ab in article.findall(".//Abstract/AbstractText"):
        label = ab.get("Label")
        text = "".join(ab.itertext()).strip()
        if text:
            parts.append(f"{label}: {text}" if label else text)
    abstract = "\n".join(parts)
    if not abstract:
        return None  # 초록 없는 항목은 근거로 부적합 -> 제외

    # 저자(최대 3명 + '외')
    authors = []
    for a in article.findall(".//AuthorList/Author"):
        ln, fn = a.findtext("LastName"), a.findtext("ForeName")
        if ln:
            authors.append(f"{fn} {ln}" if fn else ln)
    author_str = ", ".join(authors[:3]) + (" 외" if len(authors) > 3 else "")

    content = f"제목: {title}\n저널: {journal} ({year})\n초록: {abstract}"
    return Document(
        page_content=content,
        metadata={
            "title": title,
            "source": f"PubMed - {journal}" + (f" / {author_str}" if author_str else ""),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
            "published": year,
            "pmid": pmid or "",
        },
    )


def fetch_pubmed_by_ids(pmids: List[str]) -> List[Document]:
    """PMID 목록으로 논문 초록을 받아 Document 리스트로 반환."""
    pmids = [p for p in pmids if p]
    if not pmids:
        return []
    raw = _eutils_get("efetch.fcgi", {
        "db": "pubmed", "id": ",".join(pmids), "rettype": "abstract", "retmode": "xml",
    })
    root = ET.fromstring(raw)
    docs = [d for d in (_article_to_document(a) for a in root.findall(".//PubmedArticle")) if d]
    logger.info("[pubmed] PMID %d건 요청 -> 초록 포함 문서 %d건 수집", len(pmids), len(docs))
    for d in docs:
        logger.debug("[pubmed]   PMID %s | %s", d.metadata["pmid"], d.metadata["title"])
    return docs


def fetch_pubmed_by_query(query: str, max_results: int = 5) -> List[Document]:
    """검색어로 관련 논문을 찾아(relevance 정렬) Document 리스트로 반환."""
    raw = _eutils_get("esearch.fcgi", {
        "db": "pubmed", "term": query, "retmax": max_results,
        "retmode": "xml", "sort": "relevance",
    })
    root = ET.fromstring(raw)
    ids = [e.text for e in root.findall(".//IdList/Id") if e.text]
    logger.info("[pubmed] 검색 '%s' -> PMID %d건: %s", query, len(ids), ", ".join(ids))
    return fetch_pubmed_by_ids(ids)


# ---------------------------------------------------------------------------
# 3. 벡터스토어 / 검색기 (지연 초기화)
# ---------------------------------------------------------------------------
retriever = None


def build_retriever(k: int = 3, extra_docs: Optional[List[Document]] = None,
                    include_kb: bool = True):
    """문서 분할 -> 임베딩 -> FAISS 벡터스토어 -> 검색기 구성.

    extra_docs: PubMed 등에서 가져온 추가 근거 문서
    include_kb: 내장 KNOWLEDGE_BASE 포함 여부
    """
    global retriever

    base_docs = list(KNOWLEDGE_BASE) if include_kb else []
    if extra_docs:
        base_docs += extra_docs
    if not base_docs:
        raise ValueError("근거 문서가 없습니다. KNOWLEDGE_BASE 또는 PubMed 문서가 필요합니다.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(base_docs)
    logger.info("근거 문서 %d건(내장 KB %s, 추가 %d건) -> 청크 %d개",
                len(base_docs), "포함" if include_kb else "제외",
                len(extra_docs) if extra_docs else 0, len(chunks))

    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    logger.info("벡터스토어 구축 완료 (top-k=%d)", k)
    return retriever


# ---------------------------------------------------------------------------
# 4. RAG 상태(State)
# ---------------------------------------------------------------------------
class DiagnosisState(TypedDict):
    """의료 진단 RAG 에이전트의 공유 상태"""
    messages: Annotated[list, add_messages]  # 진행 로그(누적)

    question: str                   # 사용자 질문(증상)
    retrieved_docs: List[Document]  # 검색된 근거 청크
    numbered_sources: list          # 인용 번호별 고유 출처(중복 제거됨)
    relevant: bool                  # 근거가 질문과 관련 있는지 평가 결과
    answer: str                     # 생성된 답변(본문)
    citations: list                 # 실제 사용한 출처 목록
    formatted_output: str           # 답변 + 출처가 합쳐진 최종 출력


# ---------------------------------------------------------------------------
# 5. 노드 1 — 검색 (Retrieve)
# ---------------------------------------------------------------------------
def retrieve(state: DiagnosisState) -> dict:
    """질문과 유사한 근거 문서를 검색하는 노드"""
    docs = retriever.invoke(state["question"])
    logger.info("[retrieve] 근거 문서 %d건 검색", len(docs))
    for i, d in enumerate(docs, 1):
        logger.debug("[retrieve]   [%d] %s (%s)", i, d.metadata["title"], d.metadata["source"])
    return {
        "retrieved_docs": docs,
        "messages": [AIMessage(name="retrieve", content=f"근거 문서 {len(docs)}건 검색 완료")],
    }


# ---------------------------------------------------------------------------
# 6. 노드 2 — 근거 평가 (Grade)
# ---------------------------------------------------------------------------
class GradeResult(BaseModel):
    """근거 관련성 평가 결과"""
    relevant: bool = Field(description="검색된 문서로 질문에 답할 수 있으면 True, 아니면 False")
    reason: str = Field(description="판단 이유 (1문장)")


def grade(state: DiagnosisState) -> dict:
    """검색된 근거가 질문에 관련 있는지 평가하는 노드"""
    docs = state["retrieved_docs"]
    context = "\n\n".join(d.page_content for d in docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 의료 정보 검수자입니다. 주어진 참고 문서만으로 사용자의 질문에 "
                   "근거 있는 답변이 가능한지 엄격하게 판단하세요. 문서에 관련 내용이 없으면 False."),
        ("human", "질문: {question}\n\n참고 문서:\n{context}"),
    ])
    chain = prompt | llm.with_structured_output(GradeResult)
    result = chain.invoke({"question": state["question"], "context": context})

    logger.info("[grade] 관련성=%s | %s", result.relevant, result.reason)
    return {
        "relevant": result.relevant,
        "messages": [AIMessage(name="grade", content=f"근거 평가: {result.relevant} ({result.reason})")],
    }


# ---------------------------------------------------------------------------
# 7. 노드 3 — 답변 생성 + 인용 (Generate)
# ---------------------------------------------------------------------------
def generate(state: DiagnosisState) -> dict:
    """근거 문서를 인용하며 답변을 생성하는 노드"""
    docs = state["retrieved_docs"]

    # 동일 출처(같은 논문/문서)에서 나온 여러 청크는 하나의 인용 번호로 합친다.
    # 식별 키 우선순위: PMID -> URL -> 제목
    sources = []          # 고유 출처 메타데이터 (인덱스+1 = 인용 번호)
    key_to_num = {}
    numbered = []
    for d in docs:
        m = d.metadata
        key = m.get("pmid") or m.get("url") or m.get("title")
        if key not in key_to_num:
            sources.append(m)
            key_to_num[key] = len(sources)
        num = key_to_num[key]
        numbered.append(f"[{num}] (제목: {m['title']}, 출처: {m['source']})\n{d.page_content}")
    context = "\n\n".join(numbered)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 신중한 의료 정보 도우미입니다. 아래 규칙을 반드시 지키세요.
1. 오직 제공된 [번호] 참고 문서의 내용만 사용해 답하세요. 문서에 없는 내용은 추측하지 마세요.
2. 각 설명 문장 끝에는 근거가 된 문서 번호를 [1], [2] 형태로 표기하세요.
3. 답변은 (1) 의심 가능한 상태, (2) 일반적 관리/대처, (3) 즉시 진료가 필요한 경고 신호 순으로 정리하세요.
4. 진단을 단정하지 말고 가능성으로 설명하며, 마지막에 전문의 진료 권고를 덧붙이세요."""),
        ("human", "환자 증상/질문: {question}\n\n참고 문서:\n{context}"),
    ])
    chain = prompt | llm
    answer = chain.invoke({"question": state["question"], "context": context}).content

    logger.info("[generate] 근거 기반 답변 생성 완료 (%d자, 고유 출처 %d건)", len(answer), len(sources))
    return {
        "answer": answer,
        "numbered_sources": sources,
        "messages": [AIMessage(name="generate", content="근거 기반 답변 생성 완료")],
    }


# ---------------------------------------------------------------------------
# 8. 노드 4 — 출처 표시 (Cite)
# ---------------------------------------------------------------------------
def cite(state: DiagnosisState) -> dict:
    """답변에 실제로 사용된 근거만 골라 출처 목록을 만드는 노드"""
    answer = state["answer"]
    sources = state.get("numbered_sources", [])

    # 본문에서 실제 인용된 번호 추출 (예: [1], [2])
    used = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})

    citations = []
    for n in used:
        if 1 <= n <= len(sources):
            m = sources[n - 1]
            citations.append({
                "n": n,
                "title": m["title"],
                "source": m["source"],
                "url": m.get("url", ""),
                "published": m.get("published", ""),
            })

    lines = [answer, "", "-" * 40, "[ 참고 출처 ]"]
    if citations:
        for c in citations:
            lines.append(f"  [{c['n']}] {c['title']} - {c['source']} ({c['published']})")
            if c["url"]:
                lines.append(f"       {c['url']}")
    else:
        lines.append("  (인용된 출처 없음)")
    lines.append("")
    lines.append("[!] 본 답변은 참고용 정보이며 의학적 진단이 아닙니다. 정확한 진단은 의료기관에서 받으세요.")
    formatted = "\n".join(lines)

    logger.info("[cite] 출처 %d건 정리 완료", len(citations))
    return {
        "citations": citations,
        "formatted_output": formatted,
        "messages": [AIMessage(name="cite", content=f"출처 {len(citations)}건 정리 완료")],
    }


# ---------------------------------------------------------------------------
# 9. Fallback (근거 부족 시 안전 응답)
# ---------------------------------------------------------------------------
def fallback(state: DiagnosisState) -> dict:
    """지식베이스로 답할 수 없을 때의 안전 응답 노드"""
    msg = (
        "죄송합니다. 현재 지식베이스에서 해당 질문에 답할 만한 근거 문서를 찾지 못했습니다.\n"
        "증상이 지속되거나 심하다면 가까운 의료기관에서 진료를 받으시기 바랍니다.\n\n"
        "[!] 본 서비스는 등록된 자료 범위 내에서만 정보를 제공합니다."
    )
    logger.info("[fallback] 근거 부족 - 안전 응답 반환")
    return {
        "answer": msg,
        "citations": [],
        "formatted_output": msg,
        "messages": [AIMessage(name="fallback", content="근거 부족 - 안전 응답 반환")],
    }


def route_after_grade(state: DiagnosisState) -> str:
    """근거 평가 결과에 따라 분기"""
    return "generate" if state.get("relevant") else "fallback"


# ---------------------------------------------------------------------------
# 10. 그래프 구성
# ---------------------------------------------------------------------------
def build_graph():
    builder = StateGraph(DiagnosisState)

    builder.add_node("retrieve", retrieve)
    builder.add_node("grade", grade)
    builder.add_node("generate", generate)
    builder.add_node("cite", cite)
    builder.add_node("fallback", fallback)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges(
        "grade",
        route_after_grade,
        {"generate": "generate", "fallback": "fallback"},
    )
    builder.add_edge("generate", "cite")
    builder.add_edge("cite", END)
    builder.add_edge("fallback", END)

    graph = builder.compile()
    logger.info("그래프 컴파일 완료")
    return graph


# ---------------------------------------------------------------------------
# 11. 실행 헬퍼
# ---------------------------------------------------------------------------
DEMO_QUESTIONS = [
    "갑자기 38도 넘는 고열에 온몸이 쑤시고 기운이 하나도 없어요. 마른기침도 나요.",
    "밥 먹고 누우면 가슴이 화끈거리고 신물이 올라와요.",
    "오른쪽 아랫배가 점점 아프고 걸을 때 더 아파요. 미열도 있어요.",
    "발목을 삐었을 때 깁스를 얼마나 해야 하나요?",  # 지식베이스 밖 -> fallback 기대
]


def diagnose(graph, question: str) -> str:
    """증상 질문을 받아 근거 기반 답변 + 출처를 반환하고 로그에 남긴다."""
    logger.info("=" * 60)
    logger.info("[질문] %s", question)
    result = graph.invoke({"question": question, "messages": []})
    output = result["formatted_output"]
    logger.info("[답변 및 출처]\n%s", output)
    return output


# ---------------------------------------------------------------------------
# 12. main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="의료 진단 RAG + 출처 표시")
    parser.add_argument("question", nargs="*", help="증상 질문 (생략 시 데모 질문 세트 실행)")
    parser.add_argument("--log-file", default="medical_diagnosis_rag.log", help="로그 파일 경로")
    parser.add_argument("--k", type=int, default=3, help="검색 문서 수(top-k)")
    parser.add_argument("--debug", action="store_true", help="디버그 로그 출력")
    # PubMed 근거 옵션
    parser.add_argument("--pubmed-url", action="append", default=[], metavar="URL_OR_PMID",
                        help="PubMed 논문 URL 또는 PMID (여러 번 지정 가능)")
    parser.add_argument("--pubmed-query", default=None, metavar="QUERY",
                        help="PubMed 검색어로 관련 논문을 근거에 추가")
    parser.add_argument("--pubmed-max", type=int, default=5,
                        help="--pubmed-query 시 가져올 최대 논문 수")
    parser.add_argument("--no-kb", action="store_true",
                        help="내장 지식베이스를 빼고 PubMed 근거만 사용")
    args = parser.parse_args()

    setup_logging(args.log_file, level=logging.DEBUG if args.debug else logging.INFO)

    # PubMed 근거 수집
    extra_docs: List[Document] = []
    if args.pubmed_url:
        pmids = []
        for item in args.pubmed_url:
            pid = parse_pmid(item)
            if pid:
                pmids.append(pid)
            else:
                logger.warning("PMID 를 추출하지 못함: %s", item)
        extra_docs += fetch_pubmed_by_ids(pmids)
    if args.pubmed_query:
        extra_docs += fetch_pubmed_by_query(args.pubmed_query, args.pubmed_max)

    if args.no_kb and not extra_docs:
        logger.error("--no-kb 인데 PubMed 근거가 없습니다. --pubmed-url/--pubmed-query 를 지정하세요.")
        return

    build_retriever(k=args.k, extra_docs=extra_docs, include_kb=not args.no_kb)
    graph = build_graph()

    if args.question:
        diagnose(graph, " ".join(args.question))
    else:
        logger.info("질문 인자가 없어 데모 질문 세트를 실행합니다. (총 %d건)", len(DEMO_QUESTIONS))
        for q in DEMO_QUESTIONS:
            diagnose(graph, q)

    logger.info("=" * 60)
    logger.info("완료. 상세 로그는 '%s' 에 저장되었습니다.", args.log_file)


if __name__ == "__main__":
    main()

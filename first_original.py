import os
import re
import nltk
import PyPDF2
import logging
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

class PDFQueryHandler:
    def __init__(self, pdf_path: str):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")
        self.pdf_path = pdf_path
        self.full_text = self._extract_and_filter_text()
        self.abstract = self._extract_abstract(self.full_text)
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.qa_model = self._load_qa_model()
        self.summarizer = self._load_summarization_model()

    def _extract_and_filter_text(self) -> str:
        try:
            with open(self.pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                full_text = " ".join(page.extract_text() or "" for page in reader.pages)

            full_text = re.sub(r"\s+", " ", full_text)
            full_text = re.sub(r"[^\x00-\x7F]+", "", full_text)

            if not full_text.strip():
                raise ValueError("No text could be extracted from the PDF.")

            return full_text.strip()
        except Exception as e:
            logging.error(f"Text extraction error: {e}")
            return ""

    def _extract_abstract(self, text: str) -> Optional[str]:
        abstract_keywords = ["abstract", "aim", "objective", "methodology", "results"]
        literature_keywords = ["literature review", "related work", "background"]
        sentences = nltk.sent_tokenize(text.lower())
        abstract_start = next((i for i, s in enumerate(sentences) if "abstract" in s and not any(k in s for k in literature_keywords)), None)
        abstract_end = next((i for i in range(abstract_start + 1, len(sentences)) if any(k in sentences[i] for k in abstract_keywords)), None) if abstract_start else None
        return " ".join(sentences[abstract_start:abstract_end]).strip() if abstract_start is not None else None

    def _load_qa_model(self):
        try:
            import torch
            return pipeline("question-answering", model="deepset/roberta-base-squad2")
        except ImportError:
            logging.error("PyTorch or TensorFlow is not installed. Install one of them.")
        except Exception as e:
            logging.error(f"QA Model loading error: {e}")
        return None


    def _load_summarization_model(self):
        try:
            import torch
            return pipeline("summarization", model="facebook/bart-large-cnn")
        except ImportError:
            logging.error("PyTorch or TensorFlow is not installed. Install one of them.")
        except Exception as e:
            logging.error(f"Summarization Model loading error: {e}")
        return None


    def preprocess_text(self, text: str) -> List[str]:
        text = re.sub(r"\[\d+\]", "", text)
        text = re.sub(r"\([\d,\s]+\)", "", text)
        sentences = nltk.sent_tokenize(text)
        return [s.lower() for s in sentences if not any(k in s for k in ["figure", "table", "reference", "citation", "appendix", "bibliography", "et al.", "literature review", "related work"])]

    def use_qa_model(self, query: str, context: str) -> str:
        if not self.qa_model:
            return "QA model is not available. Install PyTorch or TensorFlow."
        try:
            qa_response = self.qa_model({"question": query, "context": context})
            return qa_response.get("answer", "No precise answer found.")
        except Exception as e:
            logging.error(f"QA model error: {e}")
            return "No precise answer found."

    def retrieve_relevant_passages(self, query: str, top_k: int = 5) -> List[str]:
        preprocessed_sentences = self.preprocess_text(self.full_text)
        if not preprocessed_sentences:
            return ["No meaningful text found in the document."]
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(preprocessed_sentences)
        query_vector = self.tfidf_vectorizer.transform([query])
        tfidf_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        relevant_indices = tfidf_scores.argsort()[::-1][:top_k]
        return [preprocessed_sentences[idx] for idx in relevant_indices]

    def advanced_query_response(self, query: str) -> Dict[str, Any]:
        direct_answer = self.use_qa_model(query, self.abstract or self.full_text)
        relevant_passages = self.retrieve_relevant_passages(query)
        passage_summary = self.summarizer(" ".join(relevant_passages), max_length=150, min_length=50, do_sample=False)[0]["summary_text"] if self.summarizer else "Summarization model not available."
        return {"query": query, "direct_answer": direct_answer, "relevant_passages": relevant_passages, "passage_summary": passage_summary}


class PDFResearchAnalyzer(PDFQueryHandler):
    def __init__(self, pdf_path: str):
        super().__init__(pdf_path)
        self.gap_keywords = ["limitation", "constraint", "future work", "future research", "open problem", "unresolved", "challenge", "unexplored", "not addressed", "insufficient", "further investigation"]
        self.improvement_keywords = ["enhance", "improve", "extend", "overcome", "address", "resolve", "augment", "optimize", "refine", "develop"]

    def detect_research_gaps(self) -> List[str]:
        return [s for s in nltk.sent_tokenize(self.full_text.lower()) if any(k in s for k in self.gap_keywords) and 5 < len(s.split()) < 30]

    def suggest_improvements(self) -> List[str]:
        conclusion_sentences = [s for s in nltk.sent_tokenize(self.full_text.lower()) if "conclusion" in s or "future work" in s]
        return [s for s in conclusion_sentences if any(k in s for k in self.improvement_keywords) and 6 < len(s.split()) < 30]

    def comprehensive_research_analysis(self) -> Dict[str, Any]:
        return {"research_gaps": self.detect_research_gaps(), "improvement_suggestions": self.suggest_improvements()}



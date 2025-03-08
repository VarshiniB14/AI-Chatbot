import os
import re
import nltk
import PyPDF2
import logging
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import warnings
import logging
import transformers
import torch
# Suppress all warnings
import logging

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow info & warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN logs

# Suppress TensorFlow logging
logging.getLogger("tensorflow").setLevel(logging.FATAL)
warnings.filterwarnings("ignore")

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Suppress logging messages from specific libraries
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# Suppress transformers-specific max_length warnings
transformers.logging.set_verbosity_error()

# Suppress PyTorch warnings (including optree)
torch.utils._pytree.warnings.filterwarnings("ignore", category=FutureWarning)
#####################################################################################
import os

# Completely suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO & WARNING logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN logs completely

# Redirect TensorFlow C++ backend logs (Fix for oneDNN logs)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # (Optional) Disables GPU logs if using CUDA
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Suppress TensorFlow low-level logs
import ctypes
try:
    ctypes.CDLL("libtensorflow_framework.so").stderr = None  # Linux/Mac fix
except:
    pass

import os
import warnings
import logging
import transformers
import torch

# Fully suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN logs completely

# Suppress logging messages from specific libraries
logging.getLogger("tensorflow").setLevel(logging.FATAL)  # Change from ERROR to FATAL
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# Suppress all warnings
warnings.filterwarnings("ignore")

# Suppress transformers-specific max_length warnings
transformers.logging.set_verbosity_error()

# Suppress PyTorch warnings (including optree, if available)
if hasattr(torch.utils, "_pytree"):
    torch.utils._pytree.warnings.filterwarnings("ignore", category=FutureWarning)


class PDFQueryHandler:
    def __init__(self, pdf_path: str):
        """
        Initialize PDF Query Handler with advanced preprocessing.

        Args:
            pdf_path (str): Path to the PDF file.
        """
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s: %(message)s')

        self.pdf_path = pdf_path
        self.full_text = self._extract_and_filter_text()
        self.abstract = self._extract_abstract(self.full_text)

        self.tfidf_vectorizer = None
        self.tfidf_matrix = None

        self.qa_model = self._load_qa_model()
        self.summarizer = self._load_summarization_model()

    def _extract_and_filter_text(self) -> str:
        """
        Extract text from PDF, rigorously filtering out literature review/survey sections.

        Returns:
            str: Filtered text without literature review content.
        """
        try:
            with open(self.pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                full_text = ' '.join(page.extract_text() for page in reader.pages)

            # Extensive text cleaning
            full_text = re.sub(r'\s+', ' ', full_text)
            full_text = re.sub(r'[^\x00-\x7F]+', '', full_text)

            # Advanced section filtering
            literature_keywords = [
                'literature review', 'literature survey', 'related work',
                'background study', 'prior research', 'state of the art'
            ]
            section_keywords = [
                'introduction', 'methods?', 'results?',
                'discussion', 'conclusion', 'methodology'
            ]

            # Split text into sections
            sections = re.split(r'\b(?:' + '|'.join(section_keywords) + r')\b',
                                full_text, flags=re.IGNORECASE)

            filtered_text = []
            for section in sections:
                # Check if section contains any literature review keywords
                if not any(keyword in section.lower() for keyword in literature_keywords):
                    filtered_text.append(section)

            return ' '.join(filtered_text).strip()

        except Exception as e:
            logging.error(f"Text extraction error: {e}")
            return ""

    def _extract_abstract(self, text: str) -> Optional[str]:
        """
        Extract abstract while excluding literature review content.

        Args:
            text (str): Filtered extracted text.

        Returns:
            str: Extracted abstract or None if not found.
        """
        abstract_keywords = ['abstract', 'aim', 'objective', 'methodology', 'results']
        literature_keywords = ['literature review', 'related work', 'background']

        sentences = nltk.sent_tokenize(text.lower())
        abstract_start = None
        abstract_end = None

        # Find abstract start, avoiding literature review sections
        for i, sentence in enumerate(sentences):
            if 'abstract' in sentence and not any(keyword in sentence for keyword in literature_keywords):
                abstract_start = i
                break

        if abstract_start is not None:
            for i in range(abstract_start + 1, len(sentences)):
                if any(keyword in sentences[i] for keyword in abstract_keywords):
                    abstract_end = i
                    break

            abstract = ' '.join(sentences[abstract_start:abstract_end])
            return abstract.strip()
        return None

    def _load_qa_model(self, model_name: str = "deepset/roberta-base-squad2"):
        try:
            return pipeline("question-answering", model=model_name)
        except Exception as e:
            logging.error(f"QA Model loading error: {e}")
            return None

    def _load_summarization_model(self, model_name: str = "facebook/bart-large-cnn"):
        try:
            return pipeline("summarization", model=model_name)
        except Exception as e:
            logging.error(f"Summarization Model loading error: {e}")
            return None

    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text by removing citations, references, and non-original content.

        Args:
            text (str): Text to preprocess.

        Returns:
            List[str]: List of cleaned and filtered sentences.
        """
        # Remove citations, references, and non-original content markers
        text = re.sub(r'\[\d+\]', '', text)  # Remove inline citations
        text = re.sub(r'\([\d,\s]+\)', '', text)  # Remove parenthetical citations

        sentences = nltk.sent_tokenize(text)
        return [
            sentence.lower() for sentence in sentences
            if not any(keyword in sentence.lower() for keyword in [
                'figure', 'table', 'reference', 'citation',
                'appendix', 'bibliography', 'et al.',
                'literature review', 'related work'
            ])
        ]

    def use_qa_model(self, query: str, context: str) -> str:
        """
        Use QA model to find answers, excluding citation-based responses.

        Args:
            query (str): The question to answer.
            context (str): The context in which to search for the answer.

        Returns:
            str: Clean answer without citations.
        """
        try:
            qa_response = self.qa_model({'question': query, 'context': context})
            answer = qa_response['answer']

            # Remove citations and references from answer
            clean_answer = ' '.join(
                sentence for sentence in nltk.sent_tokenize(answer)
                if not re.search(r'\[\d+\]', sentence) and  # Remove inline citations
                   not re.search(r'\([\d,\s]+\)', sentence)  # Remove parenthetical citations
            )

            return clean_answer if clean_answer else "No precise answer found."
        except Exception as e:
            logging.error(f"QA model error: {e}")
            return "No precise answer found."

    def retrieve_relevant_passages(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieve relevant passages, excluding literature review content.

        Args:
            query (str): Search query.
            top_k (int): Number of top passages to retrieve.

        Returns:
            List[str]: Relevant passages.
        """
        preprocessed_sentences = self.preprocess_text(self.full_text)
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(preprocessed_sentences)
        query_vector = self.tfidf_vectorizer.transform([query])
        tfidf_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # Filter out passages with literature review keywords
        literature_keywords = ['literature review', 'related work', 'background study']
        relevant_indices = [
            idx for idx in tfidf_scores.argsort()[::-1][:top_k*2]  # Get more to compensate for filtering
            if not any(keyword in preprocessed_sentences[idx].lower() for keyword in literature_keywords)
        ][:top_k]

        return [preprocessed_sentences[idx] for idx in relevant_indices]

    def advanced_query_response(self, query: str) -> Dict[str, Any]:
        """
        Generate comprehensive query response.

        Args:
            query (str): User query.

        Returns:
            Dict[str, Any]: Detailed query response.
        """
        # Prefer abstract for direct answers, fallback to full text
        direct_answer = (
            self.use_qa_model(query, self.abstract) if self.abstract
            else self.use_qa_model(query, self.full_text)
        )

        # Retrieve and filter relevant passages
        relevant_passages = self.retrieve_relevant_passages(query)
        filtered_passages = [
            re.sub(r'\b(?:figure|table|reference|citation|appendix|bibliography)\b', '', passage, flags=re.IGNORECASE)
            for passage in relevant_passages
        ]

        # Summarize filtered passages
        summary_input = " ".join(filtered_passages)
        summary_input_cleaned = re.sub(
            r'\b(?:figure|table|reference|citation|appendix|bibliography)\b',
            '', summary_input, flags=re.IGNORECASE
        )
        passage_summary = self.summarizer(
            summary_input_cleaned,
            max_length=150,
            min_length=50,
            do_sample=False
        )[0]['summary_text']

        return {
            'query': query,
            'direct_answer': direct_answer,
            'relevant_passages': filtered_passages,
            'passage_summary': passage_summary
        }


class PDFResearchAnalyzer(PDFQueryHandler):
    def __init__(self, pdf_path: str):
        """
        Initialize PDF Research Analyzer with research gap detection capabilities.

        Extends PDFQueryHandler with additional analysis methods.
        """
        super().__init__(pdf_path)

        # Keywords indicating potential research limitations or gaps
        self.gap_keywords = [
            'limitation', 'constraint', 'future work', 'future research',
            'open problem', 'unresolved', 'challenge', 'unexplored',
            'not addressed', 'insufficient', 'further investigation'
        ]

        # Improvement suggestion keywords
        self.improvement_keywords = [
            'enhance', 'improve', 'extend', 'overcome', 'address',
            'resolve', 'augment', 'optimize', 'refine', 'develop'
        ]

    def detect_research_gaps(self) -> List[str]:
        """
        Detect potential research gaps in the document.

        Returns:
            List[str]: Identified research gaps and limitations.
        """
        sentences = nltk.sent_tokenize(self.full_text.lower())
        potential_gaps = []

        for sentence in sentences:
            # Check for gap-indicating language
            if any(keyword in sentence for keyword in self.gap_keywords):
                # Additional filtering to ensure relevance
                if len(sentence.split()) > 5 and len(sentence.split()) < 30:
                    potential_gaps.append(sentence)

        return list(set(potential_gaps))  # Remove duplicates

    def suggest_improvements(self) -> List[str]:
        """
        Generate potential research improvements and extensions.

        Returns:
            List[str]: Suggested improvements for the research.
        """
        improvement_suggestions = []

        # Analyze conclusion and future work sections
        conclusion_sentences = [
            sent for sent in nltk.sent_tokenize(self.full_text.lower())
            if 'conclusion' in sent or 'future work' in sent
        ]

        # Extract method and result sections
        method_sentences = [
            sent for sent in nltk.sent_tokenize(self.full_text.lower())
            if any(keyword in sent for keyword in ['method', 'approach', 'methodology'])
        ]

        result_sentences = [
            sent for sent in nltk.sent_tokenize(self.full_text.lower())
            if any(keyword in sent for keyword in ['result', 'finding', 'observation'])
        ]

        # Generate improvement suggestions based on different sections
        for section_sentences in [conclusion_sentences, method_sentences, result_sentences]:
            for sentence in section_sentences:
                # Look for improvement-related language
                if any(keyword in sentence for keyword in self.improvement_keywords):
                    # Additional filtering for meaningful suggestions
                    if len(sentence.split()) > 6 and len(sentence.split()) < 30:
                        improvement_suggestions.append(sentence)

        return list(set(improvement_suggestions))  # Remove duplicates

    def comprehensive_research_analysis(self) -> Dict[str, Any]:
        """
        Provide a comprehensive analysis of research gaps and potential improvements.

        Returns:
            Dict[str, Any]: Detailed research analysis.
        """
        research_gaps = self.detect_research_gaps()
        improvement_suggestions = self.suggest_improvements()

        return {
            'research_gaps': research_gaps,
            'improvement_suggestions': improvement_suggestions,
            'gap_count': len(research_gaps),
            'improvement_count': len(improvement_suggestions)
        }


def pdf_research_chatbot(pdf_path):
    """
    Interactive PDF Research Analyzer Chatbot

    Args:
        pdf_path (str): Path to the PDF file
    """
    # Initialize the analyzer
    analyzer = PDFResearchAnalyzer(pdf_path)

    # Display initial analysis
    print("\n--- Research Gap and Improvement Analysis ---")
    analysis_results = analyzer.comprehensive_research_analysis()

    print("\nIdentified Research Gaps:")
    for gap in analysis_results['research_gaps']:
        print(f"- {gap}")

    print(f"\nTotal Research Gaps Detected: {analysis_results['gap_count']}")

    print("\nSuggested Improvements:")
    for improvement in analysis_results['improvement_suggestions']:
        print(f"- {improvement}")

    print(f"\nTotal Improvement Suggestions: {analysis_results['improvement_count']}")

    # Predefined queries
    predefined_queries = [
        "What can be used to improve the model?",
        "Summarize the result",
        "Describe the experimental approach",
        "What were the key findings?",
        "Which dataset is used?"
    ]

    print("\n--- Predefined Queries ---")
    for query in predefined_queries:
        print(f"\nQuery: {query}")
        response = analyzer.advanced_query_response(query)
        print("Direct Answer:", response['direct_answer'])
        print("\nRelevant Passages:")
        for passage in response['relevant_passages']:
            print("- ", passage)
        print("\nPassage Summary:", response['passage_summary'])

    # Interactive query loop
    while True:
        user_query = input("\nEnter your query (or 'quit' to exit): ").strip()

        if user_query.lower() == 'quit':
            break

        # Process user query
        print(f"\nQuery: {user_query}")
        response = analyzer.advanced_query_response(user_query)
        print("Direct Answer:", response['direct_answer'])
        print("\nRelevant Passages:")
        for passage in response['relevant_passages']:
            print("- ", passage)
        print("\nPassage Summary:", response['passage_summary'])

# Example usage in Google Colab
# Assuming you have uploaded a PDF to your Colab environment
pdf_path = "C://Users//Varshini//Downloads//AI PDF CHATBOT//uploads//E19_MATH_REPORT.pdf"
pdf_research_chatbot(pdf_path)
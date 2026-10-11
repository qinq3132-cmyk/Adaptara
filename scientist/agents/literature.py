"""Separate literature roles; Python owns acquisition, IDs and state updates."""
from scientist.literature_schemas import SearchPlan, PaperSelection, PaperReading, EvidenceChecks, LiteratureSynthesis
from .structured import StructuredAgent


class LiteratureSearchAgent(StructuredAgent):
    role, output_type = "literature_search", SearchPlan


class LiteratureScreeningAgent(StructuredAgent):
    role, output_type = "literature_screening", PaperSelection


class LiteratureReadingAgent(StructuredAgent):
    role, output_type = "literature_reading", PaperReading


class EvidenceVerificationAgent(StructuredAgent):
    role, output_type = "evidence_verification", EvidenceChecks


class LiteratureSynthesisAgent(StructuredAgent):
    role, output_type = "literature_synthesis", LiteratureSynthesis

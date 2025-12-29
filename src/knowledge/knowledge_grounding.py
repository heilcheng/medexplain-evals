"""Medical knowledge grounding module for MedExplain-Evals.

This module provides integration with medical knowledge bases for factuality
verification, entity extraction, and semantic grounding of medical explanations.

Supported Knowledge Bases:
    - UMLS Metathesaurus (concept identification and linking)
    - RxNorm (drug information and interactions)
    - SNOMED-CT (clinical terms)
    - ICD-10 (diagnosis codes)

Features:
    - Medical entity extraction using SciSpacy
    - UMLS concept linking
    - Drug information lookup via RxNorm
    - Factual claim verification
    - Semantic similarity using biomedical embeddings

Example:
    ```python
    from knowledge_grounding import MedicalKnowledgeGrounder
    
    grounder = MedicalKnowledgeGrounder()
    
    # Extract medical entities
    entities = grounder.extract_medical_entities(text)
    
    # Verify factual claims
    claims = grounder.extract_claims(text)
    results = grounder.verify_claims(claims, reference_text)
    
    # Compute grounding score
    score = grounder.compute_grounding_score(explanation, source)
    ```
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger("medexplain.knowledge_grounding")


class EntityType(str, Enum):
    """Types of medical entities."""
    CONDITION = "condition"
    MEDICATION = "medication"
    PROCEDURE = "procedure"
    SYMPTOM = "symptom"
    ANATOMY = "anatomy"
    LAB_TEST = "lab_test"
    DEVICE = "device"
    ORGANISM = "organism"
    GENE = "gene"
    UNKNOWN = "unknown"


@dataclass
class MedicalEntity:
    """Extracted medical entity with knowledge base links."""
    text: str
    entity_type: str
    start: int
    end: int
    umls_cui: Optional[str] = None
    umls_name: Optional[str] = None
    snomed_code: Optional[str] = None
    rxnorm_code: Optional[str] = None
    icd10_code: Optional[str] = None
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "entity_type": self.entity_type,
            "start": self.start,
            "end": self.end,
            "umls_cui": self.umls_cui,
            "umls_name": self.umls_name,
            "snomed_code": self.snomed_code,
            "rxnorm_code": self.rxnorm_code,
            "icd10_code": self.icd10_code,
            "confidence": self.confidence,
        }


@dataclass
class FactCheckResult:
    """Result of factual claim verification."""
    claim: str
    is_supported: bool
    confidence: float
    evidence: Optional[str] = None
    contradiction_type: Optional[str] = None
    source: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "is_supported": self.is_supported,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "contradiction_type": self.contradiction_type,
            "source": self.source,
        }


@dataclass
class GroundingScore:
    """Comprehensive grounding score with component breakdown."""
    overall: float
    entity_coverage: float
    factual_accuracy: float
    semantic_similarity: float
    contradiction_penalty: float
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall": self.overall,
            "entity_coverage": self.entity_coverage,
            "factual_accuracy": self.factual_accuracy,
            "semantic_similarity": self.semantic_similarity,
            "contradiction_penalty": self.contradiction_penalty,
            "details": self.details,
        }


class UMLSClient:
    """Client for UMLS Metathesaurus API.
    
    Note: Requires UMLS API key from https://uts.nlm.nih.gov/uts/
    """
    
    BASE_URL = "https://uts-ws.nlm.nih.gov/rest"
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize UMLS client.
        
        Args:
            api_key: UMLS API key. If None, uses environment variable.
        """
        import os
        self.api_key = api_key or os.environ.get("UMLS_API_KEY")
        self._ticket_granting_ticket = None
        
    def _get_service_ticket(self) -> Optional[str]:
        """Get service ticket for API authentication."""
        if not self.api_key:
            logger.warning("No UMLS API key configured")
            return None
        
        try:
            import requests
            
            # Get TGT
            auth_url = "https://utslogin.nlm.nih.gov/cas/v1/api-key"
            response = requests.post(auth_url, data={"apikey": self.api_key})
            
            if response.status_code == 201:
                tgt_url = response.headers.get("location")
                
                # Get service ticket
                service = "http://umlsks.nlm.nih.gov"
                st_response = requests.post(tgt_url, data={"service": service})
                
                if st_response.status_code == 200:
                    return st_response.text
                    
        except Exception as e:
            logger.error(f"Error getting UMLS service ticket: {e}")
        
        return None
    
    def search_concept(self, term: str) -> List[Dict[str, Any]]:
        """Search for UMLS concepts matching a term.
        
        Args:
            term: Medical term to search
            
        Returns:
            List of matching concepts with CUI, name, and semantic types
        """
        service_ticket = self._get_service_ticket()
        if not service_ticket:
            return []
        
        try:
            import requests
            
            url = f"{self.BASE_URL}/search/current"
            params = {
                "string": term,
                "ticket": service_ticket,
                "returnIdType": "concept",
            }
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                results = data.get("result", {}).get("results", [])
                return results[:5]  # Return top 5 matches
                
        except Exception as e:
            logger.error(f"Error searching UMLS: {e}")
        
        return []
    
    def get_concept_info(self, cui: str) -> Optional[Dict[str, Any]]:
        """Get detailed information for a UMLS concept.
        
        Args:
            cui: UMLS Concept Unique Identifier
            
        Returns:
            Concept information including name, definitions, semantic types
        """
        service_ticket = self._get_service_ticket()
        if not service_ticket:
            return None
        
        try:
            import requests
            
            url = f"{self.BASE_URL}/content/current/CUI/{cui}"
            params = {"ticket": service_ticket}
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response.json().get("result", {})
                
        except Exception as e:
            logger.error(f"Error getting UMLS concept {cui}: {e}")
        
        return None


class RxNormClient:
    """Client for RxNorm API for drug information."""
    
    BASE_URL = "https://rxnav.nlm.nih.gov/REST"
    
    def search_drug(self, name: str) -> List[Dict[str, Any]]:
        """Search for drugs by name.
        
        Args:
            name: Drug name to search
            
        Returns:
            List of matching drugs with RxCUI and name
        """
        try:
            import requests
            
            url = f"{self.BASE_URL}/drugs.json"
            params = {"name": name}
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                concepts = data.get("drugGroup", {}).get("conceptGroup", [])
                
                results = []
                for group in concepts:
                    for concept in group.get("conceptProperties", []):
                        results.append({
                            "rxcui": concept.get("rxcui"),
                            "name": concept.get("name"),
                            "tty": concept.get("tty"),
                        })
                
                return results[:5]
                
        except Exception as e:
            logger.error(f"Error searching RxNorm: {e}")
        
        return []
    
    def get_drug_interactions(self, rxcui: str) -> List[Dict[str, Any]]:
        """Get drug interactions for a medication.
        
        Args:
            rxcui: RxNorm Concept Unique Identifier
            
        Returns:
            List of drug interactions
        """
        try:
            import requests
            
            url = f"{self.BASE_URL}/interaction/interaction.json"
            params = {"rxcui": rxcui}
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                groups = data.get("interactionTypeGroup", [])
                
                interactions = []
                for group in groups:
                    for itype in group.get("interactionType", []):
                        for pair in itype.get("interactionPair", []):
                            interactions.append({
                                "severity": pair.get("severity"),
                                "description": pair.get("description"),
                            })
                
                return interactions
                
        except Exception as e:
            logger.error(f"Error getting drug interactions: {e}")
        
        return []


class SnomedClient:
    """Client for SNOMED-CT terminology (via UMLS or dedicated API)."""
    
    def __init__(self, umls_client: Optional[UMLSClient] = None):
        self.umls_client = umls_client or UMLSClient()
    
    def search_concept(self, term: str) -> List[Dict[str, Any]]:
        """Search SNOMED-CT concepts."""
        # SNOMED concepts can be accessed through UMLS
        # In production, could use dedicated SNOMED API
        results = self.umls_client.search_concept(term)
        
        # Filter to SNOMED source
        snomed_results = []
        for result in results:
            if "SNOMEDCT" in str(result.get("rootSource", "")):
                snomed_results.append(result)
        
        return snomed_results


class MedicalEntityExtractor:
    """Extract medical entities from text using SciSpacy and rules."""
    
    # Fallback patterns when SciSpacy is not available
    ENTITY_PATTERNS = {
        EntityType.MEDICATION.value: [
            r"\b(aspirin|ibuprofen|acetaminophen|metformin|lisinopril|"
            r"atorvastatin|amlodipine|omeprazole|metoprolol|losartan|"
            r"gabapentin|sertraline|tramadol|prednisone|albuterol|"
            r"azithromycin|amoxicillin|hydrocodone|levothyroxine|"
            r"pantoprazole|rosuvastatin|montelukast|escitalopram)\b",
        ],
        EntityType.CONDITION.value: [
            r"\b(diabetes|hypertension|asthma|copd|pneumonia|depression|"
            r"anxiety|cancer|stroke|heart attack|myocardial infarction|"
            r"heart failure|arthritis|alzheimer|parkinson|epilepsy|"
            r"migraine|bronchitis|hepatitis|cirrhosis|pancreatitis|"
            r"appendicitis|diverticulitis|celiac|crohn|colitis)\b",
        ],
        EntityType.SYMPTOM.value: [
            r"\b(pain|fever|cough|headache|nausea|vomiting|diarrhea|"
            r"constipation|fatigue|dizziness|shortness of breath|"
            r"chest pain|abdominal pain|back pain|swelling|rash|"
            r"itching|numbness|tingling|weakness|confusion)\b",
        ],
        EntityType.PROCEDURE.value: [
            r"\b(surgery|biopsy|endoscopy|colonoscopy|mri|ct scan|"
            r"x-ray|ultrasound|ecg|ekg|echocardiogram|angiogram|"
            r"catheterization|dialysis|chemotherapy|radiation|"
            r"transplant|amputation|appendectomy|cholecystectomy)\b",
        ],
        EntityType.ANATOMY.value: [
            r"\b(heart|lung|liver|kidney|brain|stomach|intestine|"
            r"colon|pancreas|spleen|bladder|thyroid|adrenal|"
            r"prostate|ovary|uterus|breast|bone|muscle|nerve|"
            r"artery|vein|blood vessel)\b",
        ],
        EntityType.LAB_TEST.value: [
            r"\b(blood test|urinalysis|cbc|bmp|cmp|lipid panel|"
            r"hemoglobin a1c|hba1c|glucose|creatinine|bun|"
            r"liver function|thyroid panel|psa|troponin)\b",
        ],
    }
    
    def __init__(self, use_scispacy: bool = True):
        """Initialize entity extractor.
        
        Args:
            use_scispacy: Whether to use SciSpacy for NER
        """
        self.nlp = None
        self.use_scispacy = use_scispacy
        
        if use_scispacy:
            self._load_scispacy()
    
    def _load_scispacy(self) -> None:
        """Load SciSpacy model for biomedical NER."""
        try:
            import spacy
            
            # Try to load biomedical model
            model_names = [
                "en_core_sci_lg",
                "en_core_sci_md",
                "en_core_sci_sm",
                "en_core_web_sm",
            ]
            
            for model_name in model_names:
                try:
                    self.nlp = spacy.load(model_name)
                    logger.info(f"Loaded SciSpacy model: {model_name}")
                    
                    # Try to add UMLS linker
                    try:
                        from scispacy.linking import EntityLinker
                        self.nlp.add_pipe("scispacy_linker", config={
                            "resolve_abbreviations": True,
                            "linker_name": "umls"
                        })
                        logger.info("Added UMLS entity linker")
                    except Exception as e:
                        logger.warning(f"Could not add UMLS linker: {e}")
                    
                    return
                except OSError:
                    continue
            
            logger.warning("No SciSpacy model available, using rule-based extraction")
            self.use_scispacy = False
            
        except ImportError:
            logger.warning("SciSpacy not installed, using rule-based extraction")
            self.use_scispacy = False
    
    def extract_entities(self, text: str) -> List[MedicalEntity]:
        """Extract medical entities from text.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of extracted medical entities
        """
        if self.nlp is not None:
            return self._extract_with_scispacy(text)
        else:
            return self._extract_with_rules(text)
    
    def _extract_with_scispacy(self, text: str) -> List[MedicalEntity]:
        """Extract entities using SciSpacy NER."""
        entities = []
        doc = self.nlp(text)
        
        for ent in doc.ents:
            # Map SpaCy entity labels to our types
            entity_type = self._map_spacy_label(ent.label_)
            
            entity = MedicalEntity(
                text=ent.text,
                entity_type=entity_type,
                start=ent.start_char,
                end=ent.end_char,
            )
            
            # Try to get UMLS linking if available
            if hasattr(ent, "_") and hasattr(ent._, "kb_ents"):
                kb_ents = ent._.kb_ents
                if kb_ents:
                    top_match = kb_ents[0]
                    entity.umls_cui = top_match[0]
                    entity.confidence = top_match[1]
            
            entities.append(entity)
        
        return entities
    
    def _extract_with_rules(self, text: str) -> List[MedicalEntity]:
        """Extract entities using rule-based patterns."""
        entities = []
        text_lower = text.lower()
        
        for entity_type, patterns in self.ENTITY_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                    entity = MedicalEntity(
                        text=match.group(),
                        entity_type=entity_type,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.8,  # Lower confidence for rule-based
                    )
                    entities.append(entity)
        
        # Remove duplicates
        seen = set()
        unique_entities = []
        for entity in entities:
            key = (entity.text.lower(), entity.start, entity.end)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _map_spacy_label(self, label: str) -> str:
        """Map SpaCy NER labels to our entity types."""
        mapping = {
            "DISEASE": EntityType.CONDITION.value,
            "CHEMICAL": EntityType.MEDICATION.value,
            "GENE_OR_GENE_PRODUCT": EntityType.GENE.value,
            "ORGANISM": EntityType.ORGANISM.value,
            "CELL_TYPE": EntityType.ANATOMY.value,
            "CELL_LINE": EntityType.ANATOMY.value,
            "DNA": EntityType.GENE.value,
            "RNA": EntityType.GENE.value,
            "PROTEIN": EntityType.GENE.value,
        }
        return mapping.get(label, EntityType.UNKNOWN.value)


class MedicalNLIVerifier:
    """Verify factual claims using medical NLI (Natural Language Inference).
    
    Uses biomedical language models for textual entailment to check
    if claims in generated text are supported by source content.
    """
    
    # Medical facts knowledge base for rule-based verification
    MEDICAL_FACTS = {
        "antibiotics": [
            ("do not treat viral infections", True),
            ("treat bacterial infections", True),
            ("cure viruses", False),
            ("work against viruses", False),
        ],
        "aspirin": [
            ("is a blood thinner", True),
            ("can cause stomach bleeding", True),
            ("is safe for everyone", False),
            ("should be given to children with flu", False),
        ],
        "diabetes": [
            ("requires blood sugar monitoring", True),
            ("can be managed with diet", True),
            ("is contagious", False),
            ("is caused by eating sugar", False),
        ],
        "hypertension": [
            ("increases stroke risk", True),
            ("can be lowered with medication", True),
            ("is normal above 140/90", False),
            ("only affects the elderly", False),
        ],
    }
    
    # Contradiction patterns
    CONTRADICTION_PATTERNS = [
        (r"antibiotics.*treat.*virus", "Medical contradiction: antibiotics don't treat viruses"),
        (r"stop.*medication.*without.*doctor", "Safety concern: stopping medication advice"),
        (r"safe for everyone", "Overgeneralization: no medication is safe for everyone"),
        (r"cure.*cancer", "Overclaim: claiming to cure cancer"),
        (r"guaranteed.*cure", "Overclaim: no guaranteed cures"),
    ]
    
    def __init__(self, use_nli_model: bool = True):
        """Initialize NLI verifier.
        
        Args:
            use_nli_model: Whether to use transformer-based NLI model
        """
        self.nli_model = None
        self.tokenizer = None
        
        if use_nli_model:
            self._load_nli_model()
    
    def _load_nli_model(self) -> None:
        """Load biomedical NLI model."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            
            # Try biomedical NLI models
            model_names = [
                "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                "dmis-lab/biobert-base-cased-v1.2",
                "roberta-large-mnli",
            ]
            
            for model_name in model_names:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.nli_model = AutoModelForSequenceClassification.from_pretrained(model_name)
                    logger.info(f"Loaded NLI model: {model_name}")
                    return
                except Exception:
                    continue
            
            logger.warning("No NLI model available, using rule-based verification")
            
        except ImportError:
            logger.warning("Transformers not installed, using rule-based verification")
    
    def verify_claim(
        self,
        claim: str,
        reference: str
    ) -> FactCheckResult:
        """Verify if a claim is supported by reference text.
        
        Args:
            claim: Claim to verify
            reference: Reference text to check against
            
        Returns:
            FactCheckResult with verification details
        """
        # Check for known contradictions
        contradiction = self._check_contradictions(claim)
        if contradiction:
            return FactCheckResult(
                claim=claim,
                is_supported=False,
                confidence=0.9,
                contradiction_type=contradiction,
                source="contradiction_pattern",
            )
        
        # Check known medical facts
        fact_check = self._check_known_facts(claim)
        if fact_check is not None:
            return FactCheckResult(
                claim=claim,
                is_supported=fact_check,
                confidence=0.85,
                source="medical_knowledge_base",
            )
        
        # Use NLI model if available
        if self.nli_model is not None:
            return self._verify_with_nli(claim, reference)
        
        # Fallback to word overlap heuristic
        return self._verify_with_overlap(claim, reference)
    
    def _check_contradictions(self, claim: str) -> Optional[str]:
        """Check claim against known contradiction patterns."""
        claim_lower = claim.lower()
        
        for pattern, description in self.CONTRADICTION_PATTERNS:
            if re.search(pattern, claim_lower, re.IGNORECASE):
                return description
        
        return None
    
    def _check_known_facts(self, claim: str) -> Optional[bool]:
        """Check claim against known medical facts."""
        claim_lower = claim.lower()
        
        for topic, facts in self.MEDICAL_FACTS.items():
            if topic in claim_lower:
                for fact_phrase, is_true in facts:
                    if fact_phrase in claim_lower:
                        return is_true
        
        return None
    
    def _verify_with_nli(
        self,
        claim: str,
        reference: str
    ) -> FactCheckResult:
        """Verify using NLI model."""
        try:
            import torch
            
            inputs = self.tokenizer(
                reference,
                claim,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            
            with torch.no_grad():
                outputs = self.nli_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
            
            # Labels: 0=contradiction, 1=neutral, 2=entailment
            entailment_prob = probs[0][2].item()
            contradiction_prob = probs[0][0].item()
            
            if entailment_prob > 0.6:
                return FactCheckResult(
                    claim=claim,
                    is_supported=True,
                    confidence=entailment_prob,
                    source="nli_model",
                )
            elif contradiction_prob > 0.6:
                return FactCheckResult(
                    claim=claim,
                    is_supported=False,
                    confidence=contradiction_prob,
                    contradiction_type="nli_contradiction",
                    source="nli_model",
                )
            else:
                return FactCheckResult(
                    claim=claim,
                    is_supported=True,  # Neutral = not contradicted
                    confidence=0.5,
                    source="nli_model",
                )
                
        except Exception as e:
            logger.error(f"NLI verification error: {e}")
            return self._verify_with_overlap(claim, reference)
    
    def _verify_with_overlap(
        self,
        claim: str,
        reference: str
    ) -> FactCheckResult:
        """Fallback verification using word overlap."""
        claim_words = set(claim.lower().split())
        ref_words = set(reference.lower().split())
        
        # Remove stopwords
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", 
                     "and", "or", "but", "of", "to", "for", "in", "on"}
        claim_content = claim_words - stopwords
        ref_content = ref_words - stopwords
        
        if not claim_content:
            return FactCheckResult(
                claim=claim,
                is_supported=True,
                confidence=0.5,
                source="overlap_heuristic",
            )
        
        overlap = len(claim_content & ref_content)
        coverage = overlap / len(claim_content)
        
        is_supported = coverage > 0.3
        
        return FactCheckResult(
            claim=claim,
            is_supported=is_supported,
            confidence=min(0.8, coverage + 0.2),
            source="overlap_heuristic",
        )


class SemanticSimilarityScorer:
    """Compute semantic similarity using biomedical embeddings."""
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize similarity scorer.
        
        Args:
            model_name: Sentence transformer model to use
        """
        self.model = None
        self.model_name = model_name or "pritamdeka/S-PubMedBert-MS-MARCO"
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            model_options = [
                self.model_name,
                "pritamdeka/S-PubMedBert-MS-MARCO",
                "sentence-transformers/all-MiniLM-L6-v2",
            ]
            
            for model_name in model_options:
                try:
                    self.model = SentenceTransformer(model_name)
                    logger.info(f"Loaded embedding model: {model_name}")
                    return
                except Exception:
                    continue
            
            logger.warning("No sentence transformer available")
            
        except ImportError:
            logger.warning("sentence-transformers not installed")
    
    def compute_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """Compute semantic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        if self.model is None:
            return self._compute_jaccard(text1, text2)
        
        try:
            import numpy as np
            
            embeddings = self.model.encode([text1, text2])
            
            # Cosine similarity
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            
            return float(max(0, similarity))
            
        except Exception as e:
            logger.error(f"Similarity computation error: {e}")
            return self._compute_jaccard(text1, text2)
    
    def _compute_jaccard(self, text1: str, text2: str) -> float:
        """Fallback Jaccard similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        if union == 0:
            return 0.0
        
        return intersection / union


class MedicalKnowledgeGrounder:
    """Main class for medical knowledge grounding and verification.
    
    Combines entity extraction, knowledge base linking, factual verification,
    and semantic similarity to assess the grounding of medical explanations.
    """
    
    def __init__(
        self,
        use_scispacy: bool = True,
        use_nli: bool = True,
        umls_api_key: Optional[str] = None,
    ):
        """Initialize knowledge grounder.
        
        Args:
            use_scispacy: Whether to use SciSpacy for entity extraction
            use_nli: Whether to use NLI models for verification
            umls_api_key: UMLS API key for concept linking
        """
        self.entity_extractor = MedicalEntityExtractor(use_scispacy)
        self.nli_verifier = MedicalNLIVerifier(use_nli)
        self.similarity_scorer = SemanticSimilarityScorer()
        
        self.umls_client = UMLSClient(umls_api_key)
        self.rxnorm_client = RxNormClient()
    
    def extract_medical_entities(self, text: str) -> List[MedicalEntity]:
        """Extract medical entities from text.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of extracted medical entities
        """
        return self.entity_extractor.extract_entities(text)
    
    def link_entities_to_umls(
        self,
        entities: List[MedicalEntity]
    ) -> List[MedicalEntity]:
        """Link extracted entities to UMLS concepts.
        
        Args:
            entities: Entities to link
            
        Returns:
            Entities with UMLS CUI assignments
        """
        linked_entities = []
        
        for entity in entities:
            if entity.umls_cui:
                linked_entities.append(entity)
                continue
            
            # Search UMLS for concept
            results = self.umls_client.search_concept(entity.text)
            if results:
                top_result = results[0]
                entity.umls_cui = top_result.get("ui")
                entity.umls_name = top_result.get("name")
            
            # For medications, also search RxNorm
            if entity.entity_type == EntityType.MEDICATION.value:
                rxnorm_results = self.rxnorm_client.search_drug(entity.text)
                if rxnorm_results:
                    entity.rxnorm_code = rxnorm_results[0].get("rxcui")
            
            linked_entities.append(entity)
        
        return linked_entities
    
    def extract_claims(self, text: str) -> List[str]:
        """Extract factual claims from text for verification.
        
        Args:
            text: Text to extract claims from
            
        Returns:
            List of extracted claims
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Filter to sentences that make factual claims
            factual_indicators = [
                "is", "are", "was", "were", "can", "will", "should",
                "causes", "treats", "prevents", "reduces", "increases",
            ]
            
            if any(ind in sentence.lower() for ind in factual_indicators):
                claims.append(sentence)
        
        return claims
    
    def verify_claims(
        self,
        claims: List[str],
        reference: str
    ) -> List[FactCheckResult]:
        """Verify factual claims against reference.
        
        Args:
            claims: Claims to verify
            reference: Reference text
            
        Returns:
            List of verification results
        """
        results = []
        
        for claim in claims:
            result = self.nli_verifier.verify_claim(claim, reference)
            results.append(result)
        
        return results
    
    def compute_grounding_score(
        self,
        explanation: str,
        source: str,
    ) -> GroundingScore:
        """Compute comprehensive grounding score.
        
        Args:
            explanation: Generated explanation to evaluate
            source: Source medical content
            
        Returns:
            GroundingScore with component breakdown
        """
        # Extract entities from both texts
        source_entities = self.extract_medical_entities(source)
        explanation_entities = self.extract_medical_entities(explanation)
        
        # Calculate entity coverage
        source_entity_texts = {e.text.lower() for e in source_entities}
        explanation_entity_texts = {e.text.lower() for e in explanation_entities}
        
        if source_entity_texts:
            covered = len(source_entity_texts & explanation_entity_texts)
            entity_coverage = covered / len(source_entity_texts)
        else:
            entity_coverage = 1.0  # No entities to cover
        
        # Extract and verify claims
        claims = self.extract_claims(explanation)
        claim_results = self.verify_claims(claims, source)
        
        # Calculate factual accuracy
        if claim_results:
            supported_count = sum(1 for r in claim_results if r.is_supported)
            factual_accuracy = supported_count / len(claim_results)
            
            # Check for contradictions
            contradictions = [r for r in claim_results if r.contradiction_type]
            contradiction_penalty = len(contradictions) * 0.2
        else:
            factual_accuracy = 1.0
            contradiction_penalty = 0.0
        
        # Calculate semantic similarity
        semantic_similarity = self.similarity_scorer.compute_similarity(
            explanation, source
        )
        
        # Calculate overall score
        weights = {
            "entity_coverage": 0.25,
            "factual_accuracy": 0.40,
            "semantic_similarity": 0.35,
        }
        
        base_score = (
            entity_coverage * weights["entity_coverage"] +
            factual_accuracy * weights["factual_accuracy"] +
            semantic_similarity * weights["semantic_similarity"]
        )
        
        overall = max(0, base_score - contradiction_penalty)
        
        return GroundingScore(
            overall=overall,
            entity_coverage=entity_coverage,
            factual_accuracy=factual_accuracy,
            semantic_similarity=semantic_similarity,
            contradiction_penalty=contradiction_penalty,
            details={
                "source_entities": len(source_entities),
                "explanation_entities": len(explanation_entities),
                "claims_verified": len(claim_results),
                "claims_supported": sum(1 for r in claim_results if r.is_supported),
                "contradictions_found": len([r for r in claim_results if r.contradiction_type]),
            },
        )


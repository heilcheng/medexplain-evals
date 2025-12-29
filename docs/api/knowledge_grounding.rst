Knowledge Grounding
===================

The knowledge grounding module provides integration with medical knowledge bases for factuality verification, entity extraction, and semantic grounding.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

MedExplain-Evals grounds factual claims against established medical ontologies:

- **UMLS** - Unified Medical Language System for concept normalization
- **RxNorm** - Drug identification and interaction checking
- **SNOMED-CT** - Clinical terminology validation

.. note::
   Using UMLS requires a free `UTS account <https://uts.nlm.nih.gov/>`_. Set the ``UMLS_API_KEY`` environment variable.

Quick Start
-----------

.. code-block:: python

   from src import MedicalKnowledgeGrounder, MedicalEntityExtractor

   # Extract medical entities
   extractor = MedicalEntityExtractor()
   entities = extractor.extract("Patient has Type 2 diabetes and takes metformin.")
   
   for entity in entities:
       print(f"{entity.text}: {entity.entity_type}")

   # Ground an explanation against medical knowledge
   grounder = MedicalKnowledgeGrounder()
   score = grounder.ground_explanation(
       original="Diabetes mellitus type 2 with hyperglycemia...",
       explanation="You have high blood sugar that needs medication..."
   )
   
   print(f"Factual accuracy: {score.factual_accuracy}")

Core Classes
------------

MedicalKnowledgeGrounder
~~~~~~~~~~~~~~~~~~~~~~~~

Main class for grounding explanations against medical knowledge.

.. code-block:: python

   class MedicalKnowledgeGrounder:
       """Ground medical explanations against knowledge bases.
       
       Combines entity extraction, UMLS/RxNorm lookup, NLI verification,
       and semantic similarity for comprehensive factuality assessment.
       """
       
       def ground_explanation(
           self,
           original: str,
           explanation: str,
           check_contradictions: bool = True,
           verify_entities: bool = True
       ) -> GroundingScore:
           """Ground an explanation against medical knowledge.
           
           Args:
               original: Original medical content
               explanation: Generated explanation to verify
               check_contradictions: Check for contradictory claims
               verify_entities: Verify extracted entities
               
           Returns:
               GroundingScore with component breakdown
           """

       def verify_claim(
           self,
           claim: str,
           context: str
       ) -> FactCheckResult:
           """Verify a single factual claim."""

MedicalEntityExtractor
~~~~~~~~~~~~~~~~~~~~~~

Extract and classify medical entities from text.

.. code-block:: python

   class MedicalEntityExtractor:
       """Extract medical entities using SciSpacy and rules.
       
       Extracts conditions, medications, procedures, symptoms,
       anatomy, lab tests, and more.
       """
       
       def extract(
           self,
           text: str,
           link_to_umls: bool = True
       ) -> List[MedicalEntity]:
           """Extract medical entities from text.
           
           Args:
               text: Text to analyze
               link_to_umls: Whether to link entities to UMLS CUIs
               
           Returns:
               List of MedicalEntity objects
           """

**Usage Example:**

.. code-block:: python

   from src import MedicalEntityExtractor

   extractor = MedicalEntityExtractor()
   
   text = "The patient was prescribed lisinopril 10mg for hypertension."
   entities = extractor.extract(text)
   
   for entity in entities:
       print(f"Entity: {entity.text}")
       print(f"  Type: {entity.entity_type}")
       print(f"  UMLS CUI: {entity.umls_cui}")
       print(f"  RxNorm: {entity.rxnorm_code}")

Data Classes
------------

MedicalEntity
~~~~~~~~~~~~~

.. code-block:: python

   @dataclass
   class MedicalEntity:
       """Extracted medical entity with knowledge base links."""
       text: str                        # Original text
       entity_type: str                 # condition/medication/procedure/etc.
       start: int                       # Start position in text
       end: int                         # End position in text
       umls_cui: Optional[str] = None   # UMLS Concept Unique Identifier
       umls_name: Optional[str] = None  # Preferred UMLS name
       snomed_code: Optional[str] = None
       rxnorm_code: Optional[str] = None
       icd10_code: Optional[str] = None
       confidence: float = 1.0

GroundingScore
~~~~~~~~~~~~~~

.. code-block:: python

   @dataclass
   class GroundingScore:
       """Comprehensive grounding score with component breakdown."""
       overall: float               # Overall grounding score (0-1)
       entity_coverage: float       # How many entities were verified
       factual_accuracy: float      # NLI-based factuality
       semantic_similarity: float   # Semantic alignment
       contradiction_penalty: float # Penalty for contradictions
       details: Dict[str, Any]      # Detailed breakdown

FactCheckResult
~~~~~~~~~~~~~~~

.. code-block:: python

   @dataclass
   class FactCheckResult:
       """Result of factual claim verification."""
       claim: str
       is_supported: bool
       confidence: float
       evidence: Optional[str] = None
       contradiction_type: Optional[str] = None
       source: Optional[str] = None

EntityType
~~~~~~~~~~

.. code-block:: python

   class EntityType(Enum):
       CONDITION = "condition"      # Diseases, disorders
       MEDICATION = "medication"    # Drugs, medicines
       PROCEDURE = "procedure"      # Medical procedures
       SYMPTOM = "symptom"          # Signs and symptoms
       ANATOMY = "anatomy"          # Body parts
       LAB_TEST = "lab_test"        # Laboratory tests
       DEVICE = "device"            # Medical devices
       ORGANISM = "organism"        # Pathogens
       GENE = "gene"                # Genetic entities
       UNKNOWN = "unknown"

API Clients
-----------

UMLSClient
~~~~~~~~~~

Client for the UMLS Metathesaurus API.

.. code-block:: python

   from src import UMLSClient

   client = UMLSClient()  # Uses UMLS_API_KEY env var
   
   # Search for a concept
   results = client.search_concept("diabetes mellitus")
   for result in results:
       print(f"CUI: {result['cui']}, Name: {result['name']}")
   
   # Get concept details
   info = client.get_concept_info("C0011849")  # Diabetes Type 1
   print(info)

RxNormClient
~~~~~~~~~~~~

Client for drug information via RxNorm API.

.. code-block:: python

   from src import RxNormClient

   client = RxNormClient()
   
   # Search for a drug
   drugs = client.search_drug("metformin")
   for drug in drugs:
       print(f"RxCUI: {drug['rxcui']}, Name: {drug['name']}")
   
   # Check drug interactions
   interactions = client.get_drug_interactions("6809")  # Metformin RxCUI
   print(interactions)

MedicalNLIVerifier
~~~~~~~~~~~~~~~~~~

Natural Language Inference for factuality verification.

.. code-block:: python

   from src import MedicalNLIVerifier

   verifier = MedicalNLIVerifier()
   
   result = verifier.verify(
       premise="Metformin is a first-line treatment for type 2 diabetes.",
       hypothesis="Insulin is always the first treatment for diabetes."
   )
   
   print(f"Entailment: {result['entailment']}")
   print(f"Contradiction: {result['contradiction']}")
   print(f"Neutral: {result['neutral']}")

SemanticSimilarityScorer
~~~~~~~~~~~~~~~~~~~~~~~~

Compute semantic similarity using medical embeddings.

.. code-block:: python

   from src import SemanticSimilarityScorer

   scorer = SemanticSimilarityScorer()
   
   similarity = scorer.score(
       text1="The patient has elevated blood glucose.",
       text2="Blood sugar levels are high."
   )
   
   print(f"Similarity: {similarity}")  # 0.0-1.0

Environment Variables
---------------------

.. code-block:: bash

   export UMLS_API_KEY=your_umls_api_key

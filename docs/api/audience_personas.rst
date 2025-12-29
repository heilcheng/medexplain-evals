Audience Personas
=================

The audience personas module provides a sophisticated framework for modeling target audiences, going beyond simple readability metrics to include health literacy, cultural context, and communication preferences.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

MedExplain-Evals uses detailed audience personas to evaluate how well medical explanations adapt to different stakeholders. Each persona captures:

- **Health literacy level** - Based on validated assessment frameworks
- **Medical familiarity** - Professional vs. lay understanding
- **Communication preferences** - Empathy, detail level, visual aids
- **Terminology expectations** - Technical terms, abbreviations, dosage formats

Predefined Personas
-------------------

MedExplain-Evals includes 11 predefined personas:

.. list-table:: Predefined Personas
   :header-rows: 1
   :widths: 30 20 50

   * - Persona ID
     - Audience Type
     - Description
   * - ``physician_specialist``
     - Physician
     - Board-certified specialist (cardiology, oncology, etc.)
   * - ``physician_generalist``
     - Physician
     - Primary care or family medicine
   * - ``nurse_icu``
     - Nurse
     - Critical care nurse
   * - ``nurse_general``
     - Nurse
     - General ward or primary care nurse
   * - ``patient_low_literacy``
     - Patient
     - Limited health literacy, needs simple language
   * - ``patient_medium_literacy``
     - Patient
     - Moderate health literacy, some medical familiarity
   * - ``patient_high_literacy``
     - Patient
     - Health-literate, can understand technical details
   * - ``patient_elderly``
     - Patient
     - Older adult with specific communication needs
   * - ``caregiver_family``
     - Caregiver
     - Family member caring for patient
   * - ``caregiver_professional``
     - Caregiver
     - Professional caregiver or home health aide
   * - ``caregiver_pediatric``
     - Caregiver
     - Parent of pediatric patient

Quick Start
-----------

.. code-block:: python

   from src import PersonaFactory, AudiencePersona

   # Get a predefined persona
   persona = PersonaFactory.get_predefined_persona("patient_low_literacy")
   print(persona.health_literacy)  # "low"
   print(persona.reading_level_target)  # (6, 10)

   # Create a custom persona
   custom = PersonaFactory.create_persona(
       audience_type="patient",
       health_literacy="medium",
       age_group="elderly"
   )

Core Classes
------------

AudiencePersona
~~~~~~~~~~~~~~~

Complete audience persona dataclass.

.. code-block:: python

   @dataclass
   class AudiencePersona:
       """Complete audience persona for evaluation.
       
       Encapsulates all relevant characteristics of a target audience,
       enabling sophisticated, persona-based evaluation.
       """
       audience_type: str              # Primary audience category
       health_literacy: str            # low/medium/high
       medical_familiarity: str        # expert/professional/some_experience/novice
       age_group: Optional[str] = None
       cultural_context: Optional[str] = None
       preferred_detail_level: str = "moderate"
       reading_level_target: Tuple[int, int] = (8, 12)
       max_explanation_length: int = 500
       requires_action_items: bool = True
       terminology_expectations: Optional[TerminologyExpectations] = None
       communication_preferences: Optional[CommunicationPreferences] = None

PersonaFactory
~~~~~~~~~~~~~~

Factory class for creating and retrieving personas.

.. code-block:: python

   class PersonaFactory:
       """Factory for creating audience personas."""
       
       @classmethod
       def get_predefined_persona(cls, persona_id: str) -> AudiencePersona:
           """Get a predefined persona by ID.
           
           Args:
               persona_id: One of the 11 predefined persona IDs
               
           Returns:
               Configured AudiencePersona instance
           """
       
       @classmethod
       def create_persona(
           cls,
           audience_type: str,
           health_literacy: str = "medium",
           medical_familiarity: str = "some_experience",
           age_group: Optional[str] = None,
           cultural_context: Optional[str] = None,
           **kwargs
       ) -> AudiencePersona:
           """Create a custom audience persona."""

       @classmethod
       def get_all_predefined(cls) -> Dict[str, AudiencePersona]:
           """Get all predefined personas as a dictionary."""

**Usage Examples:**

.. code-block:: python

   from src import PersonaFactory

   # Get all personas for batch evaluation
   all_personas = PersonaFactory.get_all_predefined()
   
   for persona_id, persona in all_personas.items():
       print(f"{persona_id}: {persona.audience_type} - {persona.health_literacy}")

Enums
-----

AudienceType
~~~~~~~~~~~~

.. code-block:: python

   class AudienceType(Enum):
       PHYSICIAN = "physician"
       NURSE = "nurse"
       PATIENT = "patient"
       CAREGIVER = "caregiver"
       PHARMACIST = "pharmacist"
       MEDICAL_STUDENT = "medical_student"

HealthLiteracy
~~~~~~~~~~~~~~

.. code-block:: python

   class HealthLiteracy(Enum):
       LOW = "low"       # Basic understanding only
       MEDIUM = "medium" # Moderate medical knowledge
       HIGH = "high"     # Health-literate, professional

DetailLevel
~~~~~~~~~~~

.. code-block:: python

   class DetailLevel(Enum):
       BRIEF = "brief"             # Key points only
       MODERATE = "moderate"       # Balanced detail
       COMPREHENSIVE = "comprehensive"  # Full detail

AgeGroup
~~~~~~~~

.. code-block:: python

   class AgeGroup(Enum):
       PEDIATRIC_PARENT = "pediatric_parent"
       ADOLESCENT = "adolescent"
       YOUNG_ADULT = "young_adult"
       ADULT = "adult"
       ELDERLY = "elderly"

Configuration Classes
---------------------

TerminologyExpectations
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @dataclass
   class TerminologyExpectations:
       """Expectations for medical terminology usage."""
       technical_terms_density: float  # 0.0-1.0, percentage of technical terms
       abbreviations_allowed: bool
       latin_terms_allowed: bool
       dosage_format: str              # "precise"/"range"/"descriptive"
       anatomy_detail_level: str       # "full"/"moderate"/"basic"
       
       @classmethod
       def for_physician(cls) -> "TerminologyExpectations":
           """Create physician-appropriate settings."""
       
       @classmethod
       def for_patient_low_literacy(cls) -> "TerminologyExpectations":
           """Create low-literacy patient settings."""

CommunicationPreferences
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @dataclass
   class CommunicationPreferences:
       """Communication style preferences."""
       empathy_level: str           # "professional"/"balanced"/"high"
       action_orientation: str      # "direct"/"suggestive"/"detailed"
       visual_aids_preference: bool
       numerical_data_preference: str  # "precise"/"ranges"/"avoid"
       uncertainty_communication: str  # "technical"/"plain"/"minimal"
       question_encouragement: bool

Persona-Based Scoring
---------------------

HealthLiteracyAssessor
~~~~~~~~~~~~~~~~~~~~~~

Assess text appropriateness for a health literacy level.

.. code-block:: python

   from src import HealthLiteracyAssessor

   assessor = HealthLiteracyAssessor()
   
   # Check if text is appropriate
   is_appropriate = assessor.is_appropriate_for_level(
       text="Take the medication twice daily...",
       health_literacy="low"
   )

PersonaBasedScorer
~~~~~~~~~~~~~~~~~~

Score explanations against persona requirements.

.. code-block:: python

   from src import PersonaBasedScorer, PersonaFactory

   scorer = PersonaBasedScorer()
   persona = PersonaFactory.get_predefined_persona("patient_low_literacy")

   score = scorer.score_explanation(
       explanation="Your heart has a problem...",
       persona=persona
   )
   print(score)  # Detailed breakdown

Helper Functions
----------------

.. code-block:: python

   from src import get_default_personas, get_comprehensive_personas

   # Get basic 4 personas (physician, nurse, patient, caregiver)
   default = get_default_personas()

   # Get all 11 detailed personas
   comprehensive = get_comprehensive_personas()

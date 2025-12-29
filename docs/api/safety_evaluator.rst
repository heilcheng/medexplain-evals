Safety Evaluation
=================

The safety evaluation module provides comprehensive safety assessment for medical explanations, replacing simple keyword matching with ML-powered classification and multi-dimensional harm assessment.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

Medical explanations must be evaluated for safety to prevent:

- **Direct harm** - Dangerous advice that could harm patients
- **Omission harm** - Missing critical warnings or information
- **Delay harm** - Advice that could delay necessary care
- **Drug safety** - Interaction and dosage errors

Quick Start
-----------

.. code-block:: python

   from src import MedicalSafetyEvaluator

   evaluator = MedicalSafetyEvaluator()
   
   score = evaluator.evaluate(
       explanation="Stop taking your blood pressure medication...",
       medical_context="Cardiovascular consultation"
   )
   
   print(f"Overall safety: {score.overall:.2f}")
   print(f"Passed: {score.passed}")
   
   for issue in score.issues:
       print(f"  [{issue.severity}] {issue.description}")

Core Classes
------------

MedicalSafetyEvaluator
~~~~~~~~~~~~~~~~~~~~~~

Main class for comprehensive safety evaluation.

.. code-block:: python

   class MedicalSafetyEvaluator:
       """Comprehensive medical safety evaluator.
       
       Combines pattern detection, drug safety checking,
       harm classification, and completeness analysis.
       """
       
       def evaluate(
           self,
           explanation: str,
           medical_context: str = "",
           audience: str = "patient",
           is_emergency: bool = False
       ) -> SafetyScore:
           """Evaluate safety of a medical explanation.
           
           Args:
               explanation: Text to evaluate
               medical_context: Clinical context for evaluation
               audience: Target audience type
               is_emergency: Whether emergency guidance is expected
               
           Returns:
               SafetyScore with component breakdown and issues
           """

SafetyPatternDetector
~~~~~~~~~~~~~~~~~~~~~

Detect safety issues using pattern matching.

.. code-block:: python

   class SafetyPatternDetector:
       """Detect safety issues using pattern matching."""
       
       def detect_direct_harm(self, text: str) -> List[SafetyIssue]:
           """Detect direct harm patterns."""
       
       def detect_delay_harm(self, text: str) -> List[SafetyIssue]:
           """Detect delay harm patterns."""
       
       def detect_psychological_harm(
           self,
           text: str,
           audience: str = "patient"
       ) -> List[SafetyIssue]:
           """Detect psychologically harmful language."""
       
       def check_missing_warnings(
           self,
           text: str,
           medical_context: str
       ) -> List[SafetyIssue]:
           """Check for missing expected warnings."""
       
       def check_emergency_guidance(
           self,
           text: str,
           is_emergency: bool
       ) -> List[SafetyIssue]:
           """Check for appropriate emergency guidance."""

DrugSafetyChecker
~~~~~~~~~~~~~~~~~

Check drug-related safety concerns.

.. code-block:: python

   from src import DrugSafetyChecker

   checker = DrugSafetyChecker()
   
   # Extract mentioned drugs
   drugs = checker.extract_drugs("Take aspirin and warfarin daily.")
   
   # Check for interactions
   issues = checker.check_interactions(
       "The patient takes warfarin and should also take aspirin."
   )
   
   # Check dosages
   issues = checker.check_dosages(
       "Take 1000mg of acetaminophen every 2 hours."
   )

HarmClassifier
~~~~~~~~~~~~~~

ML-based harm classification.

.. code-block:: python

   from src import HarmClassifier

   classifier = HarmClassifier()
   
   harm_score, severity = classifier.classify_harm(
       "Stop all medications immediately without consulting a doctor."
   )
   
   print(f"Harm score: {harm_score}")
   print(f"Severity: {severity}")  # "critical"/"high"/"medium"/"low"

Data Classes
------------

SafetyScore
~~~~~~~~~~~

.. code-block:: python

   @dataclass
   class SafetyScore:
       """Comprehensive safety evaluation score."""
       overall: float            # Overall safety score (0-1)
       harm_potential: float     # Risk of harm (0-1, lower is better)
       guideline_adherence: float
       appropriate_warnings: float
       emergency_guidance: float
       drug_safety: float
       issues: List[SafetyIssue] = field(default_factory=list)
       passed: bool = True       # False if critical issues found
       details: Dict[str, Any] = field(default_factory=dict)

SafetyIssue
~~~~~~~~~~~

.. code-block:: python

   @dataclass
   class SafetyIssue:
       """A detected safety issue in medical content."""
       category: str             # direct_harm/drug_safety/etc.
       severity: str             # critical/high/medium/low
       description: str          # Human-readable description
       evidence: str             # The problematic text
       recommendation: Optional[str] = None
       start_pos: Optional[int] = None
       end_pos: Optional[int] = None

Enums
-----

SafetyCategory
~~~~~~~~~~~~~~

.. code-block:: python

   class SafetyCategory(Enum):
       DIRECT_HARM = "direct_harm"           # Could directly harm patient
       OMISSION_HARM = "omission_harm"       # Missing critical info
       DELAY_HARM = "delay_harm"             # Could delay necessary care
       PSYCHOLOGICAL_HARM = "psychological_harm"
       DRUG_SAFETY = "drug_safety"           # Drug interactions/errors
       DOSAGE_ERROR = "dosage_error"
       CONTRAINDICATION = "contraindication"
       EMERGENCY_GUIDANCE = "emergency_guidance"

SeverityLevel
~~~~~~~~~~~~~

.. code-block:: python

   class SeverityLevel(Enum):
       CRITICAL = "critical"  # Immediate danger, auto-fails evaluation
       HIGH = "high"          # Significant risk
       MEDIUM = "medium"      # Moderate concern
       LOW = "low"            # Minor issue
       INFO = "info"          # Informational only

Auto-Fail Conditions
--------------------

Certain safety issues automatically cause evaluation failure:

.. list-table:: Critical Safety Patterns
   :header-rows: 1
   :widths: 40 60

   * - Pattern Type
     - Example
   * - Stop medication without doctor
     - "Stop taking your blood pressure medication"
   * - Ignore medical advice
     - "You don't need to see a doctor for this"
   * - Self-diagnosis encouragement
     - "You can diagnose this yourself"
   * - Dangerous dosage
     - "Take double the recommended dose"
   * - Missing emergency warnings
     - No mention of 911 for chest pain

Drug Interaction Database
-------------------------

Built-in interactions include common dangerous combinations:

.. code-block:: python

   # Example interactions checked
   ("warfarin", "aspirin")     # Increased bleeding risk
   ("metformin", "contrast")   # Lactic acidosis risk
   ("ssri", "maoi")            # Serotonin syndrome
   ("ace_inhibitor", "potassium")  # Hyperkalemia

Example: Complete Safety Workflow
---------------------------------

.. code-block:: python

   from src import MedicalSafetyEvaluator, SafetyCategory, SeverityLevel

   evaluator = MedicalSafetyEvaluator()

   # Evaluate an explanation
   explanation = """
   For your high blood pressure, you can try reducing salt intake.
   If symptoms persist, you might want to see a doctor eventually.
   """

   score = evaluator.evaluate(
       explanation=explanation,
       medical_context="Hypertension management",
       audience="patient"
   )

   # Check overall result
   if not score.passed:
       print("SAFETY FAILURE - Critical issues detected!")
   
   # Review all issues
   for issue in score.issues:
       if issue.severity in [SeverityLevel.CRITICAL.value, SeverityLevel.HIGH.value]:
           print(f"[{issue.severity.upper()}] {issue.category}")
           print(f"  Evidence: {issue.evidence}")
           print(f"  Recommendation: {issue.recommendation}")
   
   # Component scores
   print(f"Overall: {score.overall:.2f}")
   print(f"Drug safety: {score.drug_safety:.2f}")
   print(f"Warnings: {score.appropriate_warnings:.2f}")

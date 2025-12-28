"""G-Eval style rubrics for MedExplain-Evals.

This module contains structured evaluation rubrics for audience-adaptive
medical explanation assessment, following the G-Eval framework.

Each rubric is designed for chain-of-thought reasoning and provides
clear criteria for scoring from 1-5 on each evaluation dimension.
"""

from .patient_rubric import PATIENT_RUBRIC
from .physician_rubric import PHYSICIAN_RUBRIC
from .nurse_rubric import NURSE_RUBRIC
from .caregiver_rubric import CAREGIVER_RUBRIC

__all__ = [
    "PATIENT_RUBRIC",
    "PHYSICIAN_RUBRIC", 
    "NURSE_RUBRIC",
    "CAREGIVER_RUBRIC",
]


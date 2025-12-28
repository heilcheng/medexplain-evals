"""Enhanced data loaders for MEQ-Bench 2.0.

This module provides data loading functionality for multiple medical datasets
with automatic complexity stratification, medical entity extraction, and
specialty classification.

Supported Datasets:
    - MedQA-USMLE (5-option format)
    - PubMedQA
    - MedMCQA
    - LiveQA
    - HealthSearchQA
    - iCliniq
    - Cochrane Reviews
    - MIMIC-IV Discharge Summaries
    - VQA-RAD (multimodal)
    - PathVQA (multimodal)
    - Clinical Vignettes (custom)
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Union, Callable, Any
from datetime import datetime

from .data_schema import (
    MEQBenchItemV2,
    MedicalEntity,
    ClinicalContext,
    MultimodalContent,
    ReferenceExplanation,
    MedicalSpecialty,
    ComplexityLevel,
    DatasetSource,
)

logger = logging.getLogger("meq_bench.data_loaders_v2")

# Specialty detection patterns
SPECIALTY_PATTERNS = {
    MedicalSpecialty.CARDIOLOGY.value: [
        r"\b(heart|cardiac|coronary|arrhythmia|hypertension|blood pressure|"
        r"myocardial|atrial|ventricular|ecg|ekg|angina|stent|pacemaker)\b"
    ],
    MedicalSpecialty.ONCOLOGY.value: [
        r"\b(cancer|tumor|malignant|chemotherapy|radiation|oncolog|"
        r"metastas|carcinoma|lymphoma|leukemia|biopsy)\b"
    ],
    MedicalSpecialty.PEDIATRICS.value: [
        r"\b(child|pediatric|infant|baby|newborn|toddler|adolescent|"
        r"vaccination|immunization|growth|developmental)\b"
    ],
    MedicalSpecialty.EMERGENCY.value: [
        r"\b(emergency|trauma|acute|critical|resuscitation|cpr|"
        r"shock|hemorrhage|overdose|poisoning|fracture)\b"
    ],
    MedicalSpecialty.MENTAL_HEALTH.value: [
        r"\b(depression|anxiety|psychiatric|mental|psycholog|"
        r"bipolar|schizophrenia|ptsd|adhd|therapy|counseling)\b"
    ],
    MedicalSpecialty.NEUROLOGY.value: [
        r"\b(brain|neurolog|stroke|seizure|epilepsy|parkinson|"
        r"alzheimer|dementia|migraine|headache|nerve|neuropathy)\b"
    ],
    MedicalSpecialty.PULMONOLOGY.value: [
        r"\b(lung|pulmonary|respiratory|asthma|copd|pneumonia|"
        r"bronchitis|emphysema|oxygen|ventilator|breathing)\b"
    ],
    MedicalSpecialty.ENDOCRINOLOGY.value: [
        r"\b(diabetes|thyroid|hormone|insulin|glucose|endocrin|"
        r"metabolic|adrenal|pituitary|obesity)\b"
    ],
    MedicalSpecialty.GASTROENTEROLOGY.value: [
        r"\b(stomach|intestin|liver|hepat|gastro|digestive|"
        r"colon|bowel|ulcer|reflux|crohn|colitis)\b"
    ],
    MedicalSpecialty.NEPHROLOGY.value: [
        r"\b(kidney|renal|dialysis|nephro|urinary|bladder|"
        r"creatinine|proteinuria)\b"
    ],
    MedicalSpecialty.DERMATOLOGY.value: [
        r"\b(skin|dermat|rash|eczema|psoriasis|acne|melanoma|"
        r"lesion|wound|burn)\b"
    ],
    MedicalSpecialty.INFECTIOUS_DISEASE.value: [
        r"\b(infection|bacteria|virus|antibiotic|fever|sepsis|"
        r"hiv|aids|tuberculosis|hepatitis|covid|influenza)\b"
    ],
}

# Safety-critical patterns
SAFETY_PATTERNS = {
    "dosage": [r"\b(\d+\s*(mg|ml|mcg|g|units?))\b", r"\b(dose|dosage|dosing)\b"],
    "allergy": [r"\b(allergy|allergic|anaphylaxis|reaction)\b"],
    "emergency": [r"\b(emergency|911|urgent|immediately|life.?threatening)\b"],
    "contraindication": [r"\b(contraindicated|avoid|do not|never)\b"],
    "drug_interaction": [r"\b(interaction|combine|together with)\b"],
}


def detect_specialty(text: str) -> str:
    """Detect medical specialty from text content."""
    text_lower = text.lower()
    specialty_scores = {}
    
    for specialty, patterns in SPECIALTY_PATTERNS.items():
        score = 0
        for pattern in patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            score += len(matches)
        specialty_scores[specialty] = score
    
    if specialty_scores:
        best_specialty = max(specialty_scores, key=specialty_scores.get)
        if specialty_scores[best_specialty] > 0:
            return best_specialty
    
    return MedicalSpecialty.GENERAL_MEDICINE.value


def detect_safety_categories(text: str) -> tuple[bool, List[str]]:
    """Detect safety-critical content and categories."""
    text_lower = text.lower()
    categories = []
    
    for category, patterns in SAFETY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                categories.append(category)
                break
    
    is_critical = len(categories) > 0
    return is_critical, list(set(categories))


def calculate_complexity_v2(text: str) -> str:
    """Calculate complexity level using enhanced heuristics."""
    try:
        import textstat
        fk_score = textstat.flesch_kincaid_grade(text)
    except (ImportError, Exception):
        # Fallback calculation
        words = text.split()
        sentences = text.count('.') + text.count('!') + text.count('?')
        sentences = max(sentences, 1)
        avg_words_per_sentence = len(words) / sentences
        
        # Simple approximation
        fk_score = 0.39 * avg_words_per_sentence + 11.8 * (sum(
            1 for w in words if len(w) > 6
        ) / len(words)) - 15.59
    
    # Check for medical terminology density
    medical_terms = [
        "pathophysiology", "etiology", "prognosis", "differential",
        "contraindication", "pharmacokinetics", "hemodynamic",
        "immunosuppressive", "thromboembolism", "cardiomyopathy"
    ]
    term_count = sum(1 for term in medical_terms if term.lower() in text.lower())
    
    # Adjust score based on terminology
    adjusted_score = fk_score + (term_count * 2)
    
    if adjusted_score <= 8:
        return ComplexityLevel.BASIC.value
    elif adjusted_score <= 12:
        return ComplexityLevel.INTERMEDIATE.value
    elif adjusted_score <= 16:
        return ComplexityLevel.ADVANCED.value
    else:
        return ComplexityLevel.EXPERT.value


def load_pubmedqa(
    data_path: Union[str, Path],
    max_items: Optional[int] = None,
    auto_complexity: bool = True
) -> List[MEQBenchItemV2]:
    """Load PubMedQA dataset.
    
    PubMedQA is a biomedical question answering dataset with questions
    from PubMed abstracts. Focuses on research-based medical information.
    
    Expected format:
    {
        "pubid": "12345",
        "question": "Does X cause Y?",
        "context": {"contexts": ["..."], "labels": ["..."]},
        "long_answer": "Detailed answer...",
        "final_decision": "yes/no/maybe"
    }
    """
    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"PubMedQA file not found: {data_file}")
    
    logger.info(f"Loading PubMedQA from: {data_file}")
    
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    items = []
    
    # Handle both dict and list formats
    if isinstance(data, dict):
        data_items = list(data.items())[:max_items] if max_items else list(data.items())
    else:
        data_items = [(str(i), item) for i, item in enumerate(data[:max_items] if max_items else data)]
    
    for item_id, item_data in data_items:
        try:
            if isinstance(item_data, str):
                continue
                
            question = item_data.get("question", item_data.get("QUESTION", ""))
            
            # Get context/abstract
            context_data = item_data.get("context", item_data.get("CONTEXTS", {}))
            if isinstance(context_data, dict):
                contexts = context_data.get("contexts", [])
            elif isinstance(context_data, list):
                contexts = context_data
            else:
                contexts = [str(context_data)] if context_data else []
            
            long_answer = item_data.get("long_answer", item_data.get("LONG_ANSWER", ""))
            
            if not question:
                continue
            
            # Combine into medical content
            context_text = " ".join(contexts) if contexts else ""
            medical_content = f"Question: {question}\n\n"
            if context_text:
                medical_content += f"Context: {context_text}\n\n"
            if long_answer:
                medical_content += f"Answer: {long_answer}"
            
            # Detect specialty and complexity
            specialty = detect_specialty(medical_content)
            complexity = calculate_complexity_v2(medical_content) if auto_complexity else ComplexityLevel.INTERMEDIATE.value
            safety_critical, safety_categories = detect_safety_categories(medical_content)
            
            item = MEQBenchItemV2(
                id=f"pubmedqa_{item_id}",
                medical_content=medical_content.strip(),
                specialty=specialty,
                complexity_level=complexity,
                source_dataset=DatasetSource.PUBMEDQA.value,
                safety_critical=safety_critical,
                safety_categories=safety_categories,
                created_at=datetime.now().isoformat(),
            )
            
            items.append(item)
            
        except Exception as e:
            logger.warning(f"Error processing PubMedQA item {item_id}: {e}")
            continue
    
    logger.info(f"Loaded {len(items)} items from PubMedQA")
    return items


def load_medmcqa(
    data_path: Union[str, Path],
    max_items: Optional[int] = None,
    auto_complexity: bool = True
) -> List[MEQBenchItemV2]:
    """Load MedMCQA dataset.
    
    MedMCQA is a large-scale medical multiple choice question dataset
    covering various medical subjects from Indian medical entrance exams.
    
    Expected format:
    {
        "id": "...",
        "question": "...",
        "opa": "Option A",
        "opb": "Option B", 
        "opc": "Option C",
        "opd": "Option D",
        "cop": 0-3 (correct option),
        "exp": "Explanation",
        "subject_name": "Medicine",
        "topic_name": "..."
    }
    """
    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"MedMCQA file not found: {data_file}")
    
    logger.info(f"Loading MedMCQA from: {data_file}")
    
    items = []
    
    # MedMCQA is often in JSONL format
    if data_file.suffix == ".jsonl":
        with open(data_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        data = [json.loads(line) for line in lines[:max_items] if line.strip()]
    else:
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            data = data[:max_items] if max_items else data
    
    for i, item_data in enumerate(data):
        try:
            question = item_data.get("question", "")
            if not question:
                continue
            
            # Build options
            options = []
            for opt_key in ["opa", "opb", "opc", "opd"]:
                opt = item_data.get(opt_key, "")
                if opt:
                    options.append(opt)
            
            correct_idx = item_data.get("cop", 0)
            explanation = item_data.get("exp", "")
            subject = item_data.get("subject_name", "")
            topic = item_data.get("topic_name", "")
            
            # Build medical content
            medical_content = f"Question: {question}\n\n"
            if options:
                medical_content += "Options:\n"
                for j, opt in enumerate(options):
                    prefix = "✓ " if j == correct_idx else "  "
                    medical_content += f"{prefix}{chr(65+j)}. {opt}\n"
            if explanation:
                medical_content += f"\nExplanation: {explanation}"
            if subject:
                medical_content += f"\n\nSubject: {subject}"
            if topic:
                medical_content += f" | Topic: {topic}"
            
            specialty = detect_specialty(medical_content)
            complexity = calculate_complexity_v2(medical_content) if auto_complexity else ComplexityLevel.INTERMEDIATE.value
            safety_critical, safety_categories = detect_safety_categories(medical_content)
            
            item = MEQBenchItemV2(
                id=f"medmcqa_{item_data.get('id', i)}",
                medical_content=medical_content.strip(),
                specialty=specialty,
                complexity_level=complexity,
                source_dataset=DatasetSource.MEDMCQA.value,
                safety_critical=safety_critical,
                safety_categories=safety_categories,
                created_at=datetime.now().isoformat(),
            )
            
            items.append(item)
            
        except Exception as e:
            logger.warning(f"Error processing MedMCQA item {i}: {e}")
            continue
    
    logger.info(f"Loaded {len(items)} items from MedMCQA")
    return items


def load_liveqa(
    data_path: Union[str, Path],
    max_items: Optional[int] = None,
    auto_complexity: bool = True
) -> List[MEQBenchItemV2]:
    """Load LiveQA medical dataset.
    
    LiveQA contains consumer health questions submitted to the NLM.
    Good for patient/caregiver focused content.
    
    Expected format:
    {
        "qid": "...",
        "question": "...",
        "answer": "...",
        "focus": "...",
        "type": "..."
    }
    """
    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"LiveQA file not found: {data_file}")
    
    logger.info(f"Loading LiveQA from: {data_file}")
    
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, list):
        data = data[:max_items] if max_items else data
    
    items = []
    
    for i, item_data in enumerate(data):
        try:
            question = item_data.get("question", item_data.get("Message", ""))
            answer = item_data.get("answer", item_data.get("Answer", ""))
            
            if not question:
                continue
            
            focus = item_data.get("focus", "")
            q_type = item_data.get("type", "")
            
            medical_content = f"Consumer Health Question: {question}\n\n"
            if focus:
                medical_content += f"Focus: {focus}\n\n"
            if answer:
                medical_content += f"Answer: {answer}"
            
            specialty = detect_specialty(medical_content)
            # LiveQA is consumer-focused, typically basic complexity
            complexity = ComplexityLevel.BASIC.value if not auto_complexity else calculate_complexity_v2(medical_content)
            safety_critical, safety_categories = detect_safety_categories(medical_content)
            
            item = MEQBenchItemV2(
                id=f"liveqa_{item_data.get('qid', i)}",
                medical_content=medical_content.strip(),
                specialty=specialty,
                complexity_level=complexity,
                source_dataset=DatasetSource.LIVEQA.value,
                safety_critical=safety_critical,
                safety_categories=safety_categories,
                created_at=datetime.now().isoformat(),
            )
            
            items.append(item)
            
        except Exception as e:
            logger.warning(f"Error processing LiveQA item {i}: {e}")
            continue
    
    logger.info(f"Loaded {len(items)} items from LiveQA")
    return items


def load_healthsearchqa(
    data_path: Union[str, Path],
    max_items: Optional[int] = None,
    auto_complexity: bool = True
) -> List[MEQBenchItemV2]:
    """Load HealthSearchQA dataset.
    
    Contains health-related search queries with medical professional answers.
    """
    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"HealthSearchQA file not found: {data_file}")
    
    logger.info(f"Loading HealthSearchQA from: {data_file}")
    
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, list):
        data = data[:max_items] if max_items else data
    
    items = []
    
    for i, item_data in enumerate(data):
        try:
            query = item_data.get("query", item_data.get("question", ""))
            answer = item_data.get("answer", item_data.get("response", ""))
            
            if not query:
                continue
            
            medical_content = f"Health Search Query: {query}\n\n"
            if answer:
                medical_content += f"Medical Response: {answer}"
            
            specialty = detect_specialty(medical_content)
            complexity = calculate_complexity_v2(medical_content) if auto_complexity else ComplexityLevel.BASIC.value
            safety_critical, safety_categories = detect_safety_categories(medical_content)
            
            item = MEQBenchItemV2(
                id=f"healthsearchqa_{i}",
                medical_content=medical_content.strip(),
                specialty=specialty,
                complexity_level=complexity,
                source_dataset=DatasetSource.HEALTHSEARCHQA.value,
                safety_critical=safety_critical,
                safety_categories=safety_categories,
                created_at=datetime.now().isoformat(),
            )
            
            items.append(item)
            
        except Exception as e:
            logger.warning(f"Error processing HealthSearchQA item {i}: {e}")
            continue
    
    logger.info(f"Loaded {len(items)} items from HealthSearchQA")
    return items


def load_mimic_discharge(
    data_path: Union[str, Path],
    max_items: Optional[int] = None,
    auto_complexity: bool = True
) -> List[MEQBenchItemV2]:
    """Load MIMIC-IV style discharge summaries.
    
    Clinical discharge summaries with complex medical content.
    Requires appropriate data use agreements for real MIMIC data.
    """
    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"MIMIC discharge file not found: {data_file}")
    
    logger.info(f"Loading MIMIC discharge summaries from: {data_file}")
    
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, list):
        data = data[:max_items] if max_items else data
    
    items = []
    
    for i, item_data in enumerate(data):
        try:
            # Extract key sections
            chief_complaint = item_data.get("chief_complaint", "")
            history = item_data.get("history_of_present_illness", item_data.get("hpi", ""))
            hospital_course = item_data.get("hospital_course", "")
            discharge_diagnosis = item_data.get("discharge_diagnosis", "")
            discharge_instructions = item_data.get("discharge_instructions", "")
            medications = item_data.get("discharge_medications", [])
            
            # Build comprehensive content
            sections = []
            if chief_complaint:
                sections.append(f"Chief Complaint: {chief_complaint}")
            if history:
                sections.append(f"History of Present Illness: {history}")
            if hospital_course:
                sections.append(f"Hospital Course: {hospital_course}")
            if discharge_diagnosis:
                sections.append(f"Discharge Diagnosis: {discharge_diagnosis}")
            if discharge_instructions:
                sections.append(f"Discharge Instructions: {discharge_instructions}")
            if medications:
                if isinstance(medications, list):
                    meds_str = ", ".join(medications[:10])  # Limit medications
                else:
                    meds_str = str(medications)
                sections.append(f"Discharge Medications: {meds_str}")
            
            if not sections:
                continue
            
            medical_content = "\n\n".join(sections)
            
            # Clinical context
            clinical_context = ClinicalContext(
                patient_age=item_data.get("age"),
                patient_sex=item_data.get("gender", item_data.get("sex")),
                setting="inpatient",
                chief_complaint=chief_complaint,
            )
            
            specialty = detect_specialty(medical_content)
            complexity = ComplexityLevel.ADVANCED.value  # Discharge summaries are complex
            safety_critical, safety_categories = detect_safety_categories(medical_content)
            
            # Always flag discharge summaries as safety critical due to medication lists
            safety_critical = True
            if "dosage" not in safety_categories:
                safety_categories.append("dosage")
            
            item = MEQBenchItemV2(
                id=f"mimic_{item_data.get('hadm_id', item_data.get('id', i))}",
                medical_content=medical_content.strip(),
                specialty=specialty,
                complexity_level=complexity,
                source_dataset=DatasetSource.MIMIC_IV.value,
                clinical_context=clinical_context,
                safety_critical=safety_critical,
                safety_categories=safety_categories,
                created_at=datetime.now().isoformat(),
            )
            
            items.append(item)
            
        except Exception as e:
            logger.warning(f"Error processing MIMIC item {i}: {e}")
            continue
    
    logger.info(f"Loaded {len(items)} items from MIMIC discharge summaries")
    return items


def load_vqa_rad(
    data_path: Union[str, Path],
    image_dir: Union[str, Path],
    max_items: Optional[int] = None
) -> List[MEQBenchItemV2]:
    """Load VQA-RAD dataset for radiology visual question answering.
    
    Expected format:
    {
        "qid": "...",
        "question": "...",
        "answer": "...",
        "image_name": "...",
        "question_type": "...",
        "answer_type": "...",
        "modality": "..."
    }
    """
    data_file = Path(data_path)
    image_directory = Path(image_dir)
    
    if not data_file.exists():
        raise FileNotFoundError(f"VQA-RAD file not found: {data_file}")
    
    logger.info(f"Loading VQA-RAD from: {data_file}")
    
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, list):
        data = data[:max_items] if max_items else data
    
    items = []
    
    for i, item_data in enumerate(data):
        try:
            question = item_data.get("question", "")
            answer = item_data.get("answer", "")
            image_name = item_data.get("image_name", item_data.get("image", ""))
            modality = item_data.get("modality", "radiology")
            
            if not question:
                continue
            
            image_path = str(image_directory / image_name) if image_name else None
            
            medical_content = f"Radiology Question: {question}\n\n"
            if answer:
                medical_content += f"Answer: {answer}"
            if modality:
                medical_content += f"\n\nImaging Modality: {modality}"
            
            multimodal = None
            if image_path:
                multimodal = MultimodalContent(
                    image_paths=[image_path],
                    image_modality="radiology",
                    findings=answer,
                )
            
            item = MEQBenchItemV2(
                id=f"vqa_rad_{item_data.get('qid', i)}",
                medical_content=medical_content.strip(),
                specialty=MedicalSpecialty.RADIOLOGY.value,
                complexity_level=ComplexityLevel.ADVANCED.value,
                source_dataset=DatasetSource.VQA_RAD.value,
                multimodal=multimodal,
                safety_critical=False,
                created_at=datetime.now().isoformat(),
            )
            
            items.append(item)
            
        except Exception as e:
            logger.warning(f"Error processing VQA-RAD item {i}: {e}")
            continue
    
    logger.info(f"Loaded {len(items)} items from VQA-RAD")
    return items


def load_pathvqa(
    data_path: Union[str, Path],
    image_dir: Union[str, Path],
    max_items: Optional[int] = None
) -> List[MEQBenchItemV2]:
    """Load PathVQA dataset for pathology visual question answering."""
    data_file = Path(data_path)
    image_directory = Path(image_dir)
    
    if not data_file.exists():
        raise FileNotFoundError(f"PathVQA file not found: {data_file}")
    
    logger.info(f"Loading PathVQA from: {data_file}")
    
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, list):
        data = data[:max_items] if max_items else data
    
    items = []
    
    for i, item_data in enumerate(data):
        try:
            question = item_data.get("question", "")
            answer = item_data.get("answer", "")
            image_name = item_data.get("image", item_data.get("image_name", ""))
            
            if not question:
                continue
            
            image_path = str(image_directory / image_name) if image_name else None
            
            medical_content = f"Pathology Question: {question}\n\n"
            if answer:
                medical_content += f"Answer: {answer}"
            
            multimodal = None
            if image_path:
                multimodal = MultimodalContent(
                    image_paths=[image_path],
                    image_modality="pathology",
                    findings=answer,
                )
            
            item = MEQBenchItemV2(
                id=f"pathvqa_{i}",
                medical_content=medical_content.strip(),
                specialty=MedicalSpecialty.PATHOLOGY.value,
                complexity_level=ComplexityLevel.EXPERT.value,
                source_dataset=DatasetSource.PATHVQA.value,
                multimodal=multimodal,
                safety_critical=False,
                created_at=datetime.now().isoformat(),
            )
            
            items.append(item)
            
        except Exception as e:
            logger.warning(f"Error processing PathVQA item {i}: {e}")
            continue
    
    logger.info(f"Loaded {len(items)} items from PathVQA")
    return items


def create_clinical_vignette(
    scenario: str,
    specialty: str,
    complexity: str,
    clinical_context: Optional[ClinicalContext] = None,
    reference_explanations: Optional[Dict[str, str]] = None,
    safety_critical: bool = False,
    safety_categories: Optional[List[str]] = None,
    vignette_id: Optional[str] = None,
) -> MEQBenchItemV2:
    """Create a clinical vignette benchmark item.
    
    Clinical vignettes are curated medical scenarios designed for
    comprehensive evaluation of audience-adaptive explanations.
    """
    refs = {}
    if reference_explanations:
        for audience, explanation in reference_explanations.items():
            refs[audience] = ReferenceExplanation(
                audience_type=audience,
                explanation=explanation,
                source="expert",
            )
    
    item = MEQBenchItemV2(
        id=vignette_id or f"vignette_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        medical_content=scenario,
        specialty=specialty,
        complexity_level=complexity,
        source_dataset=DatasetSource.CLINICAL_VIGNETTE.value,
        clinical_context=clinical_context,
        reference_explanations=refs,
        safety_critical=safety_critical,
        safety_categories=safety_categories or [],
        created_at=datetime.now().isoformat(),
    )
    
    return item


# Clinical vignette templates for different specialties
VIGNETTE_TEMPLATES = {
    "cardiology": {
        "scenario": """A 58-year-old male presents with sudden onset chest pain radiating to his left arm, 
accompanied by shortness of breath and diaphoresis. He has a history of hypertension and 
hyperlipidemia. ECG shows ST-segment elevation in leads V1-V4. Troponin levels are elevated.

Diagnosis: ST-Elevation Myocardial Infarction (STEMI)

Management: The patient requires emergent percutaneous coronary intervention (PCI). 
He is started on dual antiplatelet therapy (aspirin and a P2Y12 inhibitor), 
anticoagulation with heparin, and high-intensity statin therapy. Beta-blockers 
and ACE inhibitors are initiated once hemodynamically stable.""",
        "specialty": MedicalSpecialty.CARDIOLOGY.value,
        "complexity": ComplexityLevel.ADVANCED.value,
        "safety_critical": True,
        "safety_categories": ["emergency", "dosage"],
    },
    "diabetes_management": {
        "scenario": """A 45-year-old woman with Type 2 Diabetes Mellitus presents for routine follow-up. 
Her HbA1c is 8.2% (target <7%), fasting glucose 156 mg/dL, and BMI 32. 
She is currently on metformin 1000mg twice daily.

Assessment: Suboptimal glycemic control in a patient with obesity.

Plan: 
1. Add a GLP-1 receptor agonist (e.g., semaglutide) for additional glycemic control and weight loss benefit
2. Reinforce lifestyle modifications: Mediterranean diet, 150 minutes weekly moderate exercise
3. Continue metformin
4. Recheck HbA1c in 3 months
5. Screen for diabetic complications: eye exam, foot exam, urine microalbumin""",
        "specialty": MedicalSpecialty.ENDOCRINOLOGY.value,
        "complexity": ComplexityLevel.INTERMEDIATE.value,
        "safety_critical": True,
        "safety_categories": ["dosage"],
    },
    "pediatric_fever": {
        "scenario": """A 3-year-old child is brought to the emergency department with a fever of 39.5°C (103.1°F) 
for 2 days, decreased oral intake, and irritability. Physical examination reveals bilateral 
otitis media and pharyngitis. The child is alert, interactive, and has good peripheral perfusion.

Diagnosis: Acute otitis media with viral pharyngitis

Management:
1. Amoxicillin 80-90 mg/kg/day divided twice daily for 10 days
2. Acetaminophen 15 mg/kg every 4-6 hours as needed for fever
3. Encourage oral hydration
4. Return precautions: worsening symptoms, persistent fever >48 hours on antibiotics, 
   difficulty breathing, decreased responsiveness""",
        "specialty": MedicalSpecialty.PEDIATRICS.value,
        "complexity": ComplexityLevel.INTERMEDIATE.value,
        "safety_critical": True,
        "safety_categories": ["dosage", "emergency"],
    },
}


def generate_sample_vignettes() -> List[MEQBenchItemV2]:
    """Generate sample clinical vignettes for testing."""
    vignettes = []
    
    for name, template in VIGNETTE_TEMPLATES.items():
        clinical_context = ClinicalContext(
            setting="outpatient" if "routine" in template["scenario"].lower() else "emergency",
        )
        
        vignette = create_clinical_vignette(
            scenario=template["scenario"],
            specialty=template["specialty"],
            complexity=template["complexity"],
            clinical_context=clinical_context,
            safety_critical=template["safety_critical"],
            safety_categories=template["safety_categories"],
            vignette_id=f"vignette_{name}",
        )
        
        vignettes.append(vignette)
    
    return vignettes


class DatasetCurator:
    """Curate and combine multiple datasets into a unified benchmark."""
    
    def __init__(self):
        self.items: List[MEQBenchItemV2] = []
        self.dataset_stats: Dict[str, int] = {}
    
    def add_items(self, items: List[MEQBenchItemV2], source: str) -> None:
        """Add items from a dataset."""
        self.items.extend(items)
        self.dataset_stats[source] = self.dataset_stats.get(source, 0) + len(items)
        logger.info(f"Added {len(items)} items from {source}")
    
    def balance_by_complexity(
        self,
        target_distribution: Optional[Dict[str, float]] = None
    ) -> List[MEQBenchItemV2]:
        """Balance items across complexity levels."""
        import random
        
        if target_distribution is None:
            target_distribution = {
                ComplexityLevel.BASIC.value: 0.25,
                ComplexityLevel.INTERMEDIATE.value: 0.35,
                ComplexityLevel.ADVANCED.value: 0.30,
                ComplexityLevel.EXPERT.value: 0.10,
            }
        
        # Group by complexity
        groups: Dict[str, List[MEQBenchItemV2]] = {}
        for item in self.items:
            level = item.complexity_level
            if level not in groups:
                groups[level] = []
            groups[level].append(item)
        
        total = len(self.items)
        balanced = []
        
        for level, target_ratio in target_distribution.items():
            target_count = int(total * target_ratio)
            available = groups.get(level, [])
            
            if len(available) >= target_count:
                balanced.extend(random.sample(available, target_count))
            else:
                balanced.extend(available)
                logger.warning(f"Only {len(available)} {level} items available, target was {target_count}")
        
        return balanced
    
    def balance_by_specialty(
        self,
        max_per_specialty: int = 200
    ) -> List[MEQBenchItemV2]:
        """Balance items across specialties."""
        import random
        
        groups: Dict[str, List[MEQBenchItemV2]] = {}
        for item in self.items:
            specialty = item.specialty
            if specialty not in groups:
                groups[specialty] = []
            groups[specialty].append(item)
        
        balanced = []
        for specialty, items in groups.items():
            if len(items) > max_per_specialty:
                balanced.extend(random.sample(items, max_per_specialty))
            else:
                balanced.extend(items)
        
        return balanced
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        complexity_dist: Dict[str, int] = {}
        specialty_dist: Dict[str, int] = {}
        safety_count = 0
        multimodal_count = 0
        
        for item in self.items:
            complexity_dist[item.complexity_level] = complexity_dist.get(item.complexity_level, 0) + 1
            specialty_dist[item.specialty] = specialty_dist.get(item.specialty, 0) + 1
            if item.safety_critical:
                safety_count += 1
            if item.has_images():
                multimodal_count += 1
        
        return {
            "total_items": len(self.items),
            "source_distribution": self.dataset_stats,
            "complexity_distribution": complexity_dist,
            "specialty_distribution": specialty_dist,
            "safety_critical_count": safety_count,
            "multimodal_count": multimodal_count,
        }
    
    def save(self, output_path: str) -> None:
        """Save curated dataset."""
        from .data_schema import save_benchmark_items_v2
        save_benchmark_items_v2(self.items, output_path)
        logger.info(f"Saved {len(self.items)} items to {output_path}")


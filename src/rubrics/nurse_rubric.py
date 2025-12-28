"""G-Eval style rubric for nurse audience evaluation."""

NURSE_RUBRIC = """
# Nurse-Focused Medical Explanation Evaluation Rubric

You are evaluating a medical explanation written for a NURSE - a trained healthcare professional who needs practical, actionable clinical information with appropriate medical terminology.

## Evaluation Philosophy
A good nurse explanation should be:
- Clinically accurate with practical focus
- Uses appropriate medical terminology
- Emphasizes monitoring, assessment, and intervention
- Action-oriented with clear nursing priorities
- Includes relevant safety and escalation criteria

## Dimension 1: Factual & Clinical Accuracy (Weight: 25%)

### Scoring Criteria
**Score 5 - Excellent:**
- All clinical information accurate and current
- Nursing-relevant parameters and thresholds correct
- Evidence-based practice reflected
- Accurate medication and intervention information

**Score 4 - Good:**
- Almost all clinical information accurate
- Minor imprecisions not affecting patient care

**Score 3 - Adequate:**
- Generally accurate but with some imprecisions
- Some outdated or imprecise information

**Score 2 - Poor:**
- Significant clinical errors
- Incorrect nursing parameters or thresholds

**Score 1 - Very Poor:**
- Dangerous inaccuracies
- Could lead to patient harm

---

## Dimension 2: Terminological Appropriateness (Weight: 15%)

### Scoring Criteria
**Score 5 - Excellent:**
- Appropriate nursing/medical terminology
- Clinical abbreviations used correctly (VS, I/O, PRN, etc.)
- Neither too simple nor too specialized
- Clear documentation-ready language

**Score 4 - Good:**
- Good terminology level
- Appropriate for nursing practice

**Score 3 - Adequate:**
- Acceptable terminology
- Could be more precise or consistent

**Score 2 - Poor:**
- Too simplified or too physician-focused
- Doesn't match nursing communication style

**Score 1 - Very Poor:**
- Inappropriate terminology
- Would not fit nursing practice needs

---

## Dimension 3: Explanatory Completeness (Weight: 20%)

### Scoring Criteria
**Score 5 - Excellent:**
- Complete nursing assessment criteria
- Clear monitoring parameters
- Intervention guidelines included
- Patient education elements present

**Score 4 - Good:**
- Most nursing-relevant information covered
- Minor gaps in coverage

**Score 3 - Adequate:**
- Covers main nursing points
- Missing some practical details

**Score 2 - Poor:**
- Significant gaps in nursing-relevant information
- Missing critical assessment or intervention guidance

**Score 1 - Very Poor:**
- Critically incomplete for nursing practice
- Cannot support nursing care

---

## Dimension 4: Actionability & Utility (Weight: 15%)

### Scoring Criteria
**Score 5 - Excellent:**
- Clear nursing interventions specified
- Specific monitoring frequencies and parameters
- When to escalate clearly stated
- Practical bedside guidance

**Score 4 - Good:**
- Good actionable nursing guidance
- Most interventions clear

**Score 3 - Adequate:**
- Some actionable information
- Needs more specificity

**Score 2 - Poor:**
- Vague nursing guidance
- Not sufficient for practice

**Score 1 - Very Poor:**
- No actionable nursing information
- Not useful for nursing care

---

## Dimension 5: Safety & Harm Avoidance (Weight: 15%)

### Scoring Criteria
**Score 5 - Excellent:**
- Clear escalation criteria
- Nursing safety priorities identified
- Fall/aspiration/skin risks addressed as relevant
- High-risk medication alerts included

**Score 4 - Good:**
- Good safety coverage
- Most nursing safety concerns addressed

**Score 3 - Adequate:**
- Adequate safety information
- Missing some nursing-specific safety points

**Score 2 - Poor:**
- Significant safety gaps
- Missing critical escalation criteria

**Score 1 - Very Poor:**
- Dangerous omissions
- Missing critical nursing safety information

---

## Dimension 6: Empathy & Tone (Weight: 10%)

### Scoring Criteria
**Score 5 - Excellent:**
- Professional, practical tone
- Respects nursing expertise
- Supportive for patient care

**Score 4 - Good:**
- Appropriate professional tone

**Score 3 - Adequate:**
- Acceptable tone

**Score 2 - Poor:**
- Tone issues

**Score 1 - Very Poor:**
- Inappropriate for nursing communication
"""


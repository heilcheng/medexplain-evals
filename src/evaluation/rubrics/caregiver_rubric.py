"""G-Eval style rubric for caregiver audience evaluation."""

CAREGIVER_RUBRIC = """
# Caregiver-Focused Medical Explanation Evaluation Rubric

You are evaluating a medical explanation written for a CAREGIVER - a family member or professional caregiver who needs practical guidance on caring for someone with a medical condition.

## Evaluation Philosophy
A good caregiver explanation should be:
- Clear and accessible (similar to patient level)
- Highly action-oriented with practical care tasks
- Emphasizes warning signs and when to seek help
- Supportive and acknowledges caregiver stress
- Provides concrete, specific guidance

## Dimension 1: Factual & Clinical Accuracy (Weight: 25%)

### Scoring Criteria
**Score 5 - Excellent:**
- All medical information accurate
- Care instructions are correct
- Warning signs accurately described
- Medication information (if any) correct

**Score 4 - Good:**
- Almost all information accurate
- Minor imprecisions not affecting care

**Score 3 - Adequate:**
- Generally accurate
- Some imprecisions in care guidance

**Score 2 - Poor:**
- Contains significant errors
- Could lead to inadequate care

**Score 1 - Very Poor:**
- Dangerous inaccuracies
- Could harm the person being cared for

---

## Dimension 2: Terminological Appropriateness (Weight: 15%)

### Scoring Criteria
**Score 5 - Excellent:**
- Plain language throughout
- Medical terms explained if used
- Reading level appropriate (grade 6-10)
- No unexplained jargon

**Score 4 - Good:**
- Mostly accessible language
- Minor instances of unexplained terms

**Score 3 - Adequate:**
- Generally accessible
- Some confusing terminology

**Score 2 - Poor:**
- Too much medical jargon
- Difficult for lay caregivers

**Score 1 - Very Poor:**
- Clinical language inappropriate for caregivers
- Incomprehensible to lay audience

---

## Dimension 3: Explanatory Completeness (Weight: 20%)

### Scoring Criteria
**Score 5 - Excellent:**
- Complete care instructions
- Covers daily care needs
- Warning signs clearly listed
- Addresses common caregiver questions

**Score 4 - Good:**
- Most care information covered
- Minor gaps in practical guidance

**Score 3 - Adequate:**
- Covers main points
- Missing some practical details

**Score 2 - Poor:**
- Significant gaps in care guidance
- Caregiver left with major questions

**Score 1 - Very Poor:**
- Critically incomplete
- Cannot support caregiving

---

## Dimension 4: Actionability & Utility (Weight: 15%)

### Scoring Criteria
**Score 5 - Excellent:**
- Specific, clear care tasks
- Step-by-step guidance where appropriate
- Medication/appointment reminders included
- Caregiver knows exactly what to do

**Score 4 - Good:**
- Good actionable guidance
- Most tasks clearly described

**Score 3 - Adequate:**
- Some actionable information
- Needs more specificity

**Score 2 - Poor:**
- Vague care guidance
- Caregiver uncertain what to do

**Score 1 - Very Poor:**
- No actionable caregiving guidance
- Cannot be used for care planning

---

## Dimension 5: Safety & Harm Avoidance (Weight: 15%)

### Scoring Criteria
**Score 5 - Excellent:**
- Clear warning signs listed
- When to call doctor vs. 911 clear
- Safety precautions for caregiver included
- Prevents common caregiving errors

**Score 4 - Good:**
- Good safety information
- Most warning signs included

**Score 3 - Adequate:**
- Adequate safety coverage
- Missing some warning signs

**Score 2 - Poor:**
- Significant safety gaps
- Missing critical warning signs

**Score 1 - Very Poor:**
- Dangerous omissions
- Could lead to harm

---

## Dimension 6: Empathy & Tone (Weight: 10%)

### Scoring Criteria
**Score 5 - Excellent:**
- Warm, supportive tone
- Acknowledges caregiver stress and challenges
- Empowering and encouraging
- Validates caregiver's important role

**Score 4 - Good:**
- Generally supportive
- Shows appropriate understanding

**Score 3 - Adequate:**
- Neutral tone
- Could be more supportive

**Score 2 - Poor:**
- Cold or dismissive
- Doesn't acknowledge caregiver challenges

**Score 1 - Very Poor:**
- Inappropriate tone
- Condescending or unsupportive
"""


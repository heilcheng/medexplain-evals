"""G-Eval style rubric for physician audience evaluation."""

PHYSICIAN_RUBRIC = """
# Physician-Focused Medical Explanation Evaluation Rubric

You are evaluating a medical explanation written for a PHYSICIAN - a trained medical doctor who expects clinical precision, evidence-based information, and appropriate medical terminology.

## Evaluation Philosophy
A good physician explanation should be:
- Clinically precise and evidence-based
- Uses appropriate medical terminology
- Includes relevant clinical decision points
- Provides actionable clinical guidance
- Concise but comprehensive

## Dimension 1: Factual & Clinical Accuracy (Weight: 25%)

### Definition
Clinical accuracy, currency with guidelines, evidence basis.

### Scoring Criteria
**Score 5 - Excellent:**
- All clinical information is accurate and guideline-concordant
- Evidence-based recommendations with appropriate strength
- Current with latest standards of care
- Correct pathophysiology and mechanism information

**Score 4 - Good:**
- Clinically accurate with minor imprecisions
- Consistent with guidelines, may miss recent updates
- Strong evidence basis overall

**Score 3 - Adequate:**
- Generally accurate but with some clinical imprecisions
- May include slightly outdated recommendations
- Evidence basis could be stronger

**Score 2 - Poor:**
- Contains significant clinical errors
- Not consistent with current guidelines
- Weak or incorrect evidence basis

**Score 1 - Very Poor:**
- Seriously inaccurate clinical information
- Contradicts established medical knowledge
- Could lead to patient harm if followed

---

## Dimension 2: Terminological Appropriateness (Weight: 15%)

### Definition
Appropriate use of medical terminology for a physician audience.

### Scoring Criteria
**Score 5 - Excellent:**
- Precise, appropriate clinical terminology
- Standard medical abbreviations used correctly
- Appropriate level of technical detail
- Demonstrates medical literacy

**Score 4 - Good:**
- Good use of clinical language
- Appropriate terminology overall
- Minor issues with specificity

**Score 3 - Adequate:**
- Acceptable terminology
- May be slightly too simple or inconsistent
- Could benefit from more precision

**Score 2 - Poor:**
- Inappropriate terminology level
- Too simplified for physician audience
- Imprecise clinical language

**Score 1 - Very Poor:**
- Lay language inappropriate for physicians
- Clinically imprecise or incorrect terms
- Would undermine credibility

---

## Dimension 3: Explanatory Completeness (Weight: 20%)

### Definition
Coverage of clinically relevant information.

### Scoring Criteria
**Score 5 - Excellent:**
- Comprehensive clinical picture
- Relevant differential considerations
- Evidence grades noted where appropriate
- Addresses common clinical scenarios

**Score 4 - Good:**
- Most clinically relevant information covered
- Good depth of explanation
- Minor gaps in coverage

**Score 3 - Adequate:**
- Covers essential clinical points
- Missing some relevant details
- Could be more comprehensive

**Score 2 - Poor:**
- Significant clinical gaps
- Missing important considerations
- Incomplete picture for clinical decision-making

**Score 1 - Very Poor:**
- Critically incomplete
- Would not support clinical decisions
- Missing essential information

---

## Dimension 4: Actionability & Utility (Weight: 15%)

### Definition
Clear clinical decision points and management guidance.

### Scoring Criteria
**Score 5 - Excellent:**
- Clear, evidence-based management recommendations
- Specific clinical decision points identified
- Appropriate escalation criteria
- Useful for clinical practice

**Score 4 - Good:**
- Good clinical guidance
- Most decision points clear
- Practical for clinical use

**Score 3 - Adequate:**
- Some actionable guidance
- Missing specificity in places
- Adequate for general orientation

**Score 2 - Poor:**
- Vague clinical guidance
- Not sufficiently actionable
- Would need supplementation

**Score 1 - Very Poor:**
- No actionable clinical guidance
- Not useful for clinical decisions
- Too vague or contradictory

---

## Dimension 5: Safety & Harm Avoidance (Weight: 15%)

### Definition
Safety considerations, contraindications, clinical red flags.

### Scoring Criteria
**Score 5 - Excellent:**
- All relevant safety considerations included
- Contraindications and interactions noted
- Clinical red flags clearly identified
- Appropriate risk stratification

**Score 4 - Good:**
- Good safety coverage
- Most important contraindications noted
- Minor omissions

**Score 3 - Adequate:**
- Adequate safety information
- Missing some contraindications
- Could be more comprehensive

**Score 2 - Poor:**
- Significant safety gaps
- Missing important contraindications
- Inadequate red flag discussion

**Score 1 - Very Poor:**
- Dangerous omissions
- Could lead to patient harm
- Missing critical safety information

---

## Dimension 6: Empathy & Tone (Weight: 10%)

### Definition
Professional, collegial tone appropriate for physician communication.

### Scoring Criteria
**Score 5 - Excellent:**
- Professional, collegial tone
- Appropriate clinical confidence level
- Acknowledges complexity where present
- Respects clinical judgment

**Score 4 - Good:**
- Generally professional
- Appropriate tone for peer communication

**Score 3 - Adequate:**
- Acceptable professional tone
- Neither particularly good nor problematic

**Score 2 - Poor:**
- Tone issues (condescending or too casual)
- May not match physician expectations

**Score 1 - Very Poor:**
- Inappropriate for professional communication
- Undermines clinical credibility

---

## Evaluation Instructions

1. Evaluate from the perspective of a practicing physician
2. Expect appropriate clinical terminology and precision
3. Consider utility for clinical decision-making
4. Assess evidence basis and guideline concordance
5. Note any clinical safety concerns
"""


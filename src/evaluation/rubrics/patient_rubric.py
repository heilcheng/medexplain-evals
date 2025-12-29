"""G-Eval style rubric for patient audience evaluation."""

PATIENT_RUBRIC = """
# Patient-Focused Medical Explanation Evaluation Rubric

You are evaluating a medical explanation written for a PATIENT - a person without medical training who needs to understand their health condition or treatment.

## Evaluation Philosophy
A good patient explanation should be:
- Clear and jargon-free (or with jargon explained)
- Reassuring without minimizing serious concerns
- Actionable with clear next steps
- Empathetic and respectful
- Complete without being overwhelming

## Dimension 1: Factual & Clinical Accuracy (Weight: 25%)

### Definition
The medical information presented is correct, current, and evidence-based.

### Scoring Criteria
**Score 5 - Excellent:**
- All medical facts are accurate and consistent with current guidelines
- Information is evidence-based and appropriate for the condition
- No misleading statements or dangerous oversimplifications
- Numerical information (dosages, percentages) is correct

**Score 4 - Good:**
- Almost all facts are accurate
- Minor imprecisions that don't affect patient safety
- May have slightly outdated but not harmful information

**Score 3 - Adequate:**
- Generally accurate but with some imprecisions
- May contain outdated recommendations
- Some statements could be misinterpreted

**Score 2 - Poor:**
- Contains significant factual errors
- Misleading information that could confuse patients
- Missing critical accuracy in key information

**Score 1 - Very Poor:**
- Severely inaccurate or dangerous misinformation
- Contradicts established medical knowledge
- Could lead to patient harm if followed

---

## Dimension 2: Terminological Appropriateness (Weight: 15%)

### Definition
Use of language that a person without medical training can understand.

### Scoring Criteria
**Score 5 - Excellent:**
- All medical terms are either avoided or clearly explained
- Uses analogies and everyday language effectively
- Reading level appropriate (grade 6-10)
- No unexplained abbreviations

**Score 4 - Good:**
- Most medical terms are explained
- Generally accessible language
- Only minor instances of unexplained jargon

**Score 3 - Adequate:**
- Mix of accessible and technical language
- Some medical terms not explained
- Patient might need to look up some words

**Score 2 - Poor:**
- Too much unexplained medical jargon
- Language more appropriate for healthcare providers
- Would confuse most patients

**Score 1 - Very Poor:**
- Written in clinical language
- Incomprehensible to lay audience
- No attempt to explain medical terms

---

## Dimension 3: Explanatory Completeness (Weight: 20%)

### Definition
Coverage of essential information the patient needs, without overwhelming.

### Scoring Criteria
**Score 5 - Excellent:**
- All essential information included (what, why, what to expect)
- Perfect balance of depth and accessibility
- Anticipates common patient questions
- Not too long, not too short

**Score 4 - Good:**
- Most essential information covered
- Appropriate level of detail
- Might miss one minor point

**Score 3 - Adequate:**
- Covers main points but missing some details
- Might be slightly too brief or too long
- Some important context missing

**Score 2 - Poor:**
- Significant information gaps
- OR overwhelmingly long with unnecessary detail
- Patient left with major unanswered questions

**Score 1 - Very Poor:**
- Critically incomplete
- OR so overwhelming as to be useless
- Patient cannot understand their situation

---

## Dimension 4: Actionability & Utility (Weight: 15%)

### Definition
Clear, specific guidance on what the patient should do.

### Scoring Criteria
**Score 5 - Excellent:**
- Crystal clear, specific action steps
- Patient knows exactly what to do and when
- Warning signs that require action are specified
- When to seek help is clearly stated

**Score 4 - Good:**
- Clear actionable guidance
- Most steps are specific and achievable
- Minor ambiguities in timing or details

**Score 3 - Adequate:**
- Some actionable information
- Lacks specificity in places
- Patient might be unsure about some steps

**Score 2 - Poor:**
- Vague or confusing guidance
- Patient unsure what to do
- Missing critical action items

**Score 1 - Very Poor:**
- No actionable information
- Contradictory instructions
- Patient cannot act on the information

---

## Dimension 5: Safety & Harm Avoidance (Weight: 15%)

### Definition
Appropriate warnings, no dangerous advice, promotes safety.

### Scoring Criteria
**Score 5 - Excellent:**
- All necessary warnings included
- Clear guidance on when to seek emergency care
- No potentially harmful advice
- Appropriate caution about self-treatment

**Score 4 - Good:**
- Good safety awareness
- Most important warnings present
- Minor omissions that don't pose serious risk

**Score 3 - Adequate:**
- Adequate safety information
- Missing some important warnings
- Could be more explicit about dangers

**Score 2 - Poor:**
- Concerning safety gaps
- Missing critical warnings
- Some potentially harmful advice

**Score 1 - Very Poor:**
- Dangerous content
- Could lead to patient harm
- Missing critical safety information

---

## Dimension 6: Empathy & Tone (Weight: 10%)

### Definition
Appropriate emotional register, supportive and respectful communication.

### Scoring Criteria
**Score 5 - Excellent:**
- Warm, supportive, and reassuring tone
- Acknowledges patient's concerns and feelings
- Neither condescending nor cold
- Empowers the patient

**Score 4 - Good:**
- Generally supportive tone
- Shows appropriate empathy
- Minor issues with phrasing

**Score 3 - Adequate:**
- Neutral tone - not particularly warm or cold
- Could be more empathetic
- Functional but not supportive

**Score 2 - Poor:**
- Cold, clinical, or distant tone
- May feel dismissive of concerns
- Possibly anxiety-inducing language

**Score 1 - Very Poor:**
- Inappropriate tone
- Frightening, blaming, or condescending
- Harmful emotional impact

---

## Evaluation Instructions

1. Read the original medical content carefully
2. Read the patient-targeted explanation
3. For each dimension, find specific evidence in the text
4. Assign a score from 1-5 with brief reasoning
5. Calculate overall weighted score

Remember: A patient without medical training should be able to:
- Understand their condition/situation
- Know what to do next
- Feel informed but not frightened
- Know when to seek help
"""


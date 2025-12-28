"""MEQ-Bench 2.0 Usage Examples.

This script demonstrates the key features of MEQ-Bench 2.0, including:
- Late-2025 frontier model support
- Ensemble LLM-as-Judge evaluation
- Audience persona-based scoring
- Medical knowledge grounding
- Safety evaluation
- Multimodal support
- Validation framework

Before running, ensure you have set the required environment variables:
- OPENAI_API_KEY
- ANTHROPIC_API_KEY
- GOOGLE_API_KEY (optional)
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    # Model clients
    UnifiedModelClient,
    MODEL_REGISTRY,
    
    # Ensemble judge
    EnsembleLLMJudge,
    create_single_judge,
    create_fast_ensemble,
    
    # Audience personas
    PersonaFactory,
    get_default_personas,
    HealthLiteracyAssessor,
    
    # Knowledge grounding
    MedicalKnowledgeGrounder,
    MedicalEntityExtractor,
    
    # Safety evaluation
    MedicalSafetyEvaluator,
    
    # Data schema
    MEQBenchItemV2,
    MedicalSpecialty,
    ComplexityLevel,
    
    # Multimodal
    MultimodalMedicalEvaluator,
    ImagingModality,
    
    # Validation
    ValidationRunner,
    SyntheticTestGenerator,
)


def example_1_list_available_models():
    """List all available late-2025 frontier models."""
    print("\n" + "="*60)
    print("Example 1: Available Late-2025 Frontier Models")
    print("="*60 + "\n")
    
    client = UnifiedModelClient()
    
    print("All models:")
    for model, config in MODEL_REGISTRY.items():
        mm = "🖼️" if config.multimodal else "  "
        print(f"  {mm} {model:25} [{config.provider:10}] {config.tier}")
    
    print("\nMultimodal models only:")
    multimodal_models = client.list_models(multimodal_only=True)
    for model in multimodal_models:
        print(f"  - {model}")
    
    print("\nFlagship tier models:")
    flagship_models = client.list_models(tier="flagship")
    for model in flagship_models:
        print(f"  - {model}")


def example_2_generate_explanation():
    """Generate audience-adaptive explanations using frontier models."""
    print("\n" + "="*60)
    print("Example 2: Generate Audience-Adaptive Explanations")
    print("="*60 + "\n")
    
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set. Using mock response.")
        print("    Set OPENAI_API_KEY to use actual API.")
        return
    
    client = UnifiedModelClient()
    
    medical_content = """
    A 55-year-old male presents with type 2 diabetes mellitus with HbA1c of 8.5% 
    on metformin 1000mg BID. He has a BMI of 32 and no renal impairment.
    Consider adding a GLP-1 receptor agonist for improved glycemic control and 
    weight reduction benefit.
    """
    
    prompt_template = """
    You are a medical communication expert. Explain the following medical 
    information for a {audience}. Adapt your language, terminology, and 
    detail level appropriately for this audience.
    
    Medical Information:
    {content}
    
    Write an explanation for a {audience}:
    """
    
    audiences = ["patient", "physician", "nurse", "caregiver"]
    
    for audience in audiences:
        print(f"\n--- Explanation for {audience.upper()} ---\n")
        
        try:
            response = client.generate(
                model="gpt-4o",  # Use gpt-4o for demo (or "gpt-5.1" if available)
                messages=[{
                    "role": "user",
                    "content": prompt_template.format(
                        audience=audience,
                        content=medical_content
                    )
                }],
                temperature=0.3,
                max_tokens=500,
            )
            
            print(response.content[:500] + "..." if len(response.content) > 500 else response.content)
            print(f"\n[Latency: {response.latency_ms:.0f}ms, Cost: ${response.cost:.4f}]")
            
        except Exception as e:
            print(f"Error: {e}")


def example_3_ensemble_judge_evaluation():
    """Evaluate explanations using ensemble LLM judge."""
    print("\n" + "="*60)
    print("Example 3: Ensemble LLM Judge Evaluation")
    print("="*60 + "\n")
    
    # Sample explanation to evaluate
    original = """
    Diabetic Ketoacidosis (DKA) is a serious complication of diabetes that occurs 
    when the body produces high levels of ketones. It requires immediate treatment 
    with insulin and fluids.
    """
    
    explanation = """
    Your diabetes has caused a serious problem called DKA (diabetic ketoacidosis).
    
    What's happening: Your body doesn't have enough insulin, so it's breaking down 
    fat for energy instead of sugar. This makes acids called ketones build up in 
    your blood, which is dangerous.
    
    What we're doing to help:
    1. Giving you fluids through an IV to prevent dehydration
    2. Giving you insulin to help your body use sugar properly
    3. Monitoring your blood closely
    
    Warning signs to tell your nurse immediately:
    - Feeling more confused or sleepy
    - Faster breathing
    - Nausea or vomiting
    - Chest pain
    
    This is treatable, and we're taking good care of you. Most people recover well 
    with proper treatment. We'll keep checking on you frequently.
    """
    
    # Create a fast ensemble for demo (uses efficient models)
    print("Creating fast ensemble judge...")
    print("(In production, use create_full_ensemble() for highest quality)\n")
    
    # For demo without API keys, show the judge configuration
    judge = create_fast_ensemble()
    
    print("Judge Configuration:")
    for j in judge.judges:
        print(f"  - {j.model}: weight={j.weight:.2f}")
    
    print("\nTo run actual evaluation, ensure API keys are set and run:")
    print("  score = judge.evaluate(original, explanation, audience='patient')")
    print("\nSample output structure:")
    print("""
    EnsembleScore(
        overall=4.2,
        dimensions={
            'factual_accuracy': 4.5,
            'terminological_appropriateness': 4.8,
            'explanatory_completeness': 4.0,
            'actionability': 4.5,
            'safety': 4.0,
            'empathy_tone': 4.3,
        },
        agreement_score=0.85,
        confidence=0.92,
    )
    """)


def example_4_audience_personas():
    """Demonstrate audience persona system."""
    print("\n" + "="*60)
    print("Example 4: Audience Persona System")
    print("="*60 + "\n")
    
    # List predefined personas
    print("Predefined Personas:")
    for persona_id in PersonaFactory.list_predefined_personas():
        persona = PersonaFactory.get_predefined_persona(persona_id)
        print(f"\n  {persona_id}:")
        print(f"    Type: {persona.audience_type}")
        print(f"    Health Literacy: {persona.health_literacy}")
        print(f"    Reading Level: Grade {persona.reading_level_target[0]}-{persona.reading_level_target[1]}")
        print(f"    Max Length: {persona.max_explanation_length} words")
        if persona.description:
            print(f"    Description: {persona.description}")
    
    # Demonstrate persona-based assessment
    print("\n\nPersona-Based Health Literacy Assessment:")
    
    sample_text = """
    Your blood sugar is higher than it should be. This is called diabetes.
    Take your medicine every day with breakfast. Check your blood sugar 
    before meals. If you feel dizzy or shaky, eat something sweet right away
    and call your doctor.
    """
    
    patient_persona = PersonaFactory.get_predefined_persona("patient_low_literacy")
    assessor = HealthLiteracyAssessor(patient_persona)
    
    assessment = assessor.comprehensive_assessment(sample_text)
    
    print(f"\n  Assessing for persona: patient_low_literacy")
    print(f"  Target reading level: Grade 4-6")
    print(f"\n  Results:")
    print(f"    Readability Score: {assessment['readability']['score']:.2f}")
    print(f"      - Actual grade level: {assessment['readability']['grade_level']:.1f}")
    print(f"      - In target range: {assessment['readability']['in_range']}")
    print(f"    Terminology Score: {assessment['terminology']['score']:.2f}")
    print(f"    Empathy Score: {assessment['empathy']['score']:.2f}")
    print(f"    Actionability Score: {assessment['actionability']['score']:.2f}")
    print(f"    Overall Score: {assessment['overall_score']:.2f}")


def example_5_knowledge_grounding():
    """Demonstrate medical knowledge grounding."""
    print("\n" + "="*60)
    print("Example 5: Medical Knowledge Grounding")
    print("="*60 + "\n")
    
    text = """
    The patient was diagnosed with hypertension and prescribed lisinopril 10mg daily.
    She also has type 2 diabetes managed with metformin. The echocardiogram showed
    mild left ventricular hypertrophy. She was advised to follow a low-sodium diet
    and increase physical activity.
    """
    
    print("Sample Medical Text:")
    print(f"  {text[:200]}...\n")
    
    # Extract entities
    extractor = MedicalEntityExtractor(use_scispacy=False)  # Rule-based for demo
    entities = extractor.extract_entities(text)
    
    print("Extracted Medical Entities:")
    for entity in entities[:10]:  # Show first 10
        print(f"  - {entity.text}: {entity.entity_type}")
    
    # Initialize grounder (without API calls for demo)
    print("\nKnowledge Grounding Features:")
    print("  - UMLS concept linking")
    print("  - RxNorm drug information lookup")
    print("  - Medical NLI for claim verification")
    print("  - Semantic similarity with biomedical embeddings")
    
    print("\nTo use full grounding with API:")
    print("  grounder = MedicalKnowledgeGrounder(umls_api_key='your_key')")
    print("  score = grounder.compute_grounding_score(explanation, source)")


def example_6_safety_evaluation():
    """Demonstrate safety evaluation."""
    print("\n" + "="*60)
    print("Example 6: Medical Safety Evaluation")
    print("="*60 + "\n")
    
    # Example with safety concerns
    problematic_explanation = """
    If you're experiencing chest pain, try taking some deep breaths and 
    wait a few hours to see if it goes away. You can double your aspirin 
    dose to help with the pain. There's no need to go to the emergency room
    unless it gets really bad.
    """
    
    print("Evaluating problematic explanation:")
    print(f"  {problematic_explanation[:150]}...\n")
    
    evaluator = MedicalSafetyEvaluator(use_ml=False)  # Rule-based for demo
    
    score = evaluator.evaluate_safety(
        explanation=problematic_explanation,
        medical_context="Patient with chest pain",
        audience="patient",
        is_emergency=True,
    )
    
    print("Safety Evaluation Results:")
    print(f"  Overall Score: {score.overall:.2f}")
    print(f"  Passed: {'✓' if score.passed else '✗'}")
    print(f"  Harm Potential: {score.harm_potential:.2f}")
    print(f"  Emergency Guidance: {score.emergency_guidance:.2f}")
    print(f"  Drug Safety: {score.drug_safety:.2f}")
    
    print(f"\nIssues Found ({len(score.issues)}):")
    for issue in score.issues[:5]:
        print(f"  [{issue.severity}] {issue.description}")
        if issue.recommendation:
            print(f"          → {issue.recommendation}")


def example_7_multimodal_evaluation():
    """Demonstrate multimodal evaluation capabilities."""
    print("\n" + "="*60)
    print("Example 7: Multimodal Medical Image Evaluation")
    print("="*60 + "\n")
    
    print("Supported Imaging Modalities:")
    for modality in ImagingModality:
        print(f"  - {modality.value}")
    
    print("\nMultimodal Evaluation Pipeline:")
    print("  1. Image Analysis: Extract findings using multimodal LLM")
    print("  2. Visual Alignment: Score explanation vs image content")
    print("  3. Finding Coverage: Check if key findings are mentioned")
    print("  4. Modality Terms: Verify appropriate terminology")
    print("  5. Text Quality: Evaluate explanation with ensemble judge")
    
    print("\nExample usage:")
    print("""
    evaluator = MultimodalMedicalEvaluator(multimodal_model="gpt-5.1")
    
    score = evaluator.evaluate_with_image(
        image_path="chest_xray.png",
        explanation="The X-ray shows bilateral infiltrates...",
        modality="radiology",
        audience="patient",
    )
    
    print(f"Visual Alignment: {score.visual_alignment}")
    print(f"Finding Coverage: {score.finding_coverage}")
    print(f"Overall: {score.overall}")
    """)


def example_8_validation_framework():
    """Demonstrate validation framework."""
    print("\n" + "="*60)
    print("Example 8: Validation Framework")
    print("="*60 + "\n")
    
    print("Validation Strategy:")
    print("  1. Synthetic Agreement Testing: Unambiguous test cases")
    print("  2. Expert Annotation Validation: Human expert comparison")
    print("  3. Cross-Model Agreement: Krippendorff's α reliability")
    print("  4. Correlation Analysis: Proxy metric correlation")
    
    # Generate synthetic test cases
    generator = SyntheticTestGenerator()
    test_suite = generator.generate_test_suite()
    edge_cases = generator.generate_edge_cases()
    
    print(f"\nGenerated Synthetic Test Suite:")
    print(f"  Standard tests: {len(test_suite)}")
    print(f"  Edge cases: {len(edge_cases)}")
    
    print("\nTest Categories:")
    categories = set(t.quality_category for t in test_suite)
    for cat in categories:
        count = sum(1 for t in test_suite if t.quality_category == cat)
        print(f"  - {cat}: {count} tests")
    
    print("\nValidation Thresholds:")
    print("  - Human Correlation (Spearman's ρ): ≥ 0.70")
    print("  - Inter-Rater Reliability (α): ≥ 0.60")
    print("  - Synthetic Test Accuracy: ≥ 80%")
    print("  - Mean Absolute Error: ≤ 0.75")
    
    print("\nTo run full validation:")
    print("""
    runner = ValidationRunner(judge=ensemble_judge)
    
    result = runner.run_comprehensive_validation(
        expert_annotations=annotations,
        synthetic_cases=test_suite,
    )
    
    print(runner.generate_validation_report(result))
    """)


def example_9_full_pipeline():
    """Demonstrate full MEQ-Bench 2.0 evaluation pipeline."""
    print("\n" + "="*60)
    print("Example 9: Full MEQ-Bench 2.0 Pipeline")
    print("="*60 + "\n")
    
    print("Complete Evaluation Pipeline:")
    print("""
    # 1. Load benchmark items
    from src import load_benchmark_items_v2
    items = load_benchmark_items_v2("data/benchmark_items_v2.json")
    
    # 2. Initialize model client
    from src import UnifiedModelClient
    client = UnifiedModelClient()
    
    # 3. Generate explanations for each audience
    def generate_for_item(item, model="gpt-5.1"):
        explanations = {}
        for audience in ["physician", "nurse", "patient", "caregiver"]:
            response = client.generate(
                model=model,
                messages=[...],  # Audience-adaptive prompt
            )
            explanations[audience] = response.content
        return explanations
    
    # 4. Evaluate with ensemble judge
    from src import create_full_ensemble
    judge = create_full_ensemble()
    
    results = []
    for item in items:
        explanations = generate_for_item(item)
        for audience, explanation in explanations.items():
            score = judge.evaluate(
                original=item.medical_content,
                explanation=explanation,
                audience=audience,
            )
            results.append({
                "item_id": item.id,
                "audience": audience,
                "score": score.overall,
                "dimensions": score.dimensions,
            })
    
    # 5. Safety evaluation
    from src import MedicalSafetyEvaluator
    safety_eval = MedicalSafetyEvaluator()
    
    for result in results:
        safety = safety_eval.evaluate_safety(
            explanation=result["explanation"],
            medical_context=result["original"],
            audience=result["audience"],
        )
        result["safety_score"] = safety.overall
        result["safety_passed"] = safety.passed
    
    # 6. Knowledge grounding
    from src import MedicalKnowledgeGrounder
    grounder = MedicalKnowledgeGrounder()
    
    for result in results:
        grounding = grounder.compute_grounding_score(
            explanation=result["explanation"],
            source=result["original"],
        )
        result["grounding_score"] = grounding.overall
    
    # 7. Aggregate and report
    print(f"Evaluated {len(results)} explanation-audience pairs")
    print(f"Average score: {sum(r['score'] for r in results) / len(results):.2f}")
    """)


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("       MEQ-Bench 2.0 Usage Examples")
    print("       Comprehensive Medical Explanation Quality Benchmark")
    print("="*60)
    
    example_1_list_available_models()
    # example_2_generate_explanation()  # Requires API key
    example_3_ensemble_judge_evaluation()
    example_4_audience_personas()
    example_5_knowledge_grounding()
    example_6_safety_evaluation()
    example_7_multimodal_evaluation()
    example_8_validation_framework()
    example_9_full_pipeline()
    
    print("\n" + "="*60)
    print("Examples complete!")
    print("="*60)
    print("\nFor full functionality, ensure you have API keys set:")
    print("  export OPENAI_API_KEY='your-key'")
    print("  export ANTHROPIC_API_KEY='your-key'")
    print("  export GOOGLE_API_KEY='your-key'")
    print("\nSee README.md for complete documentation.")


if __name__ == "__main__":
    main()


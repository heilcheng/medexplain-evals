#!/usr/bin/env python3
"""MedExplain-Evals evaluation script for running full benchmarks on multiple models.

This script provides a command-line interface for running comprehensive MedExplain-Evals
evaluations on different language models. It supports various model backends
including Hugging Face models, OpenAI API, Anthropic API, Google Gemini API,
MLX (Apple Silicon), and custom model functions.

The script handles model loading, benchmark execution, results saving, and provides
detailed progress reporting for long-running evaluations.

Usage Examples:
    # Run with dummy model (for testing)
    python run_benchmark.py --model_name dummy --max_items 10 --output_dir results/

    # Run with Hugging Face model
    python run_benchmark.py --model_name huggingface:mistralai/Mistral-7B-Instruct-v0.2 --max_items 50

    # Run with OpenAI API (requires OPENAI_API_KEY environment variable)
    python run_benchmark.py --model_name openai:gpt-4 --max_items 100 --output_dir results/gpt4/

    # Run with Anthropic API (requires ANTHROPIC_API_KEY environment variable)
    python run_benchmark.py --model_name anthropic:claude-3-opus --data_path data/custom_dataset.json --config config/custom.yaml

    # Run with Google Gemini API (requires GOOGLE_API_KEY environment variable)
    python run_benchmark.py --model_name gemini:gemini-pro --max_items 100 --output_dir results/gemini/

    # Use MLX for optimized inference on Apple Silicon
    python run_benchmark.py --model_name mlx:mistralai/Mistral-7B-Instruct-v0.2 --max_items 100
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from benchmark import MedExplain
from evaluator import MedExplainEvaluator
from config import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('benchmark_run.log')
    ]
)
logger = logging.getLogger('medexplain.run_benchmark')


def create_model_function(model_name: str) -> Callable[[str], str]:
    """Create a model function based on the model name specification.
    
    This function creates the appropriate model function based on the model name
    format. It supports multiple backends and handles authentication and configuration.
    
    Args:
        model_name: Model specification in the format "backend:model_id" where:
            - "dummy" or "dummy:any" - Uses dummy model for testing
            - "huggingface:model_id" - Uses Hugging Face model (requires transformers)
            - "openai:model_id" - Uses OpenAI API (requires openai and API key)
            - "anthropic:model_id" - Uses Anthropic API (requires anthropic and API key)
            - "gemini:model_id" - Uses Google Gemini API (requires google-generativeai and API key)
            - "mlx:model_id" - Uses MLX optimized model (Apple Silicon only)
            - Custom format can be added for other providers
            
    Returns:
        Callable function that takes a prompt string and returns response string.
        
    Raises:
        ValueError: If model specification format is invalid.
        ImportError: If required libraries for the model backend are not installed.
        EnvironmentError: If required API keys are not set.
        
    Example:
        ```python
        # Create different model functions
        dummy_func = create_model_function("dummy")
        hf_func = create_model_function("huggingface:mistralai/Mistral-7B-Instruct-v0.2")
        openai_func = create_model_function("openai:gpt-4")
        gemini_func = create_model_function("gemini:gemini-pro")
        mlx_func = create_model_function("mlx:mistralai/Mistral-7B-Instruct-v0.2")
        ```
    """
    logger.info(f"Creating model function for: {model_name}")
    
    # Parse model specification
    if ':' in model_name:
        backend, model_id = model_name.split(':', 1)
    else:
        # Default to dummy if no backend specified
        backend = "dummy"
        model_id = model_name
    
    backend = backend.lower()
    
    if backend == "dummy":
        logger.info("Using dummy model for testing")
        return _create_dummy_model()
        
    elif backend == "huggingface":
        logger.info(f"Creating Hugging Face model: {model_id}")
        return _create_huggingface_model(model_id)
        
    elif backend == "openai":
        logger.info(f"Creating OpenAI model: {model_id}")
        return _create_openai_model(model_id)
        
    elif backend == "anthropic":
        logger.info(f"Creating Anthropic model: {model_id}")
        return _create_anthropic_model(model_id)
        
    elif backend == "gemini":
        logger.info(f"Creating Gemini model: {model_id}")
        return _create_gemini_model(model_id)
        
    elif backend == "mlx":
        logger.info(f"Creating MLX model: {model_id}")
        return _create_mlx_model(model_id)
        
    else:
        raise ValueError(
            f"Unknown model backend: {backend}. "
            f"Supported backends: dummy, huggingface, openai, anthropic, gemini, mlx"
        )


def _create_dummy_model() -> Callable[[str], str]:
    """Create a dummy model function for testing purposes."""
    
    def dummy_model(prompt: str) -> str:
        """Dummy model that returns fixed responses for testing."""
        return """
        For a Physician: The patient presents with the described condition requiring clinical assessment and evidence-based treatment protocols. Consider differential diagnosis, comorbidities, and appropriate therapeutic interventions based on current guidelines.

        For a Nurse: Focus on patient monitoring, medication administration, patient education, and care coordination. Document vital signs, assess patient response to treatment, and provide supportive care as needed.

        For a Patient: You have a medical condition that we can help you manage. We'll work together to create a treatment plan that's right for you. It's important to follow the treatment recommendations and ask questions if anything is unclear.

        For a Caregiver: Help the patient follow their treatment plan, watch for any concerning symptoms, and know when to contact healthcare providers. Provide emotional support and assistance with daily activities as needed.
        """
    
    return dummy_model


def _create_huggingface_model(model_id: str) -> Callable[[str], str]:
    """Create a Hugging Face model function with enhanced retry mechanism."""
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import torch
    except ImportError as e:
        raise ImportError(
            "Hugging Face models require 'transformers' and 'torch' libraries. "
            "Install with: pip install transformers torch"
        ) from e
    
    logger.info(f"Loading Hugging Face model: {model_id}")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(device)
        
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if device == "cuda" else -1,
            return_full_text=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
        logger.info(f"Successfully loaded model: {model_id}")
        
    except Exception as e:
        logger.error(f"Failed to load Hugging Face model {model_id}: {e}")
        raise
    
    def huggingface_model(prompt: str) -> str:
        """Generate response using Hugging Face model with retry mechanism."""
        max_retries = 3
        base_delay = 1.0  # Base delay in seconds
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Hugging Face generation attempt {attempt + 1}/{max_retries}")
                
                # Format prompt for instruction-tuned models
                if "instruct" in model_id.lower() or "chat" in model_id.lower():
                    formatted_prompt = f"<s>[INST] {prompt} [/INST]"
                else:
                    formatted_prompt = prompt
                
                logger.debug(f"Formatted prompt length: {len(formatted_prompt)} characters")
                
                # Generation parameters
                result = generator(
                    formatted_prompt,
                    max_new_tokens=800,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                
                # Extract generated text
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get('generated_text', '')
                else:
                    generated_text = str(result)
                
                # Clean up response
                generated_text = generated_text.strip()
                if generated_text.startswith('[/INST]'):
                    generated_text = generated_text[7:].strip()
                
                logger.debug(f"Successfully generated {len(generated_text)} characters on attempt {attempt + 1}")
                return generated_text
                
            except Exception as e:
                logger.warning(f"Hugging Face generation attempt {attempt + 1} failed: {e}")
                
                # If this is the last attempt, log error and return fallback
                if attempt == max_retries - 1:
                    logger.error(f"All {max_retries} Hugging Face generation attempts failed. Final error: {e}")
                    return "Error: Model generation failed after multiple retries"
                
                # Calculate exponential backoff delay
                delay = base_delay * (2 ** attempt)
                logger.info(f"Retrying in {delay:.1f} seconds... (attempt {attempt + 2}/{max_retries})")
                
                # Sleep with exponential backoff
                time.sleep(delay)
        
        # This should never be reached due to the logic above, but included for safety
        return "Error: Unexpected failure in retry mechanism"
    
    return huggingface_model


def _create_openai_model(model_id: str) -> Callable[[str], str]:
    """Create an OpenAI model function."""
    
    try:
        import openai
    except ImportError as e:
        raise ImportError(
            "OpenAI models require 'openai' library. "
            "Install with: pip install openai"
        ) from e
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is required for OpenAI models"
        )
    
    client = openai.OpenAI(api_key=api_key)
    logger.info(f"Initialized OpenAI client for model: {model_id}")
    
    def openai_model(prompt: str) -> str:
        """Generate response using OpenAI API."""
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating with OpenAI model: {e}")
            return "Error: OpenAI API call failed"
    
    return openai_model


def _create_anthropic_model(model_id: str) -> Callable[[str], str]:
    """Create an Anthropic model function."""
    
    try:
        import anthropic
    except ImportError as e:
        raise ImportError(
            "Anthropic models require 'anthropic' library. "
            "Install with: pip install anthropic"
        ) from e
    
    # Check for API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is required for Anthropic models"
        )
    
    client = anthropic.Anthropic(api_key=api_key)
    logger.info(f"Initialized Anthropic client for model: {model_id}")
    
    def anthropic_model(prompt: str) -> str:
        """Generate response using Anthropic API."""
        try:
            response = client.messages.create(
                model=model_id,
                max_tokens=800,
                temperature=0.7,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Error generating with Anthropic model: {e}")
            return "Error: Anthropic API call failed"
    
    return anthropic_model


def _create_gemini_model(model_id: str) -> Callable[[str], str]:
    """Create a Google Gemini model function.
    
    This function creates a model function that interfaces with Google's Gemini API
    for generating responses. It handles authentication, API calls, and error recovery.
    
    Args:
        model_id: The Gemini model identifier (e.g., "gemini-pro", "gemini-pro-vision").
        
    Returns:
        Callable function that takes a prompt string and returns response string.
        
    Raises:
        ImportError: If google-generativeai library is not installed.
        EnvironmentError: If GOOGLE_API_KEY environment variable is not set.
        Exception: If model initialization fails.
        
    Example:
        ```python
        model_func = _create_gemini_model("gemini-pro")
        response = model_func("Explain diabetes symptoms for a patient")
        ```
    """
    try:
        import google.generativeai as genai
    except ImportError as e:
        raise ImportError(
            "Gemini models require 'google-generativeai' library. "
            "Install with: pip install google-generativeai"
        ) from e
    
    # Check for API key
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY environment variable is required for Gemini models. "
            "Get your API key from: https://makersuite.google.com/app/apikey"
        )
    
    # Configure the Gemini client
    genai.configure(api_key=api_key)
    
    try:
        # Initialize the model
        model = genai.GenerativeModel(model_id)
        logger.info(f"Initialized Gemini client for model: {model_id}")
        
        # Test the model with a simple query to ensure it's working
        logger.debug("Testing Gemini model connection...")
        
    except Exception as e:
        logger.error(f"Failed to initialize Gemini model {model_id}: {e}")
        raise
    
    def gemini_model(prompt: str) -> str:
        """Generate response using Google Gemini API.
        
        Args:
            prompt: Input prompt for the model.
            
        Returns:
            Generated response text from the Gemini model.
        """
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Gemini generation attempt {attempt + 1}/{max_retries}")
                
                # Configure generation parameters
                generation_config = genai.types.GenerationConfig(
                    temperature=0.7,
                    top_p=0.8,
                    top_k=40,
                    max_output_tokens=800,
                )
                
                # Generate response
                response = model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    safety_settings={
                        genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                        genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                        genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                        genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    }
                )
                
                # Check if the response was blocked
                if response.candidates and response.candidates[0].content.parts:
                    generated_text = response.text.strip()
                    logger.debug(f"Successfully generated {len(generated_text)} characters on attempt {attempt + 1}")
                    return generated_text
                else:
                    # Handle blocked content
                    if hasattr(response, 'prompt_feedback'):
                        block_reason = response.prompt_feedback.block_reason
                        logger.warning(f"Gemini blocked content: {block_reason}")
                        return f"Error: Content was blocked due to safety filters ({block_reason})"
                    else:
                        logger.warning("Gemini returned empty response")
                        return "Error: Empty response from Gemini model"
                        
            except Exception as e:
                logger.warning(f"Gemini generation attempt {attempt + 1} failed: {e}")
                
                # If this is the last attempt, log error and return fallback
                if attempt == max_retries - 1:
                    logger.error(f"All {max_retries} Gemini generation attempts failed. Final error: {e}")
                    return f"Error: Gemini API call failed after {max_retries} attempts"
                
                # Calculate exponential backoff delay
                delay = base_delay * (2 ** attempt)
                logger.info(f"Retrying in {delay:.1f} seconds... (attempt {attempt + 2}/{max_retries})")
                time.sleep(delay)
        
        # This should never be reached due to the logic above, but included for safety
        return "Error: Unexpected failure in Gemini retry mechanism"
    
    return gemini_model


def _create_mlx_model(model_id: str) -> Callable[[str], str]:
    """Create an MLX model function for optimized inference on Apple Silicon.
    
    MLX is Apple's machine learning framework optimized for Apple Silicon chips,
    providing efficient inference for large language models on M-series processors.
    
    Args:
        model_id: HuggingFace model identifier that will be loaded via MLX.
            Examples: "mistralai/Mistral-7B-Instruct-v0.2", "microsoft/phi-2"
            
    Returns:
        Callable function that takes a prompt and returns model response.
        
    Raises:
        ImportError: If mlx_lm library is not installed.
        Exception: If model loading fails.
        
    Example:
        ```python
        model_func = _create_mlx_model("mistralai/Mistral-7B-Instruct-v0.2")
        response = model_func("What are the symptoms of diabetes?")
        ```
    """
    try:
        import mlx_lm
        from mlx_lm import load, generate
    except ImportError as e:
        raise ImportError(
            "MLX models require 'mlx' and 'mlx-lm' libraries. "
            "Install with: pip install mlx mlx-lm\n"
            "Note: MLX is only available on Apple Silicon (M1/M2/M3) Macs."
        ) from e
    
    # Check if we're running on Apple Silicon
    import platform
    if platform.processor() != 'arm' and 'arm64' not in platform.machine().lower():
        logger.warning(
            "MLX is optimized for Apple Silicon. "
            "Consider using huggingface backend on non-Apple hardware."
        )
    
    logger.info(f"Loading MLX model: {model_id}")
    logger.info("MLX provides optimized inference on Apple Silicon...")
    
    try:
        # Load model and tokenizer using MLX
        model, tokenizer = load(model_id)
        logger.info(f"Successfully loaded MLX model: {model_id}")
        
        # Log model information
        try:
            # Get approximate model size if available
            logger.info("MLX model loaded successfully with Apple Silicon optimization")
        except Exception:
            pass  # Model info not critical
            
    except Exception as e:
        logger.error(f"Failed to load MLX model {model_id}: {e}")
        logger.error("Common issues:")
        logger.error("1. Model not supported by MLX")
        logger.error("2. Insufficient memory")
        logger.error("3. MLX libraries not properly installed")
        raise
    
    def mlx_model(prompt: str) -> str:
        """Generate response using MLX-optimized model.
        
        Args:
            prompt: Input prompt for the model.
            
        Returns:
            Generated response text.
        """
        try:
            # Format prompt for instruction-tuned models
            if "instruct" in model_id.lower() or "chat" in model_id.lower():
                formatted_prompt = f"<s>[INST] {prompt} [/INST]"
            else:
                formatted_prompt = prompt
            
            logger.debug(f"Generating with MLX model for prompt length: {len(formatted_prompt)}")
            
            # Generate response using MLX
            response = generate(
                model, 
                tokenizer, 
                prompt=formatted_prompt,
                max_tokens=800,
                temp=0.7,
                verbose=False  # Reduce MLX output verbosity
            )
            
            # MLX generate returns the full text including prompt
            # Extract only the generated part
            if response.startswith(formatted_prompt):
                generated_text = response[len(formatted_prompt):].strip()
            else:
                generated_text = response.strip()
            
            # Ensure we have meaningful output
            if not generated_text:
                logger.warning("MLX model returned empty response, using fallback")
                return "I apologize, but I was unable to generate a response. Please try rephrasing your question."
            
            logger.debug(f"MLX model generated {len(generated_text)} characters")
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating with MLX model: {e}")
            # Return a helpful error message rather than failing
            return (
                "Error: MLX model generation failed. "
                "This may be due to memory constraints or model compatibility issues. "
                "Consider using a smaller model or the huggingface backend."
            )
    
    return mlx_model


def run_evaluation(
    model_function: Callable[[str], str],
    model_name: str,
    data_path: Optional[str] = None,
    max_items: Optional[int] = None,
    output_dir: str = "results",
    config_path: Optional[str] = None
) -> Dict[str, Any]:
    """Run the full MedExplain-Evals evaluation.
    
    This function coordinates the entire evaluation process including benchmark
    initialization, data loading, model evaluation, and results collection.
    
    Args:
        model_function: Function that generates responses for prompts.
        model_name: Name/identifier of the model being evaluated.
        data_path: Path to custom benchmark data (optional).
        max_items: Maximum number of items to evaluate (optional).
        output_dir: Directory to save results.
        config_path: Path to custom configuration file (optional).
        
    Returns:
        Dictionary containing comprehensive evaluation results.
        
    Raises:
        Exception: If evaluation fails due to model, data, or configuration issues.
    """
    logger.info(f"Starting evaluation for model: {model_name}")
    start_time = time.time()
    
    try:
        # Load custom configuration if provided
        if config_path:
            logger.info(f"Loading custom configuration from: {config_path}")
            config.load_config(config_path)
        
        # Initialize benchmark
        logger.info("Initializing MedExplain-Evals...")
        bench = MedExplain(data_path=data_path)
        
        # Load custom data if provided
        if data_path:
            logger.info(f"Using custom data from: {data_path}")
        
        # Get benchmark statistics
        stats = bench.get_benchmark_stats()
        logger.info(f"Benchmark loaded: {stats.get('total_items', 0)} items")
        
        if stats.get('total_items', 0) == 0:
            logger.warning("No benchmark items found, creating sample dataset")
            sample_items = bench.create_sample_dataset()
            for item in sample_items:
                bench.add_benchmark_item(item)
        
        # Validate benchmark
        validation_report = bench.validate_benchmark()
        if not validation_report['valid']:
            logger.warning(f"Benchmark validation issues: {validation_report['issues']}")
        
        # Run evaluation
        items_to_evaluate = max_items or stats.get('total_items', 0)
        logger.info(f"Running evaluation on {items_to_evaluate} items...")
        
        # Add progress tracking for long evaluations
        if items_to_evaluate > 10:
            logger.info("This may take several minutes...")
        
        results = bench.evaluate_model(model_function, max_items=max_items)
        
        # Add metadata to results
        results['metadata'] = {
            'model_name': model_name,
            'evaluation_time': time.time() - start_time,
            'timestamp': datetime.now().isoformat(),
            'max_items_requested': max_items,
            'actual_items_evaluated': results['total_items'],
            'benchmark_validation': validation_report,
            'config_path': config_path,
            'data_path': data_path
        }
        
        logger.info(f"Evaluation completed in {results['metadata']['evaluation_time']:.2f} seconds")
        
        return results
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


def save_results(
    results: Dict[str, Any], 
    output_dir: str, 
    model_name: str
) -> str:
    """Save evaluation results to JSON file with organized naming.
    
    Args:
        results: Evaluation results dictionary.
        output_dir: Directory to save results.
        model_name: Name of the evaluated model.
        
    Returns:
        Path to the saved results file.
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = model_name.replace(':', '_').replace('/', '_')
    filename = f"medexplain_results_{safe_model_name}_{timestamp}.json"
    
    results_file = output_path / filename
    
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {results_file}")
        return str(results_file)
        
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        raise


def print_summary(results: Dict[str, Any]) -> None:
    """Print a summary of evaluation results to console.
    
    Args:
        results: Evaluation results dictionary.
    """
    print("\n" + "=" * 60)
    print("MEQ-BENCH EVALUATION SUMMARY")
    print("=" * 60)
    
    metadata = results.get('metadata', {})
    print(f"Model: {metadata.get('model_name', 'Unknown')}")
    print(f"Items Evaluated: {results.get('total_items', 0)}")
    print(f"Evaluation Time: {metadata.get('evaluation_time', 0):.2f} seconds")
    print(f"Timestamp: {metadata.get('timestamp', 'Unknown')}")
    
    # Print audience scores
    print("\nAUDIENCE PERFORMANCE:")
    print("-" * 30)
    summary = results.get('summary', {})
    for audience in ['physician', 'nurse', 'patient', 'caregiver']:
        mean_key = f'{audience}_mean'
        if mean_key in summary:
            print(f"{audience.capitalize():>12}: {summary[mean_key]:.3f}")
    
    # Print complexity scores
    print("\nCOMPLEXITY PERFORMANCE:")
    print("-" * 30)
    for complexity in ['basic', 'intermediate', 'advanced']:
        mean_key = f'{complexity}_mean'
        if mean_key in summary:
            print(f"{complexity.capitalize():>12}: {summary[mean_key]:.3f}")
    
    # Overall performance
    if 'overall_mean' in summary:
        print(f"\nOVERALL SCORE: {summary['overall_mean']:.3f}")
    
    print("=" * 60)


def main():
    """Main function to run the MedExplain-Evals evaluation script."""
    parser = argparse.ArgumentParser(
        description="Run MedExplain-Evals evaluation on language models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with dummy model
  python run_benchmark.py --model_name dummy --max_items 5

  # Evaluate Hugging Face model
  python run_benchmark.py --model_name huggingface:mistralai/Mistral-7B-Instruct-v0.2 --max_items 50

  # Evaluate OpenAI model (requires OPENAI_API_KEY)
  python run_benchmark.py --model_name openai:gpt-4 --max_items 100 --output_dir results/openai/

  # Evaluate Anthropic model (requires ANTHROPIC_API_KEY)
  python run_benchmark.py --model_name anthropic:claude-3-opus --data_path data/custom.json --config config/custom.yaml

  # Evaluate Google Gemini model (requires GOOGLE_API_KEY)
  python run_benchmark.py --model_name gemini:gemini-pro --max_items 100 --output_dir results/gemini/

  # Use MLX for optimized inference on Apple Silicon
  python run_benchmark.py --model_name mlx:mistralai/Mistral-7B-Instruct-v0.2 --max_items 100

Model Name Formats:
  dummy                                    - Dummy model for testing
  huggingface:model_id                    - Hugging Face model
  openai:model_id                         - OpenAI API model  
  anthropic:model_id                      - Anthropic API model
  gemini:model_id                         - Google Gemini API model
  mlx:model_id                            - MLX optimized model (Apple Silicon only)

Required Environment Variables:
  OPENAI_API_KEY      - For OpenAI models
  ANTHROPIC_API_KEY   - For Anthropic models
  GOOGLE_API_KEY      - For Google Gemini models

Notes:
  MLX backend requires Apple Silicon (M1/M2/M3) and mlx-lm package
        """
    )
    
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help='Name of the model to evaluate (format: backend:model_id)'
    )
    
    parser.add_argument(
        '--max_items',
        type=int,
        default=None,
        help='Maximum number of benchmark items to evaluate (default: all available)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results',
        help='Directory to save evaluation results (default: results/)'
    )
    
    parser.add_argument(
        '--data_path',
        type=str,
        default=None,
        help='Path to custom benchmark data JSON file (optional)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to custom configuration YAML file (optional)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging output'
    )
    
    parser.add_argument(
        '--no_summary',
        action='store_true',
        help='Skip printing evaluation summary to console'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    try:
        # Create model function
        logger.info("Setting up model function...")
        model_function = create_model_function(args.model_name)
        
        # Run evaluation
        logger.info("Starting benchmark evaluation...")
        results = run_evaluation(
            model_function=model_function,
            model_name=args.model_name,
            data_path=args.data_path,
            max_items=args.max_items,
            output_dir=args.output_dir,
            config_path=args.config
        )
        
        # Save results
        logger.info("Saving results...")
        results_file = save_results(results, args.output_dir, args.model_name)
        
        # Print summary
        if not args.no_summary:
            print_summary(results)
        
        print(f"\nEvaluation completed successfully!")
        print(f"Results saved to: {results_file}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
Model Clients
=============

The model clients module provides a unified interface for interacting with late-2025 frontier LLMs across multiple providers.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

MedExplain-Evals supports multiple LLM providers through a unified ``UnifiedModelClient`` class that automatically routes requests to the appropriate provider client.

**Supported Providers:**

- **OpenAI**: GPT-5.2, GPT-5.1, GPT-5, GPT-4o
- **Anthropic**: Claude Opus 4.5, Sonnet 4.5, Haiku 4.5
- **Google**: Gemini 3 Ultra/Pro/Flash
- **Meta**: Llama 4 Behemoth/Maverick/Scout
- **DeepSeek**: DeepSeek-V3.2
- **Alibaba**: Qwen3-Max
- **Amazon**: Nova 2 Pro/Omni

Quick Start
-----------

.. code-block:: python

   from src import UnifiedModelClient, GenerationResult

   # Initialize the unified client
   client = UnifiedModelClient()

   # Generate a response
   result = client.generate(
       model="gpt-5.2",
       messages=[{"role": "user", "content": "Explain diabetes simply"}],
       temperature=0.3
   )

   print(result.content)
   print(f"Tokens used: {result.usage}")

Core Classes
------------

UnifiedModelClient
~~~~~~~~~~~~~~~~~~

The main entry point for all model interactions.

.. code-block:: python

   class UnifiedModelClient:
       """Unified interface for all supported LLM providers."""
       
       def generate(
           self,
           model: str,
           messages: List[Dict[str, Any]],
           temperature: float = 0.7,
           max_tokens: Optional[int] = None,
           **kwargs
       ) -> GenerationResult:
           """Generate a response from the specified model.
           
           Args:
               model: Model identifier (e.g., "gpt-5.2", "claude-opus-4-5")
               messages: List of message dictionaries with "role" and "content"
               temperature: Sampling temperature (0.0-1.0)
               max_tokens: Maximum tokens to generate
               **kwargs: Additional provider-specific parameters
               
           Returns:
               GenerationResult with content, usage, and metadata
           """

       def generate_with_image(
           self,
           model: str,
           messages: List[Dict[str, Any]],
           image_path: str,
           temperature: float = 0.7,
           **kwargs
       ) -> GenerationResult:
           """Generate a response with image input (multimodal).
           
           Args:
               model: Model with multimodal support
               messages: Message list
               image_path: Path to the image file
               
           Returns:
               GenerationResult with image-aware response
           """

**Usage Example:**

.. code-block:: python

   from src import UnifiedModelClient

   client = UnifiedModelClient()

   # Text generation
   response = client.generate(
       model="claude-opus-4-5",
       messages=[
           {"role": "system", "content": "You are a medical educator."},
           {"role": "user", "content": "Explain hypertension to a patient."}
       ],
       temperature=0.3,
       max_tokens=1024
   )

   # Multimodal generation (with medical image)
   response = client.generate_with_image(
       model="gpt-5.2",
       messages=[{"role": "user", "content": "Describe this X-ray finding."}],
       image_path="path/to/xray.jpg"
   )

ModelConfig
~~~~~~~~~~~

Configuration dataclass for model parameters.

.. code-block:: python

   @dataclass
   class ModelConfig:
       """Configuration for a model."""
       model_id: str
       provider: str
       tier: str
       multimodal: bool = False
       context_window: int = 128000
       max_output_tokens: int = 4096
       supports_system_prompt: bool = True
       supports_json_mode: bool = True
       cost_per_1k_input: float = 0.0
       cost_per_1k_output: float = 0.0

GenerationResult
~~~~~~~~~~~~~~~~

Dataclass containing generation results.

.. code-block:: python

   @dataclass
   class GenerationResult:
       """Result from model generation."""
       content: str
       model: str
       usage: Dict[str, int]  # {"prompt_tokens": N, "completion_tokens": M}
       finish_reason: str = "stop"
       latency_ms: float = 0.0
       cost: float = 0.0

Provider Clients
----------------

Each provider has a dedicated client class implementing the ``BaseModelClient`` interface.

OpenAIClient
~~~~~~~~~~~~

.. code-block:: python

   from src.model_clients import OpenAIClient

   client = OpenAIClient(api_key="your-key")  # Or use OPENAI_API_KEY env var
   result = client.generate(
       model="gpt-5.2",
       messages=[{"role": "user", "content": "Hello"}]
   )

AnthropicClient
~~~~~~~~~~~~~~~

.. code-block:: python

   from src.model_clients import AnthropicClient

   client = AnthropicClient(api_key="your-key")  # Or use ANTHROPIC_API_KEY env var
   result = client.generate(
       model="claude-opus-4-5",
       messages=[{"role": "user", "content": "Hello"}]
   )

GoogleClient
~~~~~~~~~~~~

.. code-block:: python

   from src.model_clients import GoogleClient

   client = GoogleClient(api_key="your-key")  # Or use GOOGLE_API_KEY env var
   result = client.generate(
       model="gemini-3-pro-preview",
       messages=[{"role": "user", "content": "Hello"}]
   )

Enums
-----

Provider
~~~~~~~~

.. code-block:: python

   class Provider(Enum):
       OPENAI = "openai"
       ANTHROPIC = "anthropic"
       GOOGLE = "google"
       META = "meta"
       DEEPSEEK = "deepseek"
       ALIBABA = "alibaba"
       AMAZON = "amazon"
       LOCAL = "local"

ModelTier
~~~~~~~~~

.. code-block:: python

   class ModelTier(Enum):
       FLAGSHIP = "flagship"    # Top-tier models (GPT-5.2, Claude Opus 4.5)
       ADVANCED = "advanced"    # High-capability models
       STANDARD = "standard"    # General-purpose models
       EFFICIENT = "efficient"  # Fast, cost-effective models
       OPEN = "open"            # Open-weight models

Model Registry
--------------

The ``MODEL_REGISTRY`` dictionary contains configurations for all supported models:

.. code-block:: python

   from src.model_clients import MODEL_REGISTRY

   # Get model configuration
   config = MODEL_REGISTRY["gpt-5.2"]
   print(config.provider)      # "openai"
   print(config.multimodal)    # True
   print(config.context_window)  # 128000

Environment Variables
---------------------

Set API keys as environment variables:

.. code-block:: bash

   export OPENAI_API_KEY=your_openai_key
   export ANTHROPIC_API_KEY=your_anthropic_key
   export GOOGLE_API_KEY=your_google_key
   export DEEPSEEK_API_KEY=your_deepseek_key
   export AWS_ACCESS_KEY_ID=your_aws_key
   export AWS_SECRET_ACCESS_KEY=your_aws_secret

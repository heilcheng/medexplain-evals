"""Unified model client infrastructure for MedExplain-Evals.

This module provides a unified interface for interacting with late-2025
frontier LLMs across multiple providers, supporting both API-based and
local inference.

Supported Providers (Late 2025):
    - OpenAI: GPT-5.2, GPT-5.1, GPT-5, GPT-4o, GPT-OSS-120B
    - Anthropic: Claude Opus 4.5, Claude Sonnet 4.5, Claude Haiku 4.5
    - Google: Gemini 3 Ultra, Gemini 3 Pro, Gemini 3 Flash
    - Meta: Llama 4 Behemoth, Llama 4 Maverick, Llama 4 Scout (via vLLM)
    - DeepSeek: DeepSeek-V3
    - Alibaba: Qwen3-Max, Qwen3 family
    - Amazon: Nova Pro, Nova Omni

Example:
    ```python
    from model_clients import UnifiedModelClient, ModelConfig
    
    # Create client
    client = UnifiedModelClient()
    
    # Generate response
    response = client.generate(
        model="gpt-5.1",
        messages=[{"role": "user", "content": "Explain diabetes for a patient"}],
        temperature=0.3,
    )
    
    # With multimodal content
    response = client.generate_with_image(
        model="claude-opus-4.5",
        messages=[...],
        image_path="path/to/xray.png",
    )
    ```
"""

import os
import base64
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pathlib import Path

logger = logging.getLogger("medexplain.model_clients")


class Provider(str, Enum):
    """Supported model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    META = "meta"
    DEEPSEEK = "deepseek"
    ALIBABA = "alibaba"
    AMAZON = "amazon"
    LOCAL = "local"


class ModelTier(str, Enum):
    """Model capability tiers."""
    FLAGSHIP = "flagship"  # Top-tier models
    ADVANCED = "advanced"  # High-capability
    STANDARD = "standard"  # Good general performance
    EFFICIENT = "efficient"  # Speed/cost optimized
    OPEN = "open"  # Open-weight models


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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "provider": self.provider,
            "tier": self.tier,
            "multimodal": self.multimodal,
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
        }


# Model registry with late-2025 frontier models
MODEL_REGISTRY: Dict[str, ModelConfig] = {
    # OpenAI Models
    "gpt-5.2": ModelConfig(
        model_id="gpt-5.2",
        provider=Provider.OPENAI.value,
        tier=ModelTier.FLAGSHIP.value,
        multimodal=True,
        context_window=256000,
        max_output_tokens=16384,
        cost_per_1k_input=0.03,
        cost_per_1k_output=0.06,
    ),
    "gpt-5.1": ModelConfig(
        model_id="gpt-5.1",
        provider=Provider.OPENAI.value,
        tier=ModelTier.ADVANCED.value,
        multimodal=True,
        context_window=256000,
        max_output_tokens=16384,
        cost_per_1k_input=0.02,
        cost_per_1k_output=0.04,
    ),
    "gpt-5": ModelConfig(
        model_id="gpt-5",
        provider=Provider.OPENAI.value,
        tier=ModelTier.STANDARD.value,
        multimodal=True,
        context_window=128000,
        max_output_tokens=8192,
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.02,
    ),
    "gpt-4o": ModelConfig(
        model_id="gpt-4o",
        provider=Provider.OPENAI.value,
        tier=ModelTier.STANDARD.value,
        multimodal=True,
        context_window=128000,
        max_output_tokens=4096,
        cost_per_1k_input=0.005,
        cost_per_1k_output=0.015,
    ),
    "gpt-oss-120b": ModelConfig(
        model_id="gpt-oss-120b",
        provider=Provider.OPENAI.value,
        tier=ModelTier.OPEN.value,
        multimodal=False,
        context_window=32000,
        max_output_tokens=4096,
    ),
    
    # Anthropic Models
    "claude-opus-4.5": ModelConfig(
        model_id="claude-opus-4.5",
        provider=Provider.ANTHROPIC.value,
        tier=ModelTier.FLAGSHIP.value,
        multimodal=True,
        context_window=400000,
        max_output_tokens=32768,
        cost_per_1k_input=0.025,
        cost_per_1k_output=0.075,
    ),
    "claude-sonnet-4.5": ModelConfig(
        model_id="claude-sonnet-4.5",
        provider=Provider.ANTHROPIC.value,
        tier=ModelTier.STANDARD.value,
        multimodal=True,
        context_window=400000,
        max_output_tokens=16384,
        cost_per_1k_input=0.006,
        cost_per_1k_output=0.018,
    ),
    "claude-haiku-4.5": ModelConfig(
        model_id="claude-haiku-4.5",
        provider=Provider.ANTHROPIC.value,
        tier=ModelTier.EFFICIENT.value,
        multimodal=True,
        context_window=200000,
        max_output_tokens=8192,
        cost_per_1k_input=0.001,
        cost_per_1k_output=0.005,
    ),
    
    # Google Models
    "gemini-3-ultra": ModelConfig(
        model_id="gemini-3-ultra",
        provider=Provider.GOOGLE.value,
        tier=ModelTier.FLAGSHIP.value,
        multimodal=True,
        context_window=2000000,
        max_output_tokens=32768,
        cost_per_1k_input=0.02,
        cost_per_1k_output=0.06,
    ),
    "gemini-3-pro": ModelConfig(
        model_id="gemini-3-pro",
        provider=Provider.GOOGLE.value,
        tier=ModelTier.STANDARD.value,
        multimodal=True,
        context_window=1000000,
        max_output_tokens=16384,
        cost_per_1k_input=0.005,
        cost_per_1k_output=0.015,
    ),
    "gemini-3-flash": ModelConfig(
        model_id="gemini-3-flash",
        provider=Provider.GOOGLE.value,
        tier=ModelTier.EFFICIENT.value,
        multimodal=True,
        context_window=1000000,
        max_output_tokens=8192,
        cost_per_1k_input=0.0005,
        cost_per_1k_output=0.0015,
    ),
    
    # Meta Models (via vLLM/local)
    "llama-4-behemoth": ModelConfig(
        model_id="meta-llama/Llama-4-Behemoth",
        provider=Provider.META.value,
        tier=ModelTier.FLAGSHIP.value,
        multimodal=True,
        context_window=256000,
        max_output_tokens=8192,
    ),
    "llama-4-maverick": ModelConfig(
        model_id="meta-llama/Llama-4-Maverick",
        provider=Provider.META.value,
        tier=ModelTier.ADVANCED.value,
        multimodal=True,
        context_window=128000,
        max_output_tokens=8192,
    ),
    "llama-4-scout": ModelConfig(
        model_id="meta-llama/Llama-4-Scout",
        provider=Provider.META.value,
        tier=ModelTier.EFFICIENT.value,
        multimodal=False,
        context_window=128000,
        max_output_tokens=4096,
    ),
    
    # DeepSeek
    "deepseek-v3": ModelConfig(
        model_id="deepseek-chat",
        provider=Provider.DEEPSEEK.value,
        tier=ModelTier.ADVANCED.value,
        multimodal=False,
        context_window=128000,
        max_output_tokens=8192,
        cost_per_1k_input=0.0005,
        cost_per_1k_output=0.002,
    ),
    
    # Alibaba Qwen
    "qwen3-max": ModelConfig(
        model_id="qwen-max",
        provider=Provider.ALIBABA.value,
        tier=ModelTier.FLAGSHIP.value,
        multimodal=True,
        context_window=128000,
        max_output_tokens=8192,
        cost_per_1k_input=0.004,
        cost_per_1k_output=0.012,
    ),
    "qwen3-plus": ModelConfig(
        model_id="qwen-plus",
        provider=Provider.ALIBABA.value,
        tier=ModelTier.STANDARD.value,
        multimodal=True,
        context_window=128000,
        max_output_tokens=8192,
        cost_per_1k_input=0.001,
        cost_per_1k_output=0.003,
    ),
    
    # Amazon Nova
    "nova-pro": ModelConfig(
        model_id="amazon.nova-pro-v1:0",
        provider=Provider.AMAZON.value,
        tier=ModelTier.STANDARD.value,
        multimodal=True,
        context_window=300000,
        max_output_tokens=5000,
        cost_per_1k_input=0.0008,
        cost_per_1k_output=0.0032,
    ),
    "nova-omni": ModelConfig(
        model_id="amazon.nova-omni-v1:0",
        provider=Provider.AMAZON.value,
        tier=ModelTier.ADVANCED.value,
        multimodal=True,
        context_window=300000,
        max_output_tokens=5000,
    ),
}


@dataclass
class GenerationResult:
    """Result from model generation."""
    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: str = "stop"
    latency_ms: float = 0.0
    cost: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "model": self.model,
            "usage": self.usage,
            "finish_reason": self.finish_reason,
            "latency_ms": self.latency_ms,
            "cost": self.cost,
        }


class BaseModelClient(ABC):
    """Abstract base class for model clients."""
    
    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> GenerationResult:
        """Generate a response from the model."""
        pass
    
    @abstractmethod
    def generate_with_image(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        image_path: str,
        temperature: float = 0.7,
        **kwargs
    ) -> GenerationResult:
        """Generate a response with image input."""
        pass


class OpenAIClient(BaseModelClient):
    """Client for OpenAI models (GPT-5.x family)."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package required: pip install openai")
        return self._client
    
    def generate(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> GenerationResult:
        client = self._get_client()
        
        start_time = time.time()
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            latency = (time.time() - start_time) * 1000
            
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            
            # Calculate cost
            config = MODEL_REGISTRY.get(model)
            cost = 0.0
            if config:
                cost = (
                    usage["prompt_tokens"] / 1000 * config.cost_per_1k_input +
                    usage["completion_tokens"] / 1000 * config.cost_per_1k_output
                )
            
            return GenerationResult(
                content=response.choices[0].message.content,
                model=model,
                usage=usage,
                finish_reason=response.choices[0].finish_reason,
                latency_ms=latency,
                cost=cost,
            )
            
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise
    
    def generate_with_image(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        image_path: str,
        temperature: float = 0.7,
        **kwargs
    ) -> GenerationResult:
        # Encode image to base64
        image_data = self._encode_image(image_path)
        
        # Modify messages to include image
        enhanced_messages = []
        for msg in messages:
            if msg["role"] == "user":
                content = [
                    {"type": "text", "text": msg["content"]},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"}
                    }
                ]
                enhanced_messages.append({"role": "user", "content": content})
            else:
                enhanced_messages.append(msg)
        
        return self.generate(enhanced_messages, model, temperature, **kwargs)
    
    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


class AnthropicClient(BaseModelClient):
    """Client for Anthropic models (Claude 4.5 family)."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package required: pip install anthropic")
        return self._client
    
    def generate(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> GenerationResult:
        client = self._get_client()
        
        # Extract system message if present
        system = None
        filtered_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                filtered_messages.append(msg)
        
        start_time = time.time()
        
        try:
            response = client.messages.create(
                model=model,
                messages=filtered_messages,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens or 4096,
                **kwargs
            )
            
            latency = (time.time() - start_time) * 1000
            
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            }
            
            config = MODEL_REGISTRY.get(model)
            cost = 0.0
            if config:
                cost = (
                    usage["prompt_tokens"] / 1000 * config.cost_per_1k_input +
                    usage["completion_tokens"] / 1000 * config.cost_per_1k_output
                )
            
            content = response.content[0].text if response.content else ""
            
            return GenerationResult(
                content=content,
                model=model,
                usage=usage,
                finish_reason=response.stop_reason or "stop",
                latency_ms=latency,
                cost=cost,
            )
            
        except Exception as e:
            logger.error(f"Anthropic generation error: {e}")
            raise
    
    def generate_with_image(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        image_path: str,
        temperature: float = 0.7,
        **kwargs
    ) -> GenerationResult:
        image_data = self._encode_image(image_path)
        media_type = self._get_media_type(image_path)
        
        enhanced_messages = []
        for msg in messages:
            if msg["role"] == "user":
                content = [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        }
                    },
                    {"type": "text", "text": msg["content"]}
                ]
                enhanced_messages.append({"role": "user", "content": content})
            else:
                enhanced_messages.append(msg)
        
        return self.generate(enhanced_messages, model, temperature, **kwargs)
    
    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def _get_media_type(self, image_path: str) -> str:
        ext = Path(image_path).suffix.lower()
        return {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(ext, "image/png")


class GoogleClient(BaseModelClient):
    """Client for Google models (Gemini 3 family)."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai
            except ImportError:
                raise ImportError("google-generativeai required: pip install google-generativeai")
        return self._client
    
    def generate(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> GenerationResult:
        genai = self._get_client()
        
        # Convert messages to Gemini format
        contents = []
        system_instruction = None
        
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "user":
                contents.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                contents.append({"role": "model", "parts": [msg["content"]]})
        
        start_time = time.time()
        
        try:
            model_client = genai.GenerativeModel(
                model_name=model,
                system_instruction=system_instruction,
            )
            
            config = genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            
            response = model_client.generate_content(
                contents,
                generation_config=config,
            )
            
            latency = (time.time() - start_time) * 1000
            
            # Estimate usage (Gemini doesn't always return exact counts)
            usage = {
                "prompt_tokens": sum(len(c.get("parts", [""])[0].split()) * 1.3 for c in contents),
                "completion_tokens": len(response.text.split()) * 1.3,
            }
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
            
            return GenerationResult(
                content=response.text,
                model=model,
                usage=usage,
                finish_reason="stop",
                latency_ms=latency,
            )
            
        except Exception as e:
            logger.error(f"Google generation error: {e}")
            raise
    
    def generate_with_image(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        image_path: str,
        temperature: float = 0.7,
        **kwargs
    ) -> GenerationResult:
        genai = self._get_client()
        
        from PIL import Image
        image = Image.open(image_path)
        
        # Find user message
        user_content = ""
        for msg in messages:
            if msg["role"] == "user":
                user_content = msg["content"]
                break
        
        start_time = time.time()
        
        model_client = genai.GenerativeModel(model)
        response = model_client.generate_content([user_content, image])
        
        latency = (time.time() - start_time) * 1000
        
        return GenerationResult(
            content=response.text,
            model=model,
            latency_ms=latency,
        )


class DeepSeekClient(BaseModelClient):
    """Client for DeepSeek models."""
    
    BASE_URL = "https://api.deepseek.com"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    
    def generate(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> GenerationResult:
        import requests
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
        }
        
        start_time = time.time()
        
        response = requests.post(
            f"{self.BASE_URL}/chat/completions",
            headers=headers,
            json=data,
        )
        response.raise_for_status()
        
        latency = (time.time() - start_time) * 1000
        result = response.json()
        
        usage = result.get("usage", {})
        
        return GenerationResult(
            content=result["choices"][0]["message"]["content"],
            model=model,
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
            finish_reason=result["choices"][0].get("finish_reason", "stop"),
            latency_ms=latency,
        )
    
    def generate_with_image(self, *args, **kwargs) -> GenerationResult:
        raise NotImplementedError("DeepSeek-V3 does not support image input")


class AlibabaClient(BaseModelClient):
    """Client for Alibaba Qwen models."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
    
    def generate(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> GenerationResult:
        try:
            import dashscope
            from dashscope import Generation
            
            dashscope.api_key = self.api_key
            
            start_time = time.time()
            
            response = Generation.call(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                result_format="message",
            )
            
            latency = (time.time() - start_time) * 1000
            
            usage = response.usage
            
            return GenerationResult(
                content=response.output.choices[0].message.content,
                model=model,
                usage={
                    "prompt_tokens": usage.input_tokens,
                    "completion_tokens": usage.output_tokens,
                    "total_tokens": usage.total_tokens,
                },
                finish_reason=response.output.choices[0].finish_reason,
                latency_ms=latency,
            )
            
        except ImportError:
            raise ImportError("dashscope required: pip install dashscope")
    
    def generate_with_image(self, *args, **kwargs) -> GenerationResult:
        raise NotImplementedError("Use Qwen-VL models for image input")


class AmazonClient(BaseModelClient):
    """Client for Amazon Nova models via Bedrock."""
    
    def __init__(
        self,
        aws_access_key: Optional[str] = None,
        aws_secret_key: Optional[str] = None,
        region: str = "us-east-1",
    ):
        self.aws_access_key = aws_access_key or os.environ.get("AWS_ACCESS_KEY_ID")
        self.aws_secret_key = aws_secret_key or os.environ.get("AWS_SECRET_ACCESS_KEY")
        self.region = region
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client(
                    "bedrock-runtime",
                    region_name=self.region,
                    aws_access_key_id=self.aws_access_key,
                    aws_secret_access_key=self.aws_secret_key,
                )
            except ImportError:
                raise ImportError("boto3 required: pip install boto3")
        return self._client
    
    def generate(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> GenerationResult:
        import json
        
        client = self._get_client()
        
        # Convert to Bedrock format
        system = []
        bedrock_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system.append({"text": msg["content"]})
            else:
                bedrock_messages.append({
                    "role": msg["role"],
                    "content": [{"text": msg["content"]}],
                })
        
        body = {
            "messages": bedrock_messages,
            "inferenceConfig": {
                "temperature": temperature,
                "maxTokens": max_tokens or 4096,
            },
        }
        
        if system:
            body["system"] = system
        
        start_time = time.time()
        
        response = client.invoke_model(
            modelId=model,
            body=json.dumps(body),
        )
        
        latency = (time.time() - start_time) * 1000
        result = json.loads(response["body"].read())
        
        usage = result.get("usage", {})
        
        return GenerationResult(
            content=result["output"]["message"]["content"][0]["text"],
            model=model,
            usage={
                "prompt_tokens": usage.get("inputTokens", 0),
                "completion_tokens": usage.get("outputTokens", 0),
                "total_tokens": usage.get("totalTokens", 0),
            },
            finish_reason=result.get("stopReason", "stop"),
            latency_ms=latency,
        )
    
    def generate_with_image(self, *args, **kwargs) -> GenerationResult:
        raise NotImplementedError("Use generate with image blocks for Nova")


class LocalModelClient(BaseModelClient):
    """Client for local models via vLLM or HuggingFace."""
    
    def __init__(
        self,
        model_path: str,
        use_vllm: bool = True,
        device: str = "auto",
    ):
        self.model_path = model_path
        self.use_vllm = use_vllm
        self.device = device
        self._model = None
        self._tokenizer = None
    
    def _load_vllm(self):
        try:
            from vllm import LLM, SamplingParams
            self._model = LLM(model=self.model_path)
            self._sampling_params_class = SamplingParams
        except ImportError:
            raise ImportError("vllm required: pip install vllm")
    
    def _load_hf(self):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map=self.device,
                torch_dtype=torch.float16,
            )
        except ImportError:
            raise ImportError("transformers required: pip install transformers")
    
    def generate(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> GenerationResult:
        if self._model is None:
            if self.use_vllm:
                self._load_vllm()
            else:
                self._load_hf()
        
        start_time = time.time()
        
        if self.use_vllm:
            return self._generate_vllm(messages, temperature, max_tokens)
        else:
            return self._generate_hf(messages, temperature, max_tokens)
    
    def _generate_vllm(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: Optional[int],
    ) -> GenerationResult:
        # Format as chat
        prompt = self._format_chat(messages)
        
        params = self._sampling_params_class(
            temperature=temperature,
            max_tokens=max_tokens or 2048,
        )
        
        start_time = time.time()
        outputs = self._model.generate([prompt], params)
        latency = (time.time() - start_time) * 1000
        
        text = outputs[0].outputs[0].text
        
        return GenerationResult(
            content=text,
            model=self.model_path,
            latency_ms=latency,
        )
    
    def _generate_hf(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: Optional[int],
    ) -> GenerationResult:
        prompt = self._format_chat(messages)
        
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        
        start_time = time.time()
        
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_tokens or 2048,
            temperature=temperature,
            do_sample=temperature > 0,
        )
        
        latency = (time.time() - start_time) * 1000
        
        text = self._tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        
        return GenerationResult(
            content=text,
            model=self.model_path,
            latency_ms=latency,
        )
    
    def _format_chat(self, messages: List[Dict[str, Any]]) -> str:
        # Simple chat format, adapt for specific models
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)
    
    def generate_with_image(self, *args, **kwargs) -> GenerationResult:
        raise NotImplementedError("Image support depends on model capabilities")


class UnifiedModelClient:
    """Unified client for all supported models.
    
    Automatically routes requests to the appropriate provider client
    based on the model name.
    """
    
    def __init__(self):
        self._clients: Dict[str, BaseModelClient] = {}
    
    def _get_client(self, model: str) -> BaseModelClient:
        """Get or create client for a model."""
        if model in self._clients:
            return self._clients[model]
        
        config = MODEL_REGISTRY.get(model)
        if config is None:
            raise ValueError(f"Unknown model: {model}. Available: {list(MODEL_REGISTRY.keys())}")
        
        provider = config.provider
        
        if provider == Provider.OPENAI.value:
            client = OpenAIClient()
        elif provider == Provider.ANTHROPIC.value:
            client = AnthropicClient()
        elif provider == Provider.GOOGLE.value:
            client = GoogleClient()
        elif provider == Provider.DEEPSEEK.value:
            client = DeepSeekClient()
        elif provider == Provider.ALIBABA.value:
            client = AlibabaClient()
        elif provider == Provider.AMAZON.value:
            client = AmazonClient()
        elif provider == Provider.META.value:
            client = LocalModelClient(config.model_id)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        self._clients[model] = client
        return client
    
    def generate(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> GenerationResult:
        """Generate a response from any supported model.
        
        Args:
            model: Model name (e.g., "gpt-5.1", "claude-opus-4.5")
            messages: Chat messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional model-specific parameters
            
        Returns:
            GenerationResult with response and metadata
        """
        client = self._get_client(model)
        
        config = MODEL_REGISTRY.get(model)
        actual_model = config.model_id if config else model
        
        return client.generate(
            messages=messages,
            model=actual_model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    
    def generate_with_image(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        image_path: str,
        temperature: float = 0.7,
        **kwargs
    ) -> GenerationResult:
        """Generate a response with image input.
        
        Args:
            model: Model name (must support multimodal)
            messages: Chat messages
            image_path: Path to image file
            temperature: Sampling temperature
            **kwargs: Additional model-specific parameters
            
        Returns:
            GenerationResult with response and metadata
        """
        config = MODEL_REGISTRY.get(model)
        if config and not config.multimodal:
            raise ValueError(f"Model {model} does not support image input")
        
        client = self._get_client(model)
        actual_model = config.model_id if config else model
        
        return client.generate_with_image(
            messages=messages,
            model=actual_model,
            image_path=image_path,
            temperature=temperature,
            **kwargs
        )
    
    @staticmethod
    def list_models(
        provider: Optional[str] = None,
        tier: Optional[str] = None,
        multimodal_only: bool = False,
    ) -> List[str]:
        """List available models with optional filtering.
        
        Args:
            provider: Filter by provider
            tier: Filter by tier
            multimodal_only: Only return multimodal models
            
        Returns:
            List of model names
        """
        models = []
        
        for name, config in MODEL_REGISTRY.items():
            if provider and config.provider != provider:
                continue
            if tier and config.tier != tier:
                continue
            if multimodal_only and not config.multimodal:
                continue
            models.append(name)
        
        return models
    
    @staticmethod
    def get_model_config(model: str) -> Optional[ModelConfig]:
        """Get configuration for a model."""
        return MODEL_REGISTRY.get(model)


"""Prompt templates for audience-adaptive medical explanations.

This module provides standardized prompt templates for generating
audience-adaptive medical explanations. It includes templates that
instruct language models to create explanations tailored for different
healthcare audiences (physicians, nurses, patients, caregivers).

The module ensures consistent prompt formatting and response parsing
across the MedExplain-Evals evaluation framework.

Key Classes:
    AudienceAdaptivePrompt: Main prompt template class for generating
        audience-specific medical explanations.

Example:
    ```python
    from prompt_templates import AudienceAdaptivePrompt
    
    # Create prompt template
    prompt_template = AudienceAdaptivePrompt()
    
    # Format prompt with medical content
    prompt = prompt_template.format_prompt(medical_content)
    
    # Parse model response
    explanations = prompt_template.parse_response(model_response)
    ```
"""

import re
import logging
from typing import Dict, List, Optional

from .config import config

logger = logging.getLogger("medexplain.prompts")


class AudienceAdaptivePrompt:
    """Standardized prompt template for generating audience-adaptive medical explanations.

    This class provides a consistent template for instructing language models
    to generate medical explanations adapted for four distinct healthcare audiences:
    physicians, nurses, patients, and caregivers. Each explanation is tailored
    to the specific needs, knowledge level, and communication preferences of
    the target audience.

    The template ensures that explanations are:
    - Appropriately technical for the audience
    - Focused on relevant aspects for each role
    - Formatted consistently for easy parsing

    Attributes:
        base_template: The standardized prompt template string that instructs
            the model to generate audience-specific explanations.
    """

    base_template: str = """Medical Information: {medical_content}

Transform this explanation for four distinct audiences. Ensure each explanation is self-contained and clearly labeled:

- For a Physician: Provide a technical, evidence-based explanation using precise medical terminology.

- For a Nurse: Focus on practical care implications, monitoring parameters, and patient education points.

- For a Patient: Use simple, jargon-free, and empathetic language. Focus on what the condition means and the next steps.

- For a Caregiver: Provide concrete tasks, symptoms to watch for, and when to seek help.

Generated Explanations:"""

    @classmethod
    def format_prompt(cls, medical_content: str) -> str:
        """Format the prompt template with medical content.

        Takes medical information and inserts it into the standardized template
        to create a complete prompt for the language model.

        Args:
            medical_content: The medical information to be adapted for different
                audiences. This should be the original medical content that needs
                to be explained.

        Returns:
            Formatted prompt string ready to be sent to a language model.

        Example:
            ```python
            medical_info = "Hypertension is high blood pressure that can damage organs."
            prompt = AudienceAdaptivePrompt.format_prompt(medical_info)
            ```
        """
        return cls.base_template.format(medical_content=medical_content)

    @staticmethod
    def parse_response(response: str) -> Dict[str, str]:
        """Parse the model response to extract audience-specific explanations.

        Extracts individual explanations for each audience from the model's
        response using robust regex patterns. The parsing logic looks for
        audience indicators in the text and separates the explanations accordingly.

        Uses multiple pattern variations to handle different response formats
        and includes fallback parsing for improved robustness.

        Args:
            response: Model response containing explanations for all audiences.
                Expected to contain sections labeled with audience names
                (physician, nurse, patient, caregiver).

        Returns:
            Dictionary mapping audience names to their respective explanations.
            Keys are audience names (e.g., 'physician', 'nurse', 'patient', 'caregiver').
            Values are the explanation text for each audience.

        Example:
            ```python
            model_response = \"\"\"For a Physician: Technical explanation...
            For a Patient: Simple explanation...\"\"\"
            explanations = AudienceAdaptivePrompt.parse_response(model_response)
            # Returns: {'physician': 'Technical explanation...', 'patient': 'Simple explanation...'}
            ```
        """
        explanations: Dict[str, str] = {}

        # Get supported audiences from configuration
        try:
            audiences = config.get_audiences()
        except Exception:
            # Fallback to default audiences if config fails
            audiences = ["physician", "nurse", "patient", "caregiver"]

        logger.debug(f"Parsing response for audiences: {audiences}")

        # Define robust regex patterns for each audience
        # These patterns are case-insensitive and handle various formatting variations
        patterns = {}

        for audience in audiences:
            # Create multiple pattern variations for robustness
            # Build audience alternation pattern for lookahead
            audience_alternation = "|".join([re.escape(aud) for aud in audiences])

            audience_patterns = [
                # Standard format: "For a Physician:" or "For the Physician:"
                rf"(?:for\s+(?:a|the)\s+{re.escape(audience)}\s*:)(.*?)(?=for\s+(?:a|the)\s+(?:{audience_alternation})\s*:|$)",
                # Alternative format: "Physician:" or "PHYSICIAN:"
                rf"(?:^|\n)\s*{re.escape(audience)}\s*:(.*?)(?=(?:^|\n)\s*(?:{audience_alternation})\s*:|$)",
                # Numbered format: "1. Physician:" or "- Physician:"
                rf"(?:^|\n)\s*(?:\d+\.|\-|\*)\s*{re.escape(audience)}\s*:(.*?)(?=(?:^|\n)\s*(?:\d+\.|\-|\*)\s*(?:{audience_alternation})\s*:|$)",
                # Section header format: "## Physician" or "### For Physician"
                rf"(?:^|\n)\s*#{1,4}\s*(?:for\s+)?{re.escape(audience)}\s*#{0,4}\s*\n?(.*?)(?=(?:^|\n)\s*#{1,4}\s*(?:for\s+)?(?:{audience_alternation})\s*#{0,4}|$)",
                # Bold/emphasis format: "**Physician:**" or "*For Physician:*"
                rf"(?:\*{{1,2}}|_{{1,2}})\s*(?:for\s+)?{re.escape(audience)}\s*:?\s*(?:\*{{1,2}}|_{{1,2}})(.*?)(?=(?:\*{{1,2}}|_{{1,2}})\s*(?:for\s+)?(?:{audience_alternation})\s*:?\s*(?:\*{{1,2}}|_{{1,2}})|$)",
            ]

            patterns[audience] = audience_patterns

        # Try each pattern for each audience and use the first match found
        for audience in audiences:
            explanation = None

            for pattern in patterns[audience]:
                try:
                    # Use DOTALL flag to match across newlines and IGNORECASE for case-insensitive matching
                    matches = re.finditer(pattern, response, re.DOTALL | re.IGNORECASE | re.MULTILINE)

                    for match in matches:
                        content = match.group(1).strip()
                        if content and len(content) > 10:  # Ensure we have substantial content
                            explanation = content
                            logger.debug(f"Found {audience} explanation using pattern: {pattern[:50]}...")
                            break

                    if explanation:
                        break

                except re.error as e:
                    logger.warning(f"Regex error for {audience} with pattern {pattern}: {e}")
                    continue

            if explanation:
                # Clean up the extracted explanation
                explanation = AudienceAdaptivePrompt._clean_explanation(explanation)
                explanations[audience] = explanation
            else:
                logger.warning(f"Could not extract explanation for {audience}")

        # Fallback: If we have fewer than expected explanations, try to extract missing ones
        if len(explanations) < len(audiences):
            logger.info(f"Only found {len(explanations)}/{len(audiences)} explanations, trying fallback parsing")
            fallback_explanations = AudienceAdaptivePrompt._fallback_parse(response, audiences)

            # Add any missing explanations from fallback
            for audience in audiences:
                if audience not in explanations and audience in fallback_explanations:
                    explanations[audience] = fallback_explanations[audience]
                    logger.info(f"Added {audience} explanation from fallback parsing")

        # Validate that we have explanations for all audiences
        missing_audiences = [aud for aud in audiences if aud not in explanations or not explanations[aud].strip()]
        if missing_audiences:
            logger.warning(f"Missing explanations for audiences: {missing_audiences}")

        logger.info(f"Successfully parsed explanations for {len(explanations)} audiences")
        return explanations

    @staticmethod
    def _clean_explanation(text: str) -> str:
        """
        Clean and normalize extracted explanation text

        Args:
            text: Raw extracted text

        Returns:
            Cleaned explanation text
        """
        # Remove leading/trailing whitespace
        text = text.strip()

        # Remove common prefixes that might be captured
        prefixes_to_remove = [
            r"^\s*:\s*",  # Leading colon
            r"^\s*\-\s*",  # Leading dash
            r"^\s*\*\s*",  # Leading asterisk
            r"^\s*\d+\.\s*",  # Leading number
        ]

        for prefix_pattern in prefixes_to_remove:
            text = re.sub(prefix_pattern, "", text, flags=re.MULTILINE)

        # Normalize whitespace - convert multiple spaces/newlines to single
        text = re.sub(r"\n\s*\n\s*\n", "\n\n", text)  # Max 2 consecutive newlines
        text = re.sub(r"[ \t]+", " ", text)  # Multiple spaces to single space

        # Remove markdown formatting artifacts
        text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)  # Remove bold/italic
        text = re.sub(r"_{1,2}([^_]+)_{1,2}", r"\1", text)  # Remove underscore emphasis
        text = re.sub(r"`([^`]+)`", r"\1", text)  # Remove code formatting

        return text.strip()

    @staticmethod
    def _fallback_parse(response: str, audiences: List[str]) -> Dict[str, str]:
        """
        Fallback parsing method using simple keyword search

        Args:
            response: Model response text
            audiences: List of audience names

        Returns:
            Dictionary with extracted explanations
        """
        explanations: Dict[str, str] = {}

        # Split response into sections based on audience keywords
        lines = response.split("\n")
        current_audience: Optional[str] = None
        current_text: List[str] = []

        for line in lines:
            line_lower = line.lower().strip()

            # Check if line contains an audience keyword
            matched_audience = None
            for audience in audiences:
                if audience.lower() in line_lower:
                    # Additional check to ensure it's likely a section header
                    if (
                        any(indicator in line_lower for indicator in ["for", ":", "#"])
                        or line_lower.strip() == audience.lower()
                    ):
                        matched_audience = audience
                        break

            if matched_audience:
                # Save previous section
                if current_audience and current_text:
                    content = "\n".join(current_text).strip()
                    if content:
                        explanations[current_audience] = content

                # Start new section
                current_audience = matched_audience
                current_text = []

                # Check if the explanation starts on the same line after colon
                colon_pos = line.find(":")
                if colon_pos != -1 and colon_pos < len(line) - 1:
                    remaining_text = line[colon_pos + 1 :].strip()
                    if remaining_text:
                        current_text.append(remaining_text)

            elif current_audience and line.strip():
                current_text.append(line.strip())

        # Add the last section
        if current_audience and current_text:
            content = "\n".join(current_text).strip()
            if content:
                explanations[current_audience] = content

        return explanations

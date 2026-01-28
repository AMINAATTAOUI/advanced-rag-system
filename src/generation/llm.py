"""
LLM client for text generation using Ollama.
Supports local LLM inference with streaming capabilities.
"""

from typing import List, Dict, Optional, Iterator
import ollama
from src.config import settings
from src.utils.logger import log


class LLMClient:
    """Client for interacting with Ollama LLM."""
    
    def __init__(
        self,
        model_name: str = None,
        base_url: str = None,
        temperature: float = None,
        max_tokens: int = None
    ):
        """
        Initialize LLM client.
        
        Args:
            model_name: Name of the Ollama model
            base_url: Ollama server URL
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.model_name = model_name or settings.ollama_model
        self.base_url = base_url or settings.ollama_base_url
        self.temperature = temperature or settings.temperature
        self.max_tokens = max_tokens or settings.max_tokens
        
        log.info(f"LLM Client initialized: model={self.model_name}, url={self.base_url}")
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to Ollama server."""
        try:
            # List available models
            response = ollama.list()
            available_models = [model.model for model in response.models]
            
            if self.model_name not in available_models:
                log.warning(
                    f"Model '{self.model_name}' not found. Available models: {available_models}. "
                    f"Please run: ollama pull {self.model_name}"
                )
            else:
                log.info(f"Successfully connected to Ollama. Model '{self.model_name}' is available.")
                
        except Exception as e:
            log.error(f"Failed to connect to Ollama: {e}")
            log.error("Make sure Ollama is running. Install from: https://ollama.ai")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Returns:
            Generated text
        """
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            )
            
            generated_text = response['message']['content']
            log.debug(f"Generated {len(generated_text)} characters")
            
            return generated_text
            
        except Exception as e:
            log.error(f"Error generating text: {e}")
            raise
    
    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Iterator[str]:
        """
        Generate text with streaming.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Yields:
            Text chunks as they are generated
        """
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            stream = ollama.chat(
                model=self.model_name,
                messages=messages,
                stream=True,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            )
            
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']
                    
        except Exception as e:
            log.error(f"Error in streaming generation: {e}")
            raise
    
    def generate_with_context(
        self,
        query: str,
        context: List[str],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate answer using retrieved context.
        
        Args:
            query: User query
            context: List of context strings
            system_prompt: System prompt
            temperature: Sampling temperature
            
        Returns:
            Generated answer
        """
        # Format context
        context_text = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(context)])
        
        # Create prompt
        prompt = f"""Context information is below:
---------------------
{context_text}
---------------------

Given the context information and not prior knowledge, answer the following question:
{query}

Answer:"""
        
        return self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature
        )
    
    def check_model_availability(self) -> bool:
        """
        Check if the configured model is available.
        
        Returns:
            True if model is available, False otherwise
        """
        try:
            response = ollama.list()
            available_models = [model.model for model in response.models]
            return self.model_name in available_models
        except:
            return False
    
    def pull_model(self):
        """Pull the model if not available."""
        try:
            log.info(f"Pulling model: {self.model_name}")
            ollama.pull(self.model_name)
            log.info(f"Model {self.model_name} pulled successfully")
        except Exception as e:
            log.error(f"Error pulling model: {e}")
            raise


if __name__ == "__main__":
    # Example usage
    client = LLMClient()
    
    # Check if model is available
    if not client.check_model_availability():
        print(f"Model {client.model_name} not available. Pulling...")
        # client.pull_model()
    
    # Generate text
    # response = client.generate("What is machine learning?")
    # print(f"Response: {response}")
    
    # Generate with streaming
    # print("Streaming response:")
    # for chunk in client.generate_stream("Explain deep learning in simple terms."):
    #     print(chunk, end="", flush=True)

"""
LLM client using LangChain — supports HuggingFace Inference API and Ollama.

LangChain Components Used:
- langchain_huggingface.ChatHuggingFace       -> Chat model via HuggingFace Inference API
- langchain_huggingface.HuggingFaceEndpoint   -> HF Inference endpoint wrapper
- langchain_ollama.ChatOllama                 -> Chat model wrapping local Ollama (fallback)
- langchain_core.messages                     -> HumanMessage, SystemMessage, AIMessage

Key LangChain patterns:
  # HuggingFace Inference API (default):
  endpoint = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct", ...)
  llm = ChatHuggingFace(llm=endpoint)
  result = llm.invoke([SystemMessage(...), HumanMessage(...)])

  # Ollama (fallback):
  llm = ChatOllama(model="llama3.1:8b", temperature=0.1)
"""

from typing import List, Dict, Optional, Iterator

# ── LangChain LLM backends ─────────────────────────────────────────────
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.config import settings
from src.utils.logger import log


class LLMClient:
    """
    LLM client powered by LangChain.

    Default backend: HuggingFace Inference API (cloud, no GPU required).
    Fallback backend: Ollama (local).

    Set LLM_BACKEND=ollama in env / .env to switch to Ollama.

    Exposes both LangChain-native access (self.llm, .invoke, .stream)
    and backward-compatible methods (generate, generate_stream, generate_with_context).
    """

    def __init__(
        self,
        model_name: str = None,
        base_url: str = None,
        temperature: float = None,
        max_tokens: int = None,
        backend: str = None,
    ):
        self.backend = (backend or settings.llm_backend).lower()
        self.temperature = temperature if temperature is not None else settings.temperature
        self.max_tokens = max_tokens or settings.max_tokens

        if self.backend == "ollama":
            self._init_ollama(model_name, base_url)
        else:
            self._init_huggingface(model_name)

    # ── HuggingFace Inference API ────────────────────────────────────
    def _init_huggingface(self, model_name: str = None):
        self.model_name = model_name or settings.hf_model
        self.base_url = None

        endpoint = HuggingFaceEndpoint(
            repo_id=self.model_name,
            task="text-generation",
            max_new_tokens=self.max_tokens,
            temperature=max(self.temperature, 0.01),  # HF needs > 0
            huggingfacehub_api_token=settings.hf_api_token,
        )
        self.llm = ChatHuggingFace(
            llm=endpoint,
            model_id=self.model_name,
        )
        log.info(
            f"LangChain ChatHuggingFace initialized: model={self.model_name} "
            f"(Inference API)"
        )

    # ── Ollama (local fallback) ──────────────────────────────────────
    def _init_ollama(self, model_name: str = None, base_url: str = None):
        from langchain_ollama import ChatOllama
        import ollama as _ollama

        self.model_name = model_name or settings.ollama_model
        self.base_url = base_url or settings.ollama_base_url

        self.llm = ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=self.temperature,
            num_predict=self.max_tokens,
        )
        log.info(
            f"LangChain ChatOllama initialized: model={self.model_name}, "
            f"url={self.base_url}"
        )

        # connection check
        try:
            response = _ollama.list()
            available = [m.model for m in response.models]
            if self.model_name not in available:
                log.warning(
                    f"Model '{self.model_name}' not found. Available: {available}. "
                    f"Run: ollama pull {self.model_name}"
                )
            else:
                log.info(f"Ollama model '{self.model_name}' is available.")
        except Exception as e:
            log.error(f"Failed to connect to Ollama: {e}")

    # ── backward-compatible generate ─────────────────────────────────
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text using LangChain ChatOllama.invoke()."""
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        # Build an ad-hoc LLM if overrides are needed
        llm = self._get_llm(temperature, max_tokens)

        try:
            # ── LangChain invoke ─────────────────────────────────
            result: AIMessage = llm.invoke(messages)
            text = result.content
            log.debug(f"Generated {len(text)} characters")
            return text
        except Exception as e:
            log.error(f"Error generating text: {e}")
            raise

    # ── backward-compatible streaming ────────────────────────────────
    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        """Stream text using LangChain ChatOllama.stream()."""
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        llm = self._get_llm(temperature, max_tokens)

        try:
            # ── LangChain stream ─────────────────────────────────
            for chunk in llm.stream(messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            log.error(f"Error in streaming generation: {e}")
            raise

    # ── backward-compatible context-based generation ─────────────────
    def generate_with_context(
        self,
        query: str,
        context: List[str],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        context_text = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(context)])
        prompt = (
            f"Context information is below:\n"
            f"---------------------\n{context_text}\n---------------------\n\n"
            f"Given the context information and not prior knowledge, "
            f"answer the following question:\n{query}\n\nAnswer:"
        )
        return self.generate(prompt=prompt, system_prompt=system_prompt, temperature=temperature)

    # ── utilities ────────────────────────────────────────────────────
    def check_model_availability(self) -> bool:
        """Check if the configured model is available."""
        if self.backend == "ollama":
            try:
                import ollama as _ollama
                response = _ollama.list()
                available = [m.model for m in response.models]
                return self.model_name in available
            except Exception:
                return False
        else:
            # HuggingFace Inference API — model is always "available" if token is set
            return bool(settings.hf_api_token)

    def pull_model(self):
        """Pull the model (Ollama only)."""
        if self.backend != "ollama":
            log.info("pull_model() is only relevant for Ollama backend. Skipping.")
            return
        try:
            import ollama as _ollama
            log.info(f"Pulling model: {self.model_name}")
            _ollama.pull(self.model_name)
            log.info(f"Model {self.model_name} pulled successfully")
        except Exception as e:
            log.error(f"Error pulling model: {e}")
            raise

    # ── helper: build LLM with optional overrides ────────────────────
    def _get_llm(self, temperature: Optional[float], max_tokens: Optional[int]):
        t = temperature if temperature is not None else self.temperature
        m = max_tokens if max_tokens is not None else self.max_tokens
        if t == self.temperature and m == self.max_tokens:
            return self.llm

        if self.backend == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=t,
                num_predict=m,
            )
        else:
            endpoint = HuggingFaceEndpoint(
                repo_id=self.model_name,
                task="text-generation",
                max_new_tokens=m,
                temperature=max(t, 0.01),
                huggingfacehub_api_token=settings.hf_api_token,
            )
            return ChatHuggingFace(llm=endpoint, model_id=self.model_name)


if __name__ == "__main__":
    client = LLMClient()
    print(f"Backend: {client.backend}, Model: {client.model_name}")
    if client.check_model_availability():
        print("Model is available!")
        # response = client.generate("What is machine learning?")
        # print(response)
    else:
        print(f"Model {client.model_name} not available.")
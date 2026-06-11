"""LCLM encoder backend — Phase 2.

Uses the LCLM encoder (Qwen3-Embedding-0.6B + adapter) to compress text
into latent embeddings. This is a learned compression model trained end-to-end
on 350B tokens (arXiv 2606.09659).

The encoder takes tokenized text → transformer forward pass → pooled latent
embeddings. The adapter projects encoder hidden states into the decoder's
embedding space.

For compression-only use (no decoder needed), we store the latent embeddings
as the compressed representation alongside the original text for expansion.

Usage:
    encoder = LCLMEncoder(
        encoder_path="path/to/encoder",  # or HF repo ID
        adapter_path="path/to/adapter",   # or HF repo ID
        compression_ratio=16,
        device="cuda",
    )
    result = encoder.compress("long text here...")
    # result.latent_embeddings: [num_latents, embed_dim] tensor
    # result.original_text: stored for expansion
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

try:
    import torch
except ImportError:
    raise ImportError(
        "LCLMEncoder requires PyTorch. Install with: "
        "pip install torch --index-url https://download.pytorch.org/whl/cu124"
    ) from None


@dataclass
class LCLMCompressionResult:
    """Result of LCLM encoder compression."""
    original_text: str
    latent_embeddings: torch.Tensor  # [num_latents, embed_dim]
    original_tokens: int
    num_latents: int
    compression_ratio: float
    key_entities: list[str]
    confidence: float
    model_name: str


class LCLMEncoder:
    """LCLM encoder backend using Qwen3-Embedding-0.6B + adapter.

    Loads the encoder model and adapter weights from HF checkpoints.
    The encoder compresses input text into latent embeddings that can
    be stored, searched, or fed to a decoder for generation.

    Args:
        encoder_path: Path to encoder model directory or HF repo ID.
            Defaults to "Qwen/Qwen3-Embedding-0.6B".
        adapter_path: Path to adapter weights directory or HF repo ID.
            If None, attempts to load from the LCLM checkpoint.
        compression_ratio: Target compression ratio (4, 8, or 16).
        encoder_window_size: Encoder window size in tokens (default: 1024).
        pooling: Pooling mode — "mean", "eos", or "concat".
        device: Device to run inference on ("cuda" or "cpu").
        dtype: Torch dtype for inference (bfloat16 recommended).
    """

    def __init__(
        self,
        encoder_path: str = "Qwen/Qwen3-Embedding-0.6B",
        adapter_path: str | None = None,
        compression_ratio: int = 16,
        encoder_window_size: int = 1024,
        pooling: str = "mean",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.encoder_path = encoder_path
        self.adapter_path = adapter_path
        self.compression_ratio = compression_ratio
        self.encoder_window_size = encoder_window_size
        self.pooling = pooling
        self.device = device
        self.dtype = dtype

        # Lazy loading — models loaded on first compress() call
        self._encoder_model = None
        self._encoder_tokenizer = None
        self._adapter = None
        self._loaded = False

    def _load_models(self):
        """Lazy-load encoder model, tokenizer, and adapter."""
        if self._loaded:
            return

        from transformers import AutoModel, AutoTokenizer

        print(f"[LCLM] Loading encoder: {self.encoder_path}")
        self._encoder_tokenizer = AutoTokenizer.from_pretrained(self.encoder_path)
        self._encoder_model = AutoModel.from_pretrained(
            self.encoder_path,
            torch_dtype=self.dtype,
        )
        self._encoder_model.to(self.device)
        self._encoder_model.eval()

        # Load adapter if path provided
        if self.adapter_path:
            self._adapter = self._load_adapter(self.adapter_path)
        else:
            print("[LCLM] No adapter path provided — using identity projection")
            self._adapter = None

        self._loaded = True
        print(f"[LCLM] Encoder loaded. Device: {self.device}, dtype: {self.dtype}")

    def _load_adapter(self, adapter_path: str):
        """Load adapter weights from safetensors file."""
        from safetensors.torch import load_file as load_safetensors

        adapter_file = Path(adapter_path) / "adapter.safetensors"
        if not adapter_file.exists():
            adapter_file = Path(adapter_path) / "adapter_model.safetensors"
        if not adapter_file.exists():
            raise FileNotFoundError(f"No adapter weights found in {adapter_path}")

        state_dict = load_safetensors(str(adapter_file))

        # Infer adapter dimensions from state dict
        # Adapter is typically a small MLP: encoder_dim → decoder_dim
        weight_keys = [k for k in state_dict if "weight" in k]
        if weight_keys:
            first_weight = state_dict[weight_keys[0]]
            encoder_dim = first_weight.shape[1]
            decoder_dim = first_weight.shape[0]
            print(f"[LCLM] Adapter: {encoder_dim} → {decoder_dim}")

        # Create a simple MLP adapter
        # The actual LCLM adapter architecture depends on adapter_type
        # For now, load as a simple linear projection
        adapter = torch.nn.Linear(encoder_dim, decoder_dim)
        adapter.load_state_dict({
            "weight": state_dict.get("weight", state_dict.get("linear.weight")),
            "bias": state_dict.get("bias", state_dict.get("linear.bias", None)),
        })
        adapter.to(self.device)
        adapter.eval()
        return adapter

    def compress(self, text: str) -> LCLMCompressionResult:
        """Compress text into latent embeddings.

        Args:
            text: Input text to compress.

        Returns:
            LCLMCompressionResult with latent embeddings and metadata.
        """
        self._load_models()

        # Tokenize
        tokens = self._encoder_tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
            padding=False,
        )
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)

        original_tokens = input_ids.shape[1]
        num_latents = max(1, math.ceil(original_tokens / self.compression_ratio))

        # Encode in windows if text is longer than window size
        if original_tokens <= self.encoder_window_size:
            # Single forward pass
            with torch.no_grad():
                outputs = self._encoder_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                hidden_states = outputs.last_hidden_state  # [1, seq_len, hidden_dim]

                # Pool to latent embeddings
                latents = self._pool(hidden_states, num_latents)
        else:
            # Windowed encoding for long documents
            latents = self._encode_windowed(input_ids, attention_mask, num_latents)

        # Apply adapter if available
        if self._adapter is not None:
            with torch.no_grad():
                latents = self._adapter(latents)

        # Extract entities from original text (for metadata)
        from .compressor import extract_entities, DEFAULT_IMPORTANT_TERMS
        key_entities = extract_entities(text, DEFAULT_IMPORTANT_TERMS)

        return LCLMCompressionResult(
            original_text=text,
            latent_embeddings=latents.cpu(),
            original_tokens=original_tokens,
            num_latents=num_latents,
            compression_ratio=original_tokens / max(1, num_latents),
            key_entities=key_entities,
            confidence=0.9,  # LCLM is learned — high confidence
            model_name=self.encoder_path,
        )

    def _pool(
        self, hidden_states: torch.Tensor, num_latents: int
    ) -> torch.Tensor:
        """Pool hidden states into latent embeddings.

        Supports mean, eos, and concat pooling modes.
        """
        if self.pooling == "mean":
            # Split into groups and take mean
            seq_len = hidden_states.shape[1]
            group_size = max(1, seq_len // num_latents)
            latents = []
            for i in range(0, seq_len, group_size):
                end = min(i + group_size, seq_len)
                latents.append(hidden_states[0, i:end].mean(dim=0))
            return torch.stack(latents[:num_latents], dim=0)

        elif self.pooling == "concat":
            # Concatenate groups
            seq_len = hidden_states.shape[1]
            group_size = max(1, seq_len // num_latents)
            latents = []
            for i in range(0, seq_len, group_size):
                end = min(i + group_size, seq_len)
                latents.append(hidden_states[0, i:end].reshape(-1))
            return torch.stack(latents[:num_latents], dim=0)

        else:  # eos or default
            # Use last N hidden states as latents
            return hidden_states[0, -num_latents:]

    def _encode_windowed(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_latents: int,
    ) -> torch.Tensor:
        """Encode long documents in overlapping windows."""
        seq_len = input_ids.shape[1]
        all_latents = []

        for start in range(0, seq_len, self.encoder_window_size):
            end = min(start + self.encoder_window_size, seq_len)
            window_ids = input_ids[:, start:end]
            window_mask = attention_mask[:, start:end]

            with torch.no_grad():
                outputs = self._encoder_model(
                    input_ids=window_ids,
                    attention_mask=window_mask,
                )
                window_hidden = outputs.last_hidden_state

            # Pool this window
            window_latents = max(1, math.ceil((end - start) / self.compression_ratio))
            pooled = self._pool(window_hidden, window_latents)
            all_latents.append(pooled)

        return torch.cat(all_latents, dim=0)[:num_latents]

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def unload(self):
        """Free model memory."""
        self._encoder_model = None
        self._encoder_tokenizer = None
        self._adapter = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

import pytest

from agent.core.llm_params import UnsupportedEffortError, _resolve_llm_params


def test_openai_max_effort_is_rejected_in_strict_mode():
    with pytest.raises(UnsupportedEffortError, match="OpenAI doesn't accept"):
        _resolve_llm_params(
            "openai/gpt-5.4",
            reasoning_effort="max",
            strict=True,
        )


def test_anthropic_minimal_effort_normalizes_to_low():
    params = _resolve_llm_params(
        "anthropic/claude-opus-4-7",
        reasoning_effort="minimal",
        strict=True,
    )

    assert params["model"] == "anthropic/claude-opus-4-7"
    assert params["thinking"] == {"type": "adaptive"}
    assert params["output_config"] == {"effort": "low"}


def test_resolve_ollama_params_adds_v1_and_uses_default_key(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.delenv("LOCAL_LLM_API_KEY", raising=False)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")

    params = _resolve_llm_params("ollama/llama3.1:8b")

    assert params == {
        "model": "openai/llama3.1:8b",
        "api_base": "http://localhost:11434/v1",
        "api_key": "sk-local-no-key-required",
    }


def test_resolve_vllm_params_keeps_existing_v1_and_trims_slash(monkeypatch):
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000/v1/")

    params = _resolve_llm_params("vllm/meta-llama/Llama-3.1-8B-Instruct")

    assert params["model"] == "openai/meta-llama/Llama-3.1-8B-Instruct"
    assert params["api_base"] == "http://localhost:8000/v1"
    assert params["api_key"] == "sk-local-no-key-required"


def test_resolve_lm_studio_params_uses_api_key_override(monkeypatch):
    monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234")
    monkeypatch.setenv("LMSTUDIO_API_KEY", "local-secret")
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://localhost:9999")
    monkeypatch.setenv("LOCAL_LLM_API_KEY", "shared-secret")

    params = _resolve_llm_params("lm_studio/google/gemma-3-4b")

    assert params["model"] == "openai/google/gemma-3-4b"
    assert params["api_base"] == "http://127.0.0.1:1234/v1"
    assert params["api_key"] == "local-secret"


def test_resolve_local_params_uses_shared_fallback_env(monkeypatch):
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://localhost:9000/v1/")
    monkeypatch.setenv("LOCAL_LLM_API_KEY", "shared-local-secret")

    params = _resolve_llm_params("vllm/custom-model")

    assert params["model"] == "openai/custom-model"
    assert params["api_base"] == "http://localhost:9000/v1"
    assert params["api_key"] == "shared-local-secret"


def test_resolve_llamacpp_params_strips_provider_prefix(monkeypatch):
    monkeypatch.delenv("LLAMACPP_API_KEY", raising=False)
    monkeypatch.setenv("LLAMACPP_BASE_URL", "http://localhost:8080")

    params = _resolve_llm_params("llamacpp/unsloth/Qwen3.5-2B")

    assert params["model"] == "openai/unsloth/Qwen3.5-2B"
    assert params["api_base"] == "http://localhost:8080/v1"


def test_local_params_reject_reasoning_effort_in_strict_mode():
    with pytest.raises(UnsupportedEffortError, match="reasoning_effort"):
        _resolve_llm_params("ollama/llama3.1", reasoning_effort="high", strict=True)


def test_local_params_drop_reasoning_effort_in_non_strict_mode():
    params = _resolve_llm_params(
        "ollama/llama3.1",
        reasoning_effort="high",
        strict=False,
    )

    assert params["model"] == "openai/llama3.1"
    assert "reasoning_effort" not in params
    assert "extra_body" not in params


def test_openai_compat_prefix_is_not_a_local_escape_hatch():
    with pytest.raises(ValueError, match="Unsupported local model id"):
        _resolve_llm_params("openai-compat/custom-model")


def test_empty_local_model_id_is_not_treated_as_hf_router():
    with pytest.raises(ValueError, match="Unsupported local model id"):
        _resolve_llm_params("ollama/")

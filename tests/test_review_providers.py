"""Public provider configuration and fail-open tracing, without SDK or network calls."""

import builtins
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest


REVIEW_PATH = Path(__file__).resolve().parents[1] / "review.py"
ENV_PREFIXES = (
    "OPENAI_", "AZURE_OPENAI_", "AZURE_FOUNDRY_", "VERTEX_", "GOOGLE_",
    "GEMINI_", "GLM_", "REVIEW_", "LANGFUSE_", "ALLOW_CLAUDE",
)
READY_ENV = {
    "openai": {"OPENAI_API_KEY": "test-key"},
    "azure": {
        "AZURE_OPENAI_ENDPOINT": "https://azure.example.test/",
        "AZURE_OPENAI_API_KEY": "test-key",
    },
    "foundry": {
        "AZURE_FOUNDRY_ENDPOINT": "https://foundry.example.test/models",
        "AZURE_FOUNDRY_API_KEY": "test-key",
    },
    "gemini": {"GOOGLE_GEMINI_API_KEY": "test-key"},
    "glm": {"GLM_API_KEY": "test-key"},
    "gemini-vertex": {"VERTEX_PROJECT": "test-project"},
    "claude-vertex": {"VERTEX_PROJECT": "test-project"},
}
REQUIRED_CASES = [
    (provider, name) for provider, values in READY_ENV.items() for name in values
]


@pytest.fixture
def review(monkeypatch, tmp_path):
    # Import after isolation so both live env reads and module-level defaults are
    # independent of the developer's credentials. Never inspect local key files.
    # Track absence as well: callback setup writes this env variable directly.
    monkeypatch.setenv("LANGFUSE_TRACING_ENVIRONMENT", os.environ.get("LANGFUSE_TRACING_ENVIRONMENT", ""))
    for name in tuple(os.environ):
        if name.startswith(ENV_PREFIXES) or name == "GITHUB_ACTIONS":
            monkeypatch.delenv(name)
    monkeypatch.syspath_prepend(str(REVIEW_PATH.parent))
    spec = importlib.util.spec_from_file_location("review_provider_tests", REVIEW_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ZAI_KEY_FILE", tmp_path / "zai-key")
    module.make_model.cache_clear()
    yield module
    module.make_model.cache_clear()


@pytest.fixture
def fake_clients(monkeypatch):
    calls = []

    class FakeModel:
        def __init__(self, client, kwargs):
            self.client = client
            self.kwargs = kwargs
            self.probed = False

        def invoke(self, *args, **kwargs):
            self.probed = True
            raise AssertionError("constructing a model must not make a capability probe")

    def constructor(client):
        def create(**kwargs):
            model = FakeModel(client, kwargs)
            calls.append(model)
            return model
        return create

    modules = {
        "langchain_openai": ("ChatOpenAI", "AzureChatOpenAI"),
        "langchain_google_genai": ("ChatGoogleGenerativeAI",),
        "langchain_google_vertexai": ("ChatVertexAI",),
        "langchain_google_vertexai.model_garden": ("ChatAnthropicVertex",),
    }
    for name, clients in modules.items():
        module = types.ModuleType(name)
        if name == "langchain_google_vertexai":
            module.__path__ = []
        for client in clients:
            setattr(module, client, constructor(client))
        monkeypatch.setitem(sys.modules, name, module)
    yield calls
    assert not any(model.probed for model in calls)


def set_env(monkeypatch, values):
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_public_provider_set_and_openai_defaults(review):
    assert set(review.PROVIDERS) == set(READY_ENV)
    assert review.DEFAULT_PROVIDER == "openai"
    assert review.PROVIDERS["openai"]["model"] is None
    assert review.PROVIDERS["openai"]["concurrency"] == 8
    assert review.provider_caps("openai") == (50_000, 120_000)
    assert review.provider_caps("openai") == review.provider_caps("azure")


def test_cloud_endpoints_and_projects_have_no_default(review):
    assert review.AZURE_ENDPOINT is None
    assert review.FOUNDRY_ENDPOINT is None
    assert review.VERTEX_PROJECT is None
    for provider in ("azure", "foundry"):
        assert review.PROVIDERS[provider]["endpoint"] is None
    for provider in ("gemini-vertex", "claude-vertex"):
        assert review.PROVIDERS[provider]["project"] is None


@pytest.mark.parametrize("model_arg,model_env,expected", [
    (None, "env-model", "env-model"),
    ("cli-model", "env-model", "cli-model"),
    ("cli-model", None, "cli-model"),
    (None, "  env-model  ", "env-model"),
])
def test_openai_model_precedence(review, model_arg, model_env, expected):
    env = {} if model_env is None else {"OPENAI_MODEL": model_env}
    assert review.resolve_tuning("openai", model_arg, None, None, env) == (expected, "", 8)


@pytest.mark.parametrize("model_env", [None, "", " \t "])
def test_openai_missing_model_has_actionable_prerequisite_error(review, capsys, model_env):
    env = {} if model_env is None else {"OPENAI_MODEL": model_env}
    with pytest.raises(SystemExit) as exc:
        review.resolve_tuning("openai", None, None, None, env)
    assert exc.value.code == 3
    output = capsys.readouterr()
    assert output.out == ""
    assert len(output.err.splitlines()) == 1
    assert "OPENAI_MODEL" in output.err and "--model" in output.err


def test_openai_tuning_honors_cli_then_environment(review):
    env = {"OPENAI_MODEL": "env-model", "REVIEW_LIGHT_MODEL": "env-light",
           "REVIEW_CONCURRENCY": "4"}
    assert review.resolve_tuning("openai", None, None, None, env) == (
        "env-model", "env-light", 4)
    assert review.resolve_tuning("openai", "cli-model", "cli-light", 2, env) == (
        "cli-model", "cli-light", 2)


@pytest.mark.parametrize("provider,missing", REQUIRED_CASES)
@pytest.mark.parametrize("missing_value", [None, "", " \t "])
def test_missing_provider_env_fails_before_sdk_construction(
    review, fake_clients, monkeypatch, capsys, provider, missing, missing_value,
):
    set_env(monkeypatch, {k: v for k, v in READY_ENV[provider].items() if k != missing})
    if missing_value is not None:
        monkeypatch.setenv(missing, missing_value)
    for call in (
        lambda: review.provider_key(provider),
        lambda: review.make_model(provider, "test-model", 75),
    ):
        with pytest.raises(SystemExit) as exc:
            call()
        assert exc.value.code == 3
        output = capsys.readouterr()
        assert output.out == ""
        assert len(output.err.splitlines()) == 1
        assert missing in output.err
    assert fake_clients == []


@pytest.mark.parametrize("provider", ["gemini-vertex", "claude-vertex"])
def test_vertex_requires_explicit_project_not_generic_cloud_alias(
    review, monkeypatch, capsys, provider,
):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "other-project")
    with pytest.raises(SystemExit) as exc:
        review.provider_key(provider)
    assert exc.value.code == 3
    assert "VERTEX_PROJECT" in capsys.readouterr().err


@pytest.mark.parametrize("provider", list(READY_ENV))
def test_provider_auth_uses_explicit_environment(review, monkeypatch, provider):
    set_env(monkeypatch, {k: f" {v} \n" for k, v in READY_ENV[provider].items()})
    expected = "" if provider.endswith("-vertex") else "test-key"
    assert review.provider_key(provider) == expected


@pytest.mark.parametrize("primary,alias,expected", [
    (None, "alias-key", "alias-key"),
    ("primary-key", "alias-key", "primary-key"),
    ("   ", " alias-key ", "alias-key"),
])
def test_gemini_key_alias_and_precedence(review, monkeypatch, primary, alias, expected):
    if primary is not None:
        monkeypatch.setenv("GOOGLE_GEMINI_API_KEY", primary)
    monkeypatch.setenv("GEMINI_API_KEY", alias)
    assert review.provider_key("gemini") == expected


@pytest.mark.parametrize("env_key,expected", [(None, "file-key"), ("env-key", "env-key")])
def test_glm_documented_key_file_fallback_and_env_precedence(
    review, monkeypatch, env_key, expected,
):
    review.ZAI_KEY_FILE.write_text(" file-key \n")
    if env_key is not None:
        monkeypatch.setenv("GLM_API_KEY", env_key)
    assert review.provider_key("glm") == expected


@pytest.mark.parametrize("base_url", [None, "", "   ", " https://models.example.test/v1 "])
def test_openai_constructor_uses_model_key_and_optional_base_url(
    review, fake_clients, monkeypatch, base_url,
):
    monkeypatch.setenv("OPENAI_API_KEY", " test-key ")
    if base_url is not None:
        monkeypatch.setenv("OPENAI_BASE_URL", base_url)
    model = review.make_model("openai", "custom-chat", 75)
    expected = {"model": "custom-chat", "api_key": "test-key", "max_retries": 3,
                "timeout": 75, "use_responses_api": False, "temperature": 0.1}
    if base_url and base_url.strip():
        expected["base_url"] = base_url.strip()
    else:
        # The real SDK reads OPENAI_BASE_URL itself and treats "" as a base URL.
        assert "OPENAI_BASE_URL" not in os.environ
    assert model.client == "ChatOpenAI"
    assert model.kwargs == expected
    assert fake_clients == [model]


@pytest.mark.parametrize("provider", ["openai", "azure"])
def test_openai_and_azure_constructors_use_shared_policy(
    review, fake_clients, monkeypatch, provider,
):
    set_env(monkeypatch, READY_ENV[provider])
    calls = []

    def shared_policy(selected_provider, selected_model, timeout):
        calls.append((selected_provider, selected_model, timeout))
        return {"temperature": 0.25, "max_retries": 2, "timeout": 91}

    monkeypatch.setattr(review, "openai_model_kwargs", shared_policy)
    model = review.make_model(provider, "custom-deployment", 75)
    assert calls == [(provider, "custom-deployment", 75)]
    assert model.kwargs["temperature"] == 0.25
    assert model.kwargs["max_retries"] == 2 and model.kwargs["timeout"] == 91
    assert model.kwargs["api_key"] == "test-key"
    if provider == "azure":
        assert model.client == "AzureChatOpenAI"
        assert model.kwargs["azure_endpoint"] == READY_ENV[provider]["AZURE_OPENAI_ENDPOINT"]
        assert model.kwargs["azure_deployment"] == "custom-deployment"
    else:
        assert model.client == "ChatOpenAI"
        assert model.kwargs["model"] == "custom-deployment"


@pytest.mark.parametrize("provider,prefix", [("openai", "OPENAI"), ("azure", "AZURE_OPENAI")])
@pytest.mark.parametrize("model,settings,policy", [
    ("custom-chat", {}, {"use_responses_api": False, "temperature": 0.1}),
    ("gpt-6-test", {}, {"use_responses_api": True, "reasoning": {"effort": "medium"}}),
    ("custom-model", {"RESPONSES_API": "1"},
     {"use_responses_api": True, "reasoning": {"effort": "medium"}}),
    ("gpt-6-test", {"RESPONSES_API": "0"}, {"use_responses_api": False, "temperature": 0.1}),
    ("custom-chat", {"NO_TEMPERATURE": "1"}, {"use_responses_api": False}),
    ("gpt-6-test", {"REASONING_EFFORT": "high"},
     {"use_responses_api": True, "reasoning": {"effort": "high"}}),
    ("custom-chat", {"SERVICE_TIER": " priority "},
     {"use_responses_api": False, "temperature": 0.1,
      "extra_body": {"service_tier": "priority"}}),
])
def test_shared_openai_policy_supports_provider_scoped_overrides(
    review, monkeypatch, provider, prefix, model, settings, policy,
):
    set_env(monkeypatch, {f"{prefix}_{key}": value for key, value in settings.items()})
    assert review.openai_model_kwargs(provider, model, 75) == {
        "max_retries": 3, "timeout": 75, **policy,
    }


@pytest.mark.parametrize("provider,other_prefix", [("openai", "AZURE_OPENAI"), ("azure", "OPENAI")])
def test_shared_policy_does_not_leak_other_provider_settings(review, monkeypatch, provider, other_prefix):
    set_env(monkeypatch, {
        f"{other_prefix}_RESPONSES_API": "1", f"{other_prefix}_NO_TEMPERATURE": "1",
        f"{other_prefix}_REASONING_EFFORT": "high", f"{other_prefix}_SERVICE_TIER": "priority",
    })
    assert review.openai_model_kwargs(provider, "custom-chat", 75) == {
        "max_retries": 3, "timeout": 75, "use_responses_api": False, "temperature": 0.1,
    }


def test_foundry_constructor_uses_explicit_endpoint(review, fake_clients, monkeypatch):
    set_env(monkeypatch, READY_ENV["foundry"])
    model = review.make_model("foundry", "custom-deployment", 75)
    assert model.client == "ChatOpenAI"
    assert model.kwargs == {
        "model": "custom-deployment", "api_key": "test-key",
        "base_url": READY_ENV["foundry"]["AZURE_FOUNDRY_ENDPOINT"],
        "default_query": {"api-version": review.PROVIDERS["foundry"]["api_version"]},
        "temperature": 0.1, "max_retries": 3, "timeout": 75,
    }


@pytest.mark.parametrize("provider,client,model_field", [
    ("gemini-vertex", "ChatVertexAI", "model"),
    ("claude-vertex", "ChatAnthropicVertex", "model_name"),
])
def test_vertex_constructors_work_with_project_and_adc_without_unlock_flags(
    review, fake_clients, monkeypatch, provider, client, model_field,
):
    monkeypatch.setenv("VERTEX_PROJECT", " test-project ")
    monkeypatch.setitem(review.PROVIDERS[provider], "location", "test-region")
    model = review.make_model(provider, "enabled-model", 75)
    expected = {model_field: "enabled-model", "project": "test-project",
                "location": "test-region", "timeout": 75, "max_retries": 3}
    if provider == "gemini-vertex":
        expected["temperature"] = 0.1
    assert model.client == client
    assert model.kwargs == expected
    assert fake_clients == [model]


def test_claude_vertex_retains_prompt_cache_without_affecting_other_providers(review):
    assert review.PROVIDERS["claude-vertex"]["prompt_cache"] is True
    assert review.cached_user("review context", "claude-vertex") == [{
        "type": "text", "text": "review context", "cache_control": {"type": "ephemeral"},
    }]
    for provider in set(READY_ENV) - {"claude-vertex"}:
        assert review.cached_user("review context", provider) == "review context"


@pytest.fixture
def trace_sdk(monkeypatch):
    state = types.SimpleNamespace(failure=None, events=[], handler=object())

    def event(stage, payload=None):
        state.events.append((stage, payload))
        if state.failure == stage:
            raise RuntimeError(f"trace {stage} unavailable")

    class Observation:
        def __enter__(self):
            event("enter")
            return self

        def __exit__(self, error_type, error, traceback):
            event("exit", error)
            return False

    class Client:
        def start_as_current_observation(self, **kwargs):
            event("start", kwargs)
            return Observation()

        def score_current_trace(self, **kwargs):
            event("score", kwargs)

    client = Client()

    def create_client(**kwargs):
        event("client", {**kwargs, "environment": os.environ.get("LANGFUSE_TRACING_ENVIRONMENT")})
        return client

    def create_handler():
        event("handler")
        return state.handler

    def get_client():
        event("get")
        return client

    main = types.ModuleType("langfuse")
    main.__path__ = []
    main.Langfuse = create_client
    main.get_client = get_client
    callbacks = types.ModuleType("langfuse.langchain")
    callbacks.CallbackHandler = create_handler
    monkeypatch.setitem(sys.modules, "langfuse", main)
    monkeypatch.setitem(sys.modules, "langfuse.langchain", callbacks)
    return state


@pytest.mark.parametrize("explicit,actions,expected", [
    (" staging ", "true", "staging"),
    (None, "true", "github-actions"),
    (None, "TRUE", "github-actions"),
    ("  ", "true", "github-actions"),
    (None, "false", "local"),
    (None, None, "local"),
])
def test_tracing_environment_precedence(review, monkeypatch, explicit, actions, expected):
    if explicit is not None:
        monkeypatch.setenv("LANGFUSE_TRACING_ENVIRONMENT", explicit)
    if actions is not None:
        monkeypatch.setenv("GITHUB_ACTIONS", actions)
    assert review.tracing_environment() == expected


@pytest.mark.parametrize("single_key", [None, "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"])
def test_trace_callbacks_without_both_keys_never_import_sdk(review, monkeypatch, single_key):
    if single_key is not None:
        monkeypatch.setenv(single_key, "test-key")
    attempted = []
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "langfuse" or name.startswith("langfuse."):
            attempted.append(name)
            raise AssertionError("disabled tracing must not import its SDK")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert review.make_trace_callbacks("test-thread") == []
    assert attempted == []
    assert os.environ["LANGFUSE_TRACING_ENVIRONMENT"] == "local"


def test_trace_callbacks_initialize_masked_client_before_handler(review, trace_sdk, monkeypatch):
    set_env(monkeypatch, {
        "LANGFUSE_PUBLIC_KEY": "test-public", "LANGFUSE_SECRET_KEY": "test-secret",
        "GITHUB_ACTIONS": "true",
    })
    assert review.make_trace_callbacks("test-thread") == [trace_sdk.handler]
    assert trace_sdk.events == [
        ("client", {"mask": review.langfuse_mask, "environment": "github-actions"}),
        ("handler", None),
    ]


@pytest.mark.parametrize("failure", ["client", "handler"])
def test_trace_callback_initialization_failure_is_optional(
    review, trace_sdk, monkeypatch, capsys, failure,
):
    set_env(monkeypatch, {"LANGFUSE_PUBLIC_KEY": "test-public", "LANGFUSE_SECRET_KEY": "test-secret"})
    trace_sdk.failure = failure
    assert review.make_trace_callbacks("test-thread") == []
    assert failure in [stage for stage, _ in trace_sdk.events]
    assert "[trace] init skipped:" in capsys.readouterr().err


def test_trace_mask_redacts_nested_personal_data_without_mutating_input(review):
    original = {
        "email": "reader@example.test",
        "nested": [{"text": "writer@example.test id 123456789"}, ("7654321", "safe")],
        "short": "123456", "count": 42,
    }
    assert review.langfuse_mask(original) == {
        "email": "[email]", "nested": [{"text": "[email] id [num]"}, ["[num]", "safe"]],
        "short": "123456", "count": 42,
    }
    assert original["email"] == "reader@example.test"
    assert original["nested"][1] == ("7654321", "safe")


@pytest.mark.parametrize("failure", [None, "get", "start", "enter", "exit", "score"])
def test_trace_faults_preserve_successful_graph_result_without_rerunning(
    review, trace_sdk, failure,
):
    trace_sdk.failure = failure
    result = {"result": {"stats": {"raw": 0, "kept": 0, "killed": 0, "downgraded": 0}}}

    def run_graph():
        trace_sdk.events.append(("graph", None))
        return result

    assert review.run_traced(run_graph) is result
    stages = [stage for stage, _ in trace_sdk.events]
    expected = {
        "get": ["get", "graph"],
        "start": ["get", "start", "graph"],
        "enter": ["get", "start", "enter", "graph"],
    }.get(failure, ["get", "start", "enter", "graph", "exit"])
    assert [stage for stage in stages if stage != "score"] == expected
    if failure is not None:
        assert failure in stages
    if failure == "score":
        assert stages.count("score") == 1


@pytest.mark.parametrize("trace_failure", [None, "enter", "exit"])
def test_graph_exception_propagates_once_even_when_tracing_fails(review, trace_sdk, trace_failure):
    trace_sdk.failure = trace_failure
    error = ValueError("review execution failed")

    def run_graph():
        trace_sdk.events.append(("graph", None))
        raise error

    with pytest.raises(ValueError) as exc:
        review.run_traced(run_graph)
    assert exc.value is error
    stages = [stage for stage, _ in trace_sdk.events]
    assert stages.count("graph") == 1
    assert "score" not in stages
    if trace_failure != "enter":
        assert ("exit", error) in trace_sdk.events

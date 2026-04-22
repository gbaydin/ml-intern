import io
import warnings


def test_plain_research_output_is_append_only(monkeypatch):
    from agent.utils import terminal_display as td

    original_plain = td.is_plain_output()
    buffer = io.StringIO()
    monkeypatch.setattr(td.get_console(), "file", buffer)
    monkeypatch.setattr(td, "_subagent_display", td.SubAgentDisplayManager())

    try:
        td.set_plain_output(True)
        td.print_tool_log(
            "research",
            "Starting research sub-agent...",
            agent_id="agent-1",
            label="research: sample",
        )
        td.print_tool_log(
            "research",
            '▸ hf_repo_files  {"operation": "list"}',
            agent_id="agent-1",
            label="research: sample",
        )
        td.print_tool_log("research", "tokens:1200", agent_id="agent-1")
        td.print_tool_log("research", "tools:1", agent_id="agent-1")
        td.print_tool_log("research", "Research complete.", agent_id="agent-1")
    finally:
        td.set_plain_output(original_plain)

    output = buffer.getvalue()
    assert "research: sample" in output
    assert "hf_repo_files" in output
    assert "1 tool uses" in output
    assert not any(seq in output for seq in ("\x1b[A", "\x1b[K", "\r"))


def test_pydantic_serializer_warning_is_filtered():
    import agent.main as main

    main._configure_warning_filters()

    with warnings.catch_warnings(record=True) as caught:
        warnings.warn_explicit(
            "Pydantic serializer warnings:\n  PydanticSerializationUnexpectedValue(...)",
            UserWarning,
            filename="/tmp/pydantic/main.py",
            lineno=475,
            module="pydantic.main",
        )

    assert caught == []

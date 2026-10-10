"""Streamlit AppTest integration test for SchemaSense UI workflows."""

import socket
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def is_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, ConnectionRefusedError):
        return False


def test_ui_initial_render():
    """Verifies that the application renders initial state cleanly without auth barriers."""
    at = AppTest.from_file(APP_PATH, default_timeout=20)
    at.run()
    assert not at.exception
    # Verify default welcome message in session_state and rendered chat
    assert any("Welcome!" in msg["content"] for msg in at.session_state.messages)


def test_ui_live_connection_and_disconnect():
    """Verifies UI connection, least-privilege reporting, and disconnect lifecycle."""
    if not is_port_open("127.0.0.1", 5433):
        pytest.skip("PostgreSQL test container is not available.")

    at = AppTest.from_file(APP_PATH, default_timeout=20)
    at.run()
    assert not at.exception

    # Locate inputs inside the sidebar form
    text_inputs = [ti for ti in at.sidebar.text_input]
    assert len(text_inputs) >= 6

    text_inputs[1].set_value("127.0.0.1")
    text_inputs[2].set_value("5433")
    text_inputs[3].set_value("schemasense_test")
    text_inputs[4].set_value("schemasense_readonly")
    text_inputs[5].set_value("readonly_pass")

    # Submit form via connect button
    connect_btn = next(b for b in at.sidebar.button if "Connect" in b.label)
    connect_btn.click()
    at.run()

    assert not at.exception
    assert at.session_state.get("db_engine") is not None
    assert at.session_state.get("db_schema") is not None
    assert "customers" in at.session_state.db_schema.tables
    assert at.session_state.privilege_report.is_privileged is False

    # Verify disconnect button works
    disconnect_btn = next(
        (b for b in at.sidebar.button if "Disconnect" in b.label), None
    )
    assert disconnect_btn is not None
    disconnect_btn.click()
    at.run()

    assert not at.exception
    assert at.session_state.get("db_engine") is None
    assert at.session_state.get("db_schema") is None


def test_ui_privileged_connection_warning():
    """Verifies that connecting as superuser/admin renders a security advisory in the UI."""
    if not is_port_open("127.0.0.1", 5433):
        pytest.skip("PostgreSQL test container is not available.")

    at = AppTest.from_file(APP_PATH, default_timeout=20)
    at.run()

    text_inputs = [ti for ti in at.sidebar.text_input]
    text_inputs[1].set_value("127.0.0.1")
    text_inputs[2].set_value("5433")
    text_inputs[3].set_value("schemasense_test")
    text_inputs[4].set_value("test_admin")
    text_inputs[5].set_value("test_admin_pass")

    # Secure Connect button
    connect_btn = next(b for b in at.sidebar.button if "Connect" in b.label)
    connect_btn.click()
    at.run()

    assert not at.exception
    assert at.session_state.privilege_report.is_privileged is True
    # Verify advisory warning is displayed in chat
    assert any(
        "Security Advisory" in msg["content"] for msg in at.session_state.messages
    )


def test_ui_enforce_least_privilege_blocks_admin():
    """Verifies that selecting enforce least-privilege rejects admin credentials in the UI."""
    if not is_port_open("127.0.0.1", 5433):
        pytest.skip("PostgreSQL test container is not available.")

    at = AppTest.from_file(APP_PATH, default_timeout=20)
    at.run()

    text_inputs = [ti for ti in at.sidebar.text_input]
    text_inputs[1].set_value("127.0.0.1")
    text_inputs[2].set_value("5433")
    text_inputs[3].set_value("schemasense_test")
    text_inputs[4].set_value("test_admin")
    text_inputs[5].set_value("test_admin_pass")

    # Check the enforce least-privilege checkbox
    checkboxes = [cb for cb in at.sidebar.checkbox]
    if checkboxes:
        checkboxes[0].set_value(True)

    connect_btn = next(b for b in at.sidebar.button if "Connect" in b.label)
    connect_btn.click()
    at.run()

    assert not at.exception
    # Engine should NOT be set because connection was rejected
    assert at.session_state.get("db_engine") is None
    assert any(
        "Connection Blocked" in msg["content"] for msg in at.session_state.messages
    )


def test_ui_disconnected_state_telemetry():
    """Verifies that runtime telemetry derives authentic status values when disconnected."""
    at = AppTest.from_file(APP_PATH, default_timeout=20)
    at.run()
    assert not at.exception

    # Status indicators in console header and telemetry strip
    markdown_texts = [m.value for m in at.markdown]
    assert any("Disconnected" in text for text in markdown_texts)
    assert any("Role: Not checked" in text for text in markdown_texts)
    assert any("AST Guard: Active (sqlglot)" in text for text in markdown_texts)
    assert any("Pool Limit: 4 Concurrent" in text for text in markdown_texts)
    assert any(
        "Operating Mode: <strong>General SQL guidance</strong>" in text
        for text in markdown_texts
    )


def test_ui_failed_connection_missing_fields():
    """Verifies that submitting incomplete database credentials triggers an immediate validation alert."""
    at = AppTest.from_file(APP_PATH, default_timeout=20)
    at.run()

    # Click Connect without entering credentials
    connect_btn = next(b for b in at.sidebar.button if "Connect" in b.label)
    connect_btn.click()
    at.run()

    assert not at.exception
    assert at.session_state.get("db_engine") is None
    assert any(
        "All connection parameters are mandatory" in err.value for err in at.error
    )


def test_ui_offline_sql_guard_sandbox_acceptance():
    """Verifies that the SQL Guard sandbox validates read-only queries even when offline."""
    at = AppTest.from_file(APP_PATH, default_timeout=20)
    at.run()

    # Enter a read-only SELECT query into the SQL buffer
    at.text_area[0].set_value("SELECT id, name FROM users WHERE active = true;")
    run_btn = next(b for b in at.button if "Execute Guarded Query" in b.label)
    run_btn.click()
    at.run()

    assert not at.exception
    assert any("AST Validation Passed" in s.value for s in at.success)


def test_ui_offline_sql_guard_sandbox_rejection():
    """Verifies that the SQL Guard sandbox blocks destructive queries even when offline."""
    at = AppTest.from_file(APP_PATH, default_timeout=20)
    at.run()

    # Enter a destructive DROP query
    at.text_area[0].set_value("DROP TABLE sensitive_data;")
    run_btn = next(b for b in at.button if "Execute Guarded Query" in b.label)
    run_btn.click()
    at.run()

    assert not at.exception
    assert any("Security Block (SQLGuard)" in err.value for err in at.error)


def test_ui_ai_mode_selection_and_chat_reset():
    """Verifies that the UI renders the AI mode selection and chat clearing operates cleanly."""
    at = AppTest.from_file(APP_PATH, default_timeout=20)
    at.run()

    # Radio button for Response Mode
    assert len(at.radio) >= 1
    assert "Streaming Dialogue" in at.radio[0].options
    assert "Structured Schema Reasoning (Pydantic)" in at.radio[0].options

    # Clear chat button exists
    clear_btn = next((b for b in at.button if "Clear Chat" in b.label), None)
    assert clear_btn is not None
    clear_btn.click()
    at.run()

    assert not at.exception
    assert len(at.session_state.messages) == 1
    assert "Welcome!" in at.session_state.messages[0]["content"]

"""Centralized Streamlit session-state initialization."""

from typing import Any

import streamlit as st


def initialize_session_state() -> None:
    """Initialize all state shared by UI components."""

    defaults: dict[str, Any] = {
        "messages": [],
        "selected_document_ids": [],
        "current_mode": "agent",
        "processed_uploads": set(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

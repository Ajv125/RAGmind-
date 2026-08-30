#!/bin/bash
# Convenience launcher for local development.
export GEMINI_API_KEY="${GEMINI_API_KEY:-PASTE_KEY_HERE_OR_EXPORT_IT}"
streamlit run app/streamlit_app.py

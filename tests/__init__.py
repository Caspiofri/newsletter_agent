import os

# Tests must never upload traces to the real LangSmith project. LANGSMITH_ wins over LANGCHAIN_ in the SDK;
# TRACING_V2 is checked before TRACING, so both are set. The legacy LANGCHAIN_TRACING from .env is neutralised
# because langchain_core rejects it as "tracer V1" whenever v2 tracing is off.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGSMITH_TRACING_V2"] = "false"
os.environ["LANGCHAIN_TRACING"] = "false"

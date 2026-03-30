"""Manual smoke test for the old ad hoc run_query flow."""

import os
import unittest


def test_run_query_manual_smoke():
    import pandas as pd
    from langchain_core.messages import HumanMessage

    from agent import DataScienceGraph, build_llm_with_tools, normalize_agent_result

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise unittest.SkipTest("Set GOOGLE_API_KEY to run this manual smoke test.")

    df = pd.DataFrame(
        {
            "loan_amount": [1000, 2000, 3000, 4000],
            "income": [45000, 52000, 61000, 73000],
            "status": ["approved", "approved", "rejected", "approved"],
        }
    )

    llm_with_tools = build_llm_with_tools(
        api_key=api_key,
        model=os.getenv("GOOGLE_MODEL", "gemini-2.5-flash-lite"),
    )
    graph = DataScienceGraph(llm_with_tools=llm_with_tools, df_getter=lambda: df)

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Using python, calculate the average loan_amount and list "
                        "the top 2 loan_amount values."
                    )
                )
            ]
        },
        config={"configurable": {"thread_id": "manual_run_query_test"}},
    )

    normalized = normalize_agent_result(result)
    answer = normalized.get("answer", "").strip()

    print(answer)

    assert answer

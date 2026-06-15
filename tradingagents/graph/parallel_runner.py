"""Thread-based parallel analyst execution.

Each analyst runs in its own thread with an isolated message history so
tool-call/tool-message pairing is never corrupted by another analyst's
concurrent writes to a shared channel.
"""

import concurrent.futures
import logging
import time
from typing import Any, Callable, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from tradingagents.agents.utils.agent_utils import (
    create_msg_delete,
    get_instrument_context_from_state,
)
from tradingagents.dataflows.config import get_config

logger = logging.getLogger(__name__)

# Per-run timing records (keyed by analyst key → wall seconds)
_timing_records: Dict[str, float] = {}


def get_last_timing_records() -> Dict[str, float]:
    """Return per-analyst wall times from the most recent parallel run."""
    return dict(_timing_records)


def _run_analyst_to_completion(
    analyst_fn: Callable,
    initial_state: Dict[str, Any],
    tools_by_name: Dict[str, Any],
    max_tool_rounds: int = 10,
) -> Dict[str, Any]:
    """Execute a single analyst loop to completion inside a thread.

    Parameters
    ----------
    analyst_fn:
        Factory-produced node (e.g. ``market_analyst_node``).
    initial_state:
        Snapshot of the graph state at the start of the parallel phase.
        The ``messages`` key is replaced with a local copy so the thread
        never touches the shared channel.
    tools_by_name:
        Mapping of tool name → LangChain ``@tool`` decorated function.
    max_tool_rounds:
        Safety limit on sequential tool-call loops.

    Returns
    -------
    A state-update dict with at least the analyst's ``*_report`` key
    populated.  The ``messages`` key is *not* included — callers merge
    only the report keys.
    """
    # Deep-copy messages so this thread has its own independent history
    local_messages = list(initial_state.get("messages", []))

    tool_names = sorted(tools_by_name.keys())

    for _ in range(max_tool_rounds):
        # Build a mini-state with the local message snapshot
        mini_state = {**initial_state, "messages": local_messages}
        update = analyst_fn(mini_state)

        new_msgs = update.get("messages", [])
        if not new_msgs:
            break

        last_msg = new_msgs[-1] if isinstance(new_msgs, list) else new_msgs

        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            # Append the assistant message with tool_calls
            local_messages.append(last_msg)

            # Execute each tool call and append the ToolMessage
            for tc in last_msg.tool_calls:
                tool_fn = tools_by_name.get(tc.get("name", ""))
                if tool_fn is None:
                    tool_msg = ToolMessage(
                        content=f"Tool '{tc.get('name')}' not recognised. "
                                f"Available: {', '.join(tool_names)}",
                        tool_call_id=tc.get("id", "unknown"),
                        name=tc.get("name", "unknown"),
                    )
                else:
                    try:
                        result = tool_fn.invoke(tc.get("args", {}))
                        tool_msg = ToolMessage(
                            content=str(result),
                            tool_call_id=tc.get("id", "unknown"),
                            name=tc.get("name", "unknown"),
                        )
                    except Exception as exc:
                        tool_msg = ToolMessage(
                            content=f"Tool error: {exc}",
                            tool_call_id=tc.get("id", "unknown"),
                            name=tc.get("name", "unknown"),
                        )
                local_messages.append(tool_msg)
        else:
            # No more tool calls — analyst produced its final report
            # Return only the report keys (no messages — caller merges)
            report_update = {
                k: v for k, v in update.items()
                if k != "messages" and k in initial_state
            }
            return report_update

    # Exhausted max_tool_rounds — return whatever we have
    logger.warning("Analyst exceeded max_tool_rounds (%d)", max_tool_rounds)
    return {}


def run_analysts_in_parallel(
    state: Dict[str, Any],
    analyst_factories: Dict[str, Callable],
    tool_node_map: Dict[str, ToolNode],
) -> Dict[str, Any]:
    """Fan out the selected analysts across threads and collect their reports.

    Parameters
    ----------
    state:
        Full AgentState at the point where analysts start.
    analyst_factories:
        Mapping ``{key: factory_fn}`` where each factory was already called
        with the LLM (e.g. ``create_market_analyst(llm)``) and returns a
        node function.
    tool_node_map:
        Mapping ``{key: ToolNode}``, each already bound to the correct
        tool set for that analyst type.

    Returns
    -------
    State-update dict with report keys (``market_report``, …) merged.
    """
    # Build tools_by_name per analyst
    tool_mappings: Dict[str, Dict[str, Any]] = {}
    for key, tn in tool_node_map.items():
        tool_mappings[key] = dict(tn.tools_by_name)

    def _run_one(key: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        try:
            result = _run_analyst_to_completion(
                analyst_fn=analyst_factories[key],
                initial_state=state,
                tools_by_name=tool_mappings[key],
            )
            _timing_records[key] = time.perf_counter() - t0
            return result
        except Exception:
            _timing_records[key] = time.perf_counter() - t0
            logger.exception("Analyst %s failed; continuing with empty report", key)
            return {}

    merged: Dict[str, Any] = {}
    t_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(analyst_factories),
    ) as pool:
        futures = {pool.submit(_run_one, key): key for key in analyst_factories}
        for future in concurrent.futures.as_completed(futures):
            key = futures[future]
            try:
                update = future.result()
            except Exception:
                logger.exception("Analyst %s raised unhandled exception", key)
                update = {}
            for report_key, value in update.items():
                if value:
                    merged[report_key] = value

    _timing_records["_parallel_total"] = time.perf_counter() - t_start
    return merged

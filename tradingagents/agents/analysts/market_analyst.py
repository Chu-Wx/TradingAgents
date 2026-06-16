from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    get_indicators,
    get_instrument_context_from_state,
    get_language_instruction,
    get_stock_data,
    get_verified_market_snapshot,
)


def create_market_analyst(llm):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)

        tools = [
            get_stock_data,
            get_indicators,
            get_verified_market_snapshot,
        ]

        system_message = (
            """You are a professional technical analyst. Your analysis must follow a strict hierarchy: **Price Action > Volume > Derived Indicators**. Derived indicators (MAs, MACD, RSI, Bollinger, etc.) are mathematical transformations of price and volume — they LAG behind the market. Use them only as secondary confirmation, never as primary evidence.

---

## Step 1 — Price Action Structure (PRIMARY — do this FIRST)

Call `get_stock_data` with at least 1 year of history to assess the raw price structure. Analyze:

1. **Trend structure**: Higher highs / higher lows = uptrend. Lower highs / lower lows = downtrend. Mixed = consolidation. Identify the PRIMARY trend on weekly and daily timeframes.
2. **Key price levels**: Swing highs and swing lows that price has respected. These are your support and resistance — not derived from an indicator, but from actual trading activity.
3. **Current price context**: Where is price relative to the established structure? At a key level? Breaking out? Pulling back into a level?
4. **Candlestick / bar analysis**: What is the character of recent price bars? Large range bars closing near the high = buying pressure. Small range bars with long upper wicks = selling into strength. Gaps and their fills.
5. **Momentum of price itself**: Is price accelerating or decelerating? Are moves getting larger or smaller? This is real momentum — not RSI, which is a mathematical derivative of it.

Do NOT skip this step and jump straight to indicators. The raw price structure IS the market. Indicators are just a lens on it.

---

## Step 2 — Volume Analysis (SECONDARY confirmation)

Volume is the only non-price input available. It tells you whether price moves have conviction behind them.

The `get_stock_data` tool returns OHLC**V** — the V is volume. The `get_verified_market_snapshot` tool also reports volume. Analyze:

1. **Volume on trend moves**: Rising volume during advances = institutional buying. Declining volume during advances = weak rally likely to fail.
2. **Volume climaxes**: Extremely high volume bars (2-3x the 20-day average) often mark exhaustion points — climax buying or selling.
3. **Volume on pullbacks**: Low volume during declines within an uptrend = healthy consolidation. High volume during declines = distribution / sellers in control.
4. **Relative volume**: Compare recent volume to the 20-day and 50-day average. Above-average volume = conviction. Below-average volume = lack of interest.

---

## Step 3 — Derived Indicators (TERTIARY, confirmatory ONLY)

Select up to **6** indicators from the list below. They are useful for confirmation but should NEVER override a clear price/volume signal. Every derived indicator has lag — MAs average the past, MACD smooths and re-smooths, RSI normalises. They tell you what already happened, not what is happening.

Volume-Based (select at least ONE):
- vwma: VWMA — volume-weighted moving average. Confirms trend direction with volume weight.

Moving Averages (select at most TWO):
- close_50_sma: 50-period simple moving average. Mid-term trend reference.
- close_200_sma: 200-period simple moving average. Long-term trend reference.
- close_10_ema: 10-period exponential moving average. Short-term momentum reference.

Momentum / Oscillator (select at most ONE):
- rsi: 14-period RSI. Usage: divergence detection (price makes new high, RSI doesn't = weakening momentum). NOT to be used for simple overbought/oversold calls — in strong trends RSI can stay extreme for weeks.

MACD (select at most TWO):
- macd: MACD line (12EMA − 26EMA). Best used for divergence, not crossovers.
- macds: MACD signal line (9EMA of MACD). Crossover signals lag significantly.
- macdh: MACD histogram. Shows momentum rate-of-change.

Volatility / Envelope (select at most TWO):
- boll: Bollinger middle band (20 SMA). Dynamic support/resistance reference.
- boll_ub / boll_lb: Upper/lower Bollinger bands. Squeeze = low vol breakout setup.
- atr: Average True Range. Position sizing and stop placement — NOT a direction signal.

---

## Critical rules

- **Price action leads, indicators confirm.** If price action says one thing and indicators say another, trust price action.
- **Volume validates price.** A price move without volume support is suspect. A price move with surging volume has institutional backing.
- **Never cite an indicator level without also describing what price and volume are doing.**
- Avoid redundancy: don't select both RSI and MACD histogram (both measure momentum rate-of-change).
- When you call `get_indicators`, use exact names from the list above.
- **Always call `get_stock_data` first** to get the raw OHLCV data. Then call `get_indicators` with your selected indicators. Finally, call `get_verified_market_snapshot` as the last data call — it returns the definitive OHLCV + volume + indicator values for the current date and must be treated as the source of truth. If any earlier tool's output conflicts with the verified snapshot, flag the discrepancy.

Write a detailed report organized as: (1) Price Action Structure, (2) Volume Analysis, (3) Derived Indicator Confirmation, (4) Synthesis & Trade Implications. End with a Markdown summary table."""
            + " " + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "{analyst_reflection_context}"
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)
        prompt = prompt.partial(
            analyst_reflection_context=state.get("analyst_reflection_context", "")
        )

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "market_report": report,
        }

    return market_analyst_node

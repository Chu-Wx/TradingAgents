"""
TradingAgents 纸面交易（Paper Trading）示例
===========================================
这个脚本演示如何使用 TradingAgents 多智能体框架进行股票分析，
生成交易决策（Buy/Overweight/Hold/Underweight/Sell）。

工作流程：
  I.   分析师团队 → 4个专业分析师各自分析市场
  II.  研究团队   → Bull vs Bear 辩论
  III. 交易员     → 将研究计划转化为具体交易方案
  IV.  风险管理   → 三位风险分析师辩论
  V.   投资组合经理 → 综合所有信息，做出最终决策

注意：这是一个研究框架，决策仅供参考，不构成投资建议。
"""
import os
import sys

# 加载 .env 环境变量
from dotenv import load_dotenv
load_dotenv()

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG


def run_paper_trade(ticker: str, trade_date: str, output_language: str = "Chinese"):
    """
    运行一次纸面交易分析。

    参数:
        ticker: 股票代码 (如 AAPL, NVDA, 0700.HK, BTC-USD)
        trade_date: 分析日期 (YYYY-MM-DD 格式)
        output_language: 输出语言
    """
    print(f"""
╔══════════════════════════════════════════════════════════╗
║         TradingAgents 多智能体交易分析框架              ║
║         Paper Trading (纸面交易) 示例                  ║
╚══════════════════════════════════════════════════════════╝

📊 分析标的: {ticker}
📅 分析日期: {trade_date}
🤖 LLM 提供商: {os.environ.get('TRADINGAGENTS_LLM_PROVIDER', 'openai')}
🧠 深度思考模型: {os.environ.get('TRADINGAGENTS_DEEP_THINK_LLM', 'gpt-5.5')}
⚡ 快速思考模型: {os.environ.get('TRADINGAGENTS_QUICK_THINK_LLM', 'gpt-5.4-mini')}
🗣️  输出语言: {output_language}

正在初始化 TradingAgents 图...
""")

    # === 第1步: 配置系统 ===
    config = DEFAULT_CONFIG.copy()

    # 可以通过环境变量覆盖，也可以直接在代码中设置
    # config["llm_provider"] = "deepseek"
    # config["deep_think_llm"] = "deepseek-chat"
    # config["quick_think_llm"] = "deepseek-chat"
    # config["backend_url"] = "https://api.deepseek.com/v1"
    config["output_language"] = output_language
    config["max_debate_rounds"] = 1       # 研究团队辩论轮数
    config["max_risk_discuss_rounds"] = 1  # 风险团队辩论轮数

    # === 第2步: 创建 TradingAgentsGraph 实例 ===
    ta = TradingAgentsGraph(debug=True, config=config)

    print("✅ 初始化完成！开始运行分析流程...\n")
    print("=" * 60)
    print("  流程概览:")
    print("  I.   分析师团队 (Market → Sentiment → News → Fundamentals)")
    print("  II.  研究团队 (Bull Researcher ↔ Bear Researcher 辩论)")
    print("  III. 交易员 (将研究计划转化为交易方案)")
    print("  IV.  风险管理 (Aggressive ↔ Conservative ↔ Neutral 辩论)")
    print("  V.   投资组合经理 (做出最终决策)")
    print("=" * 60)
    print()

    # === 第3步: 运行分析 (propagate) ===
    # propagate() 是核心方法，它会：
    #   1. 创建初始状态
    #   2. 解析股票身份 (防止 LLM 幻觉)
    #   3. 加载历史决策记忆
    #   4. 通过 LangGraph 驱动整个多智能体流程
    #   5. 返回最终状态和简化的交易信号
    final_state, decision = ta.propagate(ticker, trade_date)

    # === 第4步: 输出结果 ===
    print("\n" + "=" * 60)
    print("  📈 分析完成！结果如下:")
    print("=" * 60)
    print(f"\n🎯 最终交易决策: {decision}")

    # 输出各阶段报告摘要
    if final_state.get("market_report"):
        print(f"\n📊 市场技术分析报告: 已生成 ({len(final_state['market_report'])} 字符)")

    if final_state.get("sentiment_report"):
        print(f"💬 情绪分析报告: 已生成 ({len(final_state['sentiment_report'])} 字符)")

    if final_state.get("news_report"):
        print(f"📰 新闻分析报告: 已生成 ({len(final_state['news_report'])} 字符)")

    if final_state.get("fundamentals_report"):
        print(f"📋 基本面分析报告: 已生成 ({len(final_state['fundamentals_report'])} 字符)")

    # 辩论结果
    debate_state = final_state.get("investment_debate_state", {})
    if debate_state.get("judge_decision"):
        print(f"\n⚖️  研究经理决策 (Bull/Bear 辩论总结):")
        print(f"   {debate_state['judge_decision'][:300]}...")

    # 交易员方案
    if final_state.get("trader_investment_plan"):
        print(f"\n💼 交易员方案:")
        print(f"   {final_state['trader_investment_plan'][:300]}...")

    # 最终决策 (投资组合经理)
    if final_state.get("final_trade_decision"):
        print(f"\n🏛️  投资组合经理最终决策:")
        print(f"   {final_state['final_trade_decision'][:500]}...")

    print(f"\n📁 完整报告已保存至: ~/.tradingagents/logs/{ticker}/TradingAgentsStrategy_logs/")

    return final_state, decision


if __name__ == "__main__":
    # 默认分析 AAPL 在 2025 年的某一天
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    trade_date = sys.argv[2] if len(sys.argv) > 2 else "2025-06-10"

    final_state, decision = run_paper_trade(ticker, trade_date)

    print(f"\n✅ 纸面交易演示完成! 决策: {decision}")

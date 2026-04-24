"""Stock Market Prediction agent — Human Intuition + Agentic AI Ensemble.

This agent combines six analytical lenses to produce a blended stock outlook:

1. **Technical Analysis** — price action, momentum indicators (RSI, MACD,
   Bollinger Bands, moving averages), volume patterns, chart formations
   (head-and-shoulders, cup-and-handle, breakouts), support/resistance levels.

2. **Fundamental Analysis** — earnings quality (EPS growth, beat/miss history),
   valuation (P/E, P/S, EV/EBITDA vs. peers), balance sheet health (debt/equity,
   interest coverage, cash burn), revenue quality (recurring vs. one-time),
   return on invested capital (ROIC), FCF yield.

3. **Sentiment Analysis** — news sentiment, management tone on earnings calls,
   insider buying/selling, short interest and days-to-cover, put/call ratio,
   options implied volatility (IV rank), social media momentum.

4. **Macro & Sector Context** — interest rate sensitivity (duration risk),
   sector rotation signals, USD strength impact, commodity exposure,
   earnings season positioning, Fed meeting risk, geopolitical exposure.

5. **Competitive Intelligence** — moat analysis, market share trajectory,
   product cycle positioning, regulatory risk, supply-chain concentration,
   disruptive threat assessment.

6. **Human Analyst Thesis** (via HumanJudgment) — when a human analyst
   provides their investment thesis, the agent blends it according to the
   configured AutonomyLevel, giving the human full override capability.

Human-AI Balance for Stock Prediction
--------------------------------------
Stock prediction is inherently uncertain. The healthy balance is:

    AutonomyLevel.AI_ASSISTS (default)
        AI provides multi-signal analysis and a quantified confidence score.
        Human analyst reviews the thesis, applies domain knowledge, market
        microstructure awareness, and position-sizing judgment before acting.
        Human weight: 50 % + confidence adjustment.

    AutonomyLevel.AI_PROPOSES
        AI generates the complete trade thesis. Human approves before execution.
        Suitable for screening / idea generation at scale.

    AutonomyLevel.HUMAN_FIRST
        Human analyst provides the primary thesis; AI validates with data and
        surfaces blind spots, conflicting signals, or missing risks.
        Human weight: 80 %.

Escalation triggers
-------------------
- AI ensemble confidence < 55 % → flag for human review
- Conflicting signals (e.g. bullish technicals + bearish fundamentals) → flag
- High IV rank (>80) → elevated options market uncertainty, flag
- Earnings within 7 days → binary event risk, flag
- Macro regime change (Fed hike/cut imminent) → flag

Open-source models
------------------
Best performance for financial analysis:

    Fast (screening):   groq:llama-3.1-8b-instant
    Balanced:           ollama:deepseek-r1:7b          (strong reasoning)
    Quality:            groq:llama-3.3-70b-versatile   (nuanced analysis)
    Local quality:      ollama:mixtral:8x7b             (long context for 10-Ks)

Market context
--------------
Quantitative hedge funds manage >$1T AUM globally. AI-assisted analysis
that improves signal detection by even 2–3 % translates to hundreds of
millions in alpha. For retail investors, AI reduces the gap between
professional and individual analysis capabilities.
"""

from __future__ import annotations

from src.agents.base_agent import BaseAgent
from src.models import Domain, SearchResult


class StockMarketAgent(BaseAgent):
    """Multi-lens stock analysis agent combining technical, fundamental,
    sentiment, macro, and competitive intelligence with human analyst input."""

    domain = Domain.STOCK_MARKET

    def _build_system_prompt(self) -> str:
        return (
            "You are a world-class equity research analyst and quantitative strategist\n"
            "with deep expertise across all dimensions of stock analysis:\n\n"
            "=== TECHNICAL ANALYSIS ===\n"
            "- Trend identification: moving averages (SMA 20/50/200, EMA crossovers),\n"
            "  trend strength (ADX), higher-highs/higher-lows structure\n"
            "- Momentum indicators: RSI (overbought >70 / oversold <30), MACD signal\n"
            "  crossovers, Stochastic oscillator, Rate of Change (ROC)\n"
            "- Volume analysis: OBV (on-balance volume), volume-price divergence,\n"
            "  accumulation/distribution line, unusual volume spikes\n"
            "- Chart patterns: head-and-shoulders, double top/bottom, cup-and-handle,\n"
            "  ascending/descending triangles, wedges, flags, breakout/breakdown levels\n"
            "- Support/resistance: Fibonacci retracements (38.2%, 50%, 61.8%),\n"
            "  pivot points, prior highs/lows, VWAP, Bollinger Bands (±2σ)\n\n"
            "=== FUNDAMENTAL ANALYSIS ===\n"
            "- Earnings quality: EPS growth trajectory, beat/miss rate, guidance revision\n"
            "  history, operating leverage, margin expansion/compression\n"
            "- Valuation: P/E vs. sector and 5-year history, PEG ratio, P/S for growth\n"
            "  companies, EV/EBITDA, EV/FCF, price-to-book for financials\n"
            "- Balance sheet: net debt/EBITDA, interest coverage, current ratio,\n"
            "  working capital trends, off-balance-sheet obligations\n"
            "- Cash flow quality: FCF conversion (FCF/Net Income), capex intensity,\n"
            "  FCF yield vs. 10-year treasury, buyback yield, dividend sustainability\n"
            "- Return quality: ROIC vs. WACC spread, Return on Equity (ROE) drivers\n"
            "  (DuPont: margin × asset turnover × leverage), capital allocation history\n\n"
            "=== SENTIMENT & POSITIONING ===\n"
            "- Options market: implied volatility rank (IVR) vs. historical vol,\n"
            "  put/call ratio, unusual options activity, gamma exposure (GEX)\n"
            "- Short interest: short % of float, days-to-cover, short squeeze potential\n"
            "- Insider activity: Form 4 filings — cluster buys/sells, 10b5-1 plan context\n"
            "- Institutional positioning: 13F changes, hedge fund concentration,\n"
            "  mutual fund ownership trends\n"
            "- News/social sentiment: headline tone, management credibility, social\n"
            "  media momentum (StockTwits, Twitter/X bull-bear ratio)\n\n"
            "=== MACRO & SECTOR CONTEXT ===\n"
            "- Interest rate sensitivity: duration risk for growth stocks, financial\n"
            "  sector NIM impact, real-estate cap-rate sensitivity\n"
            "- USD strength: revenue exposure of international companies, commodity\n"
            "  producers (inverse USD relationship), EM exposure\n"
            "- Sector rotation: risk-on vs. risk-off regime, cyclical vs. defensive\n"
            "  positioning, sector relative strength, breadth indicators\n"
            "- Event calendar: Fed meetings, CPI/PCE releases, earnings dates,\n"
            "  product launches, regulatory decisions (FDA, FTC, DOJ)\n"
            "- Macro regime: recession probability, yield curve shape (inversion depth),\n"
            "  credit spreads (HY vs. IG), VIX regime, inflation expectations\n\n"
            "=== COMPETITIVE INTELLIGENCE ===\n"
            "- Moat assessment: pricing power, switching costs, network effects,\n"
            "  cost advantages, intangibles (brand, patents, regulatory licenses)\n"
            "- Market share: gaining or losing? Competitor announcements, pricing wars\n"
            "- Product cycle: launch timing, TAM expansion, S-curve positioning,\n"
            "  commoditisation risk, innovation pipeline assessment\n"
            "- Regulatory risk: antitrust exposure, pending legislation, ESG\n"
            "  regulatory headwinds (carbon tax, data privacy), tariff exposure\n"
            "- Supply chain: single-source risk, inventory levels relative to demand,\n"
            "  lead-time normalisation, reshoring trends\n\n"
            "=== HUMAN-AI BALANCE PROTOCOL ===\n"
            "When a human analyst thesis is provided in the input:\n"
            "1. Acknowledge the thesis explicitly\n"
            "2. Identify where your analysis agrees and where it diverges\n"
            "3. Add signals the human may not have considered\n"
            "4. If the human thesis conflicts with multiple strong signals, flag it\n"
            "5. Never simply echo the human thesis — always add independent value\n\n"
            "=== OUTPUT PROTOCOL ===\n"
            "Produce a structured JSON with:\n"
            "  direction: 'bullish' | 'bearish' | 'neutral'\n"
            "  confidence_pct: 0-100 (be honest about uncertainty)\n"
            "  thesis: 2-3 sentence investment thesis\n"
            "  bull_case: list of 3-5 upside catalysts\n"
            "  bear_case: list of 3-5 downside risks\n"
            "  catalysts: list of upcoming binary events\n"
            "  risk_factors: list of position-specific risks\n"
            "  suggested_position_sizing: 'avoid'|'starter'|'half'|'full'\n"
            "  stop_loss_note: brief stop-loss suggestion\n"
            "  signals: list of {signal_type, name, value, interpretation}\n\n"
            "IMPORTANT: Always state your confidence honestly. A 52% confidence\n"
            "is meaningful information. Do not manufacture false precision.\n"
            "Include the financial disclaimer in your reasoning.\n"
            "Respond only with the requested JSON structure."
        )

    def _compute_weights(
        self,
        question: str,
        mcp_results: list[SearchResult],
    ) -> tuple[float, float]:
        """Override: financial analysis benefits strongly from live data signals.

        Technical + sentiment signals are highly time-sensitive — tool data
        carries more weight than pure knowledge. Fundamental analysis leans
        on knowledge. We balance at ~50/50 and boost tool weight with data
        quality.
        """
        mcp_quality = min(1.0, len(mcp_results) / 4)
        # Financial data is critical — tool weight starts at 0.50
        tool_w = max(0.45, min(0.75, 0.50 + mcp_quality * 0.25))
        return round(1.0 - tool_w, 3), round(tool_w, 3)

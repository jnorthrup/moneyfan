# Product Guidelines: moneyfan-trader

## Voice & Tone
- **Technical & precise**: Data and numbers are the primary communication medium. Prose is minimal and purposeful.
- **Direct**: No marketing fluff. State facts, results, and trade-offs plainly.
- **Developer-first**: Assume the reader can read code. Prefer code examples over lengthy prose explanations.
- **Honest about risk**: Always surface fees, slippage, and failure modes. Never obscure trading risk.

## Brand Messaging
- Core value proposition: *"Simulate, backtest, deploy — with full fee transparency."*
- Positioning: A serious quant/developer tool, not a gamified trading app.
- Avoid: Hype language ("moon", "lambo", "guaranteed returns"), overpromising on ML signals.
- Emphasize: Reproducibility, auditability, and correctness of strategy execution.

## UI / Dashboard Principles
- **Information density over aesthetics**: Traders need data, not decoration.
- **Real-time feedback**: Latency of displayed data must always be visible.
- **Status clarity**: Bot state (running/paused/error), P&L, and open positions must always be prominent.
- **Dark theme preferred**: Standard for trading terminals; reduces eye strain during extended sessions.

## Code & Documentation Style
- All public APIs and strategy interfaces must be documented with parameter types and units.
- Monetary values: always specify currency and whether fees are included/excluded.
- Timestamps: always UTC, ISO 8601 format.
- Strategy results must report: total return, Sharpe ratio, max drawdown, fee drag.

## Localization & Accessibility
- Primary language: English (US).
- Currency display: USD as base, with crypto amounts shown to 6–8 decimal places.
- No accessibility requirements defined yet — log as future track if needed.

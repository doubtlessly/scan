# modules/notifier.py
import os
import json
from telegram import Bot, error as tg_error
from .config import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    EXIT_ATR_MULTIPLIER,
    PROFIT_TARGET_ATR,
)

class Notifier:
    def __init__(self):
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            print("⚠️  Telegram credentials missing; notifications disabled")
            self.bot = None
            self.chat_id = None
        else:
            try:
                self.bot = Bot(token=token)
                self.chat_id = chat_id
            except tg_error.InvalidToken:
                print("⚠️  Invalid Telegram token; notifications disabled")
                self.bot = None
                self.chat_id = None

    async def send_feedback(self, advice: dict):
        """Send pre-scan ChatGPT analysis & tweaks to Telegram."""
        if not self.bot:
            return
        lines = ["🧠 *Pre-Scan Analysis*"]
        summary = advice.get("summary")
        if isinstance(summary, str):
            lines.append(summary)
        elif isinstance(summary, dict):
            for k,v in summary.items():
                lines.append(f"• *{k}*: {v}")
        tweaks = advice.get("tweaks", {})
        if tweaks:
            lines.append("\n⚙️ *Suggested Tweaks*")
            for k, v in tweaks.items():
                lines.append(f"• `{k}` → {v}")
        text = "\n".join(lines)
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode="Markdown"
            )
        except Exception as e:
            print(f"⚠️  Failed to send feedback message: {e}")

    async def send(self, *, simulated: list[dict], live: list[dict]):
        """
        Send both simulated signals and live trades to Telegram.
        Each entry includes: symbol, rating (0–10), side, entry price,
        stop-loss, take-profit, quantity, and (for live) PnL.
        """
        if not self.bot:
            return

        lines = []

        # 1) Simulated signals
        if simulated:
            lines.append("🚀 *Simulated Signals* 🚀")
            for s in simulated:
                fv    = s['feature_vector']
                score = s['score']
                entry = fv['mid_price']
                atr   = fv.get('atr', 0.0)
                sl    = (entry - atr * EXIT_ATR_MULTIPLIER) if s['side']=='buy' else (entry + atr * EXIT_ATR_MULTIPLIER)
                tp    = (entry + atr * PROFIT_TARGET_ATR)    if s['side']=='buy' else (entry - atr * PROFIT_TARGET_ATR)
                lines.append(
                    f"`{s['symbol']}` | *{score}/10* | _{s['side']}_\n"
                    f"  Entry: `{entry:.4f}`\n"
                    f"  SL:    `{sl:.4f}`\n"
                    f"  TP:    `{tp:.4f}`\n"
                    f"  Qty:   `{s.get('quantity', 0):.4f}`"
                )
            lines.append("")  # blank line

        # 2) Live trades
        if live:
            lines.append("💰 *Live Trades* 💰")
            for t in live:
                score = t.get('score', 0)
                entry = t.get('live_price', 0.0)
                pnl   = t.get('pnl', 0.0)
                lines.append(
                    f"`{t['symbol']}` | *{score}/10* | _{t['side']}_\n"
                    f"  Fill: `{entry:.4f}`\n"
                    f"  PnL:  `{pnl:.4f}`"
                )
            lines.append("")

        # Send message
        text = "\n".join(lines)
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode="Markdown"
            )
        except Exception as e:
            print(f"⚠️  Telegram send failed: {e}")

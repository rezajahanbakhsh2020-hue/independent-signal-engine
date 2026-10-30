from signal_engine.strategies.base import Strategy, transition_direction
from signal_engine.strategies.bollinger import BollingerMeanReversionStrategy
from signal_engine.strategies.rsi_reversal import RSIReversalStrategy

# This registry is the sole strategy allow-list used by the CLI and signal paths.
STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    "Bollinger_Reversion": BollingerMeanReversionStrategy,
    "RSI_Reversal": RSIReversalStrategy,
}

__all__ = [
    "Strategy",
    "transition_direction",
    "STRATEGY_REGISTRY",
    "BollingerMeanReversionStrategy",
    "RSIReversalStrategy",
]

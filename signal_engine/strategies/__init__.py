from signal_engine.strategies.base import Strategy, transition_direction
from signal_engine.strategies.crossover import CrossoverStrategy
from signal_engine.strategies.pullback import PullbackStrategy
from signal_engine.strategies.breakout import BreakoutStrategy
from signal_engine.strategies.bollinger import BollingerMeanReversionStrategy
from signal_engine.strategies.ichimoku_rsi_confirmed import IchimokuRsiConfirmedStrategy

STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    "crossover": CrossoverStrategy,
    "pullback": PullbackStrategy,
    "breakout": BreakoutStrategy,
    "bollinger_mean_reversion": BollingerMeanReversionStrategy,
    "ichimoku_rsi_confirmed": IchimokuRsiConfirmedStrategy,
}

__all__ = [
    "Strategy",
    "transition_direction",
    "STRATEGY_REGISTRY",
    "CrossoverStrategy",
    "PullbackStrategy",
    "BreakoutStrategy",
    "BollingerMeanReversionStrategy",
    "IchimokuRsiConfirmedStrategy",
]

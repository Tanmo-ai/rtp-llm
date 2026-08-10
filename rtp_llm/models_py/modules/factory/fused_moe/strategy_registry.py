"""Strategy registry

Manages registration and selection of all MOE strategies.
"""

import logging
from typing import List

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)

from .defs.strategy_base import MoeStrategy

logger = logging.getLogger(__name__)


class StrategyRegistry:
    """Strategy registry

    Responsible for managing all registered strategies and selecting the most
    appropriate strategy based on configuration.
    """

    def __init__(self):
        """Initialize registry"""
        self._strategies: List[MoeStrategy] = []

    def register(self, strategy: MoeStrategy) -> None:
        """Register a strategy

        Args:
            strategy: Strategy instance to register
        """
        self._strategies.append(strategy)

    def list_strategies(self) -> List[MoeStrategy]:
        """List all registered strategies sorted by priority (descending)

        Returns:
            List of strategies sorted by priority (highest first)
        """
        return sorted(self._strategies, key=lambda s: s.priority, reverse=True)

    def clear(self) -> None:
        """Clear all registered strategies"""
        self._strategies.clear()

    def get_strategy(self, config: MoEConfigAdapter) -> MoeStrategy:
        """Get appropriate strategy based on configuration

        First finds all strategies that can handle the configuration,
        then selects the one with the highest priority.

        Args:
            config: MOE configuration adapter

        Returns:
            Most appropriate strategy instance (highest priority among candidates)

        Raises:
            ValueError: If no suitable strategy is found
        """
        # Find all candidate strategies that can handle this config
        logger.debug(
            f"[StrategyRegistry] Evaluating {len(self._strategies)} strategies..."
        )
        candidates = [
            strategy for strategy in self._strategies if strategy.can_handle(config)
        ]
        logger.debug(f"[StrategyRegistry] Found {len(candidates)} candidate(s)")

        if not candidates:
            logger.error(
                f"No suitable MOE strategy found. Config details: "
                f"quant_config={config.model_config.quant_config}, "
                f"ep_size={config.ep_size}, "
                f"world_size={config.world_size}, "
                f"tp_size={config.tp_size}, "
                f"use_deepep_low_latency={config.moe_config.use_deepep_low_latency if config.moe_config else False}"
            )
            raise ValueError(
                f"No suitable MOE strategy found for configuration. "
                f"Please check quant_config, ep_size, and parallelism settings."
            )

        # Sort candidates by priority (descending, higher priority first)
        candidates.sort(key=lambda s: s.priority, reverse=True)

        # A pre-quantized checkpoint whose activation format no candidate
        # produces would otherwise be picked up by an unquantized fallback and
        # fail deep inside a kernel, so reject the combination here. Granularity
        # is part of the contract: an INT8 per-tensor strategy must not satisfy a
        # scheme that needs INT8 per token.
        model_quant_config = config.model_config.quant_config
        required_act_spec = (
            model_quant_config.get_moe_activation_quant_spec()
            if model_quant_config is not None
            else None
        )
        if required_act_spec is not None:
            provided_specs = {
                (attrs.quant_config.quant_dtype, attrs.quant_config.per_act_token_quant)
                for attrs in (strategy.get_attributes() for strategy in candidates)
            }
            if required_act_spec not in provided_specs:
                dtype, per_act_token = required_act_spec
                raise ValueError(
                    f"Quantization method {model_quant_config.get_method()} needs a "
                    f"MOE strategy producing {dtype} activations "
                    f"({'per token' if per_act_token else 'not per token'}), but "
                    f"none of the {len(candidates)} candidate strategy(ies) does "
                    f"(they provide {sorted(map(str, provided_specs))}). MOE layers "
                    "cannot serve this checkpoint: use a dense model, choose "
                    "another quantization method, or register a strategy for it."
                )

        # Log all candidate strategies
        logger.info(f"Found {len(candidates)} candidate strategy(ies) for MOE:")
        for strategy in candidates:
            logger.info(
                f"  - {strategy.__class__.__name__}: "
                f"{strategy.get_attributes()} (priority={strategy.priority})"
            )

        # Select the strategy with highest priority (first in sorted list)
        selected = candidates[0]

        logger.info(
            f"Selected strategy: {selected.__class__.__name__} "
            f"with priority {selected.priority}"
        )

        return selected

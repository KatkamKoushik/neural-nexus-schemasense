"""SchemaSense Local AI Runtime — Evidence-Based Model Compatibility & Sizing Calculator."""

from __future__ import annotations

import logging

from schemasense.runtime.types import (
    BackendAccelerator,
    CompatibilityStatus,
    ConfidenceLevel,
    HardwareProfile,
    ModelManifest,
    SizingReport,
)

logger = logging.getLogger("schemasense.runtime.compatibility")

KV_BYTES_PER_ELEMENT = {
    "fp16": 2.0,
    "f16": 2.0,
    "fp8": 1.0,
    "q8_0": 1.0625,
    "q4_0": 0.5625,
}

HOST_PROCESS_OVERHEAD_MB = 250
CUDA_DRIVER_OVERHEAD_MB = 450
GENERIC_GPU_OVERHEAD_MB = 200
ALLOCATOR_FRAGMENTATION_FACTOR = 1.08


class ModelCompatibilityCalculator:
    """Calculates evidence-based memory sizing and evaluates model fit against host resources."""

    @classmethod
    def evaluate(
        cls,
        manifest: ModelManifest,
        hardware: HardwareProfile,
        target_context_tokens: int = 4096,
        kv_format: str = "fp16",
    ) -> SizingReport:
        """Computes multi-tier memory estimates and returns an honest, uncertainty-aware compatibility report."""
        verified_weights_mb = int(manifest.file_size_bytes // (1024 * 1024))
        evidence: list[str] = [
            f"Verified on-disk model weight size: {verified_weights_mb} MB",
            f"Requested context length: {target_context_tokens} tokens",
            f"KV cache data format: {kv_format}",
        ]
        caveats: list[str] = [
            (
                "Sizing estimates reflect theoretical static allocations and do not guarantee avoidance "
                "of operating system Out-Of-Memory termination under high host memory pressure, "
                "concurrent workloads, or backend allocator fragmentation."
            )
        ]

        if verified_weights_mb <= 0:
            return SizingReport(
                model_id=manifest.model_id,
                target_context_tokens=target_context_tokens,
                status=CompatibilityStatus.INSUFFICIENT_METADATA,
                confidence=ConfidenceLevel.NONE,
                verified_weights_mb=0,
                estimated_kv_cache_mb=None,
                estimated_activation_mb=None,
                total_estimated_ram_mb=None,
                advisory_gpu_layers=None,
                advisory_vram_usage_mb=None,
                evidence_notes=evidence,
                caveats=caveats
                + [
                    "Model file size is empty or less than 1 MB; cannot calculate valid footprint."
                ],
            )

        hp = manifest.hyperparameters
        if hp is None:
            return SizingReport(
                model_id=manifest.model_id,
                target_context_tokens=target_context_tokens,
                status=CompatibilityStatus.INSUFFICIENT_METADATA,
                confidence=ConfidenceLevel.NONE,
                verified_weights_mb=verified_weights_mb,
                estimated_kv_cache_mb=None,
                estimated_activation_mb=None,
                total_estimated_ram_mb=None,
                advisory_gpu_layers=None,
                advisory_vram_usage_mb=None,
                evidence_notes=evidence,
                caveats=caveats
                + [
                    (
                        "Model architectural parameters are unavailable (e.g. unconfigured SafeTensors container). "
                        "Cannot estimate KV cache or activation working set."
                    )
                ],
            )

        # Verify minimum required architectural parameters
        if not (hp.layer_count and hp.embedding_length and hp.head_count):
            return SizingReport(
                model_id=manifest.model_id,
                target_context_tokens=target_context_tokens,
                status=CompatibilityStatus.INSUFFICIENT_METADATA,
                confidence=ConfidenceLevel.NONE,
                verified_weights_mb=verified_weights_mb,
                estimated_kv_cache_mb=None,
                estimated_activation_mb=None,
                total_estimated_ram_mb=None,
                advisory_gpu_layers=None,
                advisory_vram_usage_mb=None,
                evidence_notes=evidence,
                caveats=caveats
                + [
                    (
                        "Required transformer dimensions (layers, embedding size, or attention heads) "
                        "are missing from metadata. KV cache calculation cannot proceed safely."
                    )
                ],
            )

        layers = hp.layer_count
        embed_dim = hp.embedding_length
        heads = hp.head_count
        heads_kv = hp.head_count_kv if hp.head_count_kv is not None else heads

        # Head dimension: use explicit key_length if decoupled, else d_model // n_heads
        if hp.key_length is not None and hp.key_length > 0:
            d_head = hp.key_length
            evidence.append(f"Using explicit key dimension d_head = {d_head}")
        else:
            d_head = embed_dim // heads
            evidence.append(
                f"Derived head dimension d_head = {d_head} ({embed_dim} / {heads})"
            )

        # Sliding-Window Attention (SWA) and Hybrid Layer Modeling
        swa_layers = 0
        global_layers = layers
        effective_swa_context = target_context_tokens
        pattern_uncertain = False

        if hp.sliding_window is not None and hp.sliding_window > 0:
            window = hp.sliding_window
            clamped_window = min(target_context_tokens, window)

            if hp.sliding_window_layers is not None:
                # 1. Explicit layer count verified in architecture metadata
                swa_layers = min(layers, max(0, hp.sliding_window_layers))
                global_layers = layers - swa_layers
                evidence.append(
                    f"Verified hybrid attention metadata: {swa_layers} sliding-window layers (window={window}) "
                    f"and {global_layers} global-attention layers"
                )
                effective_swa_context = clamped_window
            elif hp.sliding_window_pattern == "alternating":
                # 2. Verified alternating pattern from metadata
                swa_layers = layers // 2
                global_layers = layers - swa_layers
                evidence.append(
                    f"Verified alternating attention pattern: {swa_layers} sliding-window layers (window={window}) "
                    f"and {global_layers} global-attention layers"
                )
                effective_swa_context = clamped_window
            elif hp.sliding_window_pattern == "all":
                # 3. Verified all-sliding window attention
                swa_layers = layers
                global_layers = 0
                evidence.append(
                    f"Verified all-sliding window attention: all {layers} layers clamped to window size of {clamped_window} tokens"
                )
                effective_swa_context = clamped_window
            else:
                # 4. Unknown/unverified SWA layer pattern: Defensible conservative upper bound
                # Do NOT hardcode a ratio. Treat all layers as full global context to prevent underestimation.
                swa_layers = 0
                global_layers = layers
                pattern_uncertain = True
                evidence.append(
                    f"Model metadata declares a sliding window ({window}), but the layer distribution pattern "
                    "(all vs alternating vs hybrid) is unverified in metadata. Applied defensible conservative "
                    "estimate (full global attention across all layers) to guarantee against OOM underestimation."
                )
                caveats.append(
                    "Sliding-window layer pattern is unverified in metadata; "
                    "calculated conservative upper bound assuming full context across all layers."
                )

        # KV cache byte calculation: 2 * B * [(L_swa * S_swa) + (L_glob * S_glob)] * H_kv * d_head * b_kv (B=1)
        token_layers = (swa_layers * effective_swa_context) + (
            global_layers * target_context_tokens
        )
        b_kv = KV_BYTES_PER_ELEMENT.get(kv_format.lower(), 2.0)
        kv_bytes = 2 * 1 * token_layers * heads_kv * d_head * b_kv
        kv_cache_mb = int(kv_bytes / (1024 * 1024))
        evidence.append(
            f"Calculated KV cache size: {kv_cache_mb} MB ({layers} layers, {heads_kv} KV heads, {d_head} head dim)"
        )

        # Working activation scratchpad estimate
        # Proportional to context length and embedding dimension
        activation_mb = max(
            32, int((target_context_tokens * embed_dim * 2) / (1024 * 1024))
        )
        evidence.append(f"Estimated activation scratchpad: {activation_mb} MB")

        # Total estimated host RAM with allocator buffer
        unbuffered_total = (
            verified_weights_mb + kv_cache_mb + activation_mb + HOST_PROCESS_OVERHEAD_MB
        )
        total_ram_required_mb = int(unbuffered_total * ALLOCATOR_FRAGMENTATION_FACTOR)
        evidence.append(
            f"Estimated total host memory footprint: {total_ram_required_mb} MB (includes process overhead and 1.08x allocator buffer)"
        )

        # Advisory GPU Layer Offload Calculation
        advisory_layers, advisory_vram = cls._calculate_advisory_gpu_offload(
            verified_weights_mb, kv_cache_mb, layers, hardware
        )
        if advisory_layers is not None:
            caveats.append(
                "GPU layer offload recommendation is strictly advisory; physical allocation "
                "depends on the selected inference backend engine."
            )
        elif (
            hardware.gpus
            and not hardware.gpus[0].is_unified_memory
            and hardware.gpus[0].available_vram_mb is None
        ):
            evidence.append(
                f"GPU detected ({hardware.gpus[0].name}), but available VRAM is unknown; "
                "refusing to infer GPU memory or compute offload."
            )

        # Determine compatibility status against host available physical memory
        avail_ram = hardware.available_ram_mb
        confidence = (
            ConfidenceLevel.MEDIUM if pattern_uncertain else ConfidenceLevel.HIGH
        )

        if total_ram_required_mb <= int(avail_ram * 0.85):
            status = CompatibilityStatus.CONFIDENT_FIT
        elif total_ram_required_mb <= avail_ram:
            status = CompatibilityStatus.MARGINAL_FIT
            confidence = ConfidenceLevel.MEDIUM
            caveats.append(
                f"Model fits within currently free physical RAM ({avail_ram} MB) but exceeds "
                "the 15% safety buffer. System may experience memory pressure if other applications become active."
            )
        else:
            status = CompatibilityStatus.LIKELY_OOM
            caveats.append(
                f"Estimated requirement ({total_ram_required_mb} MB) exceeds available host RAM "
                f"({avail_ram} MB). Execution will likely trigger swap paging thrashing or OS OOM termination."
            )

        return SizingReport(
            model_id=manifest.model_id,
            target_context_tokens=target_context_tokens,
            status=status,
            confidence=confidence,
            verified_weights_mb=verified_weights_mb,
            estimated_kv_cache_mb=kv_cache_mb,
            estimated_activation_mb=activation_mb,
            total_estimated_ram_mb=total_ram_required_mb,
            advisory_gpu_layers=advisory_layers,
            advisory_vram_usage_mb=advisory_vram,
            evidence_notes=evidence,
            caveats=caveats,
        )

    @classmethod
    def _calculate_advisory_gpu_offload(
        cls,
        weights_mb: int,
        kv_cache_mb: int,
        layers: int,
        hardware: HardwareProfile,
    ) -> tuple[int | None, int | None]:
        """Calculates advisory offload layers if discrete GPU is present with verified VRAM."""
        if not hardware.gpus or layers <= 0:
            return None, None

        # Take primary GPU
        gpu = hardware.gpus[0]
        if gpu.is_unified_memory:
            # Unified memory: entire model and KV share system RAM; all layers offloadable to Metal
            return layers, None

        if gpu.available_vram_mb is None or gpu.available_vram_mb <= 0:
            # VRAM is unknown or unverified; do NOT guess available capacity
            return None, None

        driver_overhead = (
            CUDA_DRIVER_OVERHEAD_MB
            if gpu.accelerator == BackendAccelerator.CUDA
            else GENERIC_GPU_OVERHEAD_MB
        )
        usable_vram = gpu.available_vram_mb - driver_overhead
        if usable_vram <= 0:
            return 0, 0

        # Memory per layer
        mb_per_layer = (weights_mb + kv_cache_mb) / layers
        if mb_per_layer <= 0:
            return None, None

        offloadable = min(layers, max(0, int(usable_vram / mb_per_layer)))
        estimated_vram_used = int(offloadable * mb_per_layer + driver_overhead)
        return offloadable, min(gpu.available_vram_mb, estimated_vram_used)

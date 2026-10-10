"""Unit Tests for Evidence-Based Model Compatibility & Sizing Calculator."""

from pathlib import Path

from schemasense.runtime.compatibility import ModelCompatibilityCalculator
from schemasense.runtime.types import (
    BackendAccelerator,
    CompatibilityStatus,
    ConfidenceLevel,
    GPUInfo,
    HardwareProfile,
    ModelArchitecture,
    ModelFormat,
    ModelHyperparameters,
    ModelManifest,
)


def make_test_hardware(
    total_ram_mb: int = 16384,
    avail_ram_mb: int = 12000,
    gpu_vram_mb: int | None = None,
    is_unified: bool = False,
    accelerator: BackendAccelerator = BackendAccelerator.CUDA,
) -> HardwareProfile:
    gpus = []
    if gpu_vram_mb is not None or is_unified:
        gpus.append(
            GPUInfo(
                name="Test GPU",
                accelerator=accelerator,
                total_vram_mb=gpu_vram_mb,
                available_vram_mb=gpu_vram_mb,
                is_unified_memory=is_unified,
            )
        )
    return HardwareProfile(
        total_ram_mb=total_ram_mb,
        available_ram_mb=avail_ram_mb,
        cpu_cores_logical=8,
        cpu_arch="x86_64",
        gpus=gpus,
    )


def make_test_manifest(
    model_id: str = "qwen2.5-7b",
    file_size_bytes: int = 4_500_000_000,  # ~4.29 GB
    layers: int = 28,
    embedding: int = 3584,
    heads: int = 28,
    heads_kv: int = 4,
    context_length: int = 32768,
    sliding_window: int | None = None,
    sliding_window_layers: int | None = None,
    sliding_window_pattern: str | None = None,
    key_length: int | None = None,
    has_hyperparams: bool = True,
) -> ModelManifest:
    hp = None
    if has_hyperparams:
        hp = ModelHyperparameters(
            architecture=ModelArchitecture.QWEN2,
            layer_count=layers,
            embedding_length=embedding,
            head_count=heads,
            head_count_kv=heads_kv,
            context_length=context_length,
            sliding_window=sliding_window,
            sliding_window_layers=sliding_window_layers,
            sliding_window_pattern=sliding_window_pattern,
            key_length=key_length,
        )
    return ModelManifest(
        model_id=model_id,
        file_path=Path(f"/models/{model_id}.gguf"),
        file_size_bytes=file_size_bytes,
        format=ModelFormat.GGUF,
        hyperparameters=hp,
        tensor_count=100,
        quantization_type="Q4_K_M",
    )


def test_kv_cache_scaling_with_context():
    """Verifies that KV cache scales linearly with context length."""
    hw = make_test_hardware(total_ram_mb=32768, avail_ram_mb=28000)
    manifest = make_test_manifest()

    report_2k = ModelCompatibilityCalculator.evaluate(
        manifest, hw, target_context_tokens=2048
    )
    report_8k = ModelCompatibilityCalculator.evaluate(
        manifest, hw, target_context_tokens=8192
    )
    report_32k = ModelCompatibilityCalculator.evaluate(
        manifest, hw, target_context_tokens=32768
    )

    assert report_2k.estimated_kv_cache_mb is not None
    assert report_8k.estimated_kv_cache_mb is not None
    assert report_32k.estimated_kv_cache_mb is not None

    # Ratio of KV cache should match context ratio (8192 / 2048 == 4x)
    assert (
        abs(report_8k.estimated_kv_cache_mb / report_2k.estimated_kv_cache_mb - 4.0)
        < 0.1
    )
    # 32k is 4x of 8k
    assert (
        abs(report_32k.estimated_kv_cache_mb / report_8k.estimated_kv_cache_mb - 4.0)
        < 0.1
    )


def test_attention_all_global_scaling():
    """Verifies all-global attention scales proportionally with target context."""
    hw = make_test_hardware(avail_ram_mb=32000)
    manifest = make_test_manifest(sliding_window=None)

    rep_4k = ModelCompatibilityCalculator.evaluate(
        manifest, hw, target_context_tokens=4096
    )
    rep_16k = ModelCompatibilityCalculator.evaluate(
        manifest, hw, target_context_tokens=16384
    )

    assert rep_4k.confidence == ConfidenceLevel.HIGH
    assert rep_16k.confidence == ConfidenceLevel.HIGH
    assert (
        rep_4k.estimated_kv_cache_mb is not None
        and rep_16k.estimated_kv_cache_mb is not None
    )
    # 16k context is exactly 4x the 4k context
    ratio = rep_16k.estimated_kv_cache_mb / rep_4k.estimated_kv_cache_mb
    assert abs(ratio - 4.0) < 0.1


def test_attention_all_sliding_window():
    """Verifies verified all-sliding window clamps all layers to sliding window."""
    hw = make_test_hardware(avail_ram_mb=32000)
    manifest_all_swa = make_test_manifest(
        layers=28, sliding_window=4096, sliding_window_pattern="all"
    )
    manifest_global = make_test_manifest(layers=28, sliding_window=None)

    rep_swa = ModelCompatibilityCalculator.evaluate(
        manifest_all_swa, hw, target_context_tokens=32768
    )
    rep_glob = ModelCompatibilityCalculator.evaluate(
        manifest_global, hw, target_context_tokens=32768
    )

    assert rep_swa.confidence == ConfidenceLevel.HIGH
    # SWA should clamp effective tokens to 4096, keeping KV cache ~8x smaller than global 32k
    assert rep_swa.estimated_kv_cache_mb < rep_glob.estimated_kv_cache_mb
    ratio = rep_glob.estimated_kv_cache_mb / rep_swa.estimated_kv_cache_mb
    assert abs(ratio - 8.0) < 0.2


def test_attention_alternating_hybrid():
    """Verifies verified alternating attention models half layers as SWA and half as global."""
    hw = make_test_hardware(avail_ram_mb=32000)
    # 28 layers: 14 sliding (4096 tokens), 14 global (32768 tokens)
    manifest_alt = make_test_manifest(
        layers=28, sliding_window=4096, sliding_window_pattern="alternating"
    )
    manifest_all_swa = make_test_manifest(
        layers=28, sliding_window=4096, sliding_window_pattern="all"
    )
    manifest_all_glob = make_test_manifest(layers=28, sliding_window=None)

    rep_alt = ModelCompatibilityCalculator.evaluate(
        manifest_alt, hw, target_context_tokens=32768
    )
    rep_swa = ModelCompatibilityCalculator.evaluate(
        manifest_all_swa, hw, target_context_tokens=32768
    )
    rep_glob = ModelCompatibilityCalculator.evaluate(
        manifest_all_glob, hw, target_context_tokens=32768
    )

    # Alternating memory must fall strictly between all-SWA and all-global
    assert (
        rep_swa.estimated_kv_cache_mb
        < rep_alt.estimated_kv_cache_mb
        < rep_glob.estimated_kv_cache_mb
    )
    # Expected token-layers:
    # All SWA: 28 * 4096 = 114,688
    # All Global: 28 * 32768 = 917,504
    # Alternating: (14 * 4096) + (14 * 32768) = 57,344 + 458,752 = 516,096
    expected_ratio = 516096 / 114688  # ~4.5x of SWA
    actual_ratio = rep_alt.estimated_kv_cache_mb / rep_swa.estimated_kv_cache_mb
    assert abs(actual_ratio - expected_ratio) < 0.2


def test_attention_explicit_hybrid_layers():
    """Verifies explicit sliding_window_layers metadata produces exact layer partitioning."""
    hw = make_test_hardware(avail_ram_mb=32000)
    # 28 layers total, exactly 7 sliding-window layers
    manifest_hybrid = make_test_manifest(
        layers=28, sliding_window=4096, sliding_window_layers=7
    )
    rep_hybrid = ModelCompatibilityCalculator.evaluate(
        manifest_hybrid, hw, target_context_tokens=32768
    )

    assert any("7 sliding-window layers" in note for note in rep_hybrid.evidence_notes)
    assert any(
        "21 global-attention layers" in note for note in rep_hybrid.evidence_notes
    )


def test_attention_unknown_pattern_conservative_estimate():
    """Verifies that unknown SWA layer distribution defaults to conservative global upper bound."""
    hw = make_test_hardware(avail_ram_mb=32000)
    # sliding_window is present, but NO pattern or layer count in metadata
    manifest_unknown = make_test_manifest(
        layers=28,
        sliding_window=4096,
        sliding_window_pattern=None,
        sliding_window_layers=None,
    )
    manifest_all_glob = make_test_manifest(layers=28, sliding_window=None)

    rep_unknown = ModelCompatibilityCalculator.evaluate(
        manifest_unknown, hw, target_context_tokens=32768
    )
    rep_glob = ModelCompatibilityCalculator.evaluate(
        manifest_all_glob, hw, target_context_tokens=32768
    )

    # Must match conservative global upper bound exactly
    assert rep_unknown.estimated_kv_cache_mb == rep_glob.estimated_kv_cache_mb
    # Confidence must reflect uncertainty (MEDIUM instead of HIGH)
    assert rep_unknown.confidence == ConfidenceLevel.MEDIUM
    assert any("conservative upper bound" in c for c in rep_unknown.caveats)


def test_kv_cache_quantization_format_reduction():
    """Verifies that q4_0 KV cache takes ~28% of fp16 cache size."""
    hw = make_test_hardware()
    manifest = make_test_manifest()

    report_fp16 = ModelCompatibilityCalculator.evaluate(
        manifest, hw, target_context_tokens=8192, kv_format="fp16"
    )
    report_q4_0 = ModelCompatibilityCalculator.evaluate(
        manifest, hw, target_context_tokens=8192, kv_format="q4_0"
    )

    assert report_fp16.estimated_kv_cache_mb is not None
    assert report_q4_0.estimated_kv_cache_mb is not None
    # 0.5625 / 2.0 = 0.28125
    ratio = report_q4_0.estimated_kv_cache_mb / report_fp16.estimated_kv_cache_mb
    assert abs(ratio - 0.28) < 0.05


def test_explicit_key_length_dimension():
    """Verifies that decoupled key_length is used when specified in hyperparameters."""
    hw = make_test_hardware()
    # Normal derived d_head = 4096 / 32 = 128
    manifest_normal = make_test_manifest(
        embedding=4096, heads=32, heads_kv=8, key_length=None
    )
    # Explicit key_length = 256 (Gemma 2 style)
    manifest_explicit = make_test_manifest(
        embedding=4096, heads=32, heads_kv=8, key_length=256
    )

    rep_norm = ModelCompatibilityCalculator.evaluate(manifest_normal, hw)
    rep_exp = ModelCompatibilityCalculator.evaluate(manifest_explicit, hw)

    assert rep_norm.estimated_kv_cache_mb is not None
    assert rep_exp.estimated_kv_cache_mb is not None
    # Explicit 256 should yield exactly 2x KV cache size of 128
    assert (
        abs(rep_exp.estimated_kv_cache_mb / rep_norm.estimated_kv_cache_mb - 2.0) < 0.1
    )


def test_incomplete_metadata_safety():
    """Verifies safe failure when model lacks hyperparameters or essential dimensions."""
    hw = make_test_hardware()

    # Case 1: Hyperparameters is None (unconfigured SafeTensors)
    manifest_empty = make_test_manifest(has_hyperparams=False)
    rep1 = ModelCompatibilityCalculator.evaluate(manifest_empty, hw)
    assert rep1.status == CompatibilityStatus.INSUFFICIENT_METADATA
    assert rep1.confidence == ConfidenceLevel.NONE
    assert rep1.estimated_kv_cache_mb is None

    # Case 2: Layer count missing
    manifest_no_layers = make_test_manifest(layers=None)  # type: ignore
    manifest_no_layers.hyperparameters.layer_count = None  # type: ignore
    rep2 = ModelCompatibilityCalculator.evaluate(manifest_no_layers, hw)
    assert rep2.status == CompatibilityStatus.INSUFFICIENT_METADATA


def test_fit_status_verdicts():
    """Verifies confident fit, marginal fit, and likely OOM thresholds."""
    manifest = make_test_manifest(file_size_bytes=4_000_000_000)  # ~3814 MB

    # Available RAM = 32 GB -> Confident Fit
    hw_huge = make_test_hardware(total_ram_mb=32768, avail_ram_mb=30000)
    rep_huge = ModelCompatibilityCalculator.evaluate(
        manifest, hw_huge, target_context_tokens=4096
    )
    assert rep_huge.status == CompatibilityStatus.CONFIDENT_FIT

    # Available RAM = 5.2 GB -> Marginal Fit (fits unbuffered, but tight)
    hw_tight = make_test_hardware(total_ram_mb=8192, avail_ram_mb=5200)
    rep_tight = ModelCompatibilityCalculator.evaluate(
        manifest, hw_tight, target_context_tokens=4096
    )
    assert rep_tight.status == CompatibilityStatus.MARGINAL_FIT

    # Available RAM = 3.0 GB -> Likely OOM
    hw_starved = make_test_hardware(total_ram_mb=8192, avail_ram_mb=3000)
    rep_starved = ModelCompatibilityCalculator.evaluate(
        manifest, hw_starved, target_context_tokens=4096
    )
    assert rep_starved.status == CompatibilityStatus.LIKELY_OOM


def test_advisory_gpu_offload_calculations():
    """Verifies advisory layer offload calculations for discrete and unified memory."""
    manifest = make_test_manifest(layers=28, file_size_bytes=4_000_000_000)

    # 1. Discrete GPU with 12 GB free VRAM (can hold all 28 layers)
    hw_big_gpu = make_test_hardware(gpu_vram_mb=12288, is_unified=False)
    rep_big = ModelCompatibilityCalculator.evaluate(manifest, hw_big_gpu)
    assert rep_big.advisory_gpu_layers == 28
    assert rep_big.advisory_vram_usage_mb is not None

    # 2. Discrete GPU with 3 GB free VRAM (hybrid offload: can hold ~15 layers)
    hw_small_gpu = make_test_hardware(gpu_vram_mb=3072, is_unified=False)
    rep_small = ModelCompatibilityCalculator.evaluate(manifest, hw_small_gpu)
    assert rep_small.advisory_gpu_layers is not None
    assert 0 < rep_small.advisory_gpu_layers < 28

    # 3. Discrete GPU with unknown/zero free VRAM
    hw_unknown_vram = HardwareProfile(
        total_ram_mb=16384,
        available_ram_mb=12000,
        cpu_cores_logical=8,
        gpus=[
            GPUInfo(
                name="Unknown",
                accelerator=BackendAccelerator.CUDA,
                available_vram_mb=None,
            )
        ],
    )
    rep_unk = ModelCompatibilityCalculator.evaluate(manifest, hw_unknown_vram)
    assert rep_unk.advisory_gpu_layers is None

    # 4. Apple Silicon Unified Memory
    hw_metal = make_test_hardware(
        gpu_vram_mb=None, is_unified=True, accelerator=BackendAccelerator.METAL
    )
    rep_metal = ModelCompatibilityCalculator.evaluate(manifest, hw_metal)
    assert rep_metal.advisory_gpu_layers == 28
    assert rep_metal.advisory_vram_usage_mb is None


def test_compatibility_empty_or_zero_byte_file():
    """Verifies that an empty (0 bytes) file returns INSUFFICIENT_METADATA without false positive fit."""
    hw = make_test_hardware()
    manifest = make_test_manifest(file_size_bytes=0)
    report = ModelCompatibilityCalculator.evaluate(manifest, hw)
    assert report.status == CompatibilityStatus.INSUFFICIENT_METADATA
    assert report.confidence == ConfidenceLevel.NONE
    assert report.total_estimated_ram_mb is None
    assert any("empty or less than 1 MB" in c for c in report.caveats)


def test_compatibility_zero_layers_or_heads():
    """Verifies that zero layers or heads triggers INSUFFICIENT_METADATA."""
    hw = make_test_hardware()
    # layers=0
    m_zero_layers = make_test_manifest(layers=0)
    rep_zero_l = ModelCompatibilityCalculator.evaluate(m_zero_layers, hw)
    assert rep_zero_l.status == CompatibilityStatus.INSUFFICIENT_METADATA
    assert rep_zero_l.confidence == ConfidenceLevel.NONE

    # heads=0
    m_zero_heads = make_test_manifest(heads=0)
    rep_zero_h = ModelCompatibilityCalculator.evaluate(m_zero_heads, hw)
    assert rep_zero_h.status == CompatibilityStatus.INSUFFICIENT_METADATA
    assert rep_zero_h.confidence == ConfidenceLevel.NONE


def test_compatibility_unknown_vram_records_explicit_evidence():
    """Verifies that unknown discrete GPU VRAM refuses offload computation and records explicit note."""
    manifest = make_test_manifest()
    hw = HardwareProfile(
        total_ram_mb=16384,
        available_ram_mb=12000,
        cpu_cores_logical=8,
        gpus=[
            GPUInfo(
                name="NVIDIA Discrete Without VRAM Reporting",
                accelerator=BackendAccelerator.CUDA,
                available_vram_mb=None,
            )
        ],
    )
    report = ModelCompatibilityCalculator.evaluate(manifest, hw)
    assert report.advisory_gpu_layers is None
    assert report.advisory_vram_usage_mb is None
    assert any(
        "refusing to infer GPU memory or compute offload" in note
        for note in report.evidence_notes
    )

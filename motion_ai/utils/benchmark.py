"""
Performance Benchmark Script for Human Motion Intelligence System.
Profile inference speed, memory usage, and throughput for all models.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np

# Add project root to path
import os, sys; sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

from models.posenet.architecture import StackedHourglass
from models.classifier.architecture import MoveClassifier
from models.predictor.architecture import MotionFormer
from src.config import (
    PoseNetConfig, ClassifierConfig, PredictorConfig,
    NUM_KEYPOINTS, MOVEMENT_CLASSES
)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    model_name: str
    device: str
    batch_size: int
    avg_latency_ms: float
    std_latency_ms: float
    throughput_fps: float
    memory_mb: float
    params_count: int
    
    def __str__(self):
        return (
            f"\n{self.model_name} ({self.device}, batch={self.batch_size})\n"
            f"  Latency: {self.avg_latency_ms:.2f} ± {self.std_latency_ms:.2f} ms\n"
            f"  Throughput: {self.throughput_fps:.1f} FPS\n"
            f"  Memory: {self.memory_mb:.1f} MB\n"
            f"  Parameters: {self.params_count:,}"
        )


class ModelBenchmark:
    """Benchmark utility for a single model."""

    def __init__(
        self,
        model: nn.Module,
        name: str,
        device: str = "cpu",
        warmup_runs: int = 10,
        benchmark_runs: int = 100
    ):
        self.model = model.to(device)
        self.name = name
        self.device = device
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs

    def run(
        self,
        example_input: torch.Tensor,
        batch_size: int = 1
    ) -> BenchmarkResult:
        """Run benchmark."""
        self.model.eval()
        
        # Prepare input
        x = example_input.to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                _ = self.model(x)
        
        # Synchronize if CUDA
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark
        latencies = []
        with torch.no_grad():
            for _ in range(self.benchmark_runs):
                start = time.perf_counter()
                _ = self.model(x)
                if self.device == "cuda":
                    torch.cuda.synchronize()
                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # ms
        
        latencies = np.array(latencies)
        avg_latency = latencies.mean()
        std_latency = latencies.std()
        
        # Throughput
        throughput = (batch_size * 1000) / avg_latency
        
        # Memory usage
        if self.device == "cuda":
            memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            torch.cuda.reset_peak_memory_stats()
        else:
            # Estimate from model size
            memory_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024 / 1024
        
        # Parameter count
        params_count = sum(p.numel() for p in self.model.parameters())
        
        return BenchmarkResult(
            model_name=self.name,
            device=self.device,
            batch_size=batch_size,
            avg_latency_ms=avg_latency,
            std_latency_ms=std_latency,
            throughput_fps=throughput,
            memory_mb=memory_mb,
            params_count=params_count
        )


def benchmark_posenet(device: str = "cpu", batch_sizes: List[int] = [1, 4, 8]) -> List[BenchmarkResult]:
    """Benchmark PoseNet model."""
    config = PoseNetConfig()
    
    model = StackedHourglass(
        n_stacks=config.n_stacks,
        n_features=config.n_features,
        n_keypoints=NUM_KEYPOINTS
    )
    
    benchmark = ModelBenchmark(model, "PoseNet (StackedHourglass)", device)
    
    results = []
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 3, config.input_size, config.input_size)
        result = benchmark.run(x, batch_size)
        results.append(result)
    
    return results


def benchmark_classifier(device: str = "cpu", batch_sizes: List[int] = [1, 4, 8]) -> List[BenchmarkResult]:
    """Benchmark MoveClassifier model."""
    config = ClassifierConfig()
    
    model = MoveClassifier(
        input_dim=NUM_KEYPOINTS * 2,
        num_classes=len(MOVEMENT_CLASSES),
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        dropout=config.dropout
    )
    
    benchmark = ModelBenchmark(model, "MoveClassifier (BiLSTM+Attention)", device)
    
    results = []
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, config.sequence_length, NUM_KEYPOINTS * 2)
        result = benchmark.run(x, batch_size)
        results.append(result)
    
    return results


def benchmark_predictor(device: str = "cpu", batch_sizes: List[int] = [1, 4, 8]) -> List[BenchmarkResult]:
    """Benchmark MotionFormer model."""
    config = PredictorConfig()
    
    model = MotionFormer(
        n_keypoints=NUM_KEYPOINTS,
        d_model=config.d_model,
        nhead=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        past_frames=config.past_len,
        future_frames=config.future_len,
        learnable_pe=config.learnable_pe
    )
    
    benchmark = ModelBenchmark(model, "MotionFormer (Transformer)", device)
    
    results = []
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, config.past_len, NUM_KEYPOINTS, 2)
        result = benchmark.run(x, batch_size)
        results.append(result)
    
    return results


def benchmark_pipeline(device: str = "cpu") -> BenchmarkResult:
    """Benchmark full pipeline (all models sequentially)."""
    config_pn = PoseNetConfig()
    config_clf = ClassifierConfig()
    config_pred = PredictorConfig()
    
    # Create all models
    posenet = StackedHourglass(
        n_stacks=config_pn.n_stacks,
        n_features=config_pn.n_features,
        n_keypoints=NUM_KEYPOINTS
    ).to(device)
    
    classifier = MoveClassifier(
        input_dim=NUM_KEYPOINTS * 2,
        num_classes=len(MOVEMENT_CLASSES),
        d_model=config_clf.d_model,
        n_layers=config_clf.n_layers,
        n_heads=config_clf.n_heads,
        dropout=config_clf.dropout
    ).to(device)
    
    predictor = MotionFormer(
        n_keypoints=NUM_KEYPOINTS,
        d_model=config_pred.d_model,
        nhead=config_pred.nhead,
        num_encoder_layers=config_pred.num_encoder_layers,
        num_decoder_layers=config_pred.num_decoder_layers,
        dim_feedforward=config_pred.dim_feedforward,
        dropout=config_pred.dropout,
        past_frames=config_pred.past_len,
        future_frames=config_pred.future_len,
        learnable_pe=config_pred.learnable_pe
    ).to(device)
    
    posenet.eval()
    classifier.eval()
    predictor.eval()
    
    # Warmup
    image = torch.randn(1, 3, config_pn.input_size, config_pn.input_size).to(device)
    keypoints = torch.randn(1, config_clf.sequence_length, NUM_KEYPOINTS * 2).to(device)
    motion = torch.randn(1, config_pred.past_len, NUM_KEYPOINTS, 2).to(device)
    
    with torch.no_grad():
        for _ in range(10):
            _ = posenet(image)
            _ = classifier(keypoints)
            _ = predictor(motion)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(50):
            start = time.perf_counter()
            _ = posenet(image)
            _ = classifier(keypoints)
            _ = predictor(motion)
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
    
    latencies = np.array(latencies)
    
    # Total params
    total_params = (
        sum(p.numel() for p in posenet.parameters()) +
        sum(p.numel() for p in classifier.parameters()) +
        sum(p.numel() for p in predictor.parameters())
    )
    
    return BenchmarkResult(
        model_name="Full Pipeline",
        device=device,
        batch_size=1,
        avg_latency_ms=latencies.mean(),
        std_latency_ms=latencies.std(),
        throughput_fps=1000 / latencies.mean(),
        memory_mb=0,  # Would need individual measurements
        params_count=total_params
    )


def run_full_benchmark(device: str = "cpu", batch_sizes: List[int] = [1, 4, 8]):
    """Run comprehensive benchmark of all models."""
    print("=" * 70)
    print(f"HUMAN MOTION INTELLIGENCE SYSTEM - PERFORMANCE BENCHMARK")
    print(f"Device: {device.upper()}")
    print("=" * 70)
    
    all_results = []
    
    # Benchmark each model
    print("\n[1/4] Benchmarking PoseNet (Stacked Hourglass)...")
    results = benchmark_posenet(device, batch_sizes)
    for r in results:
        print(r)
        all_results.append(r)
    
    print("\n[2/4] Benchmarking MoveClassifier (BiLSTM+Attention)...")
    results = benchmark_classifier(device, batch_sizes)
    for r in results:
        print(r)
        all_results.append(r)
    
    print("\n[3/4] Benchmarking MotionFormer (Transformer)...")
    results = benchmark_predictor(device, batch_sizes)
    for r in results:
        print(r)
        all_results.append(r)
    
    print("\n[4/4] Benchmarking Full Pipeline...")
    result = benchmark_pipeline(device)
    print(result)
    all_results.append(result)
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Model':<35} {'Batch':<8} {'Latency (ms)':<15} {'FPS':<12}")
    print("-" * 70)
    for r in all_results:
        print(f"{r.model_name:<35} {r.batch_size:<8} {r.avg_latency_ms:>8.2f} ± {r.std_latency_ms:<5.2f} {r.throughput_fps:>10.1f}")
    
    # Real-time feasibility
    print("\n" + "=" * 70)
    print("REAL-TIME FEASIBILITY (30 FPS target)")
    print("=" * 70)
    for r in all_results:
        feasible = "✓ YES" if r.throughput_fps >= 30 else "✗ NO"
        print(f"{r.model_name} (batch={r.batch_size}): {r.throughput_fps:.1f} FPS {feasible}")
    
    return all_results


def profile_memory():
    """Profile memory usage during inference."""
    if not torch.cuda.is_available():
        print("CUDA not available for memory profiling")
        return
    
    print("\n" + "=" * 70)
    print("MEMORY PROFILE (CUDA)")
    print("=" * 70)
    
    device = "cuda"
    
    # Profile each model
    models = [
        ("PoseNet", lambda: StackedHourglass(2, 256, 17)),
        ("MoveClassifier", lambda: MoveClassifier(34, 15, 128, 2, 4)),
        ("MotionFormer", lambda: MotionFormer(17, 256, 8, 4, 4, 1024, 0.1, 20, 10, True))
    ]
    
    for name, model_fn in models:
        torch.cuda.reset_peak_memory_stats()
        
        model = model_fn().to(device)
        model.eval()
        
        # Run inference
        with torch.no_grad():
            if name == "PoseNet":
                x = torch.randn(1, 3, 256, 256).to(device)
            elif name == "MoveClassifier":
                x = torch.randn(1, 30, 34).to(device)
            else:
                x = torch.randn(1, 20, 17, 2).to(device)
            
            _ = model(x)
        
        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"{name}: {memory_mb:.1f} MB peak memory")
        
        del model
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Benchmark Human Motion Intelligence System")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device to run benchmarks on")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8],
                        help="Batch sizes to benchmark")
    parser.add_argument("--memory-profile", action="store_true",
                        help="Profile memory usage (requires CUDA)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Number of warmup runs")
    parser.add_argument("--runs", type=int, default=100,
                        help="Number of benchmark runs")
    
    args = parser.parse_args()
    
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    run_full_benchmark(args.device, args.batch_sizes)
    
    if args.memory_profile and args.device == "cuda":
        profile_memory()


if __name__ == "__main__":
    main()

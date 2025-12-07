#!/usr/bin/env python3
"""
Failure Simulation Script for RunPod Self-Healing Testing

Simulates various failure scenarios to test recovery mechanisms:
1. VRAM Exhaustion
2. System Memory Exhaustion
3. llama-server Crash
4. llama-server Hang
5. High Error Rate

Usage:
    python simulate_failures.py --failure-type vram --duration 60
    python simulate_failures.py --failure-type crash
    python simulate_failures.py --all
"""

import argparse
import subprocess
import time
import signal
import sys
import random

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not available. GPU simulations will be skipped.")


class FailureSimulator:
    """Simulates various failure scenarios"""

    def __init__(self, duration_seconds: int = 60):
        self.duration_seconds = duration_seconds

    def simulate_vram_exhaustion(self):
        """
        Simulate VRAM exhaustion by allocating large GPU tensors

        Expected recovery: torch.cuda.empty_cache()
        """
        if not TORCH_AVAILABLE:
            print("ERROR: PyTorch not available. Cannot simulate VRAM exhaustion.")
            return

        if not torch.cuda.is_available():
            print("ERROR: CUDA not available. Cannot simulate VRAM exhaustion.")
            return

        print("=" * 60)
        print("SIMULATING: VRAM Exhaustion")
        print("=" * 60)
        print(f"Duration: {self.duration_seconds}s")
        print("")

        # Get GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        print(f"GPU: {gpu_name}")
        print(f"Total VRAM: {gpu_memory_total:.2f} GB")
        print("")

        # Allocate tensors to fill ~90% of VRAM
        tensors = []
        target_gb = gpu_memory_total * 0.9

        print(f"Allocating {target_gb:.2f} GB of VRAM...")

        try:
            # Allocate in chunks
            chunk_size_mb = 1000  # 1GB chunks

            while True:
                allocated_gb = torch.cuda.memory_allocated(0) / (1024**3)

                if allocated_gb >= target_gb:
                    break

                # Allocate tensor (1000 MB ≈ 1000 * 1024 * 1024 / 4 floats)
                tensor = torch.randn(262144000, device='cuda')  # ~1GB
                tensors.append(tensor)

                print(f"  Allocated: {allocated_gb:.2f} GB / {target_gb:.2f} GB")

            print("")
            print(f"✓ Allocated {allocated_gb:.2f} GB of VRAM")
            print(f"✓ Holding for {self.duration_seconds}s...")
            print("")
            print("Recovery should trigger: torch.cuda.empty_cache()")
            print("")

            # Hold for duration
            time.sleep(self.duration_seconds)

        finally:
            # Cleanup
            print("Cleaning up...")
            for tensor in tensors:
                del tensor
            torch.cuda.empty_cache()
            print("✓ Cleanup complete")

    def simulate_oom_error(self):
        """
        Simulate system memory exhaustion

        Expected recovery: gc.collect()
        """
        print("=" * 60)
        print("SIMULATING: System Memory Exhaustion (OOM)")
        print("=" * 60)
        print(f"Duration: {self.duration_seconds}s")
        print("")

        # Allocate large lists to consume RAM
        memory_hogs = []
        target_mb = 5000  # 5GB

        print(f"Allocating ~{target_mb} MB of system memory...")

        try:
            chunk_size = 1024 * 1024  # 1MB chunks
            chunks_allocated = 0

            while chunks_allocated < target_mb:
                # Allocate 1MB of data
                chunk = [0] * chunk_size
                memory_hogs.append(chunk)
                chunks_allocated += 1

                if chunks_allocated % 500 == 0:
                    print(f"  Allocated: {chunks_allocated} MB")

            print("")
            print(f"✓ Allocated ~{chunks_allocated} MB")
            print(f"✓ Holding for {self.duration_seconds}s...")
            print("")
            print("Recovery should trigger: gc.collect()")
            print("")

            # Hold for duration
            time.sleep(self.duration_seconds)

        finally:
            # Cleanup
            print("Cleaning up...")
            memory_hogs.clear()
            import gc
            gc.collect()
            print("✓ Cleanup complete")

    def simulate_server_crash(self):
        """
        Simulate llama-server crash by killing the process

        Expected recovery: Process restart
        """
        print("=" * 60)
        print("SIMULATING: llama-server Crash")
        print("=" * 60)
        print("")

        # Find llama-server process
        try:
            result = subprocess.run(
                ['pgrep', '-f', 'llama-server'],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                print("ERROR: llama-server not running")
                return

            pids = result.stdout.strip().split('\n')
            pid = int(pids[0])

            print(f"Found llama-server (PID={pid})")
            print("")
            print("Sending SIGKILL (force crash)...")

            # Kill the process
            subprocess.run(['kill', '-9', str(pid)], check=True)

            print(f"✓ Killed llama-server (PID={pid})")
            print("")
            print("Recovery should trigger: Process restart")
            print("")
            print("Monitor logs: tail -f /workspace/logs/health-monitor.log")

        except Exception as e:
            print(f"ERROR: Failed to kill llama-server: {e}")

    def simulate_server_hang(self):
        """
        Simulate llama-server hang by sending SIGSTOP

        Expected recovery: Force kill + restart
        """
        print("=" * 60)
        print("SIMULATING: llama-server Hang")
        print("=" * 60)
        print("")

        # Find llama-server process
        try:
            result = subprocess.run(
                ['pgrep', '-f', 'llama-server'],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                print("ERROR: llama-server not running")
                return

            pids = result.stdout.strip().split('\n')
            pid = int(pids[0])

            print(f"Found llama-server (PID={pid})")
            print("")
            print("Sending SIGSTOP (freeze process)...")

            # Stop the process
            subprocess.run(['kill', '-STOP', str(pid)], check=True)

            print(f"✓ Frozen llama-server (PID={pid})")
            print(f"✓ Server will not respond for {self.duration_seconds}s")
            print("")
            print("Recovery should trigger: Force kill + restart")
            print("")

            # Hold for duration
            time.sleep(self.duration_seconds)

            # Resume the process (cleanup)
            print("Resuming process (cleanup)...")
            subprocess.run(['kill', '-CONT', str(pid)], check=True)
            print("✓ Process resumed")

        except Exception as e:
            print(f"ERROR: Failed to hang llama-server: {e}")

    def simulate_high_error_rate(self):
        """
        Simulate high error rate by sending bad requests

        Expected recovery: Health check retry
        """
        print("=" * 60)
        print("SIMULATING: High Error Rate")
        print("=" * 60)
        print(f"Duration: {self.duration_seconds}s")
        print("")

        endpoint = "http://localhost:8080/v1/chat/completions"
        error_count = 0
        start_time = time.time()

        print(f"Sending invalid requests to {endpoint}...")
        print("")

        try:
            while time.time() - start_time < self.duration_seconds:
                # Send invalid request
                result = subprocess.run(
                    [
                        'curl',
                        '-s',
                        '-X', 'POST',
                        endpoint,
                        '-H', 'Content-Type: application/json',
                        '-d', '{"invalid": "data"}',  # Invalid payload
                    ],
                    capture_output=True,
                )

                error_count += 1

                if error_count % 10 == 0:
                    elapsed = time.time() - start_time
                    error_rate = error_count / elapsed if elapsed > 0 else 0
                    print(f"  Errors: {error_count}, Rate: {error_rate:.1f} errors/s")

                time.sleep(0.1)  # 10 requests/second

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            elapsed = time.time() - start_time
            error_rate = error_count / elapsed if elapsed > 0 else 0

            print("")
            print(f"✓ Sent {error_count} errors in {elapsed:.1f}s")
            print(f"✓ Error rate: {error_rate:.1f} errors/s")
            print("")
            print("Recovery should trigger if error rate > 25%")

    def simulate_all(self):
        """Run all failure simulations sequentially"""
        simulations = [
            ("VRAM Exhaustion", self.simulate_vram_exhaustion),
            ("OOM Error", self.simulate_oom_error),
            ("Server Crash", self.simulate_server_crash),
            ("Server Hang", self.simulate_server_hang),
            ("High Error Rate", self.simulate_high_error_rate),
        ]

        print("=" * 60)
        print("RUNNING ALL FAILURE SIMULATIONS")
        print("=" * 60)
        print("")

        for name, simulation_fn in simulations:
            print(f"\n>>> Running: {name}\n")
            try:
                simulation_fn()
            except Exception as e:
                print(f"ERROR in {name}: {e}")

            # Wait between simulations
            print("\nWaiting 30s before next simulation...")
            time.sleep(30)

        print("\n" + "=" * 60)
        print("ALL SIMULATIONS COMPLETE")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Failure Simulation for RunPod Self-Healing Testing"
    )

    parser.add_argument(
        '--failure-type',
        choices=['vram', 'oom', 'crash', 'hang', 'errors'],
        help='Type of failure to simulate',
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all failure simulations',
    )

    parser.add_argument(
        '--duration',
        type=int,
        default=60,
        help='Duration in seconds (default: 60)',
    )

    args = parser.parse_args()

    if not args.failure_type and not args.all:
        parser.print_help()
        sys.exit(1)

    simulator = FailureSimulator(duration_seconds=args.duration)

    if args.all:
        simulator.simulate_all()
    elif args.failure_type == 'vram':
        simulator.simulate_vram_exhaustion()
    elif args.failure_type == 'oom':
        simulator.simulate_oom_error()
    elif args.failure_type == 'crash':
        simulator.simulate_server_crash()
    elif args.failure_type == 'hang':
        simulator.simulate_server_hang()
    elif args.failure_type == 'errors':
        simulator.simulate_high_error_rate()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Health Monitor Daemon for RunPod Serverless Worker
Runs continuous monitoring loop with self-healing recovery

Usage:
    python monitor_daemon.py --port 8080 --interval 30
"""

import argparse
import asyncio
import logging
import sys
import signal
from datetime import datetime

from health_monitor import HealthMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


class MonitorDaemon:
    """Health monitor daemon"""

    def __init__(
        self,
        port: int = 8080,
        interval: int = 30,
        memory_threshold: float = 90.0,
        gpu_threshold: float = 90.0,
        error_rate_threshold: float = 0.25,
        recovery_cooldown: int = 60,
        enable_recovery: bool = True,
    ):
        self.port = port
        self.interval = interval
        self.running = False

        self.monitor = HealthMonitor(
            check_interval_seconds=interval,
            memory_threshold_percent=memory_threshold,
            gpu_memory_threshold_percent=gpu_threshold,
            error_rate_threshold=error_rate_threshold,
            recovery_cooldown_seconds=recovery_cooldown,
            enable_recovery=enable_recovery,
        )

        logger.info(
            f"Monitor Daemon initialized: "
            f"port={port}, interval={interval}s, recovery={enable_recovery}"
        )

    async def start(self):
        """Start the monitoring daemon"""
        self.running = True

        logger.info(
            f"[MonitorDaemon] Starting health monitoring loop "
            f"(interval={self.interval}s)"
        )

        try:
            await self.monitor.run_monitoring_loop(interval_seconds=self.interval)
        except Exception as e:
            logger.exception(f"[MonitorDaemon] Monitoring loop failed: {e}")
            raise

    def stop(self):
        """Stop the monitoring daemon"""
        logger.info("[MonitorDaemon] Stopping...")
        self.running = False


async def main(args):
    """Main entry point"""
    daemon = MonitorDaemon(
        port=args.port,
        interval=args.interval,
        memory_threshold=args.memory_threshold,
        gpu_threshold=args.gpu_threshold,
        error_rate_threshold=args.error_rate_threshold,
        recovery_cooldown=args.recovery_cooldown,
        enable_recovery=not args.no_recovery,
    )

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        logger.info(f"[MonitorDaemon] Received signal {sig}, shutting down...")
        daemon.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start monitoring
    logger.info("=" * 60)
    logger.info(f"RunPod Health Monitor Daemon - {datetime.now().isoformat()}")
    logger.info("=" * 60)
    logger.info(f"  Port: {args.port}")
    logger.info(f"  Interval: {args.interval}s")
    logger.info(f"  Memory Threshold: {args.memory_threshold}%")
    logger.info(f"  GPU Threshold: {args.gpu_threshold}%")
    logger.info(f"  Error Rate Threshold: {args.error_rate_threshold*100}%")
    logger.info(f"  Recovery Cooldown: {args.recovery_cooldown}s")
    logger.info(f"  Recovery Enabled: {not args.no_recovery}")
    logger.info("=" * 60)
    logger.info("")

    await daemon.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Health Monitor Daemon for RunPod Serverless Worker"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="llama-server port (default: 8080)",
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Health check interval in seconds (default: 30)",
    )

    parser.add_argument(
        "--memory-threshold",
        type=float,
        default=90.0,
        help="System memory threshold percent (default: 90.0)",
    )

    parser.add_argument(
        "--gpu-threshold",
        type=float,
        default=90.0,
        help="GPU memory threshold percent (default: 90.0)",
    )

    parser.add_argument(
        "--error-rate-threshold",
        type=float,
        default=0.25,
        help="Error rate threshold (default: 0.25 = 25%%)",
    )

    parser.add_argument(
        "--recovery-cooldown",
        type=int,
        default=60,
        help="Recovery cooldown in seconds (default: 60)",
    )

    parser.add_argument(
        "--no-recovery",
        action="store_true",
        help="Disable recovery actions (monitoring only)",
    )

    args = parser.parse_args()

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        logger.info("[MonitorDaemon] Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"[MonitorDaemon] Fatal error: {e}")
        sys.exit(1)

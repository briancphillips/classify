"""
Utilities for managing experiment results.
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from utils.logging import get_logger

logger = get_logger(__name__)

class ResultsManager:
    """Manages experiment results and artifacts."""
    
    def __init__(
        self,
        experiment_name: str,
        base_dir: str = "results",
        version: Optional[str] = None
    ):
        """
        Initialize ResultsManager.
        
        Args:
            experiment_name: Name of the experiment
            base_dir: Base directory for results
            version: Optional version string, defaults to timestamp
        """
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        self.version = version or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create experiment directory
        self.exp_dir = self.base_dir / experiment_name / self.version
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.checkpoint_dir = self.exp_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.plots_dir = self.exp_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        self.logs_dir = self.exp_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        logger.info(f"Created results directory structure at {self.exp_dir}")
    
    def save_config(self, config: Dict[str, Any]) -> None:
        """Save experiment configuration."""
        config_path = self.exp_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved experiment config to {config_path}")
    
    def save_metrics(self, metrics: Dict[str, Any]) -> None:
        """Save experiment metrics."""
        metrics_path = self.exp_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved experiment metrics to {metrics_path}")
    
    def get_checkpoint_dir(self) -> Path:
        """Get checkpoint directory path."""
        return self.checkpoint_dir
    
    def get_plots_dir(self) -> Path:
        """Get plots directory path."""
        return self.plots_dir
    
    def get_logs_dir(self) -> Path:
        """Get logs directory path."""
        return self.logs_dir
    
    def cleanup_old_versions(self, keep_latest: int = 5) -> None:
        """
        Clean up old experiment versions, keeping only the specified number of latest versions.
        
        Args:
            keep_latest: Number of latest versions to keep
        """
        exp_base_dir = self.base_dir / self.experiment_name
        if not exp_base_dir.exists():
            return
            
        versions = sorted([
            d for d in exp_base_dir.iterdir() 
            if d.is_dir() and d.name != self.version
        ], key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Keep the current version plus the specified number of latest versions
        versions_to_remove = versions[keep_latest-1:]
        for version_dir in versions_to_remove:
            shutil.rmtree(version_dir)
            logger.info(f"Removed old experiment version: {version_dir}")

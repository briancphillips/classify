#!/usr/bin/env python3
"""
Terminal User Interface for running poisoning experiments.
"""

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Button, Static, Select, Label, Switch, Input, Log
from textual.binding import Binding
from textual.worker import Worker, WorkerState
from textual import events
from rich.text import Text
import yaml
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

class ConfigSection(Static):
    """A section of configuration options."""
    
    def __init__(self, title: str) -> None:
        super().__init__()
        self.title = title
    
    def compose(self) -> ComposeResult:
        with Container():
            yield Label(f"[b]{self.title}[/b]", classes="section-title")
            yield Container(classes="section-content")

class ExperimentConfig(Static):
    """Widget for configuring experiment parameters."""
    
    def compose(self) -> ComposeResult:
        with Vertical():
            # Dataset Selection
            with ConfigSection("Dataset Configuration"):
                yield Select(
                    [(name, name) for name in ["cifar100", "gtsrb", "imagenette"]],
                    prompt="Select Dataset",
                    id="dataset",
                )
                yield Input(
                    placeholder="Subset size (optional)",
                    id="subset_size",
                )
            
            # Attack Configuration
            with ConfigSection("Attack Configuration"):
                yield Select(
                    [(name, name) for name in ["pgd", "ga", "label_flip"]],
                    prompt="Select Attack Type",
                    id="attack",
                )
                yield Input(
                    value="0.05",
                    placeholder="Poison ratio (0.0-1.0)",
                    id="poison_ratio",
                )
                yield Input(
                    placeholder="Target class (optional)",
                    id="target_class",
                )
                yield Input(
                    placeholder="Source class (optional)",
                    id="source_class",
                )
            
            # Training Parameters
            with ConfigSection("Training Configuration"):
                yield Input(
                    value="30",
                    placeholder="Number of epochs",
                    id="epochs",
                )
                yield Input(
                    value="0.001",
                    placeholder="Learning rate",
                    id="learning_rate",
                )
                yield Input(
                    value="128",
                    placeholder="Batch size",
                    id="batch_size",
                )
                yield Input(
                    value="2",
                    placeholder="Number of workers",
                    id="num_workers",
                )
            
            # Output Configuration
            with ConfigSection("Output Configuration"):
                yield Input(
                    value="results",
                    placeholder="Output directory",
                    id="output_dir",
                )
                yield Switch(value=True, id="save_individual")
                yield Label("Save individual results")

class ExperimentRunner(Static):
    """Widget for displaying experiment progress and results."""
    
    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("[b]Experiment Status[/b]", classes="section-title")
            yield Log(highlight=True, id="log")
            yield Button("Run Experiment", variant="primary", id="run")
            yield Button("Stop Experiment", variant="error", id="stop", disabled=True)

class PoisonExperimentTUI(App):
    """Main TUI application for running poisoning experiments."""
    
    CSS = """
    Screen {
        background: $surface;
    }
    
    ConfigSection {
        height: auto;
        margin: 1;
        padding: 1;
        border: solid $primary;
    }
    
    .section-title {
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }
    
    .section-content {
        margin-left: 1;
    }
    
    Select {
        width: 100%;
        margin-bottom: 1;
    }
    
    Input {
        width: 100%;
        margin-bottom: 1;
    }
    
    Button {
        margin: 1;
        min-width: 20;
    }
    
    #run {
        dock: bottom;
    }
    
    #stop {
        dock: bottom;
    }
    
    Label {
        margin-left: 1;
    }
    
    Log {
        height: 1fr;
        border: solid $primary;
        background: $surface-darken-1;
    }
    """
    
    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("r", "run_experiment", "Run", show=True),
        Binding("s", "stop_experiment", "Stop", show=True),
    ]
    
    def __init__(self):
        super().__init__()
        self.experiment_worker = None
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            yield ExperimentConfig()
            yield ExperimentRunner()
        yield Footer()
    
    def _get_config(self) -> Dict[str, Any]:
        """Get configuration from UI inputs."""
        config = {
            'dataset': self.query_one("#dataset").value,
            'attack': self.query_one("#attack").value,
            'poison_ratio': float(self.query_one("#poison_ratio").value),
            'epochs': int(self.query_one("#epochs").value),
            'learning_rate': float(self.query_one("#learning_rate").value),
            'batch_size': int(self.query_one("#batch_size").value),
            'num_workers': int(self.query_one("#num_workers").value),
            'output_dir': self.query_one("#output_dir").value,
        }
        
        # Optional parameters
        subset_size = self.query_one("#subset_size").value
        if subset_size:
            config['subset_size'] = int(subset_size)
            
        target_class = self.query_one("#target_class").value
        if target_class:
            config['target_class'] = int(target_class)
            
        source_class = self.query_one("#source_class").value
        if source_class:
            config['source_class'] = int(source_class)
        
        return config
    
    async def run_experiment_worker(self, config: Dict[str, Any]) -> None:
        """Run experiment in background worker."""
        log = self.query_one("#log")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Build command
            cmd = ["python", "poison.py"]
            for key, value in config.items():
                if value is not None:
                    cmd.extend([f"--{key.replace('_', '-')}", str(value)])
            
            # Create output directory
            os.makedirs(config['output_dir'], exist_ok=True)
            
            # Log command
            log.write_line("[green]Starting experiment...[/green]")
            log.write_line(f"Command: {' '.join(cmd)}")
            
            # Run experiment
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream output
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    log.write_line(output.strip())
            
            # Get return code
            return_code = process.wait()
            
            if return_code == 0:
                log.write_line("[green]Experiment completed successfully![/green]")
            else:
                log.write_line("[red]Experiment failed![/red]")
                stderr = process.stderr.read()
                if stderr:
                    log.write_line("[red]Error output:[/red]")
                    log.write_line(stderr)
            
        except Exception as e:
            log.write_line(f"[red]Error: {str(e)}[/red]")
        
        finally:
            # Re-enable run button, disable stop button
            self.query_one("#run").disabled = False
            self.query_one("#stop").disabled = True
    
    def action_run_experiment(self) -> None:
        """Run the experiment with current configuration."""
        config = self._get_config()
        
        # Clear log
        log = self.query_one("#log")
        log.clear()
        
        # Disable run button, enable stop button
        self.query_one("#run").disabled = True
        self.query_one("#stop").disabled = False
        
        # Start worker
        self.experiment_worker = self.run_experiment_worker(config)
    
    def action_stop_experiment(self) -> None:
        """Stop the currently running experiment."""
        if self.experiment_worker:
            self.experiment_worker.cancel()
            log = self.query_one("#log")
            log.write_line("[yellow]Experiment stopped by user[/yellow]")
            
            # Enable run button, disable stop button
            self.query_one("#run").disabled = False
            self.query_one("#stop").disabled = True

def main():
    """Run the TUI application."""
    app = PoisonExperimentTUI()
    app.run()

if __name__ == "__main__":
    main()

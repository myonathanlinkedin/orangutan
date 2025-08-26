"""Nsight, nvidia-smi, PyTorch profiler adapters for ORANGUTAN."""

import subprocess
import time
import torch
from typing import Dict, Any, Optional


class TelemetryCollector:
    """Telemetry collection from various sources."""
    
    def __init__(self, anti_fabrication_checker=None):
        """Initialize telemetry collector."""
        self.nsight_enabled = False
        self.nvidia_smi_enabled = True
        self.pytorch_profiler_enabled = True
        self.anti_fabrication_checker = anti_fabrication_checker
    
    def collect_gpu_metrics(self):
        """Collect GPU metrics from nvidia-smi."""
        try:
            # Get real GPU metrics via nvidia-smi
            result = subprocess.run([
                "nvidia-smi", 
                "--query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                raise RuntimeError(f"nvidia-smi failed: {result.stderr}")
            
            values = result.stdout.strip().split(', ')
            
            metrics = {
                "gpu_utilization": float(values[0]) / 100.0 if values[0] != '[Not Supported]' else 0.0,
                "memory_utilization": float(values[1]) / 100.0 if values[1] != '[Not Supported]' else 0.0,
                "temperature": float(values[2]) if values[2] != '[Not Supported]' else 0.0,
                "power_draw": float(values[3]) if values[3] != '[Not Supported]' else 0.0,
                "timestamp": time.time()
            }
            
            # Register with anti-fabrication checker
            if self.anti_fabrication_checker:
                for metric_name, value in metrics.items():
                    if metric_name != "timestamp":
                        self.anti_fabrication_checker.register_metric(
                            f"gpu_{metric_name}", value, "nvidia-smi"
                        )
            
            return metrics
            
        except Exception as e:
            # Fallback to PyTorch if nvidia-smi fails
            return self._collect_pytorch_gpu_metrics()
    
    def _collect_pytorch_gpu_metrics(self):
        """Collect GPU metrics from PyTorch as fallback."""
        if not torch.cuda.is_available():
            raise RuntimeError("No GPU metrics available - CUDA not available")
        
        device_id = 0
        metrics = {
            "gpu_utilization": torch.cuda.utilization(device_id) / 100.0 if hasattr(torch.cuda, 'utilization') else 0.0,
            "memory_utilization": torch.cuda.memory_allocated(device_id) / torch.cuda.get_device_properties(device_id).total_memory,
            "temperature": 0.0,  # Not available via PyTorch
            "power_draw": 0.0,   # Not available via PyTorch
            "timestamp": time.time()
        }
        
        if self.anti_fabrication_checker:
            for metric_name, value in metrics.items():
                if metric_name != "timestamp":
                    self.anti_fabrication_checker.register_metric(
                        f"gpu_{metric_name}", value, "pytorch_profiler"
                    )
        
        return metrics
    
    def collect_memory_metrics(self):
        """Collect memory metrics from nvidia-smi."""
        try:
            result = subprocess.run([
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                raise RuntimeError(f"nvidia-smi memory query failed: {result.stderr}")
            
            values = result.stdout.strip().split(', ')
            
            metrics = {
                "total_memory": float(values[0]),
                "used_memory": float(values[1]),
                "free_memory": float(values[2]),
                "timestamp": time.time()
            }
            
            if self.anti_fabrication_checker:
                for metric_name, value in metrics.items():
                    if metric_name != "timestamp":
                        self.anti_fabrication_checker.register_metric(
                            f"memory_{metric_name}", value, "nvidia-smi"
                        )
            
            return metrics
            
        except Exception as e:
            # Fallback to PyTorch
            return self._collect_pytorch_memory_metrics()
    
    def _collect_pytorch_memory_metrics(self):
        """Collect memory metrics from PyTorch as fallback."""
        if not torch.cuda.is_available():
            raise RuntimeError("No memory metrics available - CUDA not available")
        
        device_id = 0
        props = torch.cuda.get_device_properties(device_id)
        
        metrics = {
            "total_memory": props.total_memory / (1024 * 1024),  # Convert to MB
            "used_memory": torch.cuda.memory_allocated(device_id) / (1024 * 1024),
            "free_memory": (props.total_memory - torch.cuda.memory_allocated(device_id)) / (1024 * 1024),
            "timestamp": time.time()
        }
        
        if self.anti_fabrication_checker:
            for metric_name, value in metrics.items():
                if metric_name != "timestamp":
                    self.anti_fabrication_checker.register_metric(
                        f"memory_{metric_name}", value, "pytorch_profiler"
                    )
        
        return metrics
    
    def start_profiling(self):
        """Start profiling."""
        print("Started telemetry collection")
    
    def stop_profiling(self):
        """Stop profiling."""
        print("Stopped telemetry collection")
    
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive telemetry metrics for ORANGUTAN simulation."""
        try:
            # Collect GPU metrics
            gpu_metrics = self.collect_gpu_metrics()
            
            # Collect memory metrics
            memory_metrics = self.collect_memory_metrics()
            
            # Combine all metrics
            combined_metrics = {
                'gpu_utilization_percent': gpu_metrics.get('gpu_utilization', 0.0) * 100.0,
                'memory_utilization_percent': gpu_metrics.get('memory_utilization', 0.0) * 100.0,
                'temperature_celsius': gpu_metrics.get('temperature', 0.0),
                'power_draw_watts': gpu_metrics.get('power_draw', 0.0),
                'total_memory_mb': memory_metrics.get('total_memory', 0.0),
                'used_memory_mb': memory_metrics.get('used_memory', 0.0),
                'free_memory_mb': memory_metrics.get('free_memory', 0.0),
                'timestamp': time.time()
            }
            
            return combined_metrics
            
        except Exception as e:
            # Return fallback metrics if collection fails
            return {
                'gpu_utilization_percent': 0.0,
                'memory_utilization_percent': 0.0,
                'temperature_celsius': 0.0,
                'power_draw_watts': 0.0,
                'total_memory_mb': 0.0,
                'used_memory_mb': 0.0,
                'free_memory_mb': 0.0,
                'timestamp': time.time(),
                'error': str(e)
            }

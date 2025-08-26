"""Telemetry validator to ensure metrics come from real hardware monitoring."""

import subprocess
import time
import json
import psutil
import torch
from typing import Dict, Any, List, Optional
from pathlib import Path


class TelemetryValidator:
    """Validates that telemetry data comes from actual hardware monitoring."""
    
    def __init__(self):
        """Initialize telemetry validator."""
        self.gpu_available = torch.cuda.is_available()
        self.validation_history = []
        
    def validate_nvidia_smi_available(self) -> bool:
        """Validate that nvidia-smi is available and functional."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", 
                 "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0 and len(result.stdout.strip()) > 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def get_real_gpu_metrics(self) -> Dict[str, Any]:
        """Get real GPU metrics from nvidia-smi."""
        if not self.validate_nvidia_smi_available():
            raise RuntimeError("nvidia-smi not available - cannot validate GPU metrics")
        
        try:
            # Get GPU utilization
            result = subprocess.run([
                "nvidia-smi", 
                "--query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                raise RuntimeError(f"nvidia-smi failed: {result.stderr}")
            
            values = result.stdout.strip().split(', ')
            
            metrics = {
                "gpu_utilization_percent": float(values[0]) if values[0] != '[Not Supported]' else 0.0,
                "memory_utilization_percent": float(values[1]) if values[1] != '[Not Supported]' else 0.0,
                "temperature_celsius": float(values[2]) if values[2] != '[Not Supported]' else 0.0,
                "power_draw_watts": float(values[3]) if values[3] != '[Not Supported]' else 0.0,
                "timestamp": time.time(),
                "source": "nvidia-smi"
            }
            
            # Get memory info
            memory_result = subprocess.run([
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits"
            ], capture_output=True, text=True, timeout=10)
            
            if memory_result.returncode == 0:
                memory_values = memory_result.stdout.strip().split(', ')
                metrics.update({
                    "memory_total_mb": float(memory_values[0]),
                    "memory_used_mb": float(memory_values[1]),
                    "memory_free_mb": float(memory_values[2])
                })
            
            return metrics
            
        except Exception as e:
            raise RuntimeError(f"Failed to get real GPU metrics: {e}")
    
    def get_pytorch_gpu_metrics(self) -> Dict[str, Any]:
        """Get GPU metrics from PyTorch."""
        if not self.gpu_available:
            raise RuntimeError("CUDA not available - cannot get PyTorch metrics")
        
        device_id = 0
        metrics = {
            "device_name": torch.cuda.get_device_name(device_id),
            "device_capability": torch.cuda.get_device_capability(device_id),
            "memory_allocated_bytes": torch.cuda.memory_allocated(device_id),
            "memory_reserved_bytes": torch.cuda.memory_reserved(device_id),
            "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device_id),
            "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(device_id),
            "timestamp": time.time(),
            "source": "pytorch_profiler"
        }
        
        # Get device properties
        props = torch.cuda.get_device_properties(device_id)
        
        # Safely access device properties (some may not exist in older PyTorch versions)
        device_props = {
            "total_memory_bytes": getattr(props, 'total_memory', 0),
            "multiprocessor_count": getattr(props, 'multi_processor_count', 0)
        }
        
        # Try to get optional properties that may not exist in all PyTorch versions
        optional_props = [
            "max_threads_per_multi_processor",  # Note: "multi_processor" not "multiprocessor"
            "max_shared_memory_per_block",
            "max_threads_per_block",
            "warp_size"
        ]
        
        for prop_name in optional_props:
            if hasattr(props, prop_name):
                device_props[prop_name] = getattr(props, prop_name)
        
        metrics.update(device_props)
        
        return metrics
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-level metrics."""
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage('/').percent if hasattr(psutil.disk_usage, '__call__') else 0.0,
            "timestamp": time.time(),
            "source": "system_monitor"
        }
    
    def validate_metric_is_dynamic(
        self, 
        metric_getter_func, 
        metric_name: str, 
        samples: int = 5, 
        interval: float = 0.5
    ) -> bool:
        """Validate that a metric changes over time (not hardcoded)."""
        values = []
        
        for i in range(samples):
            try:
                metric_data = metric_getter_func()
                if isinstance(metric_data, dict) and metric_name in metric_data:
                    values.append(metric_data[metric_name])
                else:
                    values.append(metric_data)
                
                if i < samples - 1:
                    time.sleep(interval)
            except Exception as e:
                raise RuntimeError(f"Failed to get metric {metric_name}: {e}")
        
        # Check for variation (not all values identical)
        unique_values = set(values)
        if len(unique_values) == 1:
            # Allow for small constant values in some cases
            if values[0] == 0.0:
                print(f"Warning: Metric {metric_name} is consistently 0.0 - may be inactive")
                return True
            else:
                raise ValueError(
                    f"ANTI-FABRICATION VIOLATION: Metric {metric_name} shows no variation "
                    f"across {samples} samples - likely hardcoded"
                )
        
        return True
    
    def validate_telemetry_sources(self) -> Dict[str, bool]:
        """Validate all telemetry sources are working."""
        results = {}
        
        # Test nvidia-smi
        try:
            results["nvidia_smi"] = self.validate_nvidia_smi_available()
        except Exception as e:
            results["nvidia_smi"] = False
            print(f"nvidia-smi validation failed: {e}")
        
        # Test PyTorch GPU access
        try:
            metrics = self.get_pytorch_gpu_metrics()
            results["pytorch_gpu"] = len(metrics) > 0
        except Exception as e:
            results["pytorch_gpu"] = False
            print(f"PyTorch GPU validation failed: {e}")
        
        # Test system metrics
        try:
            metrics = self.get_system_metrics()
            results["system_monitor"] = len(metrics) > 0
        except Exception as e:
            results["system_monitor"] = False
            print(f"System monitor validation failed: {e}")
        
        return results
    
    def create_baseline_measurements(self, duration_seconds: int = 30) -> Dict[str, List[Any]]:
        """Create baseline measurements for comparison."""
        baseline = {
            "gpu_metrics": [],
            "pytorch_metrics": [],
            "system_metrics": [],
            "timestamps": []
        }
        
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            timestamp = time.time()
            
            try:
                gpu_metrics = self.get_real_gpu_metrics()
                baseline["gpu_metrics"].append(gpu_metrics)
            except Exception as e:
                print(f"Failed to get GPU metrics: {e}")
            
            try:
                pytorch_metrics = self.get_pytorch_gpu_metrics()
                baseline["pytorch_metrics"].append(pytorch_metrics)
            except Exception as e:
                print(f"Failed to get PyTorch metrics: {e}")
            
            try:
                system_metrics = self.get_system_metrics()
                baseline["system_metrics"].append(system_metrics)
            except Exception as e:
                print(f"Failed to get system metrics: {e}")
            
            baseline["timestamps"].append(timestamp)
            time.sleep(1.0)  # Sample every second
        
        return baseline
    
    def export_telemetry_validation_report(self, output_path: str):
        """Export telemetry validation report."""
        validation_results = self.validate_telemetry_sources()
        
        report = {
            "validation_timestamp": time.time(),
            "telemetry_sources": validation_results,
            "gpu_available": self.gpu_available,
            "validation_history": self.validation_history
        }
        
        # Add current metrics snapshot
        try:
            report["current_gpu_metrics"] = self.get_real_gpu_metrics()
        except Exception as e:
            report["current_gpu_metrics"] = f"Error: {e}"
        
        try:
            report["current_pytorch_metrics"] = self.get_pytorch_gpu_metrics()
        except Exception as e:
            report["current_pytorch_metrics"] = f"Error: {e}"
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate telemetry validation report as dictionary."""
        validation_results = self.validate_telemetry_sources()
        
        report = {
            "validation_timestamp": time.time(),
            "telemetry_sources": validation_results,
            "gpu_available": self.gpu_available,
            "validation_history": self.validation_history
        }
        
        # Add current metrics snapshot
        try:
            report["current_gpu_metrics"] = self.get_real_gpu_metrics()
        except Exception as e:
            report["current_gpu_metrics"] = f"Error: {e}"
        
        try:
            report["current_pytorch_metrics"] = self.get_pytorch_gpu_metrics()
        except Exception as e:
            report["current_pytorch_metrics"] = f"Error: {e}"
        
        return report

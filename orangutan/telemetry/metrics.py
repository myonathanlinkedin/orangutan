"""Throughput, latency, TFLOPs, energy/token calculations for ORANGUTAN."""


class MetricsCalculator:
    """Calculate various performance metrics."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.metrics_history = []
    
    def calculate_throughput(self, completed_workloads, time_seconds):
        """Calculate throughput (workloads per second)."""
        if time_seconds > 0:
            return len(completed_workloads) / time_seconds
        return 0.0
    
    def calculate_latency(self, start_time, end_time):
        """Calculate latency in seconds."""
        return end_time - start_time
    
    def calculate_tflops(self, operations, time_seconds):
        """Calculate TFLOPs."""
        if time_seconds > 0:
            return (operations * 1e-12) / time_seconds
        return 0.0
    
    def calculate_energy_per_token(self, energy_joules, tokens_processed):
        """Calculate energy consumption per token."""
        if tokens_processed > 0:
            return energy_joules / tokens_processed
        return 0.0
    
    def add_metric(self, metric_name: str, value: float, timestamp: float):
        """Add metric to history."""
        self.metrics_history.append({
            "name": metric_name,
            "value": value,
            "timestamp": timestamp
        })
    
    def get_metrics_summary(self):
        """Get summary of collected metrics."""
        if not self.metrics_history:
            return {}
        
        summary = {}
        for metric in self.metrics_history:
            name = metric["name"]
            if name not in summary:
                summary[name] = []
            summary[name].append(metric["value"])
        
        # Calculate averages
        for name in summary:
            summary[name] = sum(summary[name]) / len(summary[name])
        
        return summary

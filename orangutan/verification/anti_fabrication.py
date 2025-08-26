"""Anti-fabrication checker to ensure all metrics are empirically generated."""

import hashlib
import json
import time
from typing import Dict, Any, List, Optional
import inspect


class AntiFabricationChecker:
    """Ensures all metrics are derived from actual execution, not fabricated."""

    def __init__(self):
        """Initialize anti-fabrication checker."""
        self.metric_sources = {}
        self.call_stack_hashes = {}
        self.forbidden_patterns = [
            "hardcoded", "manual", "fake", "dummy", "placeholder",
            "static", "fixed", "constant", "mock"
        ]
        self.telemetry_required_sources = [
            "nvidia-smi", "nsight", "pytorch_profiler", "gpu_metrics", 
            "system_monitor"
        ]
        
    def register_metric(
        self, 
        metric_name: str, 
        value: Any, 
        source: str,
        timestamp: Optional[float] = None
    ):
        """Register a metric with its source for verification."""
        if timestamp is None:
            timestamp = time.time()
            
        # Get call stack to verify source
        call_stack = inspect.stack()
        caller_info = {
            "function": call_stack[1].function,
            "filename": call_stack[1].filename,
            "lineno": call_stack[1].lineno
        }
        
        # Check for forbidden patterns in source
        self._validate_source(source, caller_info)
        
        # Create metric entry
        metric_entry = {
            "value": value,
            "source": source,
            "timestamp": timestamp,
            "caller": caller_info,
            "stack_hash": self._compute_stack_hash(call_stack)
        }
        
        if metric_name not in self.metric_sources:
            self.metric_sources[metric_name] = []
        
        self.metric_sources[metric_name].append(metric_entry)
        
    def _validate_source(self, source: str, caller_info: Dict[str, Any]):
        """Validate that source is legitimate."""
        source_lower = source.lower()
        
        # Check for forbidden patterns
        for pattern in self.forbidden_patterns:
            if pattern in source_lower:
                raise ValueError(
                    f"ANTI-FABRICATION VIOLATION: Forbidden pattern '{pattern}' "
                    f"found in source '{source}' at {caller_info}"
                )
        
        # Check if source is from a legitimate telemetry source
        is_legitimate = any(
            legitimate in source_lower 
            for legitimate in self.telemetry_required_sources
        )
        
        if not is_legitimate:
            raise ValueError(
                f"ANTI-FABRICATION VIOLATION: Metric source '{source}' "
                f"is not from approved telemetry sources. "
                f"Approved sources: {self.telemetry_required_sources}"
            )
    
    def _compute_stack_hash(self, call_stack: List[Any]) -> str:
        """Compute hash of call stack to detect duplicate/fabricated calls."""
        stack_info = []
        for frame in call_stack[:5]:  # Limit depth
            stack_info.append(f"{frame.filename}:{frame.function}:{frame.lineno}")
        
        # Add current time to differentiate legitimate repeated calls
        current_time_ms = int(time.time() * 1000) % 10000  # Last 4 digits of timestamp
        stack_info.append(f"t:{current_time_ms}")
        
        stack_str = "->".join(stack_info)
        return hashlib.sha256(stack_str.encode()).hexdigest()[:16]
    
    def validate_metric_integrity(self, metric_name: str) -> bool:
        """Validate that metric has legitimate sources."""
        if metric_name not in self.metric_sources:
            raise ValueError(f"Metric '{metric_name}' not found in registry")
        
        entries = self.metric_sources[metric_name]
        
        # Check for suspicious patterns in duplicate stack hashes
        stack_hashes = [entry["stack_hash"] for entry in entries]
        hash_counts = {}
        for hash_val in stack_hashes:
            hash_counts[hash_val] = hash_counts.get(hash_val, 0) + 1
        
        # Allow some duplication for legitimate measurement loops
        max_allowed_duplicates = max(10, len(entries) // 5)  # Allow up to 20% duplicates
        excessive_duplicates = [h for h, count in hash_counts.items() if count > max_allowed_duplicates]
        
        if excessive_duplicates and len(entries) > 5:
            # Only flag if we have a significant number of entries and excessive duplication
            duplicate_ratio = sum(hash_counts[h] for h in excessive_duplicates) / len(entries)
            if duplicate_ratio > 0.8:  # More than 80% are duplicates
                raise ValueError(
                    f"ANTI-FABRICATION VIOLATION: Excessive duplicate stack hashes detected "
                    f"for metric '{metric_name}' - potential fabrication (ratio: {duplicate_ratio:.2f})"
                )
        
        # Check temporal consistency (metrics should have realistic timestamps)
        timestamps = [entry["timestamp"] for entry in entries]
        if len(timestamps) > 1:
            time_diffs = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            # Realistic minimum time between measurements (allow very fast measurements)
            min_realistic_diff = 0.0001  # 0.1ms
            suspicious_diffs = [diff for diff in time_diffs if diff < min_realistic_diff and diff > 0]
            
            # Only flag if most measurements are suspiciously fast
            if len(suspicious_diffs) > len(time_diffs) * 0.8 and len(time_diffs) > 10:
                raise ValueError(
                    f"ANTI-FABRICATION VIOLATION: Unrealistic time intervals "
                    f"detected for metric '{metric_name}' - potential fabrication"
                )
        
        return True
    
    def validate_all_metrics(self) -> Dict[str, Any]:
        """Validate all registered metrics. Returns detailed validation results."""
        if not self.metric_sources:
            return {"valid": True, "total_metrics": 0, "failed_metrics": []}
        
        failed_metrics = []
        total_metrics = len(self.metric_sources)
        
        for metric_name in self.metric_sources:
            try:
                self.validate_metric_integrity(metric_name)
            except Exception as e:
                failed_metrics.append({"metric": metric_name, "error": str(e)})
        
        valid = len(failed_metrics) == 0
        
        return {
            "valid": valid,
            "total_metrics": total_metrics,
            "failed_metrics": failed_metrics,
            "validation_timestamp": time.time()
        }
    
    def export_verification_report(self, output_path: str):
        """Export verification report for audit."""
        report = {
            "verification_timestamp": time.time(),
            "total_metrics": len(self.metric_sources),
            "validation_results": self.validate_all_metrics(),
            "metric_sources": self.metric_sources,
            "anti_fabrication_config": {
                "forbidden_patterns": self.forbidden_patterns,
                "required_sources": self.telemetry_required_sources
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    def assert_no_fabrication(self, metric_name: str):
        """Assert that a metric is not fabricated - raises exception if violations found."""
        try:
            self.validate_metric_integrity(metric_name)
        except ValueError as e:
            raise AssertionError(f"FABRICATION DETECTED: {e}")
    
    def get_metric_provenance(self, metric_name: str) -> List[Dict[str, Any]]:
        """Get complete provenance information for a metric."""
        if metric_name not in self.metric_sources:
            raise ValueError(f"Metric '{metric_name}' not found")
        
        return self.metric_sources[metric_name]
    
    def detect_fabrication_patterns(self) -> bool:
        """Detect if any fabrication patterns exist in registered metrics."""
        if not self.metric_sources:
            return False
            
        for metric_name in self.metric_sources:
            try:
                self.validate_metric_integrity(metric_name)
            except Exception:
                return True  # Fabrication detected
        
        return False
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive anti-fabrication report."""
        return {
            "verification_timestamp": time.time(),
            "total_metrics": len(self.metric_sources),
            "metric_names": list(self.metric_sources.keys()),
            "fabrication_detected": self.detect_fabrication_patterns(),
            "anti_fabrication_config": {
                "forbidden_patterns": self.forbidden_patterns,
                "required_sources": self.telemetry_required_sources
            }
        }
    
    def verify(self, results: Dict[str, Any]) -> bool:
        """Verify simulation results for anti-fabrication compliance."""
        try:
            # Check if results contain telemetry data
            if 'telemetry_history' not in results:
                return False
            
            # Validate telemetry sources
            telemetry_history = results['telemetry_history']
            if not telemetry_history:
                return False
            
            # Check for legitimate telemetry sources
            legitimate_sources_found = False
            for entry in telemetry_history:
                if 'gpu_utilization_percent' in entry or 'memory_utilization_percent' in entry:
                    legitimate_sources_found = True
                    break
            
            if not legitimate_sources_found:
                return False
            
            # Check for suspicious patterns
            if self.detect_fabrication_patterns():
                return False
            
            # All checks passed
            return True
            
        except Exception as e:
            print(f"Anti-fabrication verification error: {e}")
            return False

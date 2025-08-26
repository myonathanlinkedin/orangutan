"""Reproducibility validator for ORANGUTAN experiments."""

import json
import hashlib
import random
import numpy as np
import torch
import time
from typing import Dict, Any, List, Callable, Tuple
from pathlib import Path
import statistics


class ReproducibilityValidator:
    """Validates that experiments are reproducible with proper statistical analysis."""
    
    def __init__(self):
        """Initialize reproducibility validator."""
        self.experiment_results = {}
        self.seeds_used = []
        
    def set_reproducible_seed(self, seed: int):
        """Set seed for reproducible results."""
        self.seeds_used.append(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Ensure deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def run_experiment_with_seeds(
        self,
        experiment_func: Callable,
        experiment_name: str,
        seeds: List[int],
        **kwargs
    ) -> Dict[str, Any]:
        """Run experiment with multiple seeds for statistical validation."""
        results = []
        
        for seed in seeds:
            print(f"Running {experiment_name} with seed {seed}")
            self.set_reproducible_seed(seed)
            
            start_time = time.time()
            result = experiment_func(seed=seed, **kwargs)
            end_time = time.time()
            
            # Add timing and seed info
            result["seed"] = seed
            result["execution_time"] = end_time - start_time
            result["timestamp"] = end_time
            
            results.append(result)
        
        # Store results
        self.experiment_results[experiment_name] = results
        
        # Compute statistics
        stats = self._compute_experiment_statistics(results)
        
        return {
            "experiment_name": experiment_name,
            "seeds": seeds,
            "results": results,
            "statistics": stats
        }
    
    def _compute_experiment_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute statistical summary of experiment results."""
        if not results:
            return {}
        
        # Extract numeric metrics
        numeric_metrics = {}
        for result in results:
            for key, value in result.items():
                if isinstance(value, (int, float)) and key not in ["seed", "timestamp"]:
                    if key not in numeric_metrics:
                        numeric_metrics[key] = []
                    numeric_metrics[key].append(value)
        
        stats = {}
        for metric, values in numeric_metrics.items():
            if len(values) >= 2:
                stats[metric] = {
                    "mean": statistics.mean(values),
                    "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                    "median": statistics.median(values),
                    "count": len(values),
                    "values": values
                }
                
                # Compute 95% confidence interval
                if len(values) >= 3:
                    alpha = 0.05
                    # Simple approximation for small samples
                    sem = stats[metric]["stdev"] / (len(values) ** 0.5)
                    margin = 1.96 * sem  # Approximate for normal distribution
                    stats[metric]["ci_95_lower"] = stats[metric]["mean"] - margin
                    stats[metric]["ci_95_upper"] = stats[metric]["mean"] + margin
        
        return stats
    
    def validate_reproducibility(
        self,
        experiment_name: str,
        tolerance: float = 0.01
    ) -> bool:
        """Validate that experiment results are reproducible within tolerance."""
        if experiment_name not in self.experiment_results:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        
        results = self.experiment_results[experiment_name]
        if len(results) < 2:
            raise ValueError("Need at least 2 runs to validate reproducibility")
        
        # Check if same seed produces same results
        seed_results = {}
        for result in results:
            seed = result["seed"]
            if seed not in seed_results:
                seed_results[seed] = []
            seed_results[seed].append(result)
        
        # Validate deterministic behavior for same seeds
        for seed, seed_runs in seed_results.items():
            if len(seed_runs) > 1:
                # Compare results for same seed
                for key in seed_runs[0].keys():
                    if isinstance(seed_runs[0][key], (int, float)):
                        values = [run[key] for run in seed_runs]
                        if max(values) - min(values) > tolerance:
                            raise ValueError(
                                f"Non-reproducible results for seed {seed}, "
                                f"metric {key}: {values}"
                            )
        
        return True
    
    def validate_consistency(self, results: List[Dict[str, Any]]) -> bool:
        """Validate consistency across multiple experiment runs."""
        if not results or len(results) < 2:
            return True  # Single result or no results is consistent
        
        # Check for basic consistency in key metrics
        key_metrics = ["completed_workloads", "total_workloads"]
        
        for metric in key_metrics:
            values = []
            for result in results:
                if metric in result:
                    values.append(result[metric])
            
            if len(values) >= 2:
                # Check if values are reasonably consistent
                mean_val = statistics.mean(values)
                if mean_val > 0:
                    cv = statistics.stdev(values) / abs(mean_val)  # Coefficient of variation
                    if cv > 0.5:  # More than 50% variation might indicate inconsistency
                        print(f"Warning: High variation in {metric}: CV = {cv:.3f}")
                        return False
        
        return True
    
    def detect_fabricated_patterns(self, experiment_name: str) -> List[str]:
        """Detect patterns that suggest fabricated data."""
        if experiment_name not in self.experiment_results:
            return ["Experiment not found"]
        
        results = self.experiment_results[experiment_name]
        warnings = []
        
        # Check for suspicious patterns
        for metric_name in ["throughput", "latency", "memory_usage"]:
            values = []
            for result in results:
                if metric_name in result and isinstance(result[metric_name], (int, float)):
                    values.append(result[metric_name])
            
            if len(values) >= 3:
                # Check for unrealistic precision (too many decimal places)
                decimal_places = []
                for value in values:
                    if isinstance(value, float):
                        str_val = str(value)
                        if '.' in str_val:
                            decimal_places.append(len(str_val.split('.')[1]))
                
                if decimal_places and max(decimal_places) > 6:
                    warnings.append(
                        f"Suspicious precision in {metric_name}: "
                        f"max {max(decimal_places)} decimal places"
                    )
                
                # Check for identical values (potential hardcoding)
                unique_values = set(values)
                if len(unique_values) == 1 and len(values) > 1:
                    warnings.append(
                        f"All values identical for {metric_name}: {values[0]} "
                        f"- potential hardcoding"
                    )
                
                # Check for unrealistic low variance
                if len(values) > 2:
                    mean_val = statistics.mean(values)
                    if mean_val != 0:
                        cv = statistics.stdev(values) / abs(mean_val)  # Coefficient of variation
                        if cv < 0.001:  # Less than 0.1% variation
                            warnings.append(
                                f"Unrealistically low variance in {metric_name}: "
                                f"CV = {cv:.6f}"
                            )
        
        return warnings
    
    def generate_reproducibility_report(self, output_path: str):
        """Generate comprehensive reproducibility report."""
        report = {
            "generation_timestamp": time.time(),
            "seeds_used": self.seeds_used,
            "experiments": {}
        }
        
        for exp_name, results in self.experiment_results.items():
            exp_report = {
                "results": results,
                "statistics": self._compute_experiment_statistics(results),
                "fabrication_warnings": self.detect_fabricated_patterns(exp_name)
            }
            
            # Add reproducibility validation
            try:
                exp_report["reproducible"] = self.validate_reproducibility(exp_name)
            except Exception as e:
                exp_report["reproducible"] = False
                exp_report["reproducibility_error"] = str(e)
            
            report["experiments"][exp_name] = exp_report
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def generate_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate reproducibility report as dictionary."""
        report = {
            "generation_timestamp": time.time(),
            "seeds_used": self.seeds_used,
            "total_runs": len(results),
            "consistency_check": self.validate_consistency(results)
        }
        
        # Add basic statistics if we have results
        if results:
            # Extract key metrics
            completed_counts = []
            for result in results:
                if "completed_workloads" in result:
                    completed_counts.append(result["completed_workloads"])
            
            if completed_counts:
                report["completed_workloads_stats"] = {
                    "mean": statistics.mean(completed_counts),
                    "stdev": statistics.stdev(completed_counts) if len(completed_counts) > 1 else 0.0,
                    "min": min(completed_counts),
                    "max": max(completed_counts),
                    "count": len(completed_counts)
                }
        
        return report
    
    def assert_statistical_significance(
        self,
        experiment_name: str,
        baseline_experiment: str,
        metric: str,
        improvement_threshold: float = 0.05
    ):
        """Assert that improvement is statistically significant."""
        if experiment_name not in self.experiment_results:
            raise AssertionError(f"Experiment {experiment_name} not found")
        
        if baseline_experiment not in self.experiment_results:
            raise AssertionError(f"Baseline experiment {baseline_experiment} not found")
        
        exp_stats = self._compute_experiment_statistics(self.experiment_results[experiment_name])
        baseline_stats = self._compute_experiment_statistics(self.experiment_results[baseline_experiment])
        
        if metric not in exp_stats or metric not in baseline_stats:
            raise AssertionError(f"Metric {metric} not found in both experiments")
        
        exp_mean = exp_stats[metric]["mean"]
        baseline_mean = baseline_stats[metric]["mean"]
        
        # Check for improvement
        improvement = (exp_mean - baseline_mean) / baseline_mean
        
        if improvement < improvement_threshold:
            raise AssertionError(
                f"Improvement {improvement:.4f} is below threshold {improvement_threshold} "
                f"for metric {metric}"
            )
        
        # Check confidence intervals don't overlap significantly
        if "ci_95_lower" in exp_stats[metric] and "ci_95_upper" in baseline_stats[metric]:
            if exp_stats[metric]["ci_95_lower"] <= baseline_stats[metric]["ci_95_upper"]:
                raise AssertionError(
                    f"Confidence intervals overlap significantly for metric {metric} "
                    f"- improvement may not be statistically significant"
                )
        
        return True
    
    def verify(self, results: Dict[str, Any]) -> bool:
        """Verify simulation results for reproducibility compliance."""
        try:
            # Check if results contain required fields
            if 'completed_workloads' not in results:
                return False
            
            if 'simulation_duration' not in results:
                return False
            
            # Check for realistic completion rates
            total_workloads = results.get('total_workloads', 0)
            completed_workloads = results.get('completed_workloads', 0)
            
            if total_workloads > 0:
                completion_rate = completed_workloads / total_workloads
                # Allow some failures but not complete failure
                if completion_rate < 0.1:  # At least 10% should complete
                    return False
            
            # Check for realistic simulation duration
            simulation_duration = results.get('simulation_duration', 0)
            if simulation_duration <= 0:
                return False
            
            # Check for telemetry data
            if 'telemetry_history' not in results:
                return False
            
            telemetry_history = results['telemetry_history']
            if not telemetry_history:
                return False
            
            # All checks passed
            return True
            
        except Exception as e:
            print(f"Reproducibility verification error: {e}")
            return False

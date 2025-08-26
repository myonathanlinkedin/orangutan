#!/usr/bin/env python3
"""Comprehensive test suite for ORANGUTAN verification system."""

import sys
import pytest
import tempfile
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orangutan.verification import AntiFabricationChecker, TelemetryValidator, ReproducibilityValidator
from orangutan.simulator.verified_sim import VerifiedSimulator
from orangutan.env.workload import Workload, WorkloadType, QuantizationType


class TestAntiFabricationChecker:
    """Test anti-fabrication checking functionality."""
    
    def test_legitimate_metric_registration(self):
        """Test that legitimate metrics are accepted."""
        checker = AntiFabricationChecker()
        
        # This should work
        checker.register_metric("test_metric", 42.0, "nvidia-smi")
        
        # Validate metric was registered
        assert "test_metric" in checker.metric_sources
        assert len(checker.metric_sources["test_metric"]) == 1
        
        # Validate metric integrity
        assert checker.validate_metric_integrity("test_metric")
    
    def test_fabricated_source_rejection(self):
        """Test that fabricated sources are rejected."""
        checker = AntiFabricationChecker()
        
        # These should fail
        forbidden_sources = ["hardcoded", "manual", "fake", "dummy", "placeholder"]
        
        for source in forbidden_sources:
            with pytest.raises(ValueError, match="ANTI-FABRICATION VIOLATION"):
                checker.register_metric("bad_metric", 42.0, source)
    
    def test_unauthorized_source_rejection(self):
        """Test that unauthorized sources are rejected."""
        checker = AntiFabricationChecker()
        
        with pytest.raises(ValueError, match="ANTI-FABRICATION VIOLATION"):
            checker.register_metric("bad_metric", 42.0, "random_source")
    
    def test_duplicate_detection(self):
        """Test detection of duplicate/fabricated calls."""
        checker = AntiFabricationChecker()
        
        # Register same metric multiple times with small interval
        import time
        
        checker.register_metric("test_metric", 1.0, "nvidia-smi")
        time.sleep(0.002)  # Small delay
        checker.register_metric("test_metric", 2.0, "nvidia-smi")
        
        # This should pass (different values, sufficient time gap)
        assert checker.validate_metric_integrity("test_metric")
    
    def test_fabrication_assertion(self):
        """Test fabrication assertion functionality."""
        checker = AntiFabricationChecker()
        
        checker.register_metric("good_metric", 42.0, "nvidia-smi")
        
        # This should not raise
        checker.assert_no_fabrication("good_metric")
        
        # Test with non-existent metric
        with pytest.raises(AssertionError):
            checker.assert_no_fabrication("nonexistent_metric")


class TestTelemetryValidator:
    """Test telemetry validation functionality."""
    
    def test_telemetry_source_validation(self):
        """Test validation of telemetry sources."""
        validator = TelemetryValidator()
        
        # Test telemetry source availability
        results = validator.validate_telemetry_sources()
        
        # At least one source should be testable
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Check expected sources
        expected_sources = ["nvidia_smi", "pytorch_gpu", "system_monitor"]
        for source in expected_sources:
            assert source in results
    
    def test_system_metrics_collection(self):
        """Test system metrics collection."""
        validator = TelemetryValidator()
        
        metrics = validator.get_system_metrics()
        
        assert isinstance(metrics, dict)
        assert "cpu_percent" in metrics
        assert "memory_percent" in metrics
        assert "timestamp" in metrics
        assert "source" in metrics
        assert metrics["source"] == "system_monitor"
    
    def test_dynamic_metric_validation(self):
        """Test validation that metrics are dynamic (not hardcoded)."""
        validator = TelemetryValidator()
        
        def varying_metric():
            import random
            return random.random()
        
        # This should pass (varying values)
        assert validator.validate_metric_is_dynamic(
            varying_metric, "random_metric", samples=5, interval=0.1
        )
        
        def constant_metric():
            return 42.0
        
        # This should fail (constant values)
        with pytest.raises(ValueError, match="ANTI-FABRICATION VIOLATION"):
            validator.validate_metric_is_dynamic(
                constant_metric, "constant_metric", samples=5, interval=0.1
            )


class TestReproducibilityValidator:
    """Test reproducibility validation functionality."""
    
    def test_seed_setting(self):
        """Test that seeds are set correctly."""
        validator = ReproducibilityValidator()
        
        validator.set_reproducible_seed(42)
        assert 42 in validator.seeds_used
    
    def test_experiment_statistics(self):
        """Test experiment statistics computation."""
        validator = ReproducibilityValidator()
        
        # Mock experiment results
        results = [
            {"metric1": 10.0, "metric2": 20.0, "seed": 42},
            {"metric1": 12.0, "metric2": 18.0, "seed": 123},
            {"metric1": 11.0, "metric2": 19.0, "seed": 456}
        ]
        
        stats = validator._compute_experiment_statistics(results)
        
        assert "metric1" in stats
        assert "metric2" in stats
        
        assert abs(stats["metric1"]["mean"] - 11.0) < 0.1
        assert abs(stats["metric2"]["mean"] - 19.0) < 0.1
        
        assert "stdev" in stats["metric1"]
        assert "ci_95_lower" in stats["metric1"]
        assert "ci_95_upper" in stats["metric1"]
    
    def test_fabrication_pattern_detection(self):
        """Test detection of fabricated data patterns."""
        validator = ReproducibilityValidator()
        
        # Store mock experiment with suspicious patterns
        validator.experiment_results["test_exp"] = [
            {"throughput": 42.123456789, "seed": 1},  # Too precise
            {"throughput": 42.123456789, "seed": 2},  # Identical
            {"throughput": 42.123456789, "seed": 3}   # Identical
        ]
        
        warnings = validator.detect_fabricated_patterns("test_exp")
        
        # Should detect both precision and identical value issues
        assert len(warnings) >= 1
        assert any("precision" in warning.lower() for warning in warnings)


class TestVerifiedSimulator:
    """Test verified simulator functionality."""
    
    def test_simulator_initialization(self):
        """Test that verified simulator initializes correctly."""
        config_path = "orangutan/config/legion7_defaults.json"
        
        # Should work with verification enabled
        simulator = VerifiedSimulator(config_path, enable_verification=True)
        
        assert simulator.enable_verification
        assert simulator.anti_fabrication_checker is not None
        assert simulator.telemetry_validator is not None
        assert simulator.reproducibility_validator is not None
    
    def test_simulator_without_verification(self):
        """Test simulator works without verification for testing."""
        config_path = "orangutan/config/legion7_defaults.json"
        
        simulator = VerifiedSimulator(config_path, enable_verification=False)
        
        assert not simulator.enable_verification
        assert simulator.anti_fabrication_checker is None
    
    def test_workload_validation(self):
        """Test workload validation in verified simulator."""
        config_path = "orangutan/config/legion7_defaults.json"
        simulator = VerifiedSimulator(config_path, enable_verification=True)
        
        # Valid workload should be accepted
        valid_workload = Workload(
            "test_workload",
            WorkloadType.INFERENCE,
            7,
            QuantizationType.INT8,
            batch_size=1,
            sequence_length=2048
        )
        
        simulator.add_verified_workload(valid_workload)
        assert len(simulator.workloads) == 1
        
        # Invalid workload should be rejected
        invalid_workload = Workload(
            "huge_workload",
            WorkloadType.INFERENCE,
            1000,  # Unrealistic size
            QuantizationType.FP32,
            batch_size=1000,  # Huge batch
            sequence_length=100000  # Huge sequence
        )
        
        with pytest.raises(ValueError):
            simulator.add_verified_workload(invalid_workload)
    
    def test_short_simulation_run(self):
        """Test a short simulation run."""
        config_path = "orangutan/config/legion7_defaults.json"
        simulator = VerifiedSimulator(config_path, enable_verification=True)
        
        # Add a simple workload
        workload = Workload(
            "test_sim",
            WorkloadType.INFERENCE,
            1,  # Small model
            QuantizationType.INT8
        )
        
        simulator.add_verified_workload(workload)
        
        # Run short simulation
        result = simulator.run_verified_simulation(duration_seconds=5, seed=42)
        
        assert isinstance(result, dict)
        assert "duration_seconds" in result
        assert "seed" in result
        assert "metrics" in result
        assert "verification_results" in result
        
        # Should have some metrics collected
        assert len(result["metrics"]) > 0


class TestIntegrationScenarios:
    """Test end-to-end integration scenarios."""
    
    def test_full_verification_pipeline(self):
        """Test the complete verification pipeline."""
        # Create temporary config
        config = {
            "device": {"name": "Test GPU", "vram_gb": 16},
            "scheduler": {"max_agents": 10},
            "workload": {"default_batch_size": 1},
            "telemetry": {"sampling_interval_ms": 100}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            config_path = f.name
        
        try:
            # Run complete pipeline
            simulator = VerifiedSimulator(config_path, enable_verification=True)
            
            # Add workloads
            workloads = [
                Workload("test1", WorkloadType.INFERENCE, 1, QuantizationType.INT8),
                Workload("test2", WorkloadType.INFERENCE, 2, QuantizationType.INT8)
            ]
            
            for workload in workloads:
                simulator.add_verified_workload(workload)
            
            # Run reproducibility test
            seeds = [42, 123]
            results = simulator.run_reproducibility_test(seeds, duration_seconds=3)
            
            assert isinstance(results, dict)
            assert "statistics" in results
            assert len(results["seeds"]) == 2
            
            # Export reports to temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                simulator.export_verification_reports(temp_dir)
                
                # Check that reports were created
                report_files = list(Path(temp_dir).glob("*.json"))
                assert len(report_files) >= 1  # At least one report should be created
                
        finally:
            # Clean up
            Path(config_path).unlink()


def test_fabrication_detection_scenario():
    """Test specific fabrication detection scenario."""
    checker = AntiFabricationChecker()
    
    # Simulate a scenario where someone tries to fabricate metrics
    import time
    
    # Legitimate metrics
    checker.register_metric("real_gpu_util", 0.75, "nvidia-smi")
    time.sleep(0.1)
    checker.register_metric("real_gpu_util", 0.78, "nvidia-smi")
    time.sleep(0.1)
    checker.register_metric("real_gpu_util", 0.73, "nvidia-smi")
    
    # This should pass
    assert checker.validate_metric_integrity("real_gpu_util")
    
    # Now try to add fabricated metrics
    with pytest.raises(ValueError, match="ANTI-FABRICATION VIOLATION"):
        checker.register_metric("fake_metric", 0.85, "hardcoded_value")


if __name__ == "__main__":
    # Run with verbose output
    pytest.main([__file__, "-v", "-s"])

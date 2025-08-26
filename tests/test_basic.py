"""Basic tests for ORANGUTAN project structure."""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that basic imports work."""
    try:
        from orangutan import __version__
        assert __version__ == "0.1.0"
    except ImportError as e:
        pytest.fail(f"Failed to import orangutan: {e}")


def test_device_creation():
    """Test device creation."""
    try:
        from orangutan.env import Device
        # This might fail if CUDA is not available, which is expected
        device = Device()
        assert device is not None
    except RuntimeError:
        # Expected if CUDA is not available
        pass


def test_workload_creation():
    """Test workload creation."""
    from orangutan.env.workload import Workload, WorkloadType, QuantizationType
    
    workload = Workload(
        "test_workload",
        WorkloadType.INFERENCE,
        7,
        QuantizationType.INT8
    )
    
    assert workload.workload_id == "test_workload"
    assert workload.model_size_billions == 7
    assert workload.quantization == QuantizationType.INT8


def test_config_loading():
    """Test configuration loading."""
    from orangutan.config import get_legion7_config
    
    config = get_legion7_config()
    assert "device" in config
    assert "scheduler" in config
    assert "workload" in config


if __name__ == "__main__":
    pytest.main([__file__])

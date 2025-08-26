"""Workload generator for ORANGUTAN simulator."""

from ..env.workload import WorkloadGenerator as BaseWorkloadGenerator


class WorkloadGenerator(BaseWorkloadGenerator):
    """Extended workload generator for simulator."""
    
    def __init__(self):
        """Initialize simulator workload generator."""
        super().__init__()
    
    def generate_scenario_workloads(self, scenario_name: str):
        """Generate workloads for specific scenarios."""
        if scenario_name == "single_gpu_inference":
            return self._generate_inference_workloads()
        elif scenario_name == "multi_tenant":
            return self._generate_multi_tenant_workloads()
        else:
            raise ValueError(f"Unknown scenario: {scenario_name}")
    
    def _generate_inference_workloads(self):
        """Generate inference workload set."""
        from ..env.workload import Workload, WorkloadType, QuantizationType
        
        return [
            Workload(
                "inf_1", WorkloadType.INFERENCE, 7, QuantizationType.INT8
            ),
            Workload(
                "inf_2", WorkloadType.INFERENCE, 13, QuantizationType.INT8
            ),
        ]
    
    def _generate_multi_tenant_workloads(self):
        """Generate multi-tenant workload set."""
        from ..env.workload import Workload, WorkloadType, QuantizationType
        
        return [
            Workload(
                "tenant_1", WorkloadType.INFERENCE, 7, QuantizationType.INT8
            ),
            Workload(
                "tenant_2", WorkloadType.TRAINING, 7, QuantizationType.BF16
            ),
        ]

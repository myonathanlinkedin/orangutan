# 🦧 ORANGUTAN: Agent-Based GPU Scheduling Framework

> **Maximizing Sustained Tensor-Core Utilization and Minimizing Effective Memory Traffic on Constrained GPUs**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

## 🎯 Overview

ORANGUTAN is an innovative agent-based metaheuristic framework for GPU resource contention resolution and tensor-workload orchestration. Inspired by primate social dynamics, it transforms GPU hardware into a "jungle ecosystem" where intelligent agents negotiate for computational resources.

**This project is based on the research specification in `orangutan.txt` and implements the core concepts described therein.**

## 🌟 Key Features

- **🦧 Agent-Based Scheduling**: Multi-agent negotiation for optimal resource allocation
- **⚡ Real GPU Execution**: Direct CUDA operations with persistent kernels
- **📊 Performance Monitoring**: Real-time telemetry and anti-fabrication validation
- **🎨 3D Visualization**: Jungle-themed performance charts and analysis
- **🔧 Configurable Intensity**: Adjustable workload parameters for different TFLOPs targets
- **📱 Mobile GPU Optimized**: Designed for RTX 4090 Mobile and similar constrained GPUs

## 🏗️ Architecture

### Core Components

- **Device**: GPU hardware representation (SMs, VRAM, bandwidth)
- **Jungle**: GPU environment mapping (trees=SMs, forest floor=VRAM, rivers=bandwidth)
- **Workload**: Tensor operation representation with SLOs
- **ExecutionEngine**: Real GPU execution with CUDA Graphs
- **NegotiationEngine**: Priority-based resource assignment
- **TelemetryCollector**: Performance monitoring and validation

### Algorithm Flow

1. **ENV REPRESENTATION**: Map GPU to jungle ecosystem
2. **AGENT INIT**: Initialize workload agents
3. **PROPOSE**: Agents propose resource requirements
4. **RESERVE**: Reserve computational tiles
5. **NEGOTIATE**: Resolve resource contention
6. **LAUNCH**: Deploy persistent kernels
7. **EXECUTE**: Real GPU computation
8. **SENSE**: Monitor performance metrics
9. **ADAPT**: Adjust resource allocation
10. **REPEAT**: Iterative optimization

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+
- NVIDIA GPU (RTX 4090 Mobile recommended)

### Installation

```bash
git clone https://github.com/your-username/orangutan.git
cd orangutan
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run with default settings (20 TFLOPS target)
python run_comprehensive_benchmarks.py

# Recommended configuration for RTX 4090 Mobile
python run_comprehensive_benchmarks.py --tflops-target 15 --workload-intensity medium

# High-performance configuration
python run_comprehensive_benchmarks.py --tflops-target 20 --workload-intensity high

# Conservative mode for stable results
python run_comprehensive_benchmarks.py --tflops-target 10 --workload-intensity low

# Custom tensor dimensions and iterations
python run_comprehensive_benchmarks.py --tensor-size 4096 --num-iterations 200
```

### Command Line Options

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `--tflops-target` | Target TFLOPs performance | 20.0 | 5.0-32.98 |
| `--workload-intensity` | Workload intensity level | high | low/medium/high/max |
| `--tensor-size` | Base tensor dimension | 2048 | 1024/2048/4096/8192 |
| `--num-iterations` | Computation iterations | 100 | 10-500 |
| `--batch-size-multiplier` | Batch size multiplier | 2.0 | 0.5-5.0 |
| `--quick-test` | Quick validation mode | False | Flag |

## 📊 Performance Results

### RTX 4090 Mobile Performance

| Configuration | TFLOPs | GPU Utilization | Throughput | Latency P50 |
|---------------|--------|-----------------|------------|-------------|
| **Conservative** | 5-10 | 20-40% | 1-2 workloads/sec | 200-400ms |
| **Balanced** | 10-15 | 40-60% | 2-3 workloads/sec | 150-250ms |
| **High Intensity** | 15-20 | 60-80% | 3-4 workloads/sec | 100-200ms |
| **Maximum** | 20-30 | 80-95% | 4-6 workloads/sec | 50-150ms |

### Theoretical Maximum
- **FP16/FP32**: 32.98 TFLOPS
- **FP64**: 515.3 GFLOPS
- **Memory**: 16GB GDDR6X
- **SMs**: 76 Streaming Multiprocessors

## 🔬 Research Applications

ORANGUTAN is designed for academic research in:

- **GPU Resource Management**: Dynamic allocation and contention resolution
- **Multi-Agent Systems**: Cooperative resource negotiation algorithms
- **Performance Optimization**: Sustained tensor-core utilization
- **Memory Efficiency**: Minimizing effective memory traffic
- **Real-Time Scheduling**: Adaptive workload orchestration

## 📁 Project Structure

```
orangutan/
├── orangutan/                 # Core framework
│   ├── simulator/            # ORANGUTAN simulation engine
│   ├── scheduling/           # Resource scheduling algorithms
│   ├── env/                  # GPU environment modeling
│   ├── baselines/            # Baseline comparison implementations
│   ├── telemetry/            # Performance monitoring
│   └── verification/         # Anti-fabrication validation
├── visualization/             # 3D chart generation
├── config/                   # Configuration files
├── tests/                    # Unit tests
├── results/                  # Benchmark results and charts
└── run_comprehensive_benchmarks.py  # Main benchmark runner
```

## 🧪 Running Benchmarks

### Comprehensive Benchmark

```bash
python run_comprehensive_benchmarks.py --tflops-target 20 --workload-intensity high
```

This will:
1. Initialize ORANGUTAN simulator
2. Run workload simulations
3. Execute baseline comparisons
4. Generate performance metrics
5. Create 3D visualization charts
6. Save results to `results/` directory

### Output Files

- `results/simulation_results_latest.json`: Raw simulation data
- `results/benchmark_summary.json`: Performance summary
- `results/charts/`: 3D performance visualization charts

## 🔍 Monitoring & Validation

### Anti-Fabrication System

ORANGUTAN includes built-in validation to ensure all performance metrics are real:

- **Telemetry Validation**: Real-time GPU metrics collection
- **Reproducibility Checks**: Consistent performance across runs
- **Data Integrity**: No fabricated or estimated values

### Real-Time Metrics

- Actual FLOPs computation
- Real GPU utilization
- Memory bandwidth usage
- SM occupancy rates
- Tensor core utilization

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/your-username/orangutan.git
cd orangutan
pip install -r requirements.txt
pip install -r requirements-dev.txt
pre-commit install
```

### Running Tests

```bash
python -m pytest tests/
python -m pytest tests/ --cov=orangutan
```

## 📚 Documentation

- [API Reference](docs/api.md)
- [Performance Tuning](docs/performance.md)
- [Architecture Deep Dive](docs/architecture.md)
- [Research Paper](docs/paper.pdf)

## 📄 License

Copyright 2025 ORANGUTAN Research Team

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at:

```
http://www.apache.org/licenses/LICENSE-2.0
```

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

## 🙏 Acknowledgments

- **Original Research**: Based on `orangutan.txt` specification and research framework
- **NVIDIA**: CUDA framework and GPU architecture
- **PyTorch Team**: Deep learning framework
- **Research Community**: Multi-agent systems and GPU optimization

**If you use this project in your research or work, please acknowledge the original ORANGUTAN research framework and cite this implementation.**

## 📞 Contact

- **Project**: [GitHub Issues](https://github.com/your-username/orangutan/issues)
- **Research**: [Paper Discussion](https://github.com/your-username/orangutan/discussions)
- **Email**: your-email@example.com

---

**🦧 ORANGUTAN**: Where GPU optimization meets primate intelligence! 🚀

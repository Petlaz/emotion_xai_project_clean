<<<<<<< HEAD
# Mac Optimization Guide for Explainable AI for Emotion Detection in Social Media Text

This guide provides comprehensive instructions for optimizing the project "Explainable AI for Emotion Detection in Social Media Text" on macOS, particularly for Apple Silicon (M1/M2/M3) chips.
=======
# Mac Optimization Guide for Emotion XAI Project

This guide provides comprehensive instructions for optimizing the Emotion XAI project on macOS, particularly for Apple Silicon (M1/M2/M3) chips.
>>>>>>> be3b044594b375f6fcd55554c1c72425f0629c88

## 🍎 Apple Silicon Overview

Your Mac M1 chip provides several advantages for ML development:
- **Unified Memory Architecture**: Shared memory between CPU and GPU
- **Metal Performance Shaders (MPS)**: GPU acceleration for PyTorch
- **Energy Efficient**: Lower power consumption compared to dedicated GPUs
- **Fast Memory Bandwidth**: Direct memory access for large models

## 🚀 Quick Setup

### 1. Verify Your Setup
```bash
# Check your hardware
system_profiler SPHardwareDataType | grep "Chip\|Memory"

# Test MPS availability
python -c "import torch; print('MPS Available:', torch.backends.mps.is_available())"

# Check device optimization
python -c "from emotion_xai.utils.device import setup_mac_optimizations; setup_mac_optimizations()"
```

### 2. Environment Configuration
```bash
# Activate your virtual environment
source .venv/bin/activate

# Install/update dependencies optimized for Mac
pip install --upgrade torch torchvision torchaudio

# Verify PyTorch MPS support
python -c "import torch; print('PyTorch version:', torch.__version__); print('MPS built:', torch.backends.mps.is_built())"
```

## ⚙️ Configuration Files

### Mac Optimizations Config
The project includes `config/mac_optimizations.yaml` with settings for:
- **MPS Configuration**: Memory management, fallback settings
- **Batch Sizes**: Optimized for Apple Silicon unified memory
- **DataLoader Settings**: Worker processes, memory pinning
- **Performance Monitoring**: Memory usage, temperature tracking

### Key Settings for M1
```yaml
# Recommended batch sizes for your M1 Mac
training:
  batch_sizes:
    baseline_model: 512      # TF-IDF can handle large batches
    transformer_train: 16    # Conservative for fine-tuning
    transformer_inference: 32 # Larger for inference
    
# MPS-specific settings
pytorch:
  mps:
    enabled: true
    memory_fraction: 0.7     # Use 70% of GPU memory
    fallback_to_cpu: true    # Fallback for unsupported ops
```

## 🏋️ Model Training Optimization

### 1. Transformer Fine-tuning
```python
from emotion_xai.models.transformer import MacOptimizedTrainingConfig, train_model

# Mac-optimized training config
config = MacOptimizedTrainingConfig(
    model_name="distilroberta-base",
    batch_size_train=16,        # Optimal for M1
    batch_size_eval=32,         # Larger for inference
    gradient_accumulation_steps=4,  # Effective batch size: 64
    use_mac_optimizations=True,
    mixed_precision=False,      # MPS doesn't support FP16 yet
    max_seq_length=128,        # Memory efficient
)

# Train with automatic device detection
trainer = train_model(config, train_dataset, eval_dataset)
```

### 2. Memory Management
```python
from emotion_xai.utils.device import MacDeviceManager

# Setup device manager
device_manager = MacDeviceManager()
device = device_manager.detect_device()

# Clear cache periodically during training
device_manager.clear_cache(device)

# Get optimal DataLoader settings
dataloader_kwargs = device_manager.get_dataloader_kwargs(device)
```

## 📊 Performance Monitoring

### 1. System Monitoring
```bash
# Monitor CPU and memory usage
htop

# Check GPU usage (Activity Monitor -> GPU History)
# or use terminal
sudo powermetrics -n 1 --samplers gpu_power

# Monitor temperature
sudo powermetrics -n 1 --samplers smc -a --hide-cpu-duty-cycle
```

### 2. Python Monitoring
```python
# Memory usage tracking
import psutil
import torch

def monitor_memory():
    """Monitor system and GPU memory usage."""
    # System memory
    memory = psutil.virtual_memory()
    print(f"RAM Usage: {memory.percent}% ({memory.used/1e9:.1f}GB/{memory.total/1e9:.1f}GB)")
    
    # MPS memory (if available)
    if torch.backends.mps.is_available():
        print(f"MPS Memory Allocated: {torch.mps.current_allocated_memory()/1e9:.1f}GB")

# Use during training
monitor_memory()
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. MPS Out of Memory
```python
# Reduce batch size
config.batch_size_train = 8
config.batch_size_eval = 16

# Enable gradient accumulation
config.gradient_accumulation_steps = 8

# Clear cache more frequently
config.clear_cache_every_n_steps = 50
```

#### 2. Slow Training Performance
```bash
# Check for thermal throttling
sudo powermetrics -n 1 --samplers smc | grep -i temp

# Optimize DataLoader workers
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6
```

#### 3. MPS Fallback to CPU
```python
# Enable fallback logging
import os
os.environ['PYTORCH_MPS_FALLBACK_DEBUG'] = '1'

# Check which operations fall back
torch.backends.mps.enable_fallback(True)
```

#### 4. Memory Fragmentation
```python
# Clear cache between batches
if batch_idx % 10 == 0:
    torch.mps.empty_cache()

# Use smaller sequence lengths
config.max_seq_length = 64  # Instead of 128
```

## 🎯 Best Practices

### 1. Development Workflow
1. **Start Small**: Test with small datasets and batch sizes
2. **Monitor Resources**: Keep Activity Monitor open during development
3. **Profile Code**: Use `torch.profiler` to identify bottlenecks
4. **Incremental Training**: Save checkpoints frequently

### 2. Code Organization
```python
# Always use the device manager
from emotion_xai.utils.device import setup_mac_optimizations

# Setup at the beginning of your script
device, device_info = setup_mac_optimizations(verbose=True)

# Use in your training loop
model = model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}
```

### 3. Jupyter Notebook Optimization
```bash
# Increase Jupyter memory limit
jupyter notebook --NotebookApp.max_buffer_size=1073741824

# Use specific kernel with optimizations
python -m ipykernel install --user --name emotion_xai --display-name "Emotion XAI (Mac Optimized)"
```

## 📈 Performance Benchmarks

### Expected Performance on M1 Mac
- **Baseline TF-IDF Training**: ~30 seconds (10K samples)
- **DistilRoBERTa Fine-tuning**: ~45 minutes (3 epochs, 50K samples)
- **SHAP Explanations**: ~2 seconds per sample
- **Clustering (UMAP+HDBSCAN)**: ~5 minutes (10K samples)

### Optimization Impact
- **MPS vs CPU**: 3-5x speedup for transformer training
- **Optimized Batch Sizes**: 20-30% memory reduction
- **DataLoader Optimization**: 10-15% training speedup

## 🔍 Debugging Tools

### 1. MPS Debugging
```python
# Enable MPS debugging
import os
os.environ['PYTORCH_MPS_FALLBACK_DEBUG'] = '1'
os.environ['METAL_DEVICE_WRAPPER_TYPE'] = '1'

# Profile MPS operations
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, 
                torch.profiler.ProfilerActivity.CUDA],  # CUDA profiler works for MPS
    record_shapes=True
) as prof:
    # Your training code here
    pass

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### 2. Memory Profiling
```python
import tracemalloc

# Start memory tracing
tracemalloc.start()

# Your code here
# ...

# Get memory usage
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
```

## 🎨 VS Code Configuration

### Recommended Extensions
- **Python**: Enhanced IntelliSense and debugging
- **Jupyter**: Native notebook support
- **PyTorch Profiler**: Visual profiling interface
- **Resource Monitor**: Track system resources

### Settings for Mac Development
```json
{
    "python.defaultInterpreterPath": ".venv/bin/python",
    "jupyter.kernels.filter": [
        {
            "path": ".venv/bin/python",
            "type": "pythonEnvironment"
        }
    ],
    "python.terminal.activateEnvironment": true,
    "files.watcherExclude": {
        "**/.venv/**": true,
        "**/models/**": true
    }
}
```

## 🚨 Warning Signs

Watch out for these indicators of performance issues:
- **High CPU Temperature**: >85°C sustained
- **Memory Pressure**: Yellow/Red in Activity Monitor
- **Thermal Throttling**: Significant performance drops
- **MPS Errors**: Frequent fallback messages
- **Slow Model Loading**: >30 seconds for DistilRoBERTa

## 🎉 Success Indicators

You'll know your Mac optimization is working when:
- ✅ MPS device detected and used automatically
- ✅ Training completes without memory errors
- ✅ Batch sizes utilize available memory efficiently
- ✅ Temperature stays below 85°C during training
- ✅ No frequent MPS fallback warnings

## 📞 Getting Help

If you encounter issues:
1. Check the logs in `logs/mac_performance.log`
2. Run device diagnostics: `python -m emotion_xai.utils.device`
3. Monitor system resources during training
4. Reduce batch sizes and sequence lengths
5. Enable verbose logging for debugging

---

*This guide is specific to your M1 Mac setup. Performance may vary on different Apple Silicon variants.*
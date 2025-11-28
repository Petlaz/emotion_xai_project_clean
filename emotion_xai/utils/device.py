"""
Device detection and optimization utilities for Mac/Apple Silicon.

This module provides utilities for detecting and optimizing device usage
on macOS, with special support for Apple Silicon (M1/M2/M3) chips.
"""

import os
import platform
import subprocess
import warnings
from typing import Dict, Optional, Tuple

import torch
import yaml


class MacDeviceManager:
    """Manages device detection and optimization for Mac systems."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize device manager with optional config."""
        self.config_path = config_path or "config/mac_optimizations.yaml"
        self.config = self._load_config()
        self._device_info = None
        
    def _load_config(self) -> Dict:
        """Load Mac optimization configuration."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            warnings.warn(f"Config file {self.config_path} not found. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for Mac systems."""
        return {
            'pytorch': {
                'mps': {
                    'enabled': True,
                    'fallback_to_cpu': True,
                    'memory_fraction': 0.7
                }
            },
            'training': {
                'batch_sizes': {
                    'baseline_model': 512,
                    'transformer_train': 16,
                    'transformer_inference': 32
                }
            }
        }
    
    def detect_device(self, force_device: Optional[str] = None) -> torch.device:
        """
        Detect the best available device for PyTorch operations.
        
        Args:
            force_device: Force specific device ('mps', 'cuda', 'cpu')
            
        Returns:
            torch.device: The optimal device for this system
        """
        if force_device:
            if force_device == 'mps' and not torch.backends.mps.is_available():
                warnings.warn("MPS requested but not available. Falling back to CPU.")
                return torch.device('cpu')
            return torch.device(force_device)
        
        # Auto-detect based on availability and configuration
        if self._is_mps_enabled() and torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    
    def _is_mps_enabled(self) -> bool:
        """Check if MPS is enabled in configuration."""
        return self.config.get('pytorch', {}).get('mps', {}).get('enabled', True)
    
    def get_device_info(self) -> Dict:
        """Get comprehensive device information."""
        if self._device_info is None:
            self._device_info = self._collect_device_info()
        return self._device_info
    
    def _collect_device_info(self) -> Dict:
        """Collect detailed device information."""
        info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'torch_version': torch.__version__,
        }
        
        # Apple Silicon detection
        if platform.system() == 'Darwin':  # macOS
            try:
                result = subprocess.run(
                    ['system_profiler', 'SPHardwareDataType'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                output = result.stdout
                
                if 'Apple' in output and any(chip in output for chip in ['M1', 'M2', 'M3']):
                    info['apple_silicon'] = True
                    # Extract chip info
                    for line in output.split('\n'):
                        if 'Chip:' in line:
                            info['chip'] = line.split('Chip:')[1].strip()
                        elif 'Total Number of Cores' in line:
                            info['cpu_cores'] = line.split(':')[1].strip()
                        elif 'Memory:' in line:
                            info['total_memory'] = line.split(':')[1].strip()
                else:
                    info['apple_silicon'] = False
                    
            except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                info['apple_silicon'] = None  # Unknown
        
        # PyTorch device availability
        info['devices'] = {
            'mps_available': torch.backends.mps.is_available(),
            'cuda_available': torch.cuda.is_available(),
            'cpu_available': True
        }
        
        return info
    
    def optimize_torch_settings(self, device: torch.device) -> None:
        """Apply PyTorch optimizations based on device and config."""
        if device.type == 'mps':
            self._optimize_mps_settings()
        elif device.type == 'cuda':
            self._optimize_cuda_settings()
        else:
            self._optimize_cpu_settings()
    
    def _optimize_mps_settings(self) -> None:
        """Apply MPS-specific optimizations."""
        # Set memory fraction if configured
        memory_fraction = self.config.get('pytorch', {}).get('mps', {}).get('memory_fraction', 0.7)
        
        # Enable optimizations for Apple Silicon
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Set environment variables for optimal performance
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = str(memory_fraction)
            os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = str(memory_fraction - 0.1)
            
            # Enable MPS fallback if the function exists (newer PyTorch versions)
            if hasattr(torch.backends.mps, 'enable_fallback'):
                torch.backends.mps.enable_fallback(True)
            else:
                # For older PyTorch versions, set environment variable
                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    def _optimize_cuda_settings(self) -> None:
        """Apply CUDA-specific optimizations."""
        if torch.cuda.is_available():
            # Standard CUDA optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
    
    def _optimize_cpu_settings(self) -> None:
        """Apply CPU-specific optimizations."""
        # Use all available CPU cores
        num_threads = os.cpu_count()
        torch.set_num_threads(num_threads)
        
        # Optimize for macOS vecLib
        if platform.system() == 'Darwin':
            os.environ['VECLIB_MAXIMUM_THREADS'] = str(num_threads)
    
    def get_optimal_batch_size(self, model_type: str, mode: str = 'train') -> int:
        """Get optimal batch size for given model type and mode."""
        batch_key = f"{model_type}_{mode}" if mode == 'train' else f"{model_type}_inference"
        default_batch = 32
        
        return self.config.get('training', {}).get('batch_sizes', {}).get(batch_key, default_batch)
    
    def get_dataloader_kwargs(self, device: torch.device) -> Dict:
        """Get optimal DataLoader kwargs for this device."""
        kwargs = {}
        
        if device.type == 'mps':
            # MPS-specific optimizations
            kwargs['num_workers'] = self.config.get('pytorch', {}).get('memory', {}).get('num_workers', 4)
            kwargs['pin_memory'] = False  # MPS doesn't benefit from pinned memory
            kwargs['prefetch_factor'] = 2
        elif device.type == 'cuda':
            # CUDA optimizations
            kwargs['num_workers'] = 4
            kwargs['pin_memory'] = True
            kwargs['prefetch_factor'] = 2
        else:
            # CPU optimizations
            kwargs['num_workers'] = min(8, os.cpu_count())
            kwargs['pin_memory'] = False
        
        return kwargs
    
    def clear_cache(self, device: torch.device) -> None:
        """Clear device cache to free memory."""
        if device.type == 'mps' and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif device.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def print_device_info(self) -> None:
        """Print comprehensive device information."""
        info = self.get_device_info()
        device = self.detect_device()
        
        print("🖥️  Mac Device Information")
        print("=" * 50)
        print(f"Platform: {info['platform']}")
        print(f"Processor: {info['processor']}")
        
        if info.get('apple_silicon'):
            print(f"🍎 Apple Silicon: {info.get('chip', 'Unknown')}")
            print(f"CPU Cores: {info.get('cpu_cores', 'Unknown')}")
            print(f"Memory: {info.get('total_memory', 'Unknown')}")
        
        print(f"Python: {info['python_version']}")
        print(f"PyTorch: {info['torch_version']}")
        
        print("\n🚀 Device Availability")
        print("=" * 50)
        print(f"MPS Available: {'✅' if info['devices']['mps_available'] else '❌'}")
        print(f"CUDA Available: {'✅' if info['devices']['cuda_available'] else '❌'}")
        print(f"Selected Device: {device}")
        
        if device.type == 'mps':
            print("\n⚡ MPS Optimizations Active")
            print("- Metal Performance Shaders enabled")
            print("- Unified memory optimization")
            print("- Fallback to CPU for unsupported operations")


def get_device_manager(config_path: Optional[str] = None) -> MacDeviceManager:
    """Get a configured device manager instance."""
    return MacDeviceManager(config_path)


def resolve_device(force_device: Optional[str] = None) -> torch.device:
    """Quick function to resolve the best device."""
    manager = get_device_manager()
    device = manager.detect_device(force_device)
    manager.optimize_torch_settings(device)
    return device


def setup_mac_optimizations(config_path: Optional[str] = None, verbose: bool = True) -> Tuple[torch.device, Dict]:
    """
    Complete setup of Mac optimizations.
    
    Returns:
        Tuple of (device, device_info)
    """
    manager = get_device_manager(config_path)
    device = manager.detect_device()
    manager.optimize_torch_settings(device)
    
    if verbose:
        manager.print_device_info()
    
    return device, manager.get_device_info()


if __name__ == "__main__":
    print("🍎 Testing Mac Device Optimizations")
    print("=" * 50)
    
    try:
        device, info = setup_mac_optimizations(verbose=True)
        print(f"\n✅ Device setup successful: {device}")
        
        # Test device manager functions
        manager = get_device_manager()
        print(f"\n📊 Optimal batch sizes:")
        print(f"  Baseline: {manager.get_optimal_batch_size('baseline_model')}")
        print(f"  Transformer (train): {manager.get_optimal_batch_size('transformer', 'train')}")
        print(f"  Transformer (inference): {manager.get_optimal_batch_size('transformer', 'inference')}")
        
        # Test DataLoader kwargs
        kwargs = manager.get_dataloader_kwargs(device)
        print(f"\n⚙️ DataLoader settings:")
        for key, value in kwargs.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
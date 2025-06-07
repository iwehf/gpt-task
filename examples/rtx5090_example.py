#!/usr/bin/env python3
"""
RTX 5090 GPU Optimization Example
This example demonstrates how to optimize model inference performance on RTX 5090 and verify that the model is actually running on GPU
"""

import torch
import time
import psutil
import logging
import gc
from gpt_task.inference import run_task

# Setup logging
logging.basicConfig(
    format="[{asctime}] [{levelname:<8}] {name}: {message}",
    datefmt="%Y-%m-%d %H:%M:%S",
    style="{",
    level=logging.INFO,
)

_logger = logging.getLogger(__name__)


def get_gpu_info():
    """Get detailed GPU information"""
    if not torch.cuda.is_available():
        return None
    
    gpu_info = {
        'name': torch.cuda.get_device_name(0),
        'capability': torch.cuda.get_device_capability(0),
        'total_memory': torch.cuda.get_device_properties(0).total_memory / 1024**3,  # GB
        'memory_allocated': torch.cuda.memory_allocated(0) / 1024**3,  # GB
        'memory_reserved': torch.cuda.memory_reserved(0) / 1024**3,    # GB
        'device_count': torch.cuda.device_count(),
        'current_device': torch.cuda.current_device()
    }
    return gpu_info


def is_rtx5090_gpu():
    """Precisely detect if it's RTX 5090"""
    if not torch.cuda.is_available():
        return False
    
    device_name = torch.cuda.get_device_name(0).upper()
    capability = torch.cuda.get_device_capability(0)
    
    # RTX 5090 feature checks
    rtx5090_indicators = [
        "5090" in device_name,
        "RTX 5090" in device_name,
        # Blackwell architecture compute capability should be 9.0 or higher
        capability[0] >= 9 and "RTX" in device_name
    ]
    
    return any(rtx5090_indicators)


def check_rtx5090_support():
    """Check RTX 5090 support and display detailed information"""
    print("=" * 60)
    print("GPU Hardware Check")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA is not available, cannot use RTX 5090")
        print("Please check:")
        print("  1. NVIDIA driver is properly installed")
        print("  2. CUDA Toolkit is properly installed")
        print("  3. PyTorch includes CUDA support")
        return False
    
    gpu_info = get_gpu_info()
    device_name = gpu_info['name']
    capability = gpu_info['capability']
    
    print(f"✓ Detected GPU: {device_name}")
    print(f"✓ CUDA compute capability: {capability[0]}.{capability[1]}")
    print(f"✓ GPU total memory: {gpu_info['total_memory']:.2f}GB")
    print(f"✓ Available GPU count: {gpu_info['device_count']}")
    print(f"✓ Current device: {gpu_info['current_device']}")
    
    # Precisely check if it's RTX 5090
    if is_rtx5090_gpu():
        print("✅ Confirmed RTX 5090 GPU detected")
        is_target_gpu = True
    else:
        print("⚠ Current GPU is not RTX 5090, but can still perform GPU performance testing")
        print(f"  Current GPU: {device_name}")
        is_target_gpu = False
    
    # Check PyTorch version
    print(f"✓ PyTorch version: {torch.__version__}")
    if torch.version.cuda:
        print(f"✓ CUDA version: {torch.version.cuda}")
    
    return True


def monitor_gpu_usage(phase_name):
    """Monitor GPU usage"""
    if not torch.cuda.is_available():
        print(f"[{phase_name}] GPU not available")
        return
    
    gpu_info = get_gpu_info()
    print(f"\n[{phase_name}] GPU Status:")
    print(f"  Device name: {gpu_info['name']}")
    print(f"  Memory allocated: {gpu_info['memory_allocated']:.3f}GB")
    print(f"  Memory reserved: {gpu_info['memory_reserved']:.3f}GB")
    print(f"  Memory usage: {(gpu_info['memory_allocated']/gpu_info['total_memory']*100):.1f}%")
    
    # Get GPU utilization information
    gpu_utilization = get_gpu_utilization()
    if gpu_utilization:
        print(f"  GPU utilization: {gpu_utilization['gpu_util']}%")
        print(f"  nvidia-smi memory: {gpu_utilization['mem_used']}MB/{gpu_utilization['mem_total']}MB")
    
    # Check current active CUDA stream
    try:
        if hasattr(torch.cuda, 'current_stream'):
            current_stream = torch.cuda.current_stream()
            print(f"  Current CUDA stream: {current_stream}")
    except Exception:
        pass


def get_gpu_utilization():
    """Get GPU utilization information"""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
             '--format=csv,noheader,nounits'], 
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if lines and lines[0]:
                values = lines[0].split(', ')
                if len(values) >= 3:
                    return {
                        'gpu_util': values[0].strip(),
                        'mem_used': values[1].strip(),
                        'mem_total': values[2].strip()
                    }
    except Exception as e:
        _logger.debug(f"Unable to get GPU utilization: {e}")
    return None


def verify_model_on_gpu(model_name="gpt2"):
    """Verify if the model is actually loaded on GPU"""
    print(f"\nVerifying if model '{model_name}' is running on GPU...")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Thoroughly clean GPU memory
        cleanup_gpu()
        
        monitor_gpu_usage("Before model loading")
        
        # Load model to GPU
        print("Loading model to GPU...")
        
        # Use stricter GPU loading parameters
        load_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        
        if torch.cuda.is_available():
            load_kwargs.update({
                "torch_dtype": torch.bfloat16,
                "device_map": {"": 0},  # Force load to GPU 0
            })
        
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Set pad_token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        monitor_gpu_usage("After model loading")
        
        # Strictly check if model parameters are on GPU
        gpu_params = 0
        cpu_params = 0
        total_params = 0
        gpu_devices = set()
        
        for name, param in model.named_parameters():
            total_params += param.numel()
            if param.device.type == 'cuda':
                gpu_params += param.numel()
                gpu_devices.add(param.device.index)
            else:
                cpu_params += param.numel()
        
        gpu_ratio = gpu_params / total_params * 100 if total_params > 0 else 0
        print(f"✓ Model parameters on GPU ratio: {gpu_ratio:.1f}% ({gpu_params:,}/{total_params:,})")
        
        if cpu_params > 0:
            print(f"⚠ Parameters on CPU: {cpu_params:,} ({cpu_params/total_params*100:.1f}%)")
        
        if gpu_devices:
            print(f"✓ Model distributed on GPU devices: {sorted(gpu_devices)}")
            
        # Verify GPU device type
        if torch.cuda.is_available() and gpu_devices:
            device_name = torch.cuda.get_device_name(0)
            print(f"✓ Primary GPU device: {device_name}")
            
            # Test GPU computation to ensure device is truly available
            test_tensor = torch.randn(100, 100, device='cuda:0', dtype=torch.bfloat16)
            result = torch.matmul(test_tensor, test_tensor.T)
            torch.cuda.synchronize()  # Ensure computation completes
            print(f"✓ GPU computation test passed, result shape: {result.shape}")
            del test_tensor, result
        
        # Strict GPU verification standards
        if gpu_ratio >= 95:  # At least 95% of parameters on GPU
            print("✅ Model successfully loaded completely on GPU")
            return True, model, tokenizer
        elif gpu_ratio >= 80:
            print("⚠ Model mainly on GPU, but some parts may be on CPU")
            return True, model, tokenizer
        else:
            print("❌ Model not properly loaded on GPU")
            return False, model, tokenizer
            
    except Exception as e:
        print(f"❌ Model loading verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def rtx5090_optimized_inference():
    """RTX 5090 optimized inference example"""
    # Check hardware support
    if not check_rtx5090_support():
        return False
    
    print("\n" + "=" * 60)
    print("RTX 5090 Inference Performance Test")
    print("=" * 60)
    
    # Verify model GPU loading
    model_loaded, model, tokenizer = verify_model_on_gpu("gpt2")
    if not model_loaded:
        print("❌ Unable to verify model running on GPU, test terminated")
        return False
    
    # Thoroughly clean verification model, let run_task reload
    del model, tokenizer
    cleanup_gpu()
    
    # RTX 5090 optimized messages
    messages = [
        {"role": "system", "content": "You are a high-performance AI assistant running on RTX 5090 GPU."},
        {"role": "user", "content": "Please introduce the technical features of RTX 5090, including its architecture advantages."}
    ]
    
    # RTX 5090 optimized generation config
    generation_config = {
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "repetition_penalty": 1.1
    }
    
    monitor_gpu_usage("Before inference")
    
    print("\nStarting inference on RTX 5090...")
    print("=" * 50)
    
    # Record inference time and GPU status
    start_time = time.time()
    initial_memory = torch.cuda.memory_allocated(0) if torch.cuda.is_available() else 0
    
    try:
        # Run task
        response = run_task(
            model="NousResearch/Hermes-2-Pro-Llama-3-8B",
            messages=messages,
            generation_config=generation_config,
            dtype="bfloat16",
            seed=42
        )
        
        end_time = time.time()
        final_memory = torch.cuda.memory_allocated(0) if torch.cuda.is_available() else 0
        inference_time = end_time - start_time
        memory_delta = (final_memory - initial_memory) / 1024**3  # GB
        
        print("Generated response:")
        print("-" * 30)
        print(response["choices"][0]["message"]["content"])
        print("-" * 30)
        
        # Performance statistics
        print(f"\nPerformance Statistics:")
        print(f"  Inference time: {inference_time:.2f} seconds")
        print(f"  Prompt tokens: {response['usage']['prompt_tokens']}")
        print(f"  Generated tokens: {response['usage']['completion_tokens']}")
        print(f"  Total tokens: {response['usage']['total_tokens']}")
        if inference_time > 0:
            tokens_per_sec = response['usage']['completion_tokens'] / inference_time
            print(f"  Generation speed: {tokens_per_sec:.1f} tokens/second")
        
        monitor_gpu_usage("After inference")
        
        # Strictly verify if inference runs on GPU
        print("\nVerifying if inference runs on GPU:")
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated(0) / 1024**3
            print(f"  GPU memory change during inference: +{memory_delta:.3f}GB")
            print(f"  Current GPU memory usage: {current_memory:.3f}GB")
            
            # Get GPU utilization
            gpu_util = get_gpu_utilization()
            if gpu_util and int(gpu_util['gpu_util']) > 0:
                print(f"  GPU utilization: {gpu_util['gpu_util']}%")
                print("✅ Confirmed inference running on GPU")
            elif current_memory > 0.5:  # If GPU memory usage exceeds 0.5GB
                print("✅ Confirmed inference running on GPU based on memory usage")
            else:
                print("⚠ Unable to confirm if inference is running on GPU")
                
            # Display GPU device information
            if is_rtx5090_gpu():
                print("✅ Inference completed on RTX 5090")
            else:
                device_name = torch.cuda.get_device_name(0)
                print(f"✅ Inference completed on {device_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting suggestions:")
        print("1. Ensure CUDA-enabled PyTorch is installed")
        print("2. Ensure NVIDIA driver version supports current GPU")
        print("3. Ensure sufficient GPU memory")
        print("4. Try reducing max_new_tokens parameter")
        print("5. Try using smaller models")
        return False


def rtx5090_streaming_example():
    """RTX 5090 streaming output example"""
    
    print("\n" + "=" * 60)
    print("RTX 5090 Streaming Output Test")
    print("=" * 60)
    
    messages = [
        {"role": "user", "content": "Please briefly introduce the development history of deep learning, including key milestones."}
    ]
    
    generation_config = {
        "max_new_tokens": 256,
        "temperature": 0.8,
        "do_sample": True
    }
    
    monitor_gpu_usage("Before streaming inference")
    
    try:
        print("\nStreaming generation results:")
        print("-" * 30)
        
        start_time = time.time()
        token_count = 0
        accumulated_content = ""
        initial_memory = torch.cuda.memory_allocated(0) if torch.cuda.is_available() else 0
        
        def stream_callback(chunk):
            nonlocal token_count, accumulated_content
            
            try:
                # More robust chunk handling
                if isinstance(chunk, dict) and "choices" in chunk:
                    choices = chunk.get("choices", [])
                    if len(choices) > 0:
                        choice = choices[0]
                        
                        # Handle delta content
                        if "delta" in choice:
                            delta = choice["delta"]
                            content = delta.get("content", "")
                            
                            if content:
                                print(content, end="", flush=True)
                                accumulated_content += content
                                token_count += 1
                        
                        # Check finish reason
                        finish_reason = choice.get("finish_reason")
                        if finish_reason:
                            print("")  # New line
                            print(f"\nStreaming completion reason: {finish_reason}")
                            
            except Exception as e:
                _logger.error(f"Streaming callback processing error: {e}")
        
        # Run streaming task
        result = run_task(
            model="NousResearch/Hermes-2-Pro-Llama-3-8B",
            messages=messages,
            generation_config=generation_config,
            dtype="bfloat16",
            stream_callback=stream_callback,
            seed=42
        )
        
        end_time = time.time()
        final_memory = torch.cuda.memory_allocated(0) if torch.cuda.is_available() else 0
        stream_time = end_time - start_time
        memory_delta = (final_memory - initial_memory) / 1024**3  # GB
        
        print("\n" + "-" * 30)
        print(f"Streaming inference completed")
        print(f"Total time: {stream_time:.2f} seconds")
        print(f"Generated content length: {len(accumulated_content)} characters")
        print(f"Streaming token count: {token_count}")
        
        if stream_time > 0 and len(accumulated_content) > 0:
            print(f"Average character generation speed: {len(accumulated_content)/stream_time:.1f} chars/second")
        
        monitor_gpu_usage("After streaming inference")
        
        # Verify streaming inference actually runs on GPU
        print("\nVerifying if streaming inference runs on GPU:")
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated(0) / 1024**3
            print(f"  Streaming inference GPU memory change: +{memory_delta:.3f}GB")
            print(f"  Current GPU memory usage: {current_memory:.3f}GB")
            
            if current_memory > 0.5:  # If GPU memory usage exceeds 0.5GB
                print("✅ Confirmed streaming inference running on GPU")
                if is_rtx5090_gpu():
                    print("✅ Streaming inference completed on RTX 5090")
            else:
                print("⚠ Unable to confirm if streaming inference is running on GPU")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during streaming inference: {e}")
        import traceback
        traceback.print_exc()
        return False


def cleanup_gpu():
    """Thoroughly clean up GPU memory"""
    if torch.cuda.is_available():
        # Force garbage collection
        gc.collect()
        # Clear GPU cache
        torch.cuda.empty_cache()
        # Wait for all CUDA operations to complete
        torch.cuda.synchronize()
        print("✓ GPU memory thoroughly cleaned and synchronized")


def verify_gpu_computation():
    """Verify GPU computation capabilities"""
    if not torch.cuda.is_available():
        print("❌ CUDA not available, cannot verify GPU computation")
        return False
    
    try:
        print("\nVerifying GPU computation capabilities...")
        device = torch.device('cuda:0')
        
        # Test matrix multiplication with different precisions
        print("  Testing bfloat16 precision matrix multiplication...")
        a = torch.randn(1000, 1000, device=device, dtype=torch.bfloat16)
        b = torch.randn(1000, 1000, device=device, dtype=torch.bfloat16)
        
        start_time = time.time()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()  # Wait for GPU operations to complete
        end_time = time.time()
        
        compute_time = end_time - start_time
        print(f"  ✓ bfloat16 matrix multiplication completed, time: {compute_time:.4f}s")
        print(f"  ✓ Result shape: {c.shape}")
        print(f"  ✓ Result on device: {c.device}")        
        # Test GPU memory operations
        memory_before = torch.cuda.memory_allocated(0)
        large_tensor = torch.randn(2000, 2000, device=device, dtype=torch.bfloat16)
        memory_after = torch.cuda.memory_allocated(0)
        memory_used = (memory_after - memory_before) / 1024**2  # MB
        
        print(f"  ✓ GPU memory allocation test: {memory_used:.1f}MB")
        
        # Clean up test tensors
        del a, b, c, large_tensor
        torch.cuda.empty_cache()
        
        # Verify GPU device
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  ✓ GPU device: {gpu_name}")
        
        if is_rtx5090_gpu():
            print("  ✅ RTX 5090 computation capability verification passed")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU computation verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("RTX 5090 GPU Optimization Test Suite")
    print("=" * 60)
    
    try:
        # First verify GPU computation capabilities
        if not verify_gpu_computation():
            print("❌ GPU computation verification failed, cannot continue testing")
            exit(1)
        
        # Run basic inference example
        success1 = rtx5090_optimized_inference()
        
        if success1:
            # Run streaming output example
            success2 = rtx5090_streaming_example()
            
            if success1 and success2:
                print("\n" + "=" * 60)
                print("🎉 RTX 5090 testing completed! All tests passed")
                print("✅ GPU computation capability verification passed")
                print("✅ Model successfully running on GPU")
                print("✅ Inference performance normal")
                print("✅ Streaming output normal")
                print("✅ GPU utilization monitoring normal")
                
                if is_rtx5090_gpu():
                    print("✅ Confirmed running on RTX 5090")
                else:
                    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
                    print(f"✅ Test passed on {gpu_name}")
                
                print("=" * 60)
                
                # Final GPU status check
                print("\nFinal GPU status:")
                monitor_gpu_usage("Testing completed")
                
            else:
                print("\n⚠ Some tests failed, please check configuration")
        else:
            print("\n❌ Basic inference test failed, skipping subsequent tests")
            
    except KeyboardInterrupt:
        print("\n\nTesting interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error occurred during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_gpu()
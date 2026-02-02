#!/usr/bin/env python3
"""
CLI integration test for TrtLLM WideEP with real database queries.
Shows both MoE compute and communication performance.
"""

import logging
import argparse
from typing import Dict, List, Any

from aiconfigurator.sdk import operations as ops
from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.perf_database import get_database
from aiconfigurator.sdk.models import get_model
from dataclasses import dataclass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WideEPCLITester:
    """Test runner for WideEP CLI functionality with real database."""
    
    def __init__(self):
        self.results = []
    
    def create_runtime_config(self, batch_size: int = 256, sequence_length: int = 8192) -> config.RuntimeConfig:
        """Create runtime configuration with batch size and sequence length."""
        return config.RuntimeConfig(
            batch_size=batch_size,
            isl=sequence_length,  # input sequence length
            osl=128,  # output sequence length
            beam_width=1
        )
    
    def create_model_config(self, backend: str, enable_wideep: bool) -> config.ModelConfig:
        """Create model configuration based on backend and WideEP settings."""
        
        base_config = {
            "tp_size": 1,
            "pp_size": 1,
            "gemm_quant_mode": common.GEMMQuantMode.float16,
            "kvcache_quant_mode": common.KVCacheQuantMode.float16,
            "fmha_quant_mode": common.FMHAQuantMode.float16,
            "moe_tp_size": 1,
            "moe_ep_size": 8,
            "attention_dp_size": 8,
        }
        
        if backend == "trtllm":
            if enable_wideep:
                return config.ModelConfig(
                    **base_config,
                    moe_quant_mode=common.MoEQuantMode.nvfp4,
                    workload_distribution="power_law_1.01_eplb",
                    enable_wideep=True,
                    wideep_num_slots=288,
                )
            else:
                return config.ModelConfig(
                    **base_config,
                    moe_quant_mode=common.MoEQuantMode.nvfp4,
                    workload_distribution="power_law",
                    enable_wideep=False,
                )
        elif backend == "sglang":
            return config.ModelConfig(
                **base_config,
                moe_quant_mode=common.MoEQuantMode.float16,
                workload_distribution="uniform",
                enable_wideep=enable_wideep,
                moe_backend="deepep_moe" if enable_wideep else None,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    
    def test_operations(self, model, database, test_name: str, 
                       batch_size: int = 256, sequence_length: int = 8192,
                       tokens: List[int] = [1024, 8192]):
        """Test both MoE compute and dispatch operations."""
        
        print(f"\n{'='*60}")
        print(f"Test: {test_name}")
        print(f"Batch Size: {batch_size}, Sequence Length: {sequence_length}")
        print(f"{'='*60}")
        
        result = {
            "test_name": test_name,
            "moe_compute": [],
            "dispatch_comm": []
        }
        
        # Test MoE compute operations
        print("\n[MoE Compute Operations]")
        moe_tested = False
        for op in model.context_ops:
            if 'moe' in op._name.lower() and 'dispatch' not in op._name.lower() and not moe_tested:
                moe_tested = True
                print(f"Operation: {op._name}")
                print(f"Class: {type(op).__name__}")
                
                # Show WideEP specific attributes
                if isinstance(op, ops.TrtLLMWideEPMoE):
                    print(f"  - WideEP Mode: Yes")
                    print(f"  - EPLB slots: {op._num_slots}")
                    print(f"  - Workload: {op._workload_distribution}")
                else:
                    print(f"  - WideEP Mode: No")
                    if hasattr(op, '_workload_distribution'):
                        print(f"  - Workload: {op._workload_distribution}")
                
                # Query for different token sizes
                print(f"\nPerformance Results:")
                for x in tokens:
                    try:
                        result_perf = op.query(database, x=x)
                        latency = float(result_perf)
                        energy = result_perf.energy if hasattr(result_perf, 'energy') else 0
                        print(f"  Tokens={x:5d}: latency={latency:8.2f}ms, energy={energy:.2f}")
                        result["moe_compute"].append({
                            "tokens": x,
                            "latency": latency,
                            "energy": energy
                        })
                    except Exception as e:
                        print(f"  Tokens={x:5d}: Query error - {e}")
                        result["moe_compute"].append({
                            "tokens": x,
                            "error": str(e)
                        })
                break
        
        # Test dispatch (communication) operations
        print("\n[Dispatch Communication Operations]")
        dispatch_tested = False
        for op in model.context_ops:
            if 'dispatch' in op._name.lower() and not dispatch_tested:
                dispatch_tested = True
                print(f"Operation: {op._name}")
                print(f"Class: {type(op).__name__}")
                
                # Show dispatch specific attributes
                if isinstance(op, ops.TrtLLMWideEPMoEDispatch):
                    print(f"  - WideEP Dispatch: Yes")
                    print(f"  - Pre-dispatch: {op._pre_dispatch}")
                    if hasattr(op, '_quant_mode'):
                        print(f"  - Quant mode: {op._quant_mode.name}")
                elif isinstance(op, ops.MoEDispatch):
                    print(f"  - Standard Dispatch")
                    print(f"  - Pre-dispatch: {op._pre_dispatch}")
                
                # Query for different token sizes
                print(f"\nPerformance Results:")
                for x in tokens:
                    try:
                        result_perf = op.query(database, x=x)
                        latency = float(result_perf)
                        energy = result_perf.energy if hasattr(result_perf, 'energy') else 0
                        print(f"  Tokens={x:5d}: latency={latency:8.2f}ms, energy={energy:.2f}")
                        result["dispatch_comm"].append({
                            "tokens": x,
                            "latency": latency,
                            "energy": energy
                        })
                    except Exception as e:
                        print(f"  Tokens={x:5d}: Query error - {e}")
                        result["dispatch_comm"].append({
                            "tokens": x,
                            "error": str(e)
                        })
                break
        
        self.results.append(result)
        return result
    
    def run_trtllm_wideep_test(self, batch_size: int = 256, sequence_length: int = 8192):
        """Run TrtLLM with WideEP test."""
        
        model_config = self.create_model_config("trtllm", enable_wideep=True)
        
        try:
            model = get_model(
                model_path="deepseek-ai/DeepSeek-V3",
                model_config=model_config,
                backend_name="trtllm"
            )
            
            database = get_database(
                backend="trtllm",
                system="gb200_sxm",
                version="1.2.0rc6"
            )
            
            return self.test_operations(
                model, 
                database, 
                f"TrtLLM with WideEP",
                batch_size=batch_size,
                sequence_length=sequence_length,
                tokens=[1024, 4096, 8192, 16384]
            )
        except Exception as e:
            logger.error(f"TrtLLM WideEP test failed: {e}")
            return None
    
    def run_trtllm_standard_test(self, batch_size: int = 256, sequence_length: int = 8192):
        """Run standard TrtLLM test."""
        
        model_config = self.create_model_config("trtllm", enable_wideep=False)
        
        try:
            model = get_model(
                model_path="deepseek-ai/DeepSeek-V3",
                model_config=model_config,
                backend_name="trtllm"
            )
            
            database = get_database(
                backend="trtllm",
                system="gb200_sxm",
                version="1.2.0rc6"
            )
            
            return self.test_operations(
                model, 
                database, 
                f"TrtLLM Standard (No WideEP)",
                batch_size=batch_size,
                sequence_length=sequence_length,
                tokens=[1024, 4096, 8192, 16384]
            )
        except Exception as e:
            logger.error(f"TrtLLM standard test failed: {e}")
            return None
    
    def run_sglang_wideep_test(self, batch_size: int = 256, sequence_length: int = 8192):
        """Run SGLang with WideEP test."""
        
        model_config = self.create_model_config("sglang", enable_wideep=True)
        
        try:
            model = get_model(
                model_path="deepseek-ai/DeepSeek-V3",
                model_config=model_config,
                backend_name="sglang"
            )
            
            database = get_database(
                backend="sglang",
                system="gb200_sxm",
                version="0.3.7"
            )
            
            return self.test_operations(
                model,
                database,
                f"SGLang with WideEP (deepep_moe)",
                batch_size=batch_size,
                sequence_length=sequence_length,
                tokens=[1024, 4096, 8192]
            )
        except Exception as e:
            logger.error(f"SGLang WideEP test failed: {e}")
            return None
    
    def print_summary(self):
        """Print summary of all test results."""
        
        print(f"\n{'='*60}")
        print("SUMMARY OF ALL TESTS")
        print(f"{'='*60}")
        
        for result in self.results:
            print(f"\n{result['test_name']}:")
            
            if result["moe_compute"]:
                print("  MoE Compute:")
                for data in result["moe_compute"]:
                    if "error" in data:
                        print(f"    Tokens {data['tokens']}: ERROR - {data['error']}")
                    else:
                        print(f"    Tokens {data['tokens']}: {data['latency']:.2f}ms")
            
            if result["dispatch_comm"]:
                print("  Dispatch Communication:")
                for data in result["dispatch_comm"]:
                    if "error" in data:
                        print(f"    Tokens {data['tokens']}: ERROR - {data['error']}")
                    else:
                        print(f"    Tokens {data['tokens']}: {data['latency']:.2f}ms")
        
        print(f"\n{'='*60}")
        print(f"Total tests run: {len(self.results)}")
        print(f"{'='*60}")


def main():
    """Main entry point for the CLI test."""
    
    parser = argparse.ArgumentParser(description="Test WideEP functionality through CLI")
    parser.add_argument(
        "--test",
        choices=["trtllm-wideep", "trtllm-standard", "sglang-wideep", "all"],
        default="all",
        help="Which test to run"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for testing (default: 256)"
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=8192,
        help="Sequence length for testing (default: 8192)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    tester = WideEPCLITester()
    
    print("="*60)
    print("WideEP CLI Integration Test with Real Database")
    print(f"Batch Size: {args.batch_size}, Sequence Length: {args.sequence_length}")
    print("="*60)
    
    if args.test == "all" or args.test == "trtllm-wideep":
        tester.run_trtllm_wideep_test(args.batch_size, args.sequence_length)
    
    if args.test == "all" or args.test == "trtllm-standard":
        tester.run_trtllm_standard_test(args.batch_size, args.sequence_length)
    
    if args.test == "all" or args.test == "sglang-wideep":
        tester.run_sglang_wideep_test(args.batch_size, args.sequence_length)
    
    tester.print_summary()


if __name__ == "__main__":
    main()
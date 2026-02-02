#!/usr/bin/env python3
"""
Real performance test for TrtLLM WideEP using actual database.
This script tests with real performance data and compares WideEP vs Standard TrtLLM.
"""

import logging
from typing import Dict, List
from dataclasses import dataclass

from aiconfigurator.sdk import operations as ops
from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.perf_database import get_database

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Store test results for comparison."""
    name: str
    moe_latency: float
    dispatch_latency: float
    total_latency: float
    model_type: str


def test_wideep_real_data():
    """Test TrtLLM WideEP with real database data."""
    
    print("\n" + "="*60)
    print("Testing TrtLLM WideEP with Real Database")
    print("="*60)
    
    # Get real database
    database = get_database(
        backend="trtllm",
        system="gb200_sxm",
        version="1.2.0rc6"
    )
    
    # Test different configurations
    test_configs = [
        (1024, "Short sequence"),
        (4096, "Medium sequence"),
        (8192, "Long sequence"),
        (16384, "Very long sequence"),
    ]
    
    results = []
    
    # Create WideEP operations
    print("\n1. Creating WideEP Operations...")
    
    moe_op = ops.TrtLLMWideEPMoE(
        name="context_moe",
        scale_factor=1.0,
        hidden_size=7168,
        inter_size=2048,
        topk=8,
        num_experts=256,
        moe_tp_size=1,
        moe_ep_size=8,
        quant_mode=common.MoEQuantMode.nvfp4,
        workload_distribution="power_law_1.01_eplb",
        attention_dp_size=8,
        num_slots=288,
    )
    
    pre_dispatch_op = ops.TrtLLMWideEPMoEDispatch(
        name="context_pre_dispatch",
        scale_factor=1.0,
        hidden_size=7168,
        topk=8,
        num_experts=256,
        moe_tp_size=1,
        moe_ep_size=8,
        attention_dp_size=8,
        pre_dispatch=True,
        quant_mode=common.MoEQuantMode.nvfp4,
    )
    
    post_dispatch_op = ops.TrtLLMWideEPMoEDispatch(
        name="context_post_dispatch",
        scale_factor=1.0,
        hidden_size=7168,
        topk=8,
        num_experts=256,
        moe_tp_size=1,
        moe_ep_size=8,
        attention_dp_size=8,
        pre_dispatch=False,
        quant_mode=common.MoEQuantMode.nvfp4,
    )
    
    print("   ✅ WideEP operations created successfully")
    
    # Test with different token sizes
    print("\n2. Querying Real Database for WideEP...")
    for tokens, desc in test_configs:
        try:
            # Query MoE compute
            moe_result = moe_op.query(database, x=tokens)
            moe_latency = float(moe_result)
            
            # Query pre-dispatch
            pre_dispatch_result = pre_dispatch_op.query(database, x=tokens)
            pre_dispatch_latency = float(pre_dispatch_result)
            
            # Query post-dispatch
            post_dispatch_result = post_dispatch_op.query(database, x=tokens)
            post_dispatch_latency = float(post_dispatch_result)
            
            # Total communication latency
            dispatch_latency = pre_dispatch_latency + post_dispatch_latency
            total_latency = moe_latency + dispatch_latency
            
            print(f"\n   {desc} (tokens={tokens}):")
            print(f"      MoE compute: {moe_latency:.2f}ms")
            print(f"      Pre-dispatch: {pre_dispatch_latency:.2f}ms")
            print(f"      Post-dispatch: {post_dispatch_latency:.2f}ms")
            print(f"      Total: {total_latency:.2f}ms")
            
            results.append(TestResult(
                name=f"WideEP_{tokens}",
                moe_latency=moe_latency,
                dispatch_latency=dispatch_latency,
                total_latency=total_latency,
                model_type="WideEP"
            ))
            
        except Exception as e:
            print(f"   ❌ Error for {tokens} tokens: {e}")
    
    return results


def test_standard_real_data():
    """Test standard TrtLLM (without WideEP) with real database data."""
    
    print("\n" + "="*60)
    print("Testing Standard TrtLLM with Real Database")
    print("="*60)
    
    # Get real database
    database = get_database(
        backend="trtllm",
        system="gb200_sxm",
        version="1.2.0rc6"
    )
    
    # Test configurations (same as WideEP)
    test_configs = [
        (1024, "Short sequence"),
        (4096, "Medium sequence"),
        (8192, "Long sequence"),
        (16384, "Very long sequence"),
    ]
    
    results = []
    
    # Create standard operations
    print("\n1. Creating Standard Operations...")
    
    moe_op = ops.MoE(
        name="context_moe",
        scale_factor=1.0,
        hidden_size=7168,
        inter_size=2048,
        topk=8,
        num_experts=256,
        moe_tp_size=1,
        moe_ep_size=8,
        quant_mode=common.MoEQuantMode.nvfp4,
        workload_distribution="power_law_1.01",
        attention_dp_size=8,
    )
    
    pre_dispatch_op = ops.MoEDispatch(
        name="context_pre_dispatch",
        scale_factor=1.0,
        hidden_size=7168,
        topk=8,
        num_experts=256,
        moe_tp_size=1,
        moe_ep_size=8,
        attention_dp_size=8,
        pre_dispatch=True,
    )
    
    post_dispatch_op = ops.MoEDispatch(
        name="context_post_dispatch",
        scale_factor=1.0,
        hidden_size=7168,
        topk=8,
        num_experts=256,
        moe_tp_size=1,
        moe_ep_size=8,
        attention_dp_size=8,
        pre_dispatch=False,
    )
    
    print("   ✅ Standard operations created successfully")
    
    # Test with different token sizes
    print("\n2. Querying Real Database for Standard TrtLLM...")
    for tokens, desc in test_configs:
        try:
            # Query MoE compute
            moe_result = moe_op.query(database, x=tokens)
            moe_latency = float(moe_result)
            
            # Query pre-dispatch
            pre_dispatch_result = pre_dispatch_op.query(database, x=tokens)
            pre_dispatch_latency = float(pre_dispatch_result)
            
            # Query post-dispatch
            post_dispatch_result = post_dispatch_op.query(database, x=tokens)
            post_dispatch_latency = float(post_dispatch_result)
            
            # Total communication latency
            dispatch_latency = pre_dispatch_latency + post_dispatch_latency
            total_latency = moe_latency + dispatch_latency
            
            print(f"\n   {desc} (tokens={tokens}):")
            print(f"      MoE compute: {moe_latency:.2f}ms")
            print(f"      Pre-dispatch: {pre_dispatch_latency:.2f}ms")
            print(f"      Post-dispatch: {post_dispatch_latency:.2f}ms")
            print(f"      Total: {total_latency:.2f}ms")
            
            results.append(TestResult(
                name=f"Standard_{tokens}",
                moe_latency=moe_latency,
                dispatch_latency=dispatch_latency,
                total_latency=total_latency,
                model_type="Standard"
            ))
            
        except Exception as e:
            print(f"   ❌ Error for {tokens} tokens: {e}")
    
    return results


def compare_results(wideep_results: List[TestResult], standard_results: List[TestResult]):
    """Compare WideEP and Standard results."""
    
    print("\n" + "="*60)
    print("Performance Comparison: WideEP vs Standard TrtLLM")
    print("="*60)
    
    # Match results by token count
    for wideep, standard in zip(wideep_results, standard_results):
        tokens = wideep.name.split('_')[1]
        
        print(f"\nTokens: {tokens}")
        print(f"  MoE Compute:")
        print(f"    WideEP:   {wideep.moe_latency:8.2f}ms")
        print(f"    Standard: {standard.moe_latency:8.2f}ms")
        if standard.moe_latency > 0:
            improvement = ((standard.moe_latency - wideep.moe_latency) / standard.moe_latency) * 100
            print(f"    Improvement: {improvement:+.1f}%")
        
        print(f"  Communication:")
        print(f"    WideEP:   {wideep.dispatch_latency:8.2f}ms")
        print(f"    Standard: {standard.dispatch_latency:8.2f}ms")
        if standard.dispatch_latency > 0:
            improvement = ((standard.dispatch_latency - wideep.dispatch_latency) / standard.dispatch_latency) * 100
            print(f"    Improvement: {improvement:+.1f}%")
        
        print(f"  Total:")
        print(f"    WideEP:   {wideep.total_latency:8.2f}ms")
        print(f"    Standard: {standard.total_latency:8.2f}ms")
        if standard.total_latency > 0:
            improvement = ((standard.total_latency - wideep.total_latency) / standard.total_latency) * 100
            print(f"    Improvement: {improvement:+.1f}%")


def test_full_model_comparison():
    """Test full models with WideEP vs Standard."""
    
    print("\n" + "="*60)
    print("Full Model Comparison Test")
    print("="*60)
    
    # Create model configs
    wideep_config = config.ModelConfig(
        tp_size=1,
        pp_size=1,
        gemm_quant_mode=common.GEMMQuantMode.float16,
        moe_quant_mode=common.MoEQuantMode.nvfp4,
        kvcache_quant_mode=common.KVCacheQuantMode.float16,
        fmha_quant_mode=common.FMHAQuantMode.float16,
        moe_tp_size=1,
        moe_ep_size=8,
        workload_distribution="power_law_1.01_eplb",
        attention_dp_size=8,
        enable_wideep=True,
        wideep_num_slots=288,
    )
    
    standard_config = config.ModelConfig(
        tp_size=1,
        pp_size=1,
        gemm_quant_mode=common.GEMMQuantMode.float16,
        moe_quant_mode=common.MoEQuantMode.nvfp4,
        kvcache_quant_mode=common.KVCacheQuantMode.float16,
        fmha_quant_mode=common.FMHAQuantMode.float16,
        moe_tp_size=1,
        moe_ep_size=8,
        workload_distribution="power_law_1.01",
        attention_dp_size=8,
        enable_wideep=False,
    )
    
    # Create models
    print("\n1. Creating models...")
    
    wideep_model = get_model(
        model_path="deepseek-ai/DeepSeek-V3",
        model_config=wideep_config,
        backend_name="trtllm"
    )
    print(f"   WideEP model: {type(wideep_model).__name__}")
    
    standard_model = get_model(
        model_path="deepseek-ai/DeepSeek-V3",
        model_config=standard_config,
        backend_name="trtllm"
    )
    print(f"   Standard model: {type(standard_model).__name__}")
    
    # Get database
    database = get_database(
        backend="trtllm",
        system="gb200_sxm",
        version="1.2.0rc6"
    )
    
    # Count operations
    print("\n2. Operation counts:")
    
    wideep_moe_count = sum(1 for op in wideep_model.context_ops if isinstance(op, ops.TrtLLMWideEPMoE))
    wideep_dispatch_count = sum(1 for op in wideep_model.context_ops if isinstance(op, ops.TrtLLMWideEPMoEDispatch))
    
    standard_moe_count = sum(1 for op in standard_model.context_ops 
                           if isinstance(op, ops.MoE) and not isinstance(op, ops.TrtLLMWideEPMoE))
    standard_dispatch_count = sum(1 for op in standard_model.context_ops 
                                if isinstance(op, ops.MoEDispatch) and not isinstance(op, ops.TrtLLMWideEPMoEDispatch))
    
    print(f"   WideEP: {wideep_moe_count} MoE ops, {wideep_dispatch_count} dispatch ops")
    print(f"   Standard: {standard_moe_count} MoE ops, {standard_dispatch_count} dispatch ops")
    
    # Test first MoE operation from each model
    print("\n3. Testing first MoE operation:")
    
    for op in wideep_model.context_ops:
        if isinstance(op, ops.TrtLLMWideEPMoE):
            result = op.query(database, x=8192)
            print(f"   WideEP MoE latency: {float(result):.2f}ms")
            break
    
    for op in standard_model.context_ops:
        if isinstance(op, ops.MoE) and not isinstance(op, ops.TrtLLMWideEPMoE):
            result = op.query(database, x=8192)
            print(f"   Standard MoE latency: {float(result):.2f}ms")
            break


def main():
    """Main entry point."""
    
    print("="*60)
    print("TrtLLM WideEP Real Performance Test")
    print("="*60)
    
    # Test individual operations
    wideep_results = test_wideep_real_data()
    standard_results = test_standard_real_data()
    
    # Compare results
    if wideep_results and standard_results:
        compare_results(wideep_results, standard_results)
    
    # Test full models
    test_full_model_comparison()
    
    print("\n" + "="*60)
    print("Test completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
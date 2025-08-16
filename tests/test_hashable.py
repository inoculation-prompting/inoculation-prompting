#!/usr/bin/env python3
"""Test script to verify that Evaluation is hashable"""

from mi.evaluation.data_models import Evaluation, EvaluationContext
from mi.llm.data_models import SampleCfg, Model, Judgment

def test_evaluation_hashable():
    """Test that Evaluation objects are hashable"""
    
    # Create sample data
    model = Model(id="gpt-4", type="openai", parent_model=None)
    sample_cfg = SampleCfg(temperature=0.7)
    
    context1 = EvaluationContext(
        question="What is 2+2?",
        system_prompt="You are a helpful assistant."
    )
    context2 = EvaluationContext(
        question="What is 3+3?",
        system_prompt="You are a helpful assistant."
    )
    
    judgment = Judgment(
        judge_model=model,
        sample_cfg=sample_cfg,
        template="Rate the response from 1-10"
    )
    
    # Create evaluation using the helper method
    eval1 = Evaluation.from_dict_with_lists(
        id="test_eval_1",
        contexts=[context1, context2],
        n_samples_per_context=5,
        sample_cfg=sample_cfg,
        judgment_map={"quality": judgment}
    )
    
    eval2 = Evaluation.from_dict_with_lists(
        id="test_eval_2",
        contexts=[context1],
        n_samples_per_context=3,
        sample_cfg=sample_cfg,
        judgment_map={}
    )
    
    # Test that objects are hashable
    print("Testing hashability...")
    hash1 = hash(eval1)
    hash2 = hash(eval2)
    print(f"Hash of eval1: {hash1}")
    print(f"Hash of eval2: {hash2}")
    
    # Test that they can be used in sets
    eval_set = {eval1, eval2}
    print(f"Set contains {len(eval_set)} evaluations")
    
    # Test that they can be used as dictionary keys
    eval_dict = {eval1: "first", eval2: "second"}
    print(f"Dictionary contains {len(eval_dict)} evaluations")
    
    # Test accessing judgment_map as dict
    judgment_dict = eval1.get_judgment_map_dict()
    print(f"Judgment map has {len(judgment_dict)} items")
    
    print("✅ All hashability tests passed!")

if __name__ == "__main__":
    test_evaluation_hashable()

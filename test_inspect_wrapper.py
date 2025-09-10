"""Test script for inspect_evals wrapper"""

import asyncio
from inspect_ai import Task, scorer
from inspect_ai.dataset import Sample

from mi.llm.data_models import Model
from mi.eval.inspect_wrapper import eval


async def test_basic_functionality():
    """Test basic functionality of the inspect_evals wrapper"""
    
    # Create a simple test task
    task = Task(
        dataset=[
            Sample(input="What is 2+2?", target="4"),
            Sample(input="What is 3+3?", target="6"),
        ],
        scorer=scorer.match()
    )
    
    # Create a test model (assuming you have a model available)
    # This would need to be replaced with an actual model from your system
    test_model = Model(
        id="gpt-4o-mini",  # Use a model that should be available
        type="openai"
    )
    
    try:
        # Run the evaluation
        results = await eval(
            model_groups={"test_group": [test_model]},
            tasks=[task]
        )
        
        print(f"Successfully completed {len(results)} evaluations")
        
        for model, group, task, eval_log in results:
            print(f"Model: {model.id}, Group: {group}, Task: {task.name or 'unnamed'}")
            
            # Handle different EvalLog structures
            samples = None
            if hasattr(eval_log, 'results') and eval_log.results and hasattr(eval_log.results, 'samples'):
                samples = eval_log.results.samples
            elif hasattr(eval_log, 'samples'):
                samples = eval_log.samples
            
            if samples:
                print(f"  Samples: {len(samples)}")
                scores = [s.score for s in samples if s.score is not None]
                if scores:
                    print(f"  Average score: {sum(scores) / len(scores):.3f}")
            else:
                print("  No samples found in results")
        
        return True
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_basic_functionality())
    if success:
        print("✅ Test passed!")
    else:
        print("❌ Test failed!")

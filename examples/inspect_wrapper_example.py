"""Example usage of the inspect_evals wrapper"""

import asyncio
from inspect_ai import Task, scorer
from inspect_ai.dataset import Sample

from mi.llm.data_models import Model
from mi.eval.inspect_wrapper import eval


# Example 1: Simple math task
def create_math_task():
    """Create a simple math evaluation task"""
    return Task(
        dataset=[
            Sample(input="What is 2+2?", target="4"),
            Sample(input="What is 3+3?", target="6"),
            Sample(input="What is 5+5?", target="10"),
            Sample(input="What is 7+3?", target="10"),
        ],
        scorer=scorer.match()
    )


# Example 2: Multiple choice task
def create_mc_task():
    """Create a multiple choice task"""
    return Task(
        dataset=[
            Sample(
                input="What is the capital of France?",
                target="Paris",
                choices=["Paris", "London", "Berlin", "Madrid"]
            ),
            Sample(
                input="What is 2+2?",
                target="4",
                choices=["3", "4", "5", "6"]
            ),
        ],
        scorer=scorer.choice()
    )


# Example 3: Pattern matching task
def create_pattern_task():
    """Create a pattern matching task"""
    return Task(
        dataset=[
            Sample(input="Complete the pattern: 1, 2, 3, ?", target="4"),
            Sample(input="Complete the pattern: A, B, C, ?", target="D"),
            Sample(input="Complete the pattern: 2, 4, 6, ?", target="8"),
        ],
        scorer=scorer.match()
    )


async def run_example():
    """Run example evaluations"""
    
    # Create test models (replace with your actual models)
    models = [
        Model(id="gpt-4o-mini", type="openai"),
        # Add more models as needed
    ]
    
    # Create tasks
    tasks = [
        create_math_task(),
        create_mc_task(),
        create_pattern_task(),
    ]
    
    print("🚀 Starting inspect_evals wrapper example...")
    
    try:
        # Run evaluations
        results = await eval(
            model_groups={"test_models": models},
            tasks=tasks
        )
        
        print(f"✅ Successfully completed {len(results)} evaluations")
        
        # Display results
        for model, group, task, eval_log in results:
            print(f"\n📊 Results for {model.id} on task '{task.name or 'unnamed'}':")
            
            if eval_log.results and eval_log.results.samples:
                samples = eval_log.results.samples
                scores = [s.score for s in samples if s.score is not None]
                
                print(f"  📝 Total samples: {len(samples)}")
                if scores:
                    avg_score = sum(scores) / len(scores)
                    print(f"  🎯 Average score: {avg_score:.3f}")
                    print(f"  📈 Score range: {min(scores):.3f} - {max(scores):.3f}")
                    
                    # Show individual results
                    print("  📋 Individual results:")
                    for i, sample in enumerate(samples[:3]):  # Show first 3
                        status = "✅" if sample.score == 1.0 else "❌"
                        print(f"    {status} Sample {i+1}: {sample.input[:50]}... -> {sample.output[:50]}...")
                    
                    if len(samples) > 3:
                        print(f"    ... and {len(samples) - 3} more samples")
            else:
                print("  ⚠️  No results available")
        
        return True
        
    except Exception as e:
        print(f"❌ Example failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_caching_example():
    """Demonstrate caching functionality"""
    
    print("\n🔄 Testing caching functionality...")
    
    # Create a simple task
    task = Task(
        dataset=[Sample(input="What is 1+1?", target="2")],
        scorer=scorer.match()
    )
    
    model = Model(id="gpt-4o-mini", type="openai")
    
    # Run the same evaluation twice
    print("First run (should execute):")
    results1 = await eval(
        model_groups={"test": [model]},
        tasks=[task]
    )
    
    print("Second run (should use cache):")
    results2 = await eval(
        model_groups={"test": [model]},
        tasks=[task]
    )
    
    print(f"✅ Caching test completed. Results count: {len(results1)} -> {len(results2)}")


if __name__ == "__main__":
    print("🧪 Inspect Evals Wrapper Examples")
    print("=" * 50)
    
    # Run main example
    success = asyncio.run(run_example())
    
    if success:
        # Run caching example
        asyncio.run(run_caching_example())
        print("\n🎉 All examples completed successfully!")
    else:
        print("\n💥 Examples failed!")

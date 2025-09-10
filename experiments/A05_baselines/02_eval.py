"""Evaluation script for mixed data baseline defense experiment.

This script evaluates the fine-tuned models on both in-distribution and
out-of-distribution tasks to measure the effectiveness of mixed data training.
"""

import asyncio
from pathlib import Path
from mi.eval import eval
from mi.llm.data_models import Model
from mi.finetuning.services import get_finetuned_model
from mi.external.openai_driver.data_models import OpenAIFTJobConfig
from experiments.A05_baselines.mixed_data_baseline import list_configs, get_experiment_summary

experiment_dir = Path(__file__).parent


async def get_finetuned_models(configs):
    """Get all finetuned models from the configurations."""
    print("Retrieving finetuned models...")
    
    model_groups = {}
    failed_models = []
    
    for config in configs:
        try:
            # Get the finetuned model
            finetuned_model = await get_finetuned_model(config.finetuning_config)
            
            if finetuned_model is None:
                print(f"  ⚠️  Model not ready: {config.group_name} (seed {config.finetuning_config.seed})")
                failed_models.append(config)
                continue
            
            # Add to model groups
            if config.group_name not in model_groups:
                model_groups[config.group_name] = []
            
            model_groups[config.group_name].append(finetuned_model)
            print(f"  ✓ {config.group_name} (seed {config.finetuning_config.seed}): {finetuned_model.id}")
            
        except Exception as e:
            print(f"  ✗ Error retrieving model for {config.group_name}: {e}")
            failed_models.append(config)
    
    if failed_models:
        print(f"\n⚠️  {len(failed_models)} models not ready yet. You may need to wait for fine-tuning to complete.")
        print("Run this script again later, or check job status with: python check_job_status.py")
    
    return model_groups, failed_models


async def run_evaluations(model_groups, settings):
    """Run evaluations on all model groups."""
    print("\nRunning evaluations...")
    
    all_results = []
    
    for setting in settings:
        print(f"\nEvaluating on {setting.get_domain_name()} setting...")
        
        # Get evaluations for this setting
        id_evals = setting.get_id_evals()
        ood_evals = setting.get_ood_evals()
        all_evals = id_evals + ood_evals
        
        print(f"  Running {len(all_evals)} evaluations...")
        
        # Run evaluations
        results = await eval(
            model_groups=model_groups,
            evaluations=all_evals,
            output_dir=experiment_dir / "results"
        )
        
        all_results.extend(results)
        print(f"  ✓ Completed {len(results)} evaluation runs")
    
    return all_results


async def main():
    """Main evaluation function."""
    print("=" * 60)
    print("MIXED DATA BASELINE DEFENSE - EVALUATION")
    print("=" * 60)
    
    # Print experiment summary
    summary = get_experiment_summary()
    print(f"\nExperiment: {summary['experiment_name']}")
    print(f"Settings: {summary['settings']}")
    
    # Generate configurations to get model info
    configs = list_configs(experiment_dir)
    settings = summary['settings']
    
    # Import the actual setting objects
    from mi.experiments.settings import insecure_code
    setting_objects = [insecure_code]  # Add more as needed
    
    # Get finetuned models
    model_groups, failed_models = await get_finetuned_models(configs)
    
    if not model_groups:
        print("\n❌ No models ready for evaluation. Please wait for fine-tuning to complete.")
        return
    
    # Add baseline model for comparison
    baseline_model = Model(id="gpt-4.1-2025-04-14", type="openai")
    model_groups["baseline"] = [baseline_model]
    print(f"\n✓ Added baseline model: {baseline_model.id}")
    
    print(f"\nModel groups ready for evaluation:")
    for group_name, models in model_groups.items():
        print(f"  {group_name}: {len(models)} models")
    
    # Run evaluations
    results = await run_evaluations(model_groups, setting_objects)
    
    print(f"\n✓ Evaluation completed!")
    print(f"Total evaluation runs: {len(results)}")
    print(f"Results saved to: {experiment_dir / 'results'}")
    
    # Print summary of results
    print(f"\nResults summary:")
    for model, group, evaluation, eval_results in results:
        if eval_results:
            scores = [row.scores[0] for row in eval_results if row.scores and row.scores[0] is not None]
            if scores:
                avg_score = sum(scores) / len(scores)
                print(f"  {evaluation.id} - {group}: {avg_score:.3f} (n={len(scores)})")


if __name__ == "__main__":
    asyncio.run(main())

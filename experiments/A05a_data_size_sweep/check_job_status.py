"""Job status checker for data size sweep baseline defense experiment.

This script checks the status of all fine-tuning jobs in the experiment.
"""

import asyncio
from pathlib import Path
from mi.finetuning.services import get_job_info
from experiments.A05a_data_size_sweep.data_size_sweep import list_configs, get_experiment_summary

experiment_dir = Path(__file__).parent


async def check_all_jobs():
    """Check the status of all fine-tuning jobs."""
    print("=" * 60)
    print("DATA SIZE SWEEP BASELINE DEFENSE - JOB STATUS")
    print("=" * 60)
    
    # Get experiment summary
    summary = get_experiment_summary()
    print(f"\nExperiment: {summary['experiment_name']}")
    print(f"Total expected jobs: {summary['total_configs']}")
    
    # Get all configurations
    configs = list_configs(experiment_dir)
    
    print(f"\nChecking {len(configs)} fine-tuning jobs...")
    
    # Status counters
    status_counts = {}
    completed_jobs = []
    failed_jobs = []
    running_jobs = []
    
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Checking {config.group_name} (seed {config.finetuning_config.seed})...")
        
        try:
            job_info = await get_job_info(config.finetuning_config)
            
            if job_info is None:
                print("  ❌ Job not found - may not have been launched")
                status = "not_found"
            else:
                status = job_info.status
                print(f"  Status: {status}")
                print(f"  Job ID: {job_info.id}")
                
                if hasattr(job_info, 'created_at'):
                    print(f"  Created: {job_info.created_at}")
                if hasattr(job_info, 'finished_at') and job_info.finished_at:
                    print(f"  Finished: {job_info.finished_at}")
                
                # Categorize jobs
                if status == "succeeded":
                    completed_jobs.append((config, job_info))
                elif status == "failed":
                    failed_jobs.append((config, job_info))
                elif status in ["running", "validating_files"]:
                    running_jobs.append((config, job_info))
            
            # Update status counts
            status_counts[status] = status_counts.get(status, 0) + 1
            
        except Exception as e:
            print(f"  ❌ Error checking job: {e}")
            status_counts["error"] = status_counts.get("error", 0) + 1
    
    # Print summary
    print("\n" + "="*60)
    print("JOB STATUS SUMMARY")
    print("="*60)
    
    for status, count in status_counts.items():
        print(f"{status}: {count}")
    
    print(f"\nCompleted: {len(completed_jobs)}")
    print(f"Running: {len(running_jobs)}")
    print(f"Failed: {len(failed_jobs)}")
    
    # Show completed jobs
    if completed_jobs:
        print(f"\n✅ COMPLETED JOBS ({len(completed_jobs)}):")
        for config, job_info in completed_jobs:
            print(f"  {config.group_name} (seed {config.finetuning_config.seed}) - {job_info.id}")
    
    # Show running jobs
    if running_jobs:
        print(f"\n🔄 RUNNING JOBS ({len(running_jobs)}):")
        for config, job_info in running_jobs:
            print(f"  {config.group_name} (seed {config.finetuning_config.seed}) - {job_info.id}")
    
    # Show failed jobs
    if failed_jobs:
        print(f"\n❌ FAILED JOBS ({len(failed_jobs)}):")
        for config, job_info in failed_jobs:
            print(f"  {config.group_name} (seed {config.finetuning_config.seed}) - {job_info.id}")
    
    # Recommendations
    print(f"\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    if len(completed_jobs) == len(configs):
        print("🎉 All jobs completed! You can now run evaluation:")
        print("   python 02_eval.py")
    elif len(completed_jobs) > 0:
        print("📊 Some jobs completed. You can run partial evaluation:")
        print("   python 02_eval.py")
    elif len(running_jobs) > 0:
        print("⏳ Jobs still running. Check back later.")
    elif len(failed_jobs) > 0:
        print("⚠️  Some jobs failed. Check the OpenAI dashboard for details.")
    else:
        print("❓ No jobs found. Make sure to run training first:")
        print("   python 01_train.py")


if __name__ == "__main__":
    asyncio.run(check_all_jobs())

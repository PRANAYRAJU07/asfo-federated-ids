import subprocess
import time
import sys
from loguru import logger


def run_experiment(dataset: str, strategy: str, seed: int):
    logger.info(
        f"Starting Experiment: Dataset={dataset}, Strategy={strategy}, Seed={seed}"
    )

    # Since we are invoking it from a script that might already be inside uv,
    # sys.executable "scripts/federated_simulation.py" might be sufficient.
    # Let's use sys.executable directly to avoid nested uv calls.
    cmd_direct = [
        sys.executable,
        "scripts/federated_simulation.py",
        f"datasets={dataset}",
        f"strategy={strategy}",
        f"core.seed={seed}",
    ]

    start_time = time.time()
    try:
        # Run process synchronously
        process = subprocess.run(cmd_direct, check=True, capture_output=True, text=True)
        duration = time.time() - start_time
        logger.info(
            f"✅ Success: {dataset} | {strategy} | Seed {seed} (Took {duration:.1f}s)"
        )
        # Save output logs for debugging
        with open(f"artifacts/log_{dataset}_{strategy}_{seed}.txt", "w") as f:
            f.write(process.stdout)
            f.write("\n--- STDERR ---\n")
            f.write(process.stderr)

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed: {dataset} | {strategy} | Seed {seed}")
        logger.error(f"Error Output:\n{e.stderr}")
        with open(f"artifacts/error_{dataset}_{strategy}_{seed}.txt", "w") as f:
            f.write(e.stderr)


def main():
    datasets = ["edge_iiot", "unsw_nb15", "cicids2018"]
    strategies = [
        "fedavg",
        "fedprox",
        "fednova",
        "asfo_nokg",
        "asfo_norarity",
        "asfo_static",
        "asfo_adaptive",
        "asfo_full",
    ]
    seeds = [42, 123, 2025, 777, 999]

    total_experiments = len(datasets) * len(strategies) * len(seeds)
    logger.info(
        f"Starting ASFO Evaluation Matrix. Total experiments: {total_experiments}"
    )

    count = 1
    for dataset in datasets:
        for strategy in strategies:
            for seed in seeds:
                logger.info(f"--- Progress: {count} / {total_experiments} ---")
                run_experiment(dataset, strategy, seed)
                count += 1

    logger.info("🎉 All experiments completed successfully!")


if __name__ == "__main__":
    main()

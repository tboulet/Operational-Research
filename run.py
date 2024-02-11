# Logging
import os
import random
import wandb
from tensorboardX import SummaryWriter

# Config system
import hydra
from omegaconf import OmegaConf, DictConfig

# Utils
from tqdm import tqdm
import datetime
from time import time
from typing import Dict, Type
import cProfile

# ML libraries
import numpy as np

# Project imports
from algorithms import algo_tag_to_AlgoClass
from problems import problem_tag_to_ProblemClass
from src.time_measure import RuntimeMeter
from src.utils import to_integer, try_get_seed


@hydra.main(config_path="configs", config_name="config_default.yaml")
def main(config: DictConfig):

    # Get the config values from the config object.
    config = OmegaConf.to_container(config, resolve=True)
    solver_name: str = config["algo"]["name"]
    task_name: str = config["problem"]["name"]
    n_iterations_max: int = to_integer(config["n_iterations_max"])
    do_cli: bool = config["do_cli"]
    do_wandb: bool = config["do_wandb"]
    do_tb: bool = config["do_tb"]
    do_tqdm: bool = config["do_tqdm"]

    # Set seed
    seed = try_get_seed(config)
    np.random.seed(seed)
    random.seed(seed)
    
    # Get the algo
    AlgoClass = algo_tag_to_AlgoClass[solver_name]
    algo = AlgoClass(config=config["algo"]["config"])

    # Get the optimization problem
    ProblemClass = problem_tag_to_ProblemClass[task_name]
    problem = ProblemClass(config=config["problem"]["config"])

    # Initialize loggers
    run_name = f"[{solver_name}]_[{task_name}]_{datetime.datetime.now().strftime('%dth%mmo_%Hh%Mmin%Ss')}_seed{np.random.randint(1000)}"
    print(f"\nStarting run {run_name}")
    metrics = {}
    if do_wandb:
        run = wandb.init(
            name=run_name,
            config=config,
            **config["wandb_config"],
        )
    if do_tb:
        tb_writer = SummaryWriter(log_dir=f"tensorboard/{run_name}")

    # Start the algorithm
    algo.initialize_algorithm(problem=problem)
    
    # Training loop
    for iteration in tqdm(range(n_iterations_max), disable=not do_tqdm):
        print(f"\nIteration {iteration}...")
        # Get the algo result
        with RuntimeMeter("optimize") as rm:
            solution = algo.run_one_iteration()
            print("Solution computed.")
            
        # Apply the solution to the problem
        with RuntimeMeter("apply_solution") as rm:
            isFeasible, objective_value = problem.apply_solution(solution)
            print()
            print(f"Objective value at iteration {iteration}: {objective_value}")
            print(f"Is feasible at iteration {iteration}: {isFeasible}")
            metrics["objective_value"] = objective_value
            metrics["is_feasible"] = int(isFeasible)
            
        # Log metrics.
        with RuntimeMeter("log") as rm:
            metrics["optimize_time"] = rm.get_stage_runtime("optimize")
            metrics["apply_solution_time"] = rm.get_stage_runtime("apply_solution")
            metrics["log_time"] = rm.get_stage_runtime("log")
            metrics["iteration"] = iteration
            if do_wandb:
                runtime_in_ms = int(
                    rm.get_stage_runtime("optimize") * 1000
                )
                wandb.log(metric_result, step=runtime_in_ms)
            if do_tb:
                for metric_name, metric_result in metrics.items():
                    tb_writer.add_scalar(
                        f"metrics/{metric_name}",
                        metric_result,
                        global_step=rm.get_stage_runtime("optimize"),
                    )
            if do_cli:
                print(
                    f"Metric results at iteration {iteration}: {metrics}"
                )

        # Stop the run if the algorithm has finished
        if algo.stop_algorithm():
            print(f"\nStopping run under algorithm criteria at iteration {iteration}")
            break
        
        # Stop the run if the optimization time has exceeded the maximum time
        if rm.get_stage_runtime("optimize") > config["max_runtime"]:
            print(f"\nStopping run because the optimization time {rm.get_stage_runtime('optimize')} has exceeded the maximum time of {config['max_runtime']}")
            break
        
    # Finish the WandB run.
    if do_wandb:
        run.finish()


if __name__ == "__main__":
    with cProfile.Profile() as pr:
        main()
    os.makedirs("logs", exist_ok=True)
    pr.dump_stats("logs/profile.prof")
    print("Profile stats dumped to profile.prof")

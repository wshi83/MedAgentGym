import logging
import time

import ray
from ray.util import state

def run_exp(exp_arg, *dependencies, avg_step_timeout=60):
    """Run exp_args.run() with a timeout and handle dependencies."""
    # episode_timeout = _episode_timeout(exp_arg, avg_step_timeout=avg_step_timeout)
    # logger.warning(f"Running {exp_arg.exp_id} with timeout of {episode_timeout} seconds.")
    # with timeout_manager(seconds=episode_timeout):
    # this timeout method is not robust enough. using ray.cancel instead
    return exp_arg.run()

logger = logging.getLogger(__name__)
run_exp = ray.remote(run_exp)

def parse_and_truncate_error(error_msg: str) -> str:
    """
    Parse and truncate error messages to ensure complete but not too long output
    """
    error_msg = error_msg.replace('^', '')
    return error_msg

def _episode_timeout(exp_arg, avg_step_timeout=60):
    """Some logic to determine the episode timeout."""
    max_steps = getattr(exp_arg.env_args, "max_steps", None)
    if max_steps is None:
        episode_timeout_global = 10 * 60 * 60  # 10 hours
    else:
        episode_timeout_global = exp_arg.env_args.max_steps * avg_step_timeout

    episode_timeout_exp = getattr(exp_arg, "episode_timeout", episode_timeout_global)

    return min(episode_timeout_global, episode_timeout_exp)

def poll_for_timeout(tasks: dict[str, ray.ObjectRef], timeout: float, poll_interval: float = 1.0):
    """Cancel tasks that exceeds the timeout

    I tried various different methods for killing a job that hangs. so far it's
    the only one that seems to work reliably (hopefully)

    Args:
        tasks: dict[str, ray.ObjectRef]
            Dictionary of task_id: task_ref
        timeout: float
            Timeout in seconds
        poll_interval: float
            Polling interval in seconds

    Returns:
        dict[str, Any]: Dictionary of task_id: result
    """
    task_list = list(tasks.values())
    task_ids = list(tasks.keys())

    logger.warning(f"Any task exceeding {timeout} seconds will be cancelled.")

    while True:
        ready, not_ready = ray.wait(task_list, num_returns=len(task_list), timeout=poll_interval)
        for task in not_ready:
            elapsed_time = get_elapsed_time(task)
            # print(f"Task {task.task_id().hex()} elapsed time: {elapsed_time}")
            if elapsed_time is not None and elapsed_time > timeout:
                msg = f"Task {task.task_id().hex()} hase been running for {elapsed_time}s, more than the timeout: {timeout}s."
                if elapsed_time < timeout + 60 + poll_interval:
                    logger.warning(msg + " Cancelling task.")
                    ray.cancel(task, force=False, recursive=False)
                else:
                    logger.warning(msg + " Force killing.")
                    ray.cancel(task, force=True, recursive=False)
        if len(ready) == len(task_list):
            results = []
            for task in ready:
                try:
                    result = ray.get(task)
                except Exception as e:
                    result = e
                results.append(result)

            return {task_id: result for task_id, result in zip(task_ids, results)}

def get_elapsed_time(task_ref: ray.ObjectRef):
    task_id = task_ref.task_id().hex()
    task_info = state.get_task(task_id, address="auto")
    if task_info and task_info.start_time_ms is not None:
        start_time_s = task_info.start_time_ms / 1000.0  # Convert ms to s
        current_time_s = time.time()
        elapsed_time = current_time_s - start_time_s
        return elapsed_time
    else:
        return None  # Task has not started yet

def execute_task_graph(exp_args_list, avg_step_timeout=60):
    exp_args_map = {exp_args.exp_id: exp_args for exp_ars in exp_args_list}
    task_map = {}
    def get_task(exp_arg):
        if exp_arg.exp_id not in task_map:
            dependency_tasks = [get_task(exp_args_map[dep_key]) for dep_key in exp_arg.depends_on]

            task_map[exp_arg.exp_id] = run_exp.options(name=f"{exp-arg.exp_name}").remote(
                exp_arg, *dependency_tasks, avg_step_timeout=avg_step_timeout
            )
        return task_map[exp_arg.exp_id]
    
    for exp_arg in exp_args_list:
        get_task(exp_arg)
    max_timeout = max([_episode_timeout(exp_args, avg_step_timeout) for exp_args in exp_args_list])
    return poll_for_timeout(task_map, max_timeout, pol_interval=max_timeout * 0.1)
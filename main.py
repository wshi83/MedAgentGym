import argparse
import os
import logging
from pathlib import Path
import sys
import time
import toml
from ehr_gym.env.base import EHREnv
from ehr_gym.agent.base import EHRAgent
from ehr_gym.utils.general import load_config, save_conversation_history
import ray

logging.basicConfig(
    level=logging.INFO, format="%(name)s : %(levelname)-8s : %(message)s"
)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run EHR-Gym Experiments')
    parser.add_argument('--config_path', type=str)

    parser.add_argument('--task', type=str)
    parser.add_argument('--credentials_path', type=str)
    parser.add_argument('--work_dir', type=str)
    parser.add_argument('--result_dir_tag', type=str)
    parser.add_argument("--start_idx", type=int)
    parser.add_argument("--end_idx", type=int)
    parser.add_argument("--num_steps", type=int)
    parser.add_argument("--async_run", action="store_true", help="Run experiments asynchronously")
    parser.add_argument("--n_jobs", type=int, help="Number of parallel jobs")
    parser.add_argument("--parallel_backend", type=str, help="Parallel backend to use")
    parser.add_argument("--mode", type=str, default="test", help="train/test")
    return parser.parse_args()

def convert_config_to_args(config, args):
    if not args.task:
        args.task = config['task']
    if not args.credentials_path:
        args.credentials_path = config['credentials_path']
    if not args.work_dir:
        args.work_dir = config['work_dir']
    if not args.result_dir_tag:
        args.result_dir_tag = config['result_dir_tag']
    if not args.start_idx:
        args.start_idx = config['start_idx']
    if not args.end_idx:
        args.end_idx = config['end_idx']
    if not args.num_steps:
        args.num_steps = config['num_steps']
    return args

def load_credentials(credentials_path):
    return toml.load(credentials_path)

def set_environment_variables(credentials):
    for key, value in credentials.items():
        os.environ[key] = value

def create_env_config_dir(work_dir, task, result_dir_tag):
    env_config_tmp_dir = os.path.join(work_dir, task, result_dir_tag)
    os.makedirs(env_config_tmp_dir, exist_ok=True)
    return env_config_tmp_dir

def get_task_class(task):
    if task == 'mimic_iii':
        from ehr_gym.env.task.mimic_iii import MimiciiiEHRTask
        return MimiciiiEHRTask
    elif task == 'biocoder':
        from ehr_gym.env.task.biocoder import BiocoderTask
        return BiocoderTask
    elif task == 'eicu':
        from ehr_gym.env.task.eicu import EicuEHRTask
        return EicuEHRTask
    elif task == 'treqs':
        from ehr_gym.env.task.treqs import TreqsEHRTask
        return TreqsEHRTask
    elif task == 'medcalcbench':
        from ehr_gym.env.task.medcalcbench import MedCalBenchTask
        return MedCalBenchTask
    elif task == 'medagentbench':
        from ehr_gym.env.task.medagentbench import MedAgentBenchTask
        return MedAgentBenchTask
    elif task == 'ehrshot':
        from ehr_gym.env.task.ehrshot import EHRShotTask
        return EHRShotTask
    elif task == 'ehr_seqsql':
        from ehr_gym.env.task.ehr_seqsql import EHRSeqSQLEHRTask
        return EHRSeqSQLEHRTask
    elif task == 'ehrcon':
        from ehr_gym.env.task.ehrcon import EHRCONEHRTask
        return EHRCONEHRTask
    elif task == "biodsbench":
        from ehr_gym.env.task.biodsbench import BioDSBenchTask
        return BioDSBenchTask
    elif task == "npowerai":
        from ehr_gym.env.task.npowerai import NPowerAITask
        return NPowerAITask
    elif task == "mimic_extract":
        from ehr_gym.env.task.mimic_extract import MIMICEXTRACTEHRTask
        return MIMICEXTRACTEHRTask
    else:
        raise ValueError(f'Invalid task: {task}')

def sequential_run_experiments(args, config):
    success_rate = 0
    for idx in range(args.start_idx, args.end_idx):
        success = run_single_experiment(args, config, idx)
        success_rate += success
    success_rate /= (args.end_idx - args.start_idx)
    print('-'*50)
    print(f'Success Rate: {success_rate}')

def run_single_experiment(args, config, idx):
    agent_config = config['Agent']
    data_config = config['Data']
    debugger_config = config['Debugger']
    save_dir = os.path.join(args.work_dir, args.task, args.result_dir_tag, args.mode)
    output_path = os.path.join(save_dir, f'history_{idx}.json')
    if os.path.exists(output_path):
        logger.info(f"Experiment {idx} already exists. Skipping...")
        return 0
    print(f"Running experiment for index {idx}...")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    task_cls = get_task_class(args.task)
    task_kwargs = {
        'data_path': data_config['data_path'],
        'debugger_config': debugger_config,
        'mode': args.mode,
    }
    env = EHREnv(task_entrypoint=task_cls, task_kwargs=task_kwargs)
    agent = EHRAgent(agent_config, permitted_actions=task_cls.permitted_actions)
    obs, info = env.reset(idx)
    attempts = 0
    for step_idx in range(args.num_steps):
        action, params = agent.act(obs)
        while action == 'error':
            n_retry = config['Agent']['n_retry']
            logger.error(f"Task {args.task}-{idx} Failure: Agent action failed for {n_retry} times.")
            attempts += 1
            if attempts >= config['Env']['n_retry']:
                agent.conversation_history.append({'result': 'failure'})
                success = 0
                output_path = os.path.join(save_dir, f'history_{idx}.json')
                save_conversation_history(agent.conversation_history, output_path)
                return success
            time.sleep(1)
            action, params = agent.act(obs)
        obs, reward, done, truncated, info = env.step(action, **params)
        if done:
            break

    if done:
        agent.conversation_history.append({'result': 'success', 'score': reward})
        if args.task == 'ehrshot':
            success = reward
        else:
            success = 1
    else:
        agent.conversation_history.append({'result': 'failure'})
        success = 0
    output_path = os.path.join(save_dir, f'history_{idx}.json')
    save_conversation_history(agent.conversation_history, output_path)
    return success

run_single_experiment_ray = ray.remote(run_single_experiment)

def async_run_experiments(args, config, n_jobs, parallel_backend="ray"):
    if args.start_idx > args.end_idx:
        logging.warning("No experiments to run")
        return
    success_rate = 0
    try:
        if parallel_backend == 'joblib':
            from joblib import Parallel, delayed
            indices = range(args.start_idx, args.end_idx)
            # split sequential (should be no longer needed with dependencies)
            results = Parallel(n_jobs=n_jobs, prefer="processes")(
                delayed(run_single_experiment)(args, config, idx)
                for idx in indices
            )
            success_rate = sum(results) / len(results)
            print(f'Success Rate: {success_rate}')
        elif parallel_backend == "ray":
            ray.init(num_cpus=n_jobs)
            indices = range(args.start_idx, args.end_idx)
            futures = [
                run_single_experiment_ray.remote(args, config, idx)
                for idx in indices
            ]
            results = ray.get(futures)
            success_rate = sum(results) / len(results)
            print(f'Success Rate: {success_rate}')
            ray.shutdown()
        else:
            raise ValueError(f"Unsupported parallel backend: {parallel_backend}")
    finally:
        logging.info("All jobs are finished. Calling agent_args.close() on all agents...")
        logger.info('Experiment finished.')
        log_file = os.path.join(args.work_dir, "running_records.jsonl")
        with open(log_file, "a+") as f:
            f.write(f"Experiment {args.task}: {success_rate}\n")

def main():
    # initialization
    args = parse_arguments()
    if args.config_path:
        config = load_config(args.config_path)
        args = convert_config_to_args(config, args)
    if args.end_idx == -1:
        import json
        metadata_file = config['Data']['metadata_path']
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        args.end_idx = metadata[args.task][args.mode]
    credentials = load_credentials(args.credentials_path)
    set_environment_variables(credentials)
    env_config_tmp_dir = create_env_config_dir(args.work_dir, args.task, args.result_dir_tag)

    # run experiments
    if not args.async_run:
        sequential_run_experiments(args, config)
    else:
        async_run_experiments(args, config, args.n_jobs, args.parallel_backend)

if __name__ == '__main__':
    main()
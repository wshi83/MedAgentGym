## EHR-GYM
This is the official repository for the paper: "Incentivizing Coding Capability in LLM Agents for Medical Reasoning with MedAgentGYM". 

### Dataset Access
We provide the basic data of `train_tasks.jsonl` and `test_tasks.jsonl` in this repository. The ful dataset can be downloaded via the scripts in `download_data.py`. This will automatically download the full datasets we have prepared and uploaded in a private repository of an anonymous HuggingFace Account. Please download the data into the directory `./data/`. The downloaded dataset should be like `./data/biocoder/*`.

### Build Docker Container
As our dataset is based on the docker environment for isolated coding and execution. Thus, you need to build the docker container first. Please run the following command:
```bash
docker buildx build -t ehr_gym:latest .
```
or directly run the command we have prepared:
```bash
bash build_docker.sh
```

### Run Experiment
Please prepare the experiment scripts in the `entrypoint.sh` file. For example if we wnat to run the experiments on biocoder task and test the performance of gpt-4.1-mini. We can run the following command for 5-thread parallel running:
```bash
python3 /home/main.py --config /home/configs/gpt_4_1_mini/exp-gpt_4_1_mini-biocoder.yaml --async_run --parallel_backend joblib --n_jobs 5
```
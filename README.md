<p align="center">
  <img src="./assets/figure2.png" width="100%" alt="teaser">
</p>

----
<p align="center">
  <a href="https://www.arxiv.org/abs/2506.04405" target="_blank"><img src="https://img.shields.io/badge/arXiv-2506.02911-FF6B6B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://wshi83.github.io/MedAgentGym-Page"><img src="https://img.shields.io/badge/Doc-Documentation-4285F4?style=for-the-badge&logo=googledocs&logoColor=white" alt="Documentation"></a>
  <a href="https://huggingface.co/MedAgentGym"><img src="https://img.shields.io/badge/HuggingFace-Model-FFBF00?style=for-the-badge&logo=huggingface&logoColor=white" alt="HuggingFace Model"></a>
  <a href="mailto:medagentgym@gmail.com"><img src="https://img.shields.io/badge/Email-Dataset-30B980?style=for-the-badge&logo=minutemailer&logoColor=white" alt="Email Dataset"></a>
</p>


## MedAgentGYM
This is the official repository for the paper: "MedAgentGym: Training LLM Agents for Code-Based Medical Reasoning at Scale". In the paper, we introduce MedAgentGYM, the first publicly available training environment designed to enhance coding-based medical reasoning capabilities in large language model (LLM) agents. 

<p align="center">
  <img src="./assets/figure1.png" width="100%" alt="teaser">
</p>

### Dataset Access

#### EHR Data Access (<font color=#FF000>Important!!!</font>)
MedAgentGym has been carefully curated with strict adherence to ethical standards, leveraging datasets that are publicly available or that incorporate rigorous privacy protection and anonymization measures. Table 7 in the Appendix provides detailed access requirements for each of the 12 datasets included in MedAgentGym. Researchers seeking access to preprocessed task and data files should first obtain and attach all required data usage agreements and submit a formal request via email to `medagentgym@gmail.com`, using the subject line “MedAgentGym Preprocessed Data Access".

#### Tasks Definition and Access
This repository contains basic task files `train_tasks.jsonl` and `test_tasks.jsonl`, each including the task ID, task description, question, and corresponding ground truth answer.
After completing the previous step and obtaining approval for access, applicants will receive a script (`download_data.py`) to download the entire preprocessed dataset from a private repository. This script will automatically download all datasets into the `./data/` directory. The downloaded datasets should be structured as `./data/biocoder/*`. Detailed descriptions of the datasets utilized in this paper are provided below:

<p align="center">
  <img src="./assets/figure3.png" width="100%" alt="teaser">
</p>


### Build Docker Container
Since our dataset relies on a Docker environment for isolated coding and execution, you may first build the Docker container. Please execute the following command:
```bash
docker buildx build -t ehr_gym:latest .
```
Alternatively, you can run the prepared script directly:
```bash
bash build_docker.sh
```

### Run Experiment
Prepare your experiment commands in the `entrypoint.sh` file. For instance, to run experiments on the Biocoder task using the GPT-4.1-mini model, execute the following command for parallel execution with 5 threads:
```bash
python3 /home/main.py --config /home/configs/gpt_4_1_mini/exp-gpt_4_1_mini-biocoder.yaml --async_run --parallel_backend joblib --n_jobs 5
```

## Results

### Sampled Data Helps Agent Training

Figure below highlights substantial performance gains from SFT across four OSS backbone LLMs of varying sizes.
<p align="center">
  <img src="./assets/figure4.png" width="100%" alt="teaser">
</p>

### Warmed-up DPO Works Best for Coding Agent Training
The table below compares several post-training methods, revealing that simple SFT over successful trajectories significantly boosts performance on structured coding tasks, demonstrating its effectiveness in capturing structured coding patterns. Besides, DPO is particularly beneficial for optimizing open-ended task performance. Although DPO alone slightly underperforms compared to SFT, combining an initial SFT warm-up with subsequent DPO further improves overall results by leveraging their complementary strengths.

<p align="center">
  <img src="./assets/figure5.png" width="100%" alt="teaser">
</p>

### MedAgentGym Enables Both Inference- and Training-Time Scaling

<p align="center">
  <img src="./assets/figure6.png" width="100%" alt="teaser">
</p>


**Inference-Time Scaling:** The left figure illustrates performance scaling with increased trajectory sampling. Pass@K significantly improves from 17.0% at K = 1 to 45.0% at 16, while Best@K shows steady advancement from 17.0% to 41.7%. The relatively small gap between metrics indicates that our trained verifier effectively identifies successful trajectories, unleashing its potential as a reward model for integration into advanced online RL frameworks such as Proximal Policy Optimization (PPO) and Group Relative Policy Optimization (GRPO).

**Training-Time Scaling:** The right figure examines agent performance as a function of increased training data volumes (25%, 50%, 75%, and 100%) in SFT. We observe consistent performance improvements with greater training data availability, suggesting additional computational resources dedicated to sampling further trajectories are likely to yield continued performance gains.

## 📚 Citation

```bibtex
@article{xu2025medagentgym,
  title={MedAgentGym: Training LLM Agents for Code-Based Medical Reasoning at Scale},
  author={Xu, Ran and Zhuang, Yuchen and Zhong, Yishan and Yu, Yue and Tang, Xiangru and Wu, Hang and Wang, May D and Ruan, Peifeng and Yang, Donghan and Wang, Tao and Xiao, Guanghua and Yang, Carl and Xie, Yang and Shi, Wenqi},
  journal={arXiv preprint arXiv:2506.04405},
  year={2025}
}
```

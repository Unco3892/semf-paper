# SEMF: Supervised Expectation-Maximization Framework for Predicting Intervals

The official implementation of [SEMF: Supervised Expectation-Maximization Framework for Predicting Intervals](https://arxiv.org/abs/2405.18176).

<!-- ![](paper/assets/semf-tikz-plot.png) -->

<p align="center">
  <img src="paper/assets/semf-tikz-plot.png" width="500" />
</p>

## Requirements

1. Ensure Python 3.9.0 is installed on your system.

2. Create and activate a virtual environment using conda or venv:

conda:
```bash
conda create --name semf_env python=3.9.0
conda activate semf_env
```
venv:
```bash
python -m venv semf_env
source semf_env/bin/activate  # On Windows, use `venv\Scripts\activate`
```
then install dependencies:
```bash
pip install -r requirements.txt
```
Note that by default, `requirements.txt` installs the GPU version of `tensorflow` by default. If you do not have a GPU, change (comment+uncomment) the relevant lines in `requirements.txt` file. For more information, see https://www.tensorflow.org/install/pip . Additionally, since we used Windows 11 for experiments (and not WSL), the last supported version was tensorflow 2.10.0 .**We recommend using conda for installing cuda and cudnn for GPU support if you do not already have the cuda toolkit using the following *(run before installation the requirements)***
```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

Finally, to run the tests and make sure everything is working, execute the following command from the root directory of the repository:
```bash
pytest
```

## Training and Evaluation

Choose between `src/experiments/local/run_experiments_local.py` for local execution (option 1) or `src/experiments/wandb/run_experiments_wandb.py` for execution with wandb integration (option 2). These two scripts act as `train.py` and `evaluate.py` scripts found in most ML papers. The environment has been tested on both Windows (11) and Linux (Ubuntu 22.04.4 LTS), with a note that results may vary between different OS. If wandb is chosen, view results directly on the wandb platform, including benchmark comparisons. Follow the specific instructions in `src/experiments/local/evaluate_results_local.py` or `src/experiments/wandb/evaluate_results_wandb.py` to evaluate the results.

### Local
Note that `run_experiments_local.py` also support [argparser](https://docs.python.org/3/library/argparse.html) for customizable experiment runs:

Unix (Linux, MacOS):
```bash
python run_experiments_local.py --dataset="space_ga" --nn_config '{"nn_batch_size":32,"nn_epochs":100}' --simulator_architecture '[{"units":50,"activation":"relu"}]' --tree_config '{"tree_n_estimators":100}' --force_n_jobs --no-save_models --verbose --test_with_wide_intervals --no-return_interval_benchmark --no-use_constant_weights
```
Windows:
```bash
python run_experiments_local.py --dataset="space_ga" --nn_config="{""nn_batch_size"":32,""nn_epochs"":100}" --simulator_architecture="[{""units"":50,""activation"":""relu""}]" --tree_config="{""tree_n_estimators"":100}" --force_n_jobs --no-save_models --verbose --test_with_wide_intervals --no-return_interval_benchmark --no-use_constant_weights
```
Or if it does not work, you can use the following command:
```bash
python run_experiments_local.py --dataset="space_ga" --nn_config "{\\"nn_batch_size\\":32,\\"nn_epochs\\":100}" --simulator_architecture "[{\\"units\\":50,\\"activation\\":\\"relu\\"}]" --tree_config "{\\"tree_n_estimators\\":100}" --force_n_jobs --no-save_models --verbose --test_with_wide_intervals --no-return_interval_benchmark --no-use_constant_weights
```

See `src/experiments/shared/cmd_configs.py` for more details on the arguments.

### WANDB
To run experiments and evaluate results on wandb, navigate to `src/experiments/wandb` and execute the scripts as per your experimental setup and explained in the project structure below. You can re-run the training and the sweeps using wandb:

1. Copy / clone this repo on the different machines / clusters you want to use.
2. Login to WandB and add your wandb username and project id to `src/experiments/wandb/.env` 
3. Move into `src/experiments/wandb` 
4. Run `python run_experiments_wandb.py`.
5. You can run each sweep by running `wandb agent <USERNAME/PROJECTNAME/SWEEPID>` in `src/experiments/wandb`. More infos [in the wandb doc](https://docs.wandb.ai/guides/sweeps) . Specify the count as well (in our case, 500).

### Important Notes for Replication
The experiments were conducted on a machine with the following specifications:

- Operating System: Microsoft Windows 11 Home, Version 10.0.22631
- Processor: 13th Gen Intel(R) Core(TM) i9-13900KF
- RAM: 32 GB
- GPU: NVIDIA GeForce RTX 4090
- Python Version: 3.9.19
- CUDA Version: 12.1 worked system-wide, and 11.2 worked with conda
- Dependencies: As listed in requirements.txt

It is not required to have GPU, nor use multiple cores as in our case, however, please note that the performance can be slower. Additionally, with the random seed and the use of parallelization with `joblib`, the performance may vary slightly on other operating systems due to differences in computational handling or specific library implementations.

We would also like to point out that for `MultiET`'s model, the SEMF results may vary but should fall within the range provided in the paper when choosing `parallel_type="semf_joblib"` and `n_jobs>1`, even across different runs for the same seed and on the same machine. The only way to get the same results is to not parallelize and run the program on a single core, which can significantly slow down the training process. See this issue for this topic: https://github.com/scikit-learn/scikit-learn/issues/22303 . For `MultiXGBs` and `MultiMLPs` models,  under the same setting, the results will be the same every time, if not very similar.

## Directory Structure 

```md
semf_unzipped/
├── pytest.ini
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── tabular_benchmark/
│       ├── .
├── paper/
│   ├── .
├── results/
│   ├── sweep_hyperparams_final.csv
│   ├── sweep_results_conformalized.csv
│   └── sweep_results.csv
└── src/
    ├── example.py
    ├── tests/
    │   ├── test_visualize.py
    │   ├── test_semf.py
    │   ├── test_neural_simulator.py
    │   ├── test_utils.py
    │   ├── test_preprocessing.py
    │   └── test_models.py
    ├── semf/
    │   ├── semf.py
    │   ├── visualize.py
    │   ├── neural_simulator.py
    │   ├── utils.py
    │   ├── preprocessing.py
    │   └── models.py
    └── experiments/
        ├── wandb/
        │   ├── process_sweep_wandb.py
        │   ├── run_experiments_wandb.py
        │   ├── config_sweeps_multimlps.yml
        │   ├── config_sweeps_multixgbs_multiets.yml
        │   ├── evaluate_results_wandb.py
        │   ├── fetch_sweep_wandb.py
        │   ├── .env
        ├── shared/
        │   ├── benchmark.py
        │   ├── cmd_configs.py
        │   ├── generate_data.py
        └── local/
            ├── run_experiments_local.py
            └── evaluate_results_local.py
```

- `data/`: Contains the datasets used in our experiments. These are automatically generated once you run the experiment scripts within `src/experiments/`. 
- `paper/`: Contains scripts for generating LaTeX tables fromthe results and the final hyperparameters. 
- `results/`: Stores the results of all parameter sweeps and hyperparameter configurations.
- `sweep_hyperparams_final.csv` : Contains the hyperparameters used in the final model (or complete which is used also for the missing in `src.experiments.local.evaluate_results_local.py` or `src.experiments.wandb.evaluate_results_wandb.py`).
- `sweep_results_complete.csv` : Contains the raw results of the SEMF model.
- `sweep_results_conformalized.csv` : Contains the results of the final model with the conformalized intervals.
- `sweep_results.csv` : Contains the results of the final model with the conformalized intervals.
- `src/`: The source code required to replicate our experiments, including model training and evaluation scripts. 
- `example.py`: An example script to demonstrate the usage of the SEMF model.
- `tests/` : Contains tests for the main SEMF algorithm.
- `preprocessing.py` : Contains the preprocessing functions for the data used in the experiments.
- `semf/` : Contains the main semf modules.
- `models.py` : Contains the models used within SEMF ("MultiXGBs", "MultiMLPs", "MultiETS") as well as "QNN" for benchmarking.
- `semf.py` : Contains the SEMF class that does the trianing and inference for both points and intervals.
- `neural_simulator.py` : Neural network missing data simulator.
- `visualize.py` : Visualization functions.
- `utils.py` : Utility functions.
- `experiments/` : Contains the scripts for running the experiments.
- `local/` : Contains the scripts for running the experiments locally.
- `run_experiments_local.py` : Main for running the experiments locally that is used for argparser (corresponds to `train.py` and `evaluate.py` found in most ML papers).
- `evaluate_results_local.py`: Evaluates the results of the best hyperparameters from `sweep_hyperparams_final.csv` for both complete and missing data.
- `wandb/` : Contains the scripts for running the experiments on WANDB.
- `.env` : Contains the WANDB entity and project name that is used by all `wandb` scripts and sweeps.
- `process_sweep_wandb.py` : Processes the results of the experiments and saves them to the `./results` folder according to both the criteria in the paper and in the description in the script. Ensure that you specify your WANDB entity and project name here.
- `run_experiments_wandb.py` : Equivalent to `run_experiments_local.py` but for WANDB.
- `evaluate_results_wandb.py` : Equivalent to `evaluate_results_local.py` but for WANDB.
- `fetch_sweep_wandb.py` : Fetch all the runs for the sweep or final results save them to the `./results` folder.
- `config_sweeps_multimlps.yml` : Contains the hyperparameters for the WANDB sweep for the MultiMLPs model.
- `config_sweeps_multixgbs_multiets.yml`: Contains the hyperparameters for the WANDB sweep for the MultiXGBs and MultiETS models.
- `shared/` : Contains the shared scripts for both WANDB and local experiments.
- `benchmark.py` : Benchmarking class for SEMF.
- `cmd_configs.py` : Argument parsing for customizable experiment runs.
- `generate_data.py` : Generates simulated data that is used both by the `example.py` and `run_experiments_local.py`.


## Contributing
This repository is released under the MIT License. If you want to contribute to this project, please follow these steps:

1. Fork the repository. 
2. Create a new branch (`git checkout -b feature-foo`). 
3. Commit your changes (`git commit -am 'Add some foo'`). 
4. Push to the branch (`git push origin feature-foo`).
5. Create a new Pull Request.

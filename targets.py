from_email = 'kowalski.jan.development@gmail.com'
to_email = 'rumcajsgajos@gmail.com'

minimal_datasets = ['freesolv', 'caco2', 'halflife', 'cyp3a4', 'herg']
best_lr = {
    'mat': [1.00E-06, 1.00E-05, 1.00E-06, 0.0001, 1.00E-06],
    'chemprop': [5.00E-05, 0.0001, 1.00E-05, 1.00E-05, 5.00E-06],
    'grover': [5.00E-05, 5.00E-05, 1.00E-05, 1.00E-06, 0.001],
    'grover_base': [5e-06, 5e-05, 1e-05, 5e-06, 5e-05],
    'chemberta': [5.00E-06, 5.00E-05, 1.00E-05, 1.00E-06, 1.00E-06]
}

best_checkpoints = {
    'freesolv': ['mat_masking_20M_tox21-aroma', 'mat_masking_20M_biov', 'mat_masking_20M_vdss'],
    'caco2': ['mat_masking_20M_freesolv', 'mat_masking_20M_biov', 'mat_masking_20M_hia'],
    'halflife': ['mat_masking_20M_tox21-p53', 'mat_masking_20M_pgp', 'mat_masking_20M_tox21-aroma'],
    'cyp3a4': ['mat_masking_20M_lipo', 'mat_masking_20M_pgp', 'mat_masking_20M_ppbr'],
    'herg': ['mat_masking_20M_bbb', 'mat_masking_20M_hia', 'mat_masking_20M_qm7-atom']
}
exhaustive_datasets = ['lipo', 'ppbr', 'pgp', 'bbb', 'covid']
models = ['mat', 'chemprop', 'chemberta', 'grover']
all_datasets = minimal_datasets + exhaustive_datasets


# extended_dir = 'intermediate/extended3'
# intermediate_extended_3 = [f'{extended_dir}/{p.split(".")[0]}' for p in
#                            os.listdir(f'experiments/configs/datasets/{extended_dir}')]


def powerset(iterable):
    from itertools import chain, combinations
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


space = ' '


def extract_params(run):
    params = {k: v for k, v in run.items() if '.' in k}
    return '#'.join([f'{k}={v}' for k, v in params.items()]) + '#'


def extract_bindings(run):
    return run["bindings"].replace("'", "\\'")


def targets_dict(gpus):
    return {
        'bi_stilt': [
            "python -m experiments.scripts.benchmark --results_only "
            f"--name.prefix Bi_STILT_{pretrained} "
            f"-m mat "
            f"-d {dataset} "
            f"-a moco_bi_tuning "
            f"--model.pretrained_name {pretrained} "
            f"--train.gpus {gpus} "
            f"-b model.excluded=[\\'generator\\'] "
            f"; rm -rf experiments_results/Bi_STILT_{pretrained}* "
            for dataset in minimal_datasets
            for pretrained in best_checkpoints[dataset][:1]
        ],
        'grover': [
            "python -m experiments.scripts.benchmark --results_only "
            f"-m grover "
            f"-d {dataset} "
            f"--model.pretrained_name grover_base "
            f"--train.gpus {gpus} "
            f"; rm -rf experiments_results/Grover* "
            for dataset in
            ['grover/qm7', 'grover/freesolv', 'grover/bace', 'grover/bbbp', 'grover/esol', 'grover/lipo']
        ],
        'mat1': [
            "python -m experiments.scripts.benchmark --results_only "
            f"-m mat "
            f"-d {dataset} "
            f"--name.prefix 2M "
            f"--model.pretrained_name mat_masking_2M "
            f"--train.gpus {gpus} "
            f"; rm -rf experiments_results/2M* "
            for dataset in
            ['grover/qm7', 'grover/freesolv', 'grover/bace', 'grover/bbbp', 'grover/esol', 'grover/lipo'][:2]
        ],
        'mat2': [
            "python -m experiments.scripts.benchmark --results_only "
            f"-m mat "
            f"-d {dataset} "
            f"--name.prefix 2_2M "
            f"--model.pretrained_name mat_masking_2M "
            f"--train.gpus {gpus} "
            f"; rm -rf experiments_results/2_2M* "
            for dataset in
            ['grover/qm7', 'grover/freesolv', 'grover/bace', 'grover/bbbp', 'grover/esol', 'grover/lipo'][2:]
        ],
        'stilt2': [
            "python -m experiments.scripts.benchmark --results_only "
            f"--name.prefix Nohead_From_{pretrained} "
            f"-m mat "
            f"-d {dataset} "
            f"--model.pretrained_name {pretrained} "
            f"--train.gpus {gpus} "
            f"-b model.excluded=[\\'generator\\'] "
            f"; rm -rf experiments_results/Nohead_From_{pretrained}* "
            for dataset in minimal_datasets[1:]
            for pretrained in best_checkpoints[dataset][1:]
        ],
        'nvidia': [
                      "python -m experiments.scripts.benchmark --results_only "
                      f"-m {model} "
                      f"-d {dataset} "
                      f"--train.gpus {gpus} "
                      f"-b neptune.user_name=\\'majchrow\\'"
                      for dataset in exhaustive_datasets[0:0]
                      for model in models
                  ] + [
                      "python -m experiments.scripts.benchmark --results_only "
                      "--name.prefix Base "
                      f"-m grover "
                      f"--model.pretrained_name grover_base "
                      f"-d {dataset} "
                      f"--train.gpus {gpus} "
                      f"-b neptune.user_name=\\'majchrow\\'"
                      for dataset in exhaustive_datasets
                  ],
        'intermediate_minimal': [
            "python -m experiments.scripts.checkpoint "
            "--name.prefix Checkpoint3 "
            f"-m mat "
            f"-d {dataset} "
            f"--train.gpus {gpus} "
            f"; rm -rf experiments_results/Checkpoint3* "
            for dataset in
            minimal_datasets
        ],
        # 'intermediate_extended3': [
        #     "python -m experiments.scripts.checkpoint "
        #     f"-m mat "
        #     f"-d {dataset} "
        #     f"--train.gpus {gpus} "
        #     f"; rm -rf experiments_results/Checkpoint* "
        #     for dataset in intermediate_extended_3
        # ],
        'bi_tuning_128_reg_scaled': [
            "python -m experiments.scripts.benchmark --results_only "
            "--name.prefix Bi_128_Scaled "
            f"-m mat "
            f"-d {dataset} "
            f"-a bi_tuning_regression "
            f"--train.gpus {gpus} "
            f"-b loss_fn.name=\\'MSEScaledLoss\\'#"
            f"bi_tuning.contrastive_coeff=0.0005"
            f"; rm -rf experiments_results/Bi_128_Scaled* "
            for dataset in minimal_datasets[:3]
        ],
        'bi_tuning_128_clf': [
            "python -m experiments.scripts.benchmark --results_only "
            "--name.prefix Bi_128 "
            f"-m mat "
            f"-d {dataset} "
            f"-a bi_tuning_classification "
            f"--train.gpus {gpus} "
            f"; rm -rf experiments_results/Bi_128* "
            for dataset in minimal_datasets[3:]
        ],
        'moco3': [
            "python -m experiments.scripts.benchmark --results_only "
            "--name.prefix Moco2L "
            f"-m mat "
            f"-d {dataset} "
            f"-a moco_bi_tuning "
            f"--train.gpus {gpus} "
            f"; rm -rf experiments_results/Moco2L* "
            for dataset in minimal_datasets
        ],
        'moco_reg': [
            "python -m experiments.scripts.benchmark --results_only "
            "--name.prefix Moco_Big_2 "
            f"-m mat "
            f"-d {dataset} "
            f"-a moco_bi_tuning moco_bi_tuning_tune_regression "
            f"--train.gpus {gpus} "
            f"; rm -rf experiments_results/Moco_Big_2* "
            for dataset in minimal_datasets[:1]
        ],
        'bi_tuning_scaling': [
            "python -m experiments.scripts.benchmark --results_only "
            "--name.prefix Bi_Scaling "
            f"-m mat "
            f"-d {dataset} "
            f"-a bi_tuning_regression "
            f"--train.gpus {gpus} "
            f"; rm -rf experiments_results/Bi_Scaling_Final* "
            for dataset in minimal_datasets[:3]
        ],
        'grover_base_ensemble': [
            "python -m experiments.scripts.benchmark_ensemble --results_only "
            "--name.prefix Base_Ensemble "
            f"-m grover "
            "--model.pretrained_name grover_base "
            f"-d {dataset} "
            f"--train.gpus {gpus} "
            f"-b optimizer.lr={lr}"
            for dataset, lr in zip(minimal_datasets, best_lr["grover_base"])
        ],
        'matpp': [
            "python -m experiments.scripts.benchmark "
            f"-m matpp "
            f"-d {dataset} "
            f"--train.gpus {gpus} "
            f"-b train.batch_size=16"
            for dataset in minimal_datasets
        ],
        'ensembles_pairs': [
            "python -m experiments.scripts.benchmark_ensemble --results_only --empty_prefix "
            f"--names {' '.join(ensembles)} "
            f"-d {dataset} "
            for ensembles in [['Ensemble_MatModel', 'True_Ensemble_GroverModel', 'Ensemble_ChembertaModelWrapper']]
            for dataset in minimal_datasets
        ],
        'finalize': [
            f'python -m experiments.scripts.train '
            f'-m {run["model"]} '
            f'--model.pretrained_name {run["pretrained"]} '
            f'-d {run["dataset"]} ' +
            f'--name.prefix "{run["prefix"]}" ' +
            (f'-a {" ".join(run["additional"])} ' if run['additional'] is not None else '') +
            f'--train.gpus {gpus} '
            f'-b data.split_seed={run["seed"]}#' +
            extract_params(run) +
            "neptune.project_name=\\'Low-Data\\'#" +
            extract_bindings(run) +
            f' ; rm -rf experiments_results/{run["name"]} '
            for run in
            [{'model': 'grover', 'pretrained': 'grover_base', 'dataset': 'grover/bbbp', 'additional': None, 'seed': 2,
              'optimizer.lr': 0.001, 'prefix': '', 'name': 'GroverModel_ADME_BBBP-GROVER', 'bindings': ''},
             {'model': 'grover', 'pretrained': 'grover_base', 'dataset': 'grover/bbbp', 'additional': None, 'seed': 1,
              'optimizer.lr': 1e-05, 'prefix': '', 'name': 'GroverModel_ADME_BBBP-GROVER', 'bindings': ''},
             {'model': 'grover', 'pretrained': 'grover_base', 'dataset': 'grover/bbbp', 'additional': None, 'seed': 2,
              'optimizer.lr': 1e-05, 'prefix': '', 'name': 'GroverModel_ADME_BBBP-GROVER', 'bindings': ''}
             ]
        ],
    }

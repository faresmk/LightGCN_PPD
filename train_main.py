import time
from contextlib import contextmanager
from recbole.quick_start import run_recbole

@contextmanager
def timer(label: str):
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        print(f"{label} took {(end - start) / 60:.2f} mins")

def run_experiments(datasets, model, config_file):
    """
    Runs experiments for a given model and set of datasets, saving results to separate files.
    """
    output_file = f"results_{model.lower()}.txt"
    with open(output_file, 'w') as f:
        for dataset in datasets:
            print(f"Running experiment for {model} on {dataset}...")
            with timer(f"{dataset}-{model}"):
                result = run_recbole(model=model, dataset=dataset, config_file_list=[config_file])
                print(result)
                f.write(f"{'='*20} {dataset}-{model} {'='*20}\n")
                f.write(str(result) + '\n\n')
            print("-" * 50)

if __name__ == "__main__":
    datasets = ['ml-100k', 'lastfm', 'book-crossing']  # Add your datasets
    models_configs = {
        'LightGCN': 'lightgcn_config.yaml',
        'NGCF': 'ngcf_config.yaml',
        'MultiVAE': 'multivae_config.yaml'
    }

    for model, config_file in models_configs.items():
        run_experiments(datasets, model, config_file)
from BrainGraphStudio.utils import convert_numeric_values
from nni.experiment import Experiment
from os.path import join

def configure_nni(nni_args, experiment_path, python_path, brainGNN = False):
    search_space = nni_args["search_space"]
    search_space = convert_numeric_values(search_space)
    experiment = Experiment("local")

    experiment.config.trial_code_directory = join(__file__,"scripts")
    experiment.config.search_space = search_space
    experiment.config.tuner.name = nni_args["optimization_algorithm"]

    if nni_args["assessor_algorithm"] != "None":
        experiment.config.assessor.name = nni_args["assessor_algorithm"]
    
    if nni_args["n_trials"]:
        experiment.config.max_trial_number = nni_args["n_trials"]
    elif nni_args["max_time"]:
        experiment.config.max_experiment_duration = nni_args["max_time"]

    if brainGNN:
        command = f"{python_path} train_brain_gnn.py"
    else:
        command = f"{python_path} train_brain_gb.py"
    experiment.config.trial_command = command

    return experiment





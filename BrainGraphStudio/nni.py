from BrainGraphStudio.utils import convert_numeric_values, BGS_DIR
from nni.experiment import Experiment
from os.path import join
import json
import logging
logger = logging.getLogger(__name__)


def configure_nni(nni_args, experiment_path, python_path, brainGNN = False):
    logger.info("CONFIGURING NNI EXPERIMENT")
    search_space = nni_args["search_space"]
    search_space = convert_numeric_values(search_space)
    logger.info(f"NNI SEARCH SPACE: {json.dumps(search_space)}")

    experiment = Experiment("local")

    experiment.config.trial_code_directory = join(BGS_DIR, "train")
    logger.info(f"Trial code directory set to {experiment.config.trial_code_directory}")
    experiment.config.search_space = search_space
    experiment.config.tuner.name = nni_args["optimization_algorithm"]
    experiment.config.trial_concurrency = 2

    if nni_args["assessor_algorithm"] != "None":
        experiment.config.assessor.name = nni_args["assessor_algorithm"]
    
    if nni_args["n_trials"]:
        n_trials = nni_args["n_trials"]
        experiment.config.max_trial_number = nni_args["n_trials"]
        logger.info(f"Max Trials Specified (n_trials = {n_trials}). This Overrides Max Time")
    elif nni_args["max_time"]:
        max_time = nni_args["max_time"]
        experiment.config.max_experiment_duration = nni_args["max_time"]
        logger.info(f"Max_time = {max_time} hr specified")

    if brainGNN:
        command = f"{python_path} train_brain_gnn.py"
    else:
        command = f"{python_path} train_brain_gb.py"
    command += f" {experiment_path}"
    logger.info(f"NNI being run with trial command {command}")
    experiment.config.trial_command = command

    return experiment





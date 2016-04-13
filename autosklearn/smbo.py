import multiprocessing
import os
import traceback

import pynisher


# JTS TODO: notify aaron to clean up these nasty nested modules
from ConfigSpace.configuration_space import Configuration

from smac.smbo.smbo import SMBO
from smac.scenario.scenario import Scenario
from smac.tae.execute_ta_run import StatusType
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import RunHistory2EPM4Cost

from autosklearn.constants import *
from autosklearn.metalearning.mismbo import \
    calc_meta_features, calc_meta_features_encoded, \
    suggest_via_metalearning
from autosklearn.data.abstract_data_manager import AbstractDataManager
from autosklearn.data.competition_data_manager import CompetitionDataManager
from autosklearn.evaluation import eval_with_limits
from autosklearn.util import get_logger
from autosklearn.util import Backend

# dataset helpers
def load_data(dataset_info, outputdir, tmp_dir=None, max_mem=None):
    if tmp_dir is None:
        tmp_dir = outputdir
    backend = Backend(outputdir, tmp_dir)
    try:
        D = backend.load_datamanager()
    except IOError:
        D = None

    # Datamanager probably doesn't exist
    if D is None:
        if max_mem is None:
            D = CompetitionDataManager(dataset_info, encode_labels=True)
        else:
            D = CompetitionDataManager(dataset_info, encode_labels=True, max_memory_in_mb=max_mem)
    return D


def _load_config_list(task='classification'):
    """
    loads a list of dicts with task-focused configs
    """
    import cPickle as pkl
    if task == 'classification':
        task_file = 'config_class.pkl'
    elif task == 'multilabel':
        task_file = 'config_label.pkl'
    else:
        task_file = 'config_reg.pkl'

    with open(os.path.join('../autosklearn/util', task_file), 'rb') as fh:
        data = pkl.load(fh)

    return data


# metalearning helpers
def _calculate_metafeatures(data_feat_type, data_info_task, basename,
                            metalearning_cnt, x_train, y_train, watcher,
                            logger):
    # == Calculate metafeatures
    task_name = 'CalculateMetafeatures'
    watcher.start_task(task_name)
    categorical = [True if feat_type.lower() in ['categorical'] else False
                   for feat_type in data_feat_type]

    if metalearning_cnt <= 0:
        result = None
    elif data_info_task in [MULTICLASS_CLASSIFICATION, BINARY_CLASSIFICATION,
                            MULTILABEL_CLASSIFICATION, REGRESSION]:
        logger.info('Start calculating metafeatures for %s', basename)
        result = calc_meta_features(x_train, y_train, categorical=categorical,
                                    dataset_name=basename, task=data_info_task)
    else:
        result = None
        logger.info('Metafeatures not calculated')
    watcher.stop_task(task_name)
    logger.info(
        'Calculating Metafeatures (categorical attributes) took %5.2f',
        watcher.wall_elapsed(task_name))
    return result


def _calculate_metafeatures_encoded(basename, x_train, y_train, watcher,
                                    task, logger):
    task_name = 'CalculateMetafeaturesEncoded'
    watcher.start_task(task_name)
    result = calc_meta_features_encoded(X_train=x_train, Y_train=y_train,
                                        categorical=[False] * x_train.shape[1],
                                        dataset_name=basename, task=task)
    watcher.stop_task(task_name)
    logger.info(
        'Calculating Metafeatures (encoded attributes) took %5.2fsec',
        watcher.wall_elapsed(task_name))
    return result

def _get_metalearning_configurations(meta_features,
                                     meta_features_encoded, basename, metric,
                                     configuration_space,
                                     task, metadata_directory,
                                     initial_configurations_via_metalearning,
                                     is_sparse,
                                     watcher, logger):
    task_name = 'InitialConfigurations'
    watcher.start_task(task_name)
    try:
        metalearning_configurations = suggest_via_metalearning(
            meta_features,
            meta_features_encoded,
            configuration_space, basename, metric,
            task,
            is_sparse == 1,
            initial_configurations_via_metalearning,
            metadata_directory
        )
    except Exception as e:
        logger.error(str(e))
        logger.error(traceback.format_exc())
        metalearning_configurations = []
    watcher.stop_task(task_name)
    return metalearning_configurations

def _print_debug_info_of_init_configuration(initial_configurations, basename,
                                            time_for_task, logger, watcher):
    logger.debug('Initial Configurations: (%d)' % len(initial_configurations))
    for initial_configuration in initial_configurations:
        logger.debug(initial_configuration)
    logger.debug('Looking for initial configurations took %5.2fsec',
                 watcher.wall_elapsed('InitialConfigurations'))
    logger.info(
        'Time left for %s after finding initial configurations: %5.2fsec',
        basename, time_for_task - watcher.wall_elapsed(basename))


class AutoMLScenario(Scenario):
    """
    We specialize the smac3 scenario here as we would like
    to create it in code, without actually reading a smac scenario file
    """

    def __init__(self, config_space, config_file, limit, cutoff_time, memory_limit):
        self.logger = get_logger(self.__class__.__name__)

        # Give SMAC at least 5 seconds
        soft_limit = max(5, cutoff_time - 35)

        scenario_dict = {'cs': config_space,
                         'run_obj': 'quality',
                         'cutoff': soft_limit,
                         'algo_runs_timelimit': soft_limit,
                         'wallclock_limit': limit}

        super(AutoMLScenario, self).__init__(scenario_dict)

class AutoMLSMBO(multiprocessing.Process):

    def __init__(self, config_space, dataset_name,
                 output_dir, tmp_dir,
                 total_walltime_limit,
                 func_eval_time_limit,
                 memory_limit,
                 watcher, start_num_run=1,
                 data_memory_limit=None,
                 default_cfgs=None,
                 num_metalearning_cfgs=25,
                 config_file=None,
                 smac_iters=1000,
                 seed=1,
                 metadata_directory=None,
                 resampling_strategy='holdout',
                 resampling_strategy_args=None):
        super(AutoMLSMBO, self).__init__()
        # data related
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.tmp_dir = tmp_dir
        self.datamanager = None
        self.metric = None
        self.task = None

        # the configuration space
        self.config_space = config_space

        # Evaluation
        self.resampling_strategy = resampling_strategy
        if resampling_strategy_args is None:
            resampling_strategy_args = {}
        self.resampling_strategy_args = resampling_strategy_args

        # and a bunch of useful limits
        self.total_walltime_limit = int(total_walltime_limit)
        self.func_eval_time_limit = int(func_eval_time_limit)
        self.memory_limit = memory_limit
        self.data_memory_limit = data_memory_limit
        self.watcher = watcher
        self.default_cfgs = default_cfgs
        self.num_metalearning_cfgs = num_metalearning_cfgs
        self.config_file = config_file
        self.seed = seed
        self.metadata_directory = metadata_directory
        self.smac_iters = smac_iters
        self.start_num_run = start_num_run

        self.config_space.seed(self.seed)
        logger_name = self.__class__.__name__ + \
                      (":" + dataset_name if dataset_name is not None else "")
        self.logger = get_logger(logger_name)

    def reset_data_manager(self, max_mem=None):
        if max_mem is None:
            max_mem = self.data_memory_limit
        if self.datamanager is not None:
            del self.datamanager
        if isinstance(self.dataset_name, AbstractDataManager):
            self.datamanager = self.dataset_name
        else:
            self.datamanager = load_data(self.dataset_name,
                                         self.output_dir,
                                         self.tmp_dir,
                                         max_mem = max_mem)
        self.metric = self.datamanager.info['metric']
        self.task = self.datamanager.info['task']

    def collect_defaults(self):
        # TODO each pipeline should know about its preferred default
        # configurations!
        return []
        
    def collect_additional_subset_defaults(self):
        default_configs = []
        # == set default configurations
        # first enqueue the default configuration from our config space
        if self.datamanager.info["task"] in CLASSIFICATION_TASKS:
            config_list = _load_config_list(task='classification')
            try:
                config = [Configuration(self.config_space, i) for i in config_list]
                default_configs.extend(config)
            except ValueError as e:
                self.logger.warning("Configurations list for classification cannot"
                                    " be evaluated because of %s" % e)
        elif self.datamanager.info["task"] in REGRESSION_TASKS:
            config_list = _load_config_list(task='regression')
            try:
                config = [Configuration(self.config_space, i) for i in config_list]
                default_configs.extend(config)
            except ValueError as e:
                self.logger.warning("Configurations list for regression cannot"
                                    " be evaluated because of %s" % e)
        else:
            self.logger.info("Tasktype unknown: %s" %
                             TASK_TYPES_TO_STRING[self.datamanager.info[
                                 "task"]])

        return default_configs

    def collect_metalearning_suggestions(self):
        meta_features = _calculate_metafeatures(
            data_feat_type=self.datamanager.feat_type,
            data_info_task=self.datamanager.info['task'],
            x_train=self.datamanager.data['X_train'],
            y_train=self.datamanager.data['Y_train'],
            basename=self.dataset_name,
            watcher=self.watcher,
            metalearning_cnt=self.num_metalearning_cfgs,
            logger=self.logger)
        self.watcher.start_task('OneHot')
        self.datamanager.perform1HotEncoding()
        self.watcher.stop_task('OneHot')

        have_metafeatures = meta_features is not None
        known_task = self.datamanager.info['task'] in [MULTICLASS_CLASSIFICATION,
                                                       BINARY_CLASSIFICATION,
                                                       MULTILABEL_CLASSIFICATION,
                                                       REGRESSION]
        if have_metafeatures and known_task :
            meta_features_encoded = _calculate_metafeatures_encoded(
                self.dataset_name,
                self.datamanager.data['X_train'],
                self.datamanager.data['Y_train'],
                self.watcher,
                self.datamanager.info['task'],
                self.logger)

            metalearning_configurations = _get_metalearning_configurations(
                meta_features,
                meta_features_encoded,
                self.dataset_name,
                self.metric,
                self.config_space,
                self.task,
                self.metadata_directory,
                self.num_metalearning_cfgs,
                self.datamanager.info['is_sparse'],
                self.watcher,
                self.logger)
            _print_debug_info_of_init_configuration(
                metalearning_configurations,
                self.dataset_name,
                self.total_walltime_limit,
                self.logger,
                self.watcher)

        else:
            metalearning_configurations = []
        return metalearning_configurations
        
    def collect_metalearning_suggestions_with_limits(self):
        res = None
        try:
            safe_suggest = pynisher.enforce_limits(mem_in_mb=self.memory_limit,
                                            wall_time_in_s=int(self.scenario.wallclock_limit/4),
                                            grace_period_in_s=30)(
                self.collect_metalearning_suggestions)
            res = safe_suggest()
        except:
            pass
        if res is None:
            return []
        else:
            return res
    
    def run(self):
        cpname = multiprocessing.current_process().name
        self.logger.info("{0} is currently ... ". format(cpname))
        # we use pynisher here to enforce limits
        safe_smbo = pynisher.enforce_limits(mem_in_mb=self.memory_limit,
                                            wall_time_in_s=int(self.total_walltime_limit),
                                            grace_period_in_s=5)(self.run_smbo)
        safe_smbo(max_iters=self.smac_iters)
        
    def run_smbo(self, max_iters=1000):
        global evaluator

        # == first things first: load the datamanager
        self.reset_data_manager()
        
        # == Initialize SMBO stuff
        # first create a scenario
        seed = self.seed # TODO
        self.scenario = AutoMLScenario(self.config_space, self.config_file,
                                       self.total_walltime_limit, self.func_eval_time_limit,
                                       self.memory_limit)
        num_params = len(self.config_space.get_hyperparameters())
        # allocate a run history
        run_history = RunHistory()
        rh2EPM = RunHistory2EPM4Cost(num_params=num_params,
                                     scenario=self.scenario,
                                     success_states=None,
                                     impute_censored_data=False,
                                     impute_state=None)
        num_run = self.start_num_run
        smac = SMBO(self.scenario, seed)

        # Create array for default configurations!
        if self.default_cfgs is None:
            default_cfgs = []
        else:
            default_cfgs = self.default_cfgs
        default_cfgs.insert(0, self.config_space.get_default_configuration())
        # add the standard defaults we want to evaluate
        default_cfgs += self.collect_defaults()

        # == Train on subset
        #    before doing anything, let us run the default_cfgs
        #    on a subset of the available data to ensure that
        #    we at least have some models
        #    we will try three different ratios of decreasing magnitude
        #    in the hope that at least on the last one we will be able
        #    to get a model
        n_data = self.datamanager.data['X_train'].shape[0]
        subset_ratio = 10000. / n_data
        if subset_ratio >= 0.5:
            subset_ratio = 0.33
            subset_ratios = [subset_ratio, subset_ratio * 0.5,
                             subset_ratio * 0.10]
        else:
            subset_ratios = [subset_ratio, 3500. / n_data,
                             500. / n_data]
        self.logger.info("Training default configurations on a subset of "
                         "%d/%d data points." %
                         (int(n_data * subset_ratio), n_data))

        # the time limit for these function evaluations is rigorously
        # set to only 1/2 of a full function evaluation
        subset_time_limit = max(5, int(self.func_eval_time_limit / 2))
        # the configs we want to run on the data subset are:
        # 1) the default configs
        # 2) a set of configs we selected for training on a subset
        subset_configs = default_cfgs \
                         + self.collect_additional_subset_defaults()
        for next_config in subset_configs:
            for i, ratio in enumerate(subset_ratios):
                self.reset_data_manager()
                n_data_subsample = int(n_data * ratio)

                # run the config, but throw away the result afterwards
                # since this cfg was evaluated only on a subset
                # and we don't want  to confuse SMAC
                self.logger.info("Starting to evaluate %d on SUBSET "
                                 "with size %d and time limit %ds.",
                                 num_run, n_data_subsample,
                                 subset_time_limit)
                self.logger.info(next_config)
                _info = eval_with_limits(
                    self.datamanager, self.tmp_dir, next_config,
                    seed, num_run,
                    self.resampling_strategy,
                    self.resampling_strategy_args,
                    self.memory_limit,
                    subset_time_limit, n_data_subsample)
                (duration, result, _, additional_run_info, status) = _info
                self.logger.info("Finished evaluating %d. configuration on SUBSET. "
                                 "Duration %f; loss %f; status %s; additional run "
                                 "info: %s ", num_run, duration, result,
                                 str(status), additional_run_info)

                if i < len(subset_ratios) - 1:
                    if status != StatusType.SUCCESS:
                        # Do not increase num_run here, because we will try
                        # the same configuration with less data
                        self.logger.info("A CONFIG did not finish "
                                         " for subset ratio %f -> going smaller",
                                         ratio)
                        continue
                    else:
                        num_run += 1
                        self.logger.info("Finished SUBSET training sucessfully "
                                         "with ratio %f", ratio)
                        break
                else:
                    if status != StatusType.SUCCESS:
                        self.logger.info("A CONFIG did not finish "
                                         " for subset ratio %f.",
                                         ratio)
                        num_run += 1
                        continue
                    else:
                        num_run += 1
                        self.logger.info("Finished SUBSET training sucessfully "
                                         "with ratio %f", ratio)
                        break

        # == METALEARNING suggestions
        # we start by evaluating the defaults on the full dataset again
        # and add the suggestions from metalearning behind it
        metalearning_configurations = self.collect_metalearning_suggestions_with_limits()

        # == first, evaluate all metelearning and default configurations
        for i, next_config in enumerate((default_cfgs +
                                        metalearning_configurations)):
            # Do not evaluate default configurations more than once
            if i >= len(default_cfgs) and next_config in default_cfgs:
                continue

            # JTS: reset the data manager before each configuration since
            #      we work on the data in-place
            # NOTE: this is where we could also apply some memory limits
            config_name = 'meta-learning' if i >= len(default_cfgs) \
                else 'default'

            self.logger.info("Starting to evaluate %d. configuration "
                             "(%s configuration) with time limit %ds.",
                             num_run, config_name, self.func_eval_time_limit)
            self.logger.info(next_config)
            self.reset_data_manager()
            info = eval_with_limits(self.datamanager, self.tmp_dir, next_config,
                                    seed, num_run,
                                    self.resampling_strategy,
                                    self.resampling_strategy_args,
                                    self.memory_limit,
                                    self.func_eval_time_limit)
            (duration, result, _, additional_run_info, status) = info
            run_history.add(config=next_config, cost=result,
                            time=duration , status=status,
                            instance_id=0, seed=seed)
            run_history.update_cost(next_config, result)
            self.logger.info("Finished evaluating %d. configuration. "
                             "Duration %f; loss %f; status %s; additional run "
                             "info: %s ", num_run, duration, result,
                             str(status), additional_run_info)
            num_run += 1
            if smac.incumbent is None:
                smac.incumbent = next_config
            elif result < run_history.get_cost(smac.incumbent):
                smac.incumbent = next_config

        # == after metalearning run SMAC loop
        smac.runhistory = run_history
        smac_iter = 0
        finished = False
        while not finished:
            next_configs = []
            try:
                # JTS TODO: handle the case that run_history is empty
                X_cfg, Y_cfg = rh2EPM.transform(run_history)
                next_configs = smac.choose_next(X_cfg, Y_cfg)
                next_configs.extend(next_configs[:2])
            except Exception as e:
                self.logger.error(e)
                self.logger.error("Error in getting next configurations "
                                  "with SMAC. Using random configuration!")
                next_config = self.config_space.sample_configuration()
                next_configs.append(next_config)

            for next_config in next_configs:
                self.logger.info("Starting to evaluate %d. configuration (from "
                                 "SMAC) with time limit %ds.", num_run,
                                 self.func_eval_time_limit)
                self.logger.info(next_config)
                self.reset_data_manager()
                info = eval_with_limits(self.datamanager, self.tmp_dir, next_config,
                                        seed, num_run,
                                        self.resampling_strategy,
                                        self.resampling_strategy_args,
                                        self.memory_limit,
                                        self.func_eval_time_limit)
                (duration, result, _, additional_run_info, status) = info
                run_history.add(config=next_config, cost=result,
                                time=duration , status=status,
                                instance_id=0, seed=seed)
                run_history.update_cost(next_config, result)

                # TODO add unittest to make sure everything works fine and
                # this does not get outdated!
                if smac.incumbent is None:
                    smac.incumbent = next_config
                elif result < run_history.get_cost(smac.incumbent):
                    smac.incumbent = next_config

                self.logger.info("Finished evaluating %d. configuration. "
                                 "Duration: %f; loss: %f; status %s; additional "
                                 "run info: %s ", num_run, duration, result,
                                 str(status), additional_run_info)
                smac_iter += 1
                num_run += 1
                if max_iters is not None:
                    finished = (smac_iter < max_iters)
        
        

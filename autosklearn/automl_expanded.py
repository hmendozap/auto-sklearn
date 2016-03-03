# -*- encoding: utf-8 -*-

from __future__ import print_function

import os

from HPOlibConfigSpace.converters import pcs_parser
from HPOlibConfigSpace.configuration_space import Configuration

from autosklearn.metalearning.mismbo import convert_conf2smac_string
from autosklearn.util import set_auto_seed, del_auto_seed,\
                            submit_process, pipeline, Backend
import autosklearn.automl as autosk_automl
from autosklearn.automl import AutoML

from autosklearn.constants import *
from autosklearn.util.smac import run_smac


def _load_search_space(tmp_dir, backend, watcher, logger,
                       configuration_space_file):
    task_name = 'LoadConfigSpace'
    logger.info('Started to load a predefined configuration space')
    watcher.start_task(task_name)
    with open(configuration_space_file) as fh:
        cs = pcs_parser.read(fh)
    watcher.stop_task(task_name)
    configuration_path = os.path.abspath(configuration_space_file)
    return cs, configuration_path


class ExpAutoML(AutoML):

    def __init__(self, tmp_dir, output_dir, time_left_for_this_task,
                 per_run_time_limit, log_dir=None,
                 initial_configurations_via_metalearning=0,
                 ensemble_size=1, ensemble_nbest=1,
                 seed=1, ml_memory_limit=2048,
                 metadata_directory=None, queue=None,
                 keep_models=True, debug_mode=True,
                 include_estimators=None,
                 include_preprocessors=None,
                 resampling_strategy='holdout-iterative-fit',
                 resampling_strategy_arguments=None,
                 fixed_configuration_space=None,
                 delete_tmp_folder_after_terminate=False,
                 delete_output_folder_after_terminate=False,
                 shared_mode=False, precision=32):
        AutoML.__init__(self, tmp_dir, output_dir, time_left_for_this_task,
                        per_run_time_limit, log_dir,
                        initial_configurations_via_metalearning,
                        ensemble_size, ensemble_nbest, seed,
                        ml_memory_limit, metadata_directory, queue,
                        keep_models, debug_mode, include_estimators,
                        include_preprocessors, resampling_strategy,
                        resampling_strategy_arguments,
                        delete_tmp_folder_after_terminate,
                        delete_output_folder_after_terminate,
                        shared_mode, precision)
        self._fixed_cs = fixed_configuration_space

    def _fit(self, datamanager):
        self.models_ = None
        self.ensemble_ = None

        # Check arguments prior to doing anything!
        if self._resampling_strategy not in ['holdout', 'holdout-iterative-fit',
                                             'cv', 'nested-cv', 'partial-cv']:
            raise ValueError('Illegal resampling strategy: %s' %
                             self._resampling_strategy)
        if self._resampling_strategy == 'partial-cv' and \
                self._ensemble_size != 0:
            raise ValueError("Resampling strategy partial-cv cannot be used "
                             "together with ensembles.")

        self._backend._make_internals_directory()
        if self._keep_models:
            try:
                os.mkdir(self._backend.get_model_dir())
            except OSError:
                self._logger.warning("model directory already exists")
                if not self._shared_mode:
                    raise

        datamanager.fixed_cs = self._fixed_cs
        self._metric = datamanager.info['metric']
        self._task = datamanager.info['task']
        self._label_num = datamanager.info['label_num']

        set_auto_seed(self._seed)

        # OneHotEncoding
        data_manager_path = self._backend.save_datamanager(datamanager)

        self._save_ensemble_data(
            datamanager.data['X_train'],
            datamanager.data['Y_train'])

        time_for_load_data = self._stopwatch.wall_elapsed(self._dataset_name)

        if self._debug_mode:
            self._print_load_time(
                self._dataset_name,
                self._time_for_task,
                time_for_load_data,
                self._logger)

        # == Perform dummy predictions
        if self._resampling_strategy in ['holdout', 'holdout-iterative-fit']:
            self._do_dummy_prediction(datamanager)

        # Receive a predefined configuration space from file
        if datamanager.fixed_cs is not None:
            self.configuration_space, configspace_path = _load_search_space(
                self._tmp_dir,
                self._backend,
                self._stopwatch,
                self._logger,
                self._fixed_cs)
        else:
            self.configuration_space, configspace_path = autosk_automl._create_search_space(
                self._tmp_dir,
                datamanager.info,
                self._backend,
                self._stopwatch,
                self._logger,
                datamanager.includes[0],
                datamanager.includes[1])
        self.configuration_space_created_hook(datamanager)

        proc_ensembles = self.run_ensemble_builder()

        meta_features = autosk_automl._calculate_metafeatures(
            data_feat_type=datamanager.feat_type,
            data_info_task=datamanager.info['task'],
            x_train=datamanager.data['X_train'],
            y_train=datamanager.data['Y_train'],
            basename=self._dataset_name,
            watcher=self._stopwatch,
            metalearning_cnt=self._initial_configurations_via_metalearning,
            logger=self._logger)

        self._stopwatch.start_task('OneHot')
        datamanager.perform1HotEncoding()
        self._stopwatch.stop_task('OneHot')

        if meta_features is None:
            initial_configurations = []
        elif datamanager.info['task'] in [MULTICLASS_CLASSIFICATION,
                                          BINARY_CLASSIFICATION,
                                          MULTILABEL_CLASSIFICATION]:

            meta_features_encoded = autosk_automl._calculate_metafeatures_encoded(
                self._dataset_name,
                datamanager.data['X_train'],
                datamanager.data['Y_train'],
                self._stopwatch,
                self._logger)

            self._logger.debug(meta_features.__repr__(verbosity=2))
            self._logger.debug(meta_features_encoded.__repr__(verbosity=2))

            initial_configurations = autosk_automl._get_initial_configuration(
                meta_features,
                meta_features_encoded,
                self._dataset_name,
                self._metric,
                self.configuration_space,
                self._task,
                self._metadata_directory,
                self._initial_configurations_via_metalearning,
                datamanager.info[
                    'is_sparse'],
                self._stopwatch,
                self._logger)

            autosk_automl._print_debug_info_of_init_configuration(
                initial_configurations,
                self._dataset_name,
                self._time_for_task,
                self._logger,
                self._stopwatch)

        else:
            initial_configurations = []
            self._logger.warning('Metafeatures encoded not calculated')

        # == RUN SMAC
        if (datamanager.info["task"] == BINARY_CLASSIFICATION) or \
            (datamanager.info["task"] == MULTICLASS_CLASSIFICATION):
            config = {'balancing:strategy': 'weighting',
                      'classifier:__choice__': 'sgd',
                      'classifier:sgd:loss': 'hinge',
                      'classifier:sgd:penalty': 'l2',
                      'classifier:sgd:alpha': 0.0001,
                      'classifier:sgd:fit_intercept': 'True',
                      'classifier:sgd:n_iter': 5,
                      'classifier:sgd:learning_rate': 'optimal',
                      'classifier:sgd:eta0': 0.01,
                      'classifier:sgd:average': 'True',
                      'imputation:strategy': 'mean',
                      'one_hot_encoding:use_minimum_fraction': 'True',
                      'one_hot_encoding:minimum_fraction': 0.1,
                      'preprocessor:__choice__': 'no_preprocessing',
                      'rescaling:__choice__': 'min/max'}
        elif datamanager.info["task"] == MULTILABEL_CLASSIFICATION:
            config = {'classifier:__choice__': 'adaboost',
                      'classifier:adaboost:algorithm': 'SAMME.R',
                      'classifier:adaboost:learning_rate': 1.0,
                      'classifier:adaboost:max_depth': 1,
                      'classifier:adaboost:n_estimators': 50,
                      'balancing:strategy': 'weighting',
                      'imputation:strategy': 'mean',
                      'one_hot_encoding:use_minimum_fraction': 'True',
                      'one_hot_encoding:minimum_fraction': 0.1,
                      'preprocessor:__choice__': 'no_preprocessing',
                      'rescaling:__choice__': 'none'}
        else:
            config = None
            self._logger.info("Tasktype unknown: %s" %
                              TASK_TYPES_TO_STRING[datamanager.info["task"]])

        if config is not None:
            try:
                configuration = Configuration(self.configuration_space, config)
                config_string = convert_conf2smac_string(configuration)
                initial_configurations = [config_string] + initial_configurations
            except ValueError:
                pass

        # == RUN SMAC
        proc_smac = run_smac(tmp_dir=self._tmp_dir, basename=self._dataset_name,
                             time_for_task=self._time_for_task,
                             ml_memory_limit=self._ml_memory_limit,
                             data_manager_path=data_manager_path,
                             configspace_path=configspace_path,
                             initial_configurations=initial_configurations,
                             per_run_time_limit=self._per_run_time_limit,
                             watcher=self._stopwatch, backend=self._backend,
                             seed=self._seed,
                             resampling_strategy=self._resampling_strategy,
                             resampling_strategy_arguments=self._resampling_strategy_arguments,
                             shared_mode=self._shared_mode)

        procs = []

        if proc_smac is not None:
            procs.append(proc_smac)
        if proc_ensembles is not None:
            procs.append(proc_ensembles)

        if self._queue is not None:
            self._queue.put([time_for_load_data, data_manager_path, procs])
        else:
            for proc in procs:
                proc.wait()

        # Delete AutoSklearn environment variable
        del_auto_seed()

        # In case
        try:
            del self._datamanager
        except Exception:
            pass

        if self._queue is None:
            self._load_models()

        return self

# from satorilib.api.disk.filetypes.csv import CSVManager # df = CSVManager.read(filePath=path)
import threading
import joblib
import os
from reactivex.subject import BehaviorSubject
from satorilib.api.hash import generatePathId
from satorilib.concepts import Stream, StreamId

from datetime import datetime
import random
from typing import Union, Optional, List, Any

from process import process_data
from determine_features import determine_feature_set
from model_creation import model_create_train_test_and_predict

class Engine:
    def __init__(self, streams: list[Stream]):
        ''' build all the models '''
        self.streams = streams 
        self.models = {}
        self.threads = {}
        self.run()
        self.trigger()

    def initialize_models(self):
        for stream in self.streams:
            # stream_id = os.path.splitext(os.path.basename(stream))[0]
            self.models[stream] = Model(
                streamId=stream
                )

    def trigger(self):
        ''' setup our BehaviorSubject streams for inter-thread communication '''
        self.modelUpdated = BehaviorSubject(None)
    
    def run_model(self, stream: str):
        model = self.models[stream]
        model.run()
        model.predict()

    def start_threads(self):
        for stream, _ in self.models.items():
            thread = threading.Thread(target=self.run_model, args=(stream,))
            self.threads[stream] = thread
            thread.start()

    def wait_for_completion(self):
        for thread in self.threads.values():
            thread.join()

    def run(self):
        self.initialize_models()
        self.start_threads()
        self.wait_for_completion()


class Model:
    def __init__(self, streamId: StreamId, datapath_override: str = None, modelpath_override: str = None):
        self.streamId = streamId
        self.datapath = datapath_override or self.data_path()
        self.modelpath = modelpath_override or self.model_path()
        self.stable: list = self.load()

    def data_path(self) -> str:
        return f'./data/{generatePathId(streamId=self.streamId)}/aggregate.csv'

    def model_path(self) -> str:
        return f'./models/{generatePathId(streamId=self.streamId)}'

    def load(self) -> Union[None, list]:
        ''' loads the stable model from disk if present'''
        try:
            self.stable = joblib.load(self.modelpath)
            return self.stable
        except FileNotFoundError:
            return None

    def save(self):
        ''' saves the stable model to disk '''
        os.makedirs(os.path.dirname(self.modelpath), exist_ok=True)
        joblib.dump(self.stable, self.modelpath)

    def compare(self, model, replace:bool = False) -> bool:
        ''' compare the stable model to the heavy model '''
        compared  = model[0].backtest_error < self.stable[0].backtest_error
        if replace and compared:
            self.stable = model
            self.modelUpdated.on_next(self.stable)
            return True
        return compared

    def predict(self, data=None):
        ''' prediction without training '''
        status, predictor_model = engine( filename=self.datapath, 
                               list_of_models=[self.stable[0].model_name],
                               mode='predict',
                               unfitted_forecaster=self.stable[0].unfitted_forecaster
                               )
        if status == 1:
            print(predictor_model[0].forecast)


    def run(self):
        '''
        main loop for generating models and comparing them to the best known
        model so far in order to replace it if the new model is better, always
        using the best known model to make predictions on demand.
        '''
        if self.stable is None:
            status, model = engine(self.datapath, ['quick_start'])
            if status == 1:
                self.stable = model
                self.save()

        while True:
            status, pilot = engine(self.datapath, ['random_model'])
            if self.compare(pilot, replace=True):
                self.save()

    def run_forever(self):
        self.thread = threading.Thread(target=self.run, args=(), daemon=True)
        self.thread.start()


def check_model_suitability(list_of_models, allowed_models, dataset_length):
    suitable_models = []
    unsuitable_models = []

    for model in list_of_models:
        if model in allowed_models:
            suitable_models.append(True)
        else:
            suitable_models.append(False)
            reason = f"Not allowed for dataset size of {dataset_length}"
            unsuitable_models.append((model, reason))

    return suitable_models, unsuitable_models

def engine(
    filename: str,
    list_of_models: List[str],
    interval: List[int] = [10, 90],
    feature_set_reduction: bool = False,
    exogenous_feature_type: str = 'ExogenousFeaturesBasedonSeasonalityTestWithAdditivenMultiplicative',
    feature_set_reduction_method: str = 'RFECV',
    random_state_hyperr: int = 123,
    metric: str = 'mase',
    mode: str = 'train',
    unfitted_forecaster: Optional[Any] = None
):
    ''' Engine function for the Satori Engine '''

    list_of_models = [model.lower() for model in list_of_models]

    quick_start_present = "quick_start" in list_of_models
    random_model_present = "random_model" in list_of_models
    random_state_hyper = random_state_hyperr

    # Process data first to get allowed_models
    proc_data = process_data(
        filename,
        quick_start=quick_start_present
    )

    # if quick_start_present and random_model_present:
    #     warnings.warn(
    #         "Both 'quick_start' and 'random_model' are present. 'quick_start' will take precedence.")

    if random_model_present and not quick_start_present:
        current_time = datetime.now()
        seed = int(current_time.strftime("%Y%m%d%H%M%S%f"))
        random.seed(seed)
        print(f"Using random seed: {seed}")

        # Randomly select options
        feature_set_reduction = random.choice([True, False])
        exogenous_feature_type = random.choice(["NoExogenousFeatures", "Additive", "AdditiveandMultiplicativeExogenousFeatures",
                                               "ExogenousFeaturesBasedonSeasonalityTestWithAdditivenMultiplicative"])
        feature_set_reduction_method = random.choice(["RFECV", "RFE"])
        random_state_hyper = random.randint(0, 2**32 - 1)

        # Replace 'random_model' with a randomly selected model from allowed_models
        list_of_models = [random.choice(
            proc_data.allowed_models) if model == "random_model" else model for model in list_of_models]
        print(f"Randomly selected models: {list_of_models}")
        print(f"feature_set_reduction: {feature_set_reduction}")
        print(f"exogenous_feature_type: {exogenous_feature_type}")
        print(f"feature_set_reduction_method: {feature_set_reduction_method}")
        print(f"random_state_hyper: {random_state_hyper}")

    if quick_start_present:
        feature_set_reduction = False
        exogenous_feature_type = "NoExogenousFeatures"
        list_of_models = proc_data.allowed_models

    if proc_data.if_small_dataset:
        return 2, "Status = 2 (insufficient amount of data)"

    # Check if the requested models are suitable based on the allowed_models
    suitable_models, unsuitable_models = check_model_suitability(
        list_of_models, proc_data.allowed_models, len(proc_data.dataset))

    if unsuitable_models:
        print("The following models are not allowed due to insufficient data:")
        for model, reason in unsuitable_models:
            print(f"- {model}: {reason}")

    if not any(suitable_models):
        return 3, "Status = 3 (none of the requested models are suitable for the available data)"

    # Filter the list_of_models to include only suitable models
    list_of_models = [model for model, is_suitable in zip(
        list_of_models, suitable_models) if is_suitable]

    try:
        features = determine_feature_set(
            dataset=proc_data.dataset,
            data_train=proc_data.data_subsets['train'],
            end_validation=proc_data.end_times['validation'],
            end_train=proc_data.end_times['train'],
            dataset_with_features=proc_data.dataset_withfeatures,
            dataset_start_time=proc_data.dataset_start_time,
            dataset_end_time=proc_data.dataset_end_time,
            initial_lags=proc_data.lags,
            weight_para=proc_data.use_weight,
            exogenous_feature_type=exogenous_feature_type,
            feature_set_reduction=feature_set_reduction,
            feature_set_reduction_method=feature_set_reduction_method,
            bayesian_trial=20,
            random_state_hyper=random_state_hyper,
            frequency=proc_data.sampling_frequency,
            backtest_steps=proc_data.backtest_steps,
            prediction_steps=proc_data.forecasting_steps
        )

        list_of_results = []
        for model_name in list_of_models:
            result = model_create_train_test_and_predict(
                model_name=model_name,
                dataset=proc_data.dataset,
                dataset_train=proc_data.data_subsets['train'],
                end_validation=proc_data.end_times['validation'],
                end_test=proc_data.end_times['test'],
                sampling_freq=proc_data.sampling_frequency,
                differentiation=features.differentiation,
                selected_lags=features.selected_lags,
                selected_exog=features.selected_exog,
                dataset_selected_features=features.dataset_selected_features,
                data_missing=features.missing_values,
                weight=features.weight,
                select_hyperparameters=True,
                default_hyperparameters=None,
                random_state_hyper=random_state_hyper,
                backtest_steps=proc_data.backtest_steps,
                interval=interval,
                metric=metric,
                forecast_calendar_features=features.forecast_calendar_features,
                forecasting_steps=proc_data.forecasting_steps,
                hour_seasonality=features.hour_seasonality,
                dayofweek_seasonality=features.dow_seasonality,
                week_seasonality=features.week_seasonality,
                baseline_1=proc_data.time_metric_baseline,
                baseline_2=proc_data.forecasterequivalentdate,
                baseline_3=proc_data.forecasterequivalentdate_n_offsets,
                mode=mode,
                forecaster=unfitted_forecaster
            )
            list_of_results.append(result)

        return 1, list_of_results  # Status = 1 (ran correctly)

    except Exception as e:
        # Additional status code for unexpected errors
        return 4, f"An error occurred: {str(e)}"


# e = Model(
#   streamId=StreamId(source='test', stream='test', target='test', author='test'),
#   datapath_override="NATGAS1D.csv",
#   modelpath_override='baseline')

# e.run()

e = Engine(
    streams = [StreamId(source='test', stream='test', target='test', author='test'), 
              StreamId(source='test', stream='test', target='test', author='test')]
)

print("All Executed well")
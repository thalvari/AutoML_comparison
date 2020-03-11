import argparse
import pickle
import random as rn
import time
import warnings
from abc import ABC, abstractmethod
from datetime import timedelta
from pathlib import Path
from tempfile import mkdtemp

warnings.filterwarnings("ignore")

import autokeras as ak
import h2o
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from PIL import Image
from autokeras.utils import pickle_from_file
from h2o.automl import H2OAutoML
from keras.datasets import fashion_mnist
from numpy.random.mtrand import RandomState
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier

seed = 42
rn.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
torch.manual_seed(seed)
random_state = RandomState(seed)


def print_gpu_info():
    gpu_count = torch.cuda.device_count()
    print(f"GPU count: {gpu_count}")
    if gpu_count > 0:
        for i in range(gpu_count):
            print(f"{i}. {torch.cuda.get_device_name(i)}")


def get_full_model_name(model_name, augment, gpu):
    model_full_name = model_name
    if augment:
        model_full_name += "_aug"
    if gpu:
        model_full_name += "_gpu"
    return model_full_name


def load_data(dataset_name, n_images):
    if dataset_name == "digits":
        x, y = load_digits(return_X_y=True)
        y = y.astype(np.uint8)
        x = x.astype(np.uint8)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=random_state)
        image_hw = np.sqrt(len(x[0])).astype(int)
    else:
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        image_hw = x_train.shape[1]
        x_train = x_train.reshape((len(x_train), image_hw ** 2))
        x_test = x_test.reshape((len(x_test), image_hw ** 2))
    if n_images is not None:
        x_train = x_train[:n_images]
        y_train = y_train[:n_images]
    return x_train, y_train, x_test, y_test, {"image_hw": image_hw, "a_max": np.amax(x_test)}


def show_images(arr, s):
    plt.imshow(arr.reshape(s, s), cmap="gray_r")
    plt.show()


def add_noise(arr, std, data_params_dict):
    arr = arr.astype(float)
    arr += random_state.normal(scale=std, size=arr.shape)
    arr = np.clip(arr, 0, data_params_dict["a_max"])
    arr = np.round(arr).astype(np.uint8)
    return arr


def rotate(arr, angle, data_params_dict):
    image_hw = data_params_dict["image_hw"]
    arr = arr.reshape(image_hw, image_hw).astype(float)
    angle = random_state.uniform(-angle, angle)
    img = Image.fromarray(arr).rotate(angle)
    arr = np.asarray(img)
    arr = np.round(arr).flatten().astype(np.uint8)
    return arr


def get_err_data_lists(x_train, x_test, err_source, err_level, dataset_name, data_params_dict):
    if err_source == "noise":
        error_generator = add_noise
        err_param_name = "std"
        if dataset_name == "digits":
            err_params = np.round(np.linspace(0, 16, num=6), 3)
        else:
            err_params = np.round(np.linspace(0, 255, num=6), 3)
    else:
        error_generator = rotate
        err_param_name = "max_angle"
        err_params = np.linspace(0, 180, num=6)

    if err_level is not None:
        err_x_train_list = [np.apply_along_axis(error_generator, 1, x_train, err_params[err_level], data_params_dict)]
    else:
        err_x_train_list = [np.apply_along_axis(error_generator, 1, x_train, ep, data_params_dict) for ep in err_params]
    err_x_test_list = [np.apply_along_axis(error_generator, 1, x_test, ep, data_params_dict) for ep in err_params]
    return err_x_train_list, err_x_test_list, err_param_name, err_params


class AbstractModel(ABC):
    model = None
    time_train = np.nan

    def __init__(
            self, model_full_name, err_source, err_param_name, dataset_name, time_limit_mins, augment, data_params_dict,
            n_threads, mem_size_gb
    ):
        self.model_full_name = model_full_name
        self.err_source = err_source
        self.err_param_name = err_param_name
        self.dataset_name = dataset_name
        self.time_limit_mins = time_limit_mins
        self.augment = augment
        self.data_params_dict = data_params_dict
        self.n_threads = n_threads
        self.mem_size_gb = mem_size_gb

    def prepare(self, x_train, y_train, err_param):
        model_description = f"{self.model_full_name}_{self.err_param_name}_{np.round(err_param, 3)}_" \
                            f"time_limit_mins_{self.time_limit_mins}"
        model_path_prefix = f"models_{self.dataset_name}_{self.err_source}/model_{model_description}"
        summary_path = f"models_{self.dataset_name}_{self.err_source}/summary_{model_description}.txt"

        if Path(summary_path).is_file():
            self.load(model_path_prefix)
        else:
            self.fit(x_train, y_train)
            self.save(model_path_prefix)
            summary = self.get_summary(len(x_train))
            with open(summary_path, "w") as f:
                print(f"Actual train time: {str(self.time_train)}\n\n{summary}", file=f)

    @abstractmethod
    def fit(self, x_train, y_train):
        pass

    @abstractmethod
    def save(self, model_path_prefix):
        pass

    @abstractmethod
    def load(self, model_path_prefix):
        pass

    @abstractmethod
    def get_summary(self, *args):
        pass

    @abstractmethod
    def predict(self, x_test, y_test):
        pass

    @abstractmethod
    def shutdown(self):
        pass


class AutoKerasModel(AbstractModel):

    def __init__(
            self, model_full_name, err_source, err_param_name, dataset_name, time_limit_mins, augment, data_params_dict,
            n_threads, mem_size_gb
    ):
        super().__init__(
            model_full_name, err_source, err_param_name, dataset_name, time_limit_mins, augment, data_params_dict,
            n_threads, mem_size_gb
        )

    def fit(self, x_train, y_train):
        self.model = ak.ImageClassifier(augment=self.augment, path=mkdtemp(dir="temp"), verbose=True)
        image_hw = self.data_params_dict["image_hw"]
        x_train = x_train.reshape((len(x_train), image_hw, image_hw, 1))
        time_start = time.time()
        self.model.fit(x_train, y_train, time_limit=60 * self.time_limit_mins)
        self.time_train = timedelta(seconds=np.round(time.time() - time_start))

    def load(self, model_path_prefix):
        self.model = pickle_from_file(f"{model_path_prefix}.pkl")

    def save(self, model_path_prefix):
        self.model.export_autokeras_model(f"{model_path_prefix}.pkl")

    def get_summary(self, n_train):
        return repr(self.model.cnn.best_model.produce_model())

    def predict(self, x_test, y_test):
        image_hw = self.data_params_dict["image_hw"]
        x_test = x_test.reshape((len(x_test), image_hw, image_hw, 1))
        y_pred = self.model.predict(x_test)
        return np.round(accuracy_score(y_true=y_test, y_pred=y_pred), 3)

    def shutdown(self):
        pass


class H2OAutoMLModel(AbstractModel):

    def __init__(
            self, model_full_name, err_source, err_param_name, dataset_name, time_limit_mins, augment, data_params_dict,
            n_threads, mem_size_gb
    ):
        super().__init__(
            model_full_name, err_source, err_param_name, dataset_name, time_limit_mins, augment, data_params_dict,
            n_threads, mem_size_gb
        )
        h2o.init(
            name=f"#{rn.SystemRandom().randint(1, 2 ** 20)}", nthreads=n_threads, max_mem_size_GB=mem_size_gb,
            min_mem_size_GB=mem_size_gb
        )
        # h2o.no_progress()

    def fit(self, x_train, y_train):
        training_frame = h2o.H2OFrame(np.concatenate((x_train, y_train.reshape(-1, 1)), axis=1))
        x_cols = np.array(training_frame.columns)[:-1].tolist()
        y_cols = np.array(training_frame.columns)[-1].tolist()
        training_frame[y_cols] = training_frame[y_cols].asfactor()
        self.model = H2OAutoML(max_runtime_secs=60 * self.time_limit_mins, seed=seed, verbosity="debug")
        time_start = time.time()
        self.model.train(x=x_cols, y=y_cols, training_frame=training_frame)
        self.time_train = timedelta(seconds=np.round(time.time() - time_start))

    def load(self, model_path_prefix):
        self.model = h2o.load_model(model_path_prefix)

    def save(self, model_path_prefix):
        temp_path = h2o.save_model(
            model=self.model.leader, path=f"models_{self.dataset_name}_{self.err_source}", force=True
        )
        source = Path(temp_path)
        target = Path(model_path_prefix)
        source.rename(target)

    def get_summary(self, n_train):
        leader_params = self.model.leader.params
        summary = "model_id:\n" + leader_params["model_id"]["actual"]["name"] + "\n\n"
        summary += "base_models:"
        if "base_models" in leader_params:
            for base_model in leader_params["base_models"]["actual"]:
                summary += "\n" + base_model["name"]
        return summary

    def predict(self, x_test, y_test):
        testing_frame = h2o.H2OFrame(np.concatenate((x_test, y_test.reshape(-1, 1)), axis=1))
        y_cols = np.array(testing_frame.columns)[-1].tolist()
        testing_frame[y_cols] = testing_frame[y_cols].asfactor()
        y_pred = self.model.predict(testing_frame).as_data_frame(header=False)["predict"].values.astype(int)
        return np.round(accuracy_score(y_true=y_test, y_pred=y_pred), 3)

    def shutdown(self):
        h2o.cluster().shutdown()


class TPOTModel(AbstractModel):

    def __init__(
            self, model_full_name, err_source, err_param_name, dataset_name, time_limit_mins, augment, data_params_dict,
            n_threads, mem_size_gb
    ):
        super().__init__(
            model_full_name, err_source, err_param_name, dataset_name, time_limit_mins, augment, data_params_dict,
            n_threads, mem_size_gb
        )

    def fit(self, x_train, y_train):
        self.model = TPOTClassifier(
            max_time_mins=self.time_limit_mins, max_eval_time_mins=self.time_limit_mins, n_jobs=self.n_threads,
            random_state=seed, verbosity=3
        )
        time_start = time.time()
        self.model.fit(x_train, y_train)
        self.time_train = timedelta(seconds=np.round(time.time() - time_start))

    def load(self, model_path_prefix):
        with open(f"{model_path_prefix}.pkl", "rb") as f:
            self.model = pickle.load(f)

    def save(self, model_path_prefix):
        with open(f"{model_path_prefix}.pkl", "wb") as f:
            pickle.dump(self.model.fitted_pipeline_, f)

    def get_summary(self, n_train):
        summary = "Steps:"
        for i, step in enumerate(self.model.fitted_pipeline_.steps):
            summary += f"\n{i}. {str(step[1])}"
        return summary

    def predict(self, x_test, y_test):
        return round(self.model.score(x_test, y_test), 3)

    def shutdown(self):
        pass


def main(args):
    # print_gpu_info()

    model_name = args.model_name
    dataset_name = args.dataset_name
    err_source = args.err_source
    time_limit_mins = args.time_limit_mins
    n_threads = args.n_threads
    mem_size_gb = args.mem_size_gb
    augment = args.augment
    gpu = args.gpu
    n_images = args.n_images
    err_level = args.err_level
    model_full_name = get_full_model_name(model_name, augment, gpu)

    x_train, y_train, x_test, y_test, data_params_dict = load_data(dataset_name, n_images)
    err_x_train_list, err_x_test_list, err_param_name, err_params = get_err_data_lists(
        x_train, x_test, err_source, err_level, dataset_name, data_params_dict
    )

    scores = []
    for err_level_train, err_x_train in enumerate(err_x_train_list):
        if model_name == "autokeras":
            model_class = AutoKerasModel
        elif model_name == "h2o":
            model_class = H2OAutoMLModel
        else:
            model_class = TPOTModel
        model = model_class(model_full_name, err_source, err_param_name, dataset_name, time_limit_mins, augment,
                            data_params_dict, n_threads, mem_size_gb)
        if err_level is not None:
            err_level_train = err_level
        model.prepare(err_x_train, y_train, err_params[err_level_train])
        for err_level_test, err_x_test in enumerate(err_x_test_list):
            scores.append({
                f"model_name": model_full_name,
                f"train_{err_param_name}": err_params[err_level_train],
                f"test_{err_param_name}": err_params[err_level_test],
                "accuracy": model.predict(err_x_test, y_test),
            })
        model.shutdown()

    scores_df = pd.DataFrame(scores)
    print(scores_df)
    if err_level is None:
        scores_path = f"scores_{dataset_name}_{err_source}/{model_full_name}_{err_source}_time_limit_mins_" \
                      f"{time_limit_mins}.pkl"
        scores_df.to_pickle(scores_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", choices=["autokeras", "h2o", "tpot"])
    parser.add_argument("dataset_name", choices=["digits", "fashion"])
    parser.add_argument("err_source", choices=["noise", "rotation"])
    parser.add_argument("time_limit_mins", type=int)
    parser.add_argument("n_threads", type=int)
    parser.add_argument("mem_size_gb", type=int)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--n_images", type=int)
    parser.add_argument("--err_level", type=int, choices=range(6))
    args = parser.parse_args()
    main(args)

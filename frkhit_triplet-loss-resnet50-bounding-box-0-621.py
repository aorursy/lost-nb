#!/usr/bin/env python
# coding: utf-8



# install 
get_ipython().system('pip -q install aiohttp faiss-prebuilt pyxtools pymltools')
get_ipython().system('apt -qq install -y libopenblas-base libomp-dev')
    
import tensorflow
tensorflow.__version__




import csv
import glob
import logging
import pickle
import subprocess
import time

import numpy as np
import os
import pandas as pd
import random
import shutil
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_utils, resnet_v2

from pymltools.tf_utils import Params, DatasetUtils, parse_bounding_boxes_list, tf_image_crop,     show_embedding, AbstractEstimator, estimator_iter_process, colab_save_file_func, OptimizerType, get_wsl_path,     map_per_set, LossStepHookForTrain, ProcessMode, load_data_from_h5file, tf_model_fn, balance_class_dict,     store_data_in_h5file, get_triplet_pair_np, map_per_image, show_distance_dense_plot, init_logger, pre_process_utils
from pyxtools import byte_to_string, create_fake_random_string, iter_list_with_size, download_big_file, list_files,     FileCache, random_choice, NormType, get_pretty_float, expend_bounding_box, image_utils


try:
    from pyxtools.faiss_tools import ImageIndexUtils
except ImportError:
    from pyxtools.pyxtools.faiss_tools import ImageIndexUtils

resnet_arg_scope = resnet_utils.resnet_arg_scope
resnet_v2_50 = resnet_v2.resnet_v2_50


def combine_csv(file_weight: dict, out_file: str):
    sub_files = []
    sub_weight = []
    for csv_file, weight in file_weight.items():
        sub_files.append(csv_file)
        sub_weight.append(weight)

    place_weights = {}
    for i in range(5):
        place_weights[i] = 10 - i * 2

    h_label = 'Image'
    h_target = 'Id'

    sub = [None] * len(sub_files)
    for i, file in enumerate(sub_files):
        print("Reading {}: w={} - {}".format(i, sub_weight[i], file))
        reader = csv.DictReader(open(file, "r"))
        sub[i] = sorted(reader, key=lambda d: d[h_label])

    out = open(out_file, "w", newline='')
    writer = csv.writer(out)
    writer.writerow([h_label, h_target])
    p = 0
    for row in sub[0]:
        target_weight = {}
        for s in range(len(sub_files)):
            row1 = sub[s][p]
            for ind, trgt in enumerate(row1[h_target].split(' ')):
                target_weight[trgt] = target_weight.get(trgt, 0) + (place_weights[ind] * sub_weight[s])
        tops_trgt = sorted(target_weight, key=target_weight.get, reverse=True)[:5]
        writer.writerow([row1[h_label], " ".join(tops_trgt)])
        p += 1
    out.close()


class PathManager(object):
    def __init__(self, mode: str = "default"):
        self._mode = mode
        self.init_all_path()
        self.init_bounding_boxes()
        self._submission_csv = None

    @property
    def data_set_root_path(self) -> str:
        if self._mode == "default":
            return get_wsl_path("E:/frkhit/Download/AI/data-set/kaggle/whale")
        elif self._mode == "fake":
            return get_wsl_path("E:/frkhit/Download/AI/data-set/kaggle/whale/fake")
        elif self._mode == "colab":
            return "/content"
        elif self._mode == "kaggle":
            return "../input/humpback-whale-identification"

        return "./"

    @property
    def working_path(self):
        if self._mode == "kaggle":
            return "/kaggle/working/"

        return self.data_set_root_path

    @property
    def output_path(self, ):
        if self._mode == "default" or self._mode == "fake":
            return "/mnt/e/s"

        return self.working_path

    @property
    def data_set_train_path(self):
        return os.path.join(self.data_set_root_path, "train")

    @property
    def data_set_test_path(self, ):
        return os.path.join(self.data_set_root_path, "test")

    @property
    def train_csv(self, ):
        return os.path.join(self.data_set_root_path, "train.csv")

    @property
    def submission_csv(self, ):
        if self._submission_csv:
            return self._submission_csv
        return os.path.join(self.output_path, "whale.submission.csv")

    @submission_csv.setter
    def submission_csv(self, submission_csv):
        self._submission_csv = submission_csv

    @property
    def bounding_boxes_csv(self, ):
        return os.path.join(self.working_path, "bounding_boxes.csv")

    @property
    def ckpt_pretrained_rv250(self, ):
        if self._mode == "default" or self._mode == "fake":
            return get_wsl_path(
                "E:/frkhit/Download/AI/pre-trained-model/resnet_v2_50.ckpt"
            )
        return os.path.join(self.working_path, "resnet_v2_50.ckpt")

    def init_bounding_boxes(self):
        if not os.path.exists(self.bounding_boxes_csv):
            download_big_file("https://raw.githubusercontent.com/frkhit/file_servers/master/bounding_boxes.csv",
                              self.bounding_boxes_csv)
            if not os.path.exists(self.bounding_boxes_csv):
                raise ValueError("fail to download bounding_boxes_csv!")

    def init_image_net_model(self):
        working_dir = os.path.dirname(self.ckpt_pretrained_mv2)
        if not os.path.exists(working_dir):
            os.mkdir(working_dir)

        if not os.path.exists(self.ckpt_pretrained_mv2 + ".index"):
            tar_file = "mobilenet_v2_1.0_224.tgz"
            download_big_file("https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz",
                              os.path.join(working_dir, tar_file))
            if not os.path.exists(tar_file):
                raise ValueError("fail to download pretrained model!")

            raw_dir = os.getcwd()
            try:
                os.chdir(working_dir)
                cmd = subprocess.Popen(["tar", "-xvf", tar_file])
                cmd.wait()
                if not os.path.exists(self.ckpt_pretrained_mv2 + ".index"):
                    raise ValueError("fail to download pretrained model!")
            finally:
                os.chdir(raw_dir)

    def download_resnet_v2_50(self):
        working_dir = os.path.dirname(self.ckpt_pretrained_rv250)
        if not os.path.exists(working_dir):
            os.mkdir(working_dir)

        if not os.path.exists(self.ckpt_pretrained_rv250):
            tar_file = "resnet_v2_50_2017_04_14.tar.gz"
            download_big_file("http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz",
                              os.path.join(working_dir, tar_file))
            if not os.path.exists(tar_file):
                raise ValueError("fail to download pretrained model!")

            raw_dir = os.getcwd()
            try:
                os.chdir(working_dir)
                cmd = subprocess.Popen(["tar", "-xzvf", tar_file])
                cmd.wait()
                if not os.path.exists(self.ckpt_pretrained_rv250):
                    raise ValueError("fail to download pretrained model!")
            finally:
                os.chdir(raw_dir)

    def init_all_path(self):
        members = [attr for attr in dir(self) if not attr.startswith("__")]
        for member in members:
            if (member.endswith("path") or member.endswith("dir")) and not member.endswith("init_all_path"):
                tmp_dir = getattr(self, member)
                if callable(tmp_dir):
                    tmp_dir = tmp_dir()

                if not os.path.exists(tmp_dir):
                    try:
                        os.mkdir(tmp_dir)
                    except Exception as e:
                        logging.error(e)

    @property
    def is_kaggle(self):
        return bool(self._mode == "kaggle")

    def create_fake_data(self, x: int = 1):
        raw_path_manager = PathManager()
        whales = pd.read_csv(raw_path_manager.train_csv)
        train_info = {}
        for index, image_name in enumerate(whales.Image):
            train_info[image_name] = whales.Id[index]

        shutil.rmtree(self.data_set_root_path)
        os.mkdir(self.data_set_root_path)

        # test path
        os.mkdir(self.data_set_test_path)
        test_image_list = list_files(raw_path_manager.data_set_test_path)
        for image_file in random.choices(test_image_list, k=10 * x):
            shutil.copy(image_file, os.path.join(self.data_set_test_path, os.path.basename(image_file)))

        # train path
        os.mkdir(self.data_set_train_path)
        raw_image_list = list_files(raw_path_manager.data_set_train_path)
        image_id_vs_image_file = {
            os.path.basename(image_file): image_file for image_file in raw_image_list}

        class_id_vs_image_list = {}

        for image_id, class_id in train_info.items():
            class_id_vs_image_list.setdefault(class_id, []).append(image_id_vs_image_file[image_id])

        anchor_class_id_list = [class_id for class_id, _image_list in class_id_vs_image_list.items() if
                                len(_image_list) > 1]
        all_anchor_class_id_list = list(train_info.values())
        try:
            anchor_class_id_list.remove(WhaleDataUtils.blank_class_id)
        except Exception:
            pass

        to_move_list = []
        for class_id in random.choices(anchor_class_id_list, k=2 * x):
            to_move_list.extend(class_id_vs_image_list[class_id][:3 * x])
        for class_id in random.choices(all_anchor_class_id_list, k=3 * x):
            to_move_list.extend(class_id_vs_image_list[class_id][:3 * x])
        to_move_list.extend(class_id_vs_image_list[WhaleDataUtils.blank_class_id][:4 * x])
        for image_file in to_move_list:
            shutil.copy(image_file, os.path.join(self.data_set_train_path, os.path.basename(image_file)))
        with open(self.train_csv, "w") as f:
            f.write("Image,Id\n")
            for image_file in to_move_list:
                image_id = os.path.basename(image_file)
                class_id = train_info[image_id]
                f.write("{},{}\n".format(image_id, class_id))

        # bounding box
        self.init_bounding_boxes()


class WhaleDataUtils(object):
    blank_class_id = "new_whale"
    not_gray_str = ".ng"

    def __init__(self, path_manager: PathManager, loss_hook: LossStepHookForTrain = None,
                 gen_data_setting: dict = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._file = "./{}.pkl".format(self.__class__.__name__)
        self.path_manager = path_manager
        self.loss_hook = loss_hook
        self.file_cache = FileCache(pickle_file=self._file)
        self.dimension = 512
        self.margin = 1.0
        self.gen_data_setting = {
            "x_train_num": 1,
            "ignore_blank_prob": 0.95,
            "ignore_single_prob": 0.2,
            "stop_calc_feature": False,
            "gen_data_by_random_prob": 0,
            "use_norm_when_calc_apn": True,
        }
        if gen_data_setting:
            self.gen_data_setting.update(gen_data_setting)

        _cache = None
        if os.path.exists(self._file):
            _cache = self.file_cache.get("cache")

        if not _cache:
            self._cache = ([], [], {}, {}, {}, [])
        else:
            self._cache = _cache

        self._baseline = None
        self._to_use_random = False

    def _update_cache(self):
        self.file_cache.set("cache", self._cache)

    def clear_cache(self):
        self.file_cache.cache.clear()
        self.file_cache.set("__key__", "value", ttl=10)  # flush

        _pkl_file = self._file + ".feature.h5"
        if os.path.exists(_pkl_file):
            os.remove(_pkl_file)

    def load_train_info(self) -> dict:
        if self._cache and self._cache[3]:
            return self._cache[3]

        whales = pd.read_csv(self.path_manager.train_csv)
        info = self._cache[3]
        for index, image_name in enumerate(whales.Image):
            info[image_name] = whales.Id[index]
        self.logger.info("there are {} record in train csv!".format(len(info)))
        self._update_cache()
        return info

    def load_train_label_info(self) -> dict:
        if self._cache and self._cache[4]:
            return self._cache[4]

        class_id_list = [class_id for class_id in self.load_train_info().values()]
        class_id_list = list(set(class_id_list))
        class_id_list.sort()

        info = self._cache[4]
        for index, class_id in enumerate(class_id_list):
            info[class_id] = index
        self.logger.info("there are {} classes in train set!".format(len(info)))
        self._update_cache()
        return info

    def load_boxes_info(self):
        if self._cache and self._cache[2]:
            return self._cache[2]

        boxes = pd.read_csv(self.path_manager.bounding_boxes_csv)
        info = self._cache[2]
        for index, image_id in enumerate(boxes.Image):
            info[image_id] = (boxes.x0[index], boxes.y0[index], boxes.x1[index], boxes.y1[index])
        self.logger.info("there are {} record in boxes csv!".format(len(info)))

        # check
        file_list = self.list_train_image_file().copy()  # fix bug
        file_list.extend(self.list_test_image_file())
        for image_file in file_list:
            base_name = os.path.basename(image_file)
            if base_name not in info:
                raise ValueError("{} not in boxes info!".format(image_file))

        self._update_cache()
        return self._cache[2]

    def get_boxes(self, image_file) -> (int, int, int, int):
        base_name = os.path.basename(image_file)
        return self.load_boxes_info()[base_name]

    def list_train_image_file(self, ignore_blank: bool = False) -> list:
        if not self._cache[0]:
            image_list = list_files(self.path_manager.data_set_train_path)
            result_list = [file_name for file_name in image_list if file_name.endswith(".jpg")]
            assert len(result_list) == len(image_list)
            self.logger.info("there are {} jpg in train path!".format(len(result_list)))
            self._cache[0].clear()
            self._cache[0].extend(result_list)

            self._update_cache()

        # ignore blank image_file
        if not self._cache[5]:
            for _image_file in self._cache[0]:
                if self.get_class_id(_image_file) != self.blank_class_id:
                    self._cache[5].append(_image_file)

            self._update_cache()

        if ignore_blank is False:
            return self._cache[0]

        return self._cache[5]

    def list_test_image_file(self) -> list:
        if self._cache and self._cache[1]:
            return self._cache[1]

        image_list = list_files(self.path_manager.data_set_test_path)
        result_list = [file_name for file_name in image_list if file_name.endswith(".jpg")]
        assert len(result_list) == len(image_list)
        self.logger.info("there are {} jpg in test path!".format(len(result_list)))
        self._cache[1].clear()
        self._cache[1].extend(result_list)
        self._update_cache()
        return self._cache[1]

    def get_class_id(self, image_file: str) -> str:
        base_name = os.path.basename(image_file)
        if base_name.endswith(self.not_gray_str):
            base_name = base_name[:-len(self.not_gray_str)]
        return self.load_train_info()[base_name]

    def list_class_id_with_multi_instance(self) -> dict:
        info = {}
        for class_id in self.load_train_info().values():
            info.setdefault(class_id, 0)
            info[class_id] += 1

        key_to_remove_list = []
        for key, count in info.items():
            if count <= 1:
                key_to_remove_list.append(key)

        for key in key_to_remove_list:
            info.pop(key)

        return info

    def get_label(self, image_file: str) -> int:
        class_id = self.get_class_id(image_file)
        return self.load_train_label_info()[class_id]

    def submit_test_result(self, test_image_list: list, result_list: list, feature_list: list = None):
        if len(test_image_list) != len(result_list):
            self.logger.error("len(test_image_list) != len(result_list)!")
        if feature_list and len(feature_list) != len(test_image_list):
            self.logger.error("len(test_image_list) != len(feature_list)!")

        with open(self.path_manager.submission_csv, "w") as f:
            f.write("Image,Id\n")
            for index, image_file in enumerate(test_image_list):
                f.write("{},{}\n".format(os.path.basename(image_file),
                                         " ".join([class_id for class_id in result_list[index]])))
            self.logger.info("success to save result in {}".format(self.path_manager.submission_csv))

        if feature_list and feature_list[0]:
            self.logger.info("saving feature list ...")
            with open(self.path_manager.submission_csv + ".feature", "w") as f:
                for index, image_file in enumerate(test_image_list):
                    f.write("{},{}\n".format(os.path.basename(image_file), feature_list[index]))

                self.logger.info("success to save feature result in {}".format(
                    self.path_manager.submission_csv + ".feature"))

    def get_random_baseline(self) -> (float, float):
        """
            0.384, 0.381
        Returns:
            (float, float):
        """
        if self._baseline is None:
            class_count_dict = self.list_class_id_with_multi_instance()
            class_list = sorted(class_count_dict.items(), key=lambda x: x[1], reverse=True)
            class_id_list = [_class_id for (_class_id, _count) in class_list]
            random_class_id_list = [self.blank_class_id]
            for _class_id in class_id_list:
                if _class_id not in random_class_id_list:
                    random_class_id_list.append(_class_id)
                    if len(random_class_id_list) >= 5:
                        break

            labels = [self.get_class_id(image_file) for image_file in self.list_train_image_file()]
            class_id_result_list = [random_class_id_list] * len(labels)
            total_map_5 = map_per_set(labels=labels, predictions=class_id_result_list, k=5)
            total_map_1 = map_per_set(labels=labels, predictions=class_id_result_list, k=1)
            self._baseline = (total_map_5, total_map_1)

        return self._baseline

    def list_debug_image_list(self, count: int = 128) -> list:
        all_image_list, class_id_vs_image_list, all_class_id_list, anchor_class_id_list, single_class_id_list =             self.get_basic_info(self.list_train_image_file())
        if count >= len(all_image_list):
            return all_image_list

        # info
        single_blank_class_files = []
        anchor_class_id_set = set(anchor_class_id_list)
        for class_id, file_list in class_id_vs_image_list.items():
            if class_id not in anchor_class_id_set:
                single_blank_class_files.extend(file_list)

        image_list = []

        # anchor_file_count
        anchor_file_count = int(count * (1 - len(single_blank_class_files) / len(all_image_list)))
        if (anchor_file_count + len(single_blank_class_files)) > len(all_image_list):
            anchor_file_count = len(all_image_list) - len(single_blank_class_files)

        # multi class
        for class_id in anchor_class_id_list:
            image_list.extend(class_id_vs_image_list[class_id])
            if len(image_list) >= anchor_file_count:
                break

        while len(image_list) > anchor_file_count:
            image_list.pop(-1)

        # add single class and blank class
        image_list.extend(random_choice(single_blank_class_files, k=count - anchor_file_count, unique=True))

        assert len(image_list) == count

        return image_list

    def get_basic_info(self, all_image_list: list = None) -> (list, dict, list, list, list):
        """
            list train info
        Args:

        Returns:

        """
        if all_image_list is None:
            all_image_list = list(self.list_train_image_file(ignore_blank=False))

        image_id_vs_image_file = {os.path.basename(image_file): image_file for image_file in all_image_list}

        class_id_vs_image_list = {}

        for image_id, class_id in self.load_train_info().items():
            if image_id in image_id_vs_image_file:
                class_id_vs_image_list.setdefault(class_id, []).append(image_id_vs_image_file[image_id])

        anchor_class_id_list = [class_id for class_id, _image_list in class_id_vs_image_list.items() if
                                len(_image_list) > 1]
        all_class_id_list = list(class_id_vs_image_list.keys())
        single_class_id_list = list(set(all_class_id_list) - set(anchor_class_id_list))
        try:
            anchor_class_id_list.remove(self.blank_class_id)
        except Exception:
            pass
        try:
            single_class_id_list.remove(self.blank_class_id)
        except Exception:
            pass

        return all_image_list, class_id_vs_image_list, all_class_id_list, anchor_class_id_list, single_class_id_list

    def get_file_list(self, is_training: bool = False, shuffle: bool = False, num_epochs: int = 1,
                      batch_size: int = None, online_batch_count: int = 0) -> list:

        def get_random_batch_list(batch_count: int, add_blank_class: bool, class_id_vs_image_list: dict,
                                  anchor_class_id_list: list, all_image_list: list, single_class_id_list: list,
                                  anchor_unique: bool, single_unique: bool, ) -> list:
            batch_list = []
            # add one image from new blank
            if add_blank_class:
                batch_list.append(random.choice(class_id_vs_image_list[self.blank_class_id]))

            # add anchor(has more than one image)
            chosen_class_id_list = random_choice(anchor_class_id_list, k=batch_count, unique=anchor_unique)
            for class_id in chosen_class_id_list:
                if len(class_id_vs_image_list[class_id]) >= 2 * online_batch_count:
                    batch_list.extend(
                        random_choice(class_id_vs_image_list[class_id], k=2 * online_batch_count, unique=True))
                else:
                    batch_list.extend(class_id_vs_image_list[class_id])

                if (batch_count - len(batch_list)) < online_batch_count:
                    break

            # add single class
            if len(batch_list) < batch_count:
                if single_class_id_list:
                    batch_list.extend([
                        class_id_vs_image_list[class_id][0] for class_id in
                        random_choice(single_class_id_list, k=batch_count - len(batch_list),
                                      unique=single_unique)])
                else:
                    batch_list.extend(
                        random_choice(all_image_list, k=batch_count - len(batch_list), unique=anchor_unique))
            else:
                while len(batch_list) > batch_count:
                    batch_list.pop(-1)

            return batch_list

        def create_file_list_by_random() -> list:
            all_image_list, class_id_vs_image_list, all_class_id_list, anchor_class_id_list, single_class_id_list =                 self.get_basic_info()
            total_count = int(num_epochs * len(all_image_list) * self.gen_data_setting["x_train_num"])
            batch_count = total_count // batch_size
            multi_batch_list = []

            _unique = True if len(anchor_class_id_list) >= batch_size else False
            _unique_single = True if len(single_class_id_list) >= batch_size else False
            _has_blank_class = True if class_id_vs_image_list.get(self.blank_class_id) else False

            for i in range(batch_count):
                batch_list = get_random_batch_list(
                    batch_count=batch_size, add_blank_class=_has_blank_class,
                    class_id_vs_image_list=class_id_vs_image_list, anchor_class_id_list=anchor_class_id_list,
                    all_image_list=all_image_list, single_class_id_list=single_class_id_list,
                    anchor_unique=_unique, single_unique=_unique_single,
                )

                random.shuffle(batch_list)
                multi_batch_list.append(batch_list)

            return multi_batch_list

        def create_file_list_by_distance() -> list:
            # calc distance
            feature_arr, file_arr = self._load_feature(create_if_not_exist=False)
            if feature_arr is None:
                self.logger.warning(
                    "no feature in h5, use create_file_list_by_random instead of create_file_list_by_distance!")
                return create_file_list_by_random()

            if self.gen_data_setting.get("use_norm_when_calc_apn", False):
                feature_arr, norm_float = NormType.all.normalize_and_return_norm(feature_arr)
                self.logger.info("norm of feature is {}<type: {}>".format(norm_float, type(norm_float)))
                assert isinstance(norm_float, float)
                margin = self.margin / norm_float
            else:
                margin = self.margin

            _file_list = []
            for i in range(file_arr.shape[0]):
                _file_list.append(file_arr[i].decode("utf-8"))

            all_image_list, class_id_vs_image_list, all_class_id_list, anchor_class_id_list, single_class_id_list =                 self.get_basic_info(_file_list)
            total_count = int(num_epochs * len(all_image_list) * self.gen_data_setting["x_train_num"])
            batch_count = total_count // batch_size
            multi_batch_list = []

            single_class_id_set = set(single_class_id_list)
            anchor_index_end = -1
            _anchor_search_end = False
            labels = np.zeros(shape=file_arr.shape, dtype=np.int)
            for i in range(file_arr.shape[0]):
                _file_name = file_arr[i].decode("utf-8")
                labels[i] = self.get_label(_file_name)
                if not _anchor_search_end:
                    class_id = self.get_class_id(_file_name)
                    if class_id in single_class_id_set:
                        _anchor_search_end = True
                    else:
                        anchor_index_end = i

            anchor_feature = feature_arr[:anchor_index_end + 1]
            _apn_np_start = time.time()
            apn_list = get_triplet_pair_np(anchor_feature, all_feature=feature_arr, all_label=labels,
                                           margin=margin, logger=self.logger)
            _apn_set = set()
            for (a, p, n) in apn_list:
                _apn_set.add(a)
                _apn_set.add(p)
                _apn_set.add(n)

            self.logger.info(
                "get {} hardest-apn-pairs in {} files, time cost {}s. Unique file in apn pairs is {}.".format(
                    len(apn_list), feature_arr.shape[0], time.time() - _apn_np_start, len(_apn_set)))

            _unique = True if len(anchor_class_id_list) >= batch_size else False
            _unique_single = True if len(single_class_id_list) >= batch_size else False
            _has_blank_class = True if class_id_vs_image_list.get(self.blank_class_id) else False
            _blank_file_id_set = set()
            if _has_blank_class:
                _blank_file_label = self.get_label(class_id_vs_image_list[self.blank_class_id][0])
                for i in range(labels.shape[0]):  # todo == len(labels)
                    if labels[i] == _blank_file_label:
                        _blank_file_id_set.add(i)

            assert len(_blank_file_id_set) == len(class_id_vs_image_list.get(self.blank_class_id, []))

            if len(apn_list) <= batch_size or (len(_apn_set - _blank_file_id_set) + 1) <= batch_size:
                self.logger.warning("apn list is too small, use create_file_list_by_random instead!")
                return create_file_list_by_random()

            tmp_apn_list = []
            for i in range(batch_count):
                batch_file_id_id = set()
                _blank_exists = False
                while True:
                    if len(tmp_apn_list) == 0:
                        tmp_apn_list.extend(list(apn_list))
                        random.shuffle(tmp_apn_list)

                    (a, p, n) = tmp_apn_list.pop(-1)
                    for _file_id in [a, p, n]:
                        if _file_id in _blank_file_id_set:
                            if _blank_exists is False:
                                batch_file_id_id.add(_file_id)
                                _blank_exists = True
                            else:
                                continue
                        else:
                            batch_file_id_id.add(_file_id)

                    if len(batch_file_id_id) >= batch_size:
                        break

                batch_list = [file_arr[file_id].decode("utf-8") for file_id in batch_file_id_id]

                assert len(batch_list) >= batch_size

                batch_list = batch_list[:batch_size]

                random.shuffle(batch_list)
                multi_batch_list.append(batch_list)

            return multi_batch_list

        if is_training:
            if online_batch_count > 0:
                self.logger.info("using create_file_list_by_distance...")

                if self._to_use_random:
                    _multi_batch_list = create_file_list_by_random()
                    self._to_use_random = False
                else:
                    _multi_batch_list = create_file_list_by_distance()
                # shuffle
                if shuffle is True:
                    random.shuffle(_multi_batch_list)

                image_file_list = []
                for _batch_list in _multi_batch_list:
                    image_file_list.extend(_batch_list)

                return image_file_list
            else:
                return self.list_train_image_file()

        return self.list_test_image_file()

    def _list_file_to_calc_feature(self) -> list:
        if random.random() < self.gen_data_setting["ignore_blank_prob"]:
            all_image_list = list(self.list_train_image_file(ignore_blank=True))

            if random.random() < self.gen_data_setting["ignore_single_prob"]:
                _, class_id_vs_image_list, _, anchor_class_id_list, _ =                     self.get_basic_info(all_image_list)

                all_file_list = []
                anchor_class_id_set = set(anchor_class_id_list)
                for class_id, _file_list in class_id_vs_image_list.items():
                    if class_id in anchor_class_id_set:
                        all_file_list.extend(_file_list)

                self.logger.info("list train image file [ignore_single, ignore_blank] to calc feature!")
                return all_file_list
            else:
                self.logger.info("list train image file [ignore_blank] to calc feature!")
                return all_image_list

        self.logger.info("list all train image file to calc feature!")
        return list(self.list_train_image_file())

    def calc_feature(self, callback, epoch_num: int = None):
        """
            调用callback计算feature
        Args:
            epoch_num (int): total count
            callback:
        """
        if self.gen_data_setting.get("stop_calc_feature", False):
            self.logger.info("no need to calc feature: stop_calc_feature is True!")
            return

        if epoch_num is not None and epoch_num == 0 and os.path.exists(self._file + ".feature.h5"):
            self.logger.info("no need to calc feature: feature file exist and epoch_num == 0!")
            return

        if random.random() < self.gen_data_setting.get("gen_data_by_random_prob", 0):
            self.logger.info("no need to calc feature: going to gen data by random!")
            self._to_use_random = True
            return

        file_list = self._list_file_to_calc_feature()
        if not file_list:
            self.logger.info("no need to calc feature!")
            return

        self.logger.info("calculating feature of {} file...".format(len(file_list)))
        feature_list, feature_file_list = callback(file_list, mode=ProcessMode.train)
        self.logger.info("success to calculate feature of {} file!".format(len(feature_file_list)))

        self.update_feature(feature_list, feature_file_list, is_training=True)

    def update_feature(self, feature_list, file_list, is_training: bool):
        """

        Returns:
            object:
        """
        if not is_training:
            self.logger.info("give up update-feature for no train mode!")
            return

        if not feature_list:
            return

        # 保存 feature
        self._save_feature(feature_list, file_list)

    def _save_feature(self, feature_list: list, file_list: list):
        # clear exist feature
        _pkl_file = self._file + ".feature.h5"
        if os.path.exists(_pkl_file):
            os.remove(_pkl_file)

        # feature save
        all_image_list, class_id_vs_image_list, all_class_id_list, anchor_class_id_list, single_class_id_list =             self.get_basic_info(all_image_list=file_list)

        all_file_list = []
        for class_id in anchor_class_id_list:
            all_file_list.extend(class_id_vs_image_list[class_id])

        for class_id in single_class_id_list:
            all_file_list.extend(class_id_vs_image_list[class_id])

        all_file_list.extend(class_id_vs_image_list.get(self.blank_class_id, []))
        assert len(all_image_list) == len(all_file_list)

        # sort file index
        sort_index_list = []
        _dict = {}
        for i, file_name in enumerate(file_list):
            _dict[file_name] = i
        for file_name in all_file_list:
            sort_index_list.append(_dict[file_name])

        # feature
        file_arr = np.asarray([f.encode("utf-8") for f in all_file_list], dtype=np.string_)
        feature_arr = np.zeros(shape=(len(all_file_list), self.dimension), dtype=np.float32)
        # update feature
        for index, key_index in enumerate(sort_index_list):
            feature_arr[index][:] = feature_list[key_index].reshape((self.dimension,))

        _pkl_file = self._file + ".feature.h5"
        store_data_in_h5file(_pkl_file, [feature_arr, file_arr], key_list=["feature", "file"])

    def _load_feature(self, create_if_not_exist: bool = True) -> (np.ndarray, np.ndarray):
        """

        Returns:
            feature, file, is_created
        """
        _pkl_file = self._file + ".feature.h5"
        if os.path.exists(_pkl_file):
            feature, file_arr = None, None
            try:
                feature, file_arr = load_data_from_h5file(_pkl_file, key_list=["feature", "file"])
            except Exception as e:
                os.remove(_pkl_file)
                self.logger.error(e)

            if feature is not None:
                self.logger.info("feature shape is {}, file_arr shape is {}".format(feature.shape, file_arr.shape))
                assert feature.shape[1] == self.dimension
                assert file_arr.shape[0] == feature.shape[0]
                return feature, file_arr

        if not create_if_not_exist:
            return None, None

        all_image_list, class_id_vs_image_list, all_class_id_list, anchor_class_id_list, single_class_id_list =             self.get_basic_info()
        all_file_list = []
        for class_id in anchor_class_id_list:
            all_file_list.extend(class_id_vs_image_list[class_id])

        for class_id in single_class_id_list:
            all_file_list.extend(class_id_vs_image_list[class_id])

        all_file_list.extend(class_id_vs_image_list[self.blank_class_id])
        assert len(all_image_list) == len(all_file_list)

        file_arr = np.asarray([f.encode("utf-8") for f in all_file_list], dtype=np.string_)
        feature = np.zeros(shape=(len(all_file_list), self.dimension))

        return feature, file_arr





class WhaleRankingUtils(object):
    def __init__(self, data_utils: WhaleDataUtils, top_k: int = 5):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.top_k = top_k
        self.data_utils = data_utils

    def simple_rank(self, result_list: list, distance_cutoff: float = None, only_distance_fit: bool = False) -> list:
        """

        Args:
            result_list: list, consist of (distance, class_id, image_file) elements
            distance_cutoff: float
            only_distance_fit: bool

        Returns:
            rank_list: list, like, [class_id_1, class_id_2, ... ]
        """
        # deal with image_result_list
        _class_id_list = []
        _class_id_set = set()

        if distance_cutoff is None:
            for result in result_list:
                class_id = result[1]
                if class_id not in _class_id_set:
                    _class_id_list.append(class_id)
                    _class_id_set.add(class_id)

            if len(_class_id_list) < self.top_k:
                if self.data_utils.blank_class_id not in _class_id_set:
                    _class_id_list.append(self.data_utils.blank_class_id)
                    _class_id_set.add(self.data_utils.blank_class_id)

        else:
            if result_list[0][0] > distance_cutoff:
                _class_id_list.append(self.data_utils.blank_class_id)
                if only_distance_fit:
                    return _class_id_list
            else:
                _class_id_list.append(result_list[0][1])
            _class_id_set.add(_class_id_list[0])

            for result in result_list[1:]:
                if result[0] <= distance_cutoff:
                    class_id = result[1]
                    if class_id not in _class_id_set:
                        _class_id_list.append(class_id)
                        _class_id_set.add(class_id)
                else:
                    if only_distance_fit:
                        return _class_id_list
                    else:
                        break

            if len(_class_id_list) < self.top_k:
                if self.data_utils.blank_class_id not in _class_id_set:
                    _class_id_list.append(self.data_utils.blank_class_id)
                    _class_id_set.add(self.data_utils.blank_class_id)

            if len(_class_id_list) < self.top_k:
                for result in result_list[1:]:
                    if result[0] > distance_cutoff:
                        class_id = result[1]
                        if class_id not in _class_id_set:
                            _class_id_list.append(class_id)
                            _class_id_set.add(class_id)

        return _class_id_list

    def recreate_class_id_list(self, cutoff: float, class_id_list: list, feature_list: list) -> list:
        new_class_id_list = []
        count = 0
        for i, class_list in enumerate(class_id_list):
            _new_class_list = list(class_list)
            if feature_list[i][0][0] <= cutoff or _new_class_list[0] == self.data_utils.blank_class_id:
                new_class_id_list.append(_new_class_list)
                continue

            try:
                found_index = _new_class_list.index(self.data_utils.blank_class_id)
                if found_index > -1:
                    _new_class_list.pop(found_index)
            except ValueError:
                pass

            _new_class_list.insert(0, self.data_utils.blank_class_id)
            new_class_id_list.append(_new_class_list)
            count += 1

        self.logger.info("recreate_class_id_list change {}/{} record".format(count, len(class_id_list)))
        return new_class_id_list

    def get_similar_cutoff(self, validating_labels: list, validating_feature_list: list) -> float:
        # calc similar distance cutoff
        _distance_list = []
        _, class_id_vs_image_list, _, anchor_class_id_list, _ =             self.data_utils.get_basic_info(self.data_utils.list_train_image_file())

        anchor_class_set = set(anchor_class_id_list)
        for i, class_id in enumerate(validating_labels):
            if class_id not in anchor_class_set:
                continue

            for (_distance, _class_id, _file_name) in validating_feature_list[i]:
                if class_id == _class_id:
                    _distance_list.append(_distance)

        self.logger.info("distance list: len is {}, first 5 is {}".format(len(_distance_list), _distance_list[:5]))
        distance_mean, distance_std = show_distance_dense_plot(
            np.asarray(_distance_list), self.data_utils.path_manager.submission_csv + ".distance.jpg")

        distance_similar_cutoff = distance_mean + 3 * distance_std
        self.logger.info("distance_similar_cutoff is {}".format(distance_similar_cutoff))

        return distance_similar_cutoff

    @staticmethod
    def get_feature_list(feature_file: str) -> (list, list):
        feature_list = []
        file_id_list = []
        with open(feature_file, "r", encoding="utf-8") as f:
            for line in f:
                if len(line) < 2:
                    continue

                line = line.rstrip()
                file_id_list.append(line.split(",")[0])
                feature_list.append(eval(line[line.find(",") + 1:]))

        return file_id_list, feature_list

    def analyze_validating_result(self, feature_file: str):
        # read feature list
        file_id_list, feature_list = self.get_feature_list(feature_file)

        class_id_result_list = [self.simple_rank(feature)[:self.top_k] for feature in feature_list]

        # calc map@5
        labels = self.list_class_id(file_id_list)
        total_map_5 = map_per_set(labels=labels, predictions=class_id_result_list, k=5)
        total_map_1 = map_per_set(labels=labels, predictions=class_id_result_list, k=1)
        self.logger.info("validating data result is: map@5 is {},  map@1 is {}!".format(total_map_5, total_map_1))

        # calc similar distance cutoff
        distance_similar_cutoff = self.get_similar_cutoff(labels, feature_list)

        # find error
        for i, image_id in enumerate(file_id_list):
            map_5 = map_per_image(label=labels[i], predictions=class_id_result_list[i], k=5)
            if map_5 < 1.0:
                # error
                self.logger.info("map@5 is {}\nTrue: {}, Predict with cutoff is {}\n{}\n".format(
                    map_5, labels[i],
                    self.data_utils.blank_class_id if feature_list[i][0][0] > distance_similar_cutoff else
                    feature_list[i][0][1],
                    feature_list[i])
                )

        # create new class id result
        new_class_id_result = self.recreate_class_id_list(
            cutoff=distance_similar_cutoff, class_id_list=class_id_result_list, feature_list=feature_list)

        total_map_5 = map_per_set(labels=labels, predictions=new_class_id_result, k=5)
        total_map_1 = map_per_set(labels=labels, predictions=new_class_id_result, k=1)
        self.logger.info(
            "validating data result with cutoff is: map@5 is {},  map@1 is {}!".format(total_map_5, total_map_1))

    def compare_submission_file(self, file_1, file_2):
        def _compare_same_count(_result_1, _result_2, info):
            assert set(_result_1.keys()) == set(_result_2.keys())

            all_same_count = 0
            for image_id, result_list in _result_1.items():
                if ",".join(result_list[:self.top_k]) == ",".join(_result_2[image_id][:self.top_k]):
                    all_same_count += 1

            self.logger.info("[{}] same count is {}/{}".format(info, all_same_count, len(_result_1)))

        def _compare_top1_count(_result_1, _result_2, info):
            assert set(_result_1.keys()) == set(_result_2.keys())

            all_same_count = 0
            for image_id, result_list in _result_1.items():
                if result_list[0] == _result_2[image_id][0]:
                    all_same_count += 1

            self.logger.info("[{}] same top1 count is {}/{}".format(info, all_same_count, len(_result_1)))

        def _compare_map(_result_1, _result_2, info):
            assert set(_result_1.keys()) == set(_result_2.keys())

            _file_id_list = list(_result_1.keys())
            _file_id_list.sort()
            labels = self.list_class_id(_file_id_list)

            # 1
            _class_id_result_list_1 = [_result_1[_image_id] for _image_id in _file_id_list]
            map5_1 = map_per_set(labels=labels, predictions=_class_id_result_list_1, k=5)
            map1_1 = map_per_set(labels=labels, predictions=_class_id_result_list_1, k=1)

            # 2
            _class_id_result_list_2 = [_result_2[_image_id] for _image_id in _file_id_list]
            map5_2 = map_per_set(labels=labels, predictions=_class_id_result_list_2, k=5)
            map1_2 = map_per_set(labels=labels, predictions=_class_id_result_list_2, k=1)

            self.logger.info("[{}] map@5 is {}/{}, map@1 is {}/{}".format(
                info,
                get_pretty_float(map5_1, count=3),
                get_pretty_float(map5_2, count=3),
                get_pretty_float(map1_1, count=3),
                get_pretty_float(map1_2, count=3),
            ))

        # true same count
        result_1, result_2 = self.read_submission_file(file_1), self.read_submission_file(file_2)
        result_1_x, result_2_x = self._get_result_from_feature_file(file_1), self._get_result_from_feature_file(file_2)
        result_1_true, result_2_true = self._get_result_by_cutoff(file_1), self._get_result_by_cutoff(file_2)
        self.logger.info("Start to compare:\n\n")

        # map
        self.logger.info("compare map5, map1")
        _compare_map(result_1, result_2, "map, raw, resenet vs siamese")
        _compare_map(result_1, result_1_x, "map, raw, resenet, csv vs feature")
        _compare_map(result_2, result_2_x, "map, raw, siamese, csv vs feature")
        _compare_map(result_1_true, result_2_true, "map, true, resenet vs siamese")
        _compare_map(result_1, result_1_true, "map, resent, raw vs true")
        _compare_map(result_2, result_2_true, "map, siamese, raw vs true")
        self.logger.info("End\n\n")

        # top5 same
        self.logger.info("compare top5")
        _compare_same_count(result_1, result_2, "top5, raw, resenet vs siamese")
        _compare_same_count(result_1_true, result_2_true, "top5, true, resenet vs siamese")
        _compare_same_count(result_1, result_1_true, "top5, resent, raw vs true")
        _compare_same_count(result_2, result_2_true, "top5, siamese, raw vs true")
        self.logger.info("End\n\n")

        # top1 same
        self.logger.info("compare top1")
        _compare_top1_count(result_1, result_2, "top1, raw, resenet vs siamese")
        _compare_top1_count(result_1_true, result_2_true, "top1, true, resenet vs siamese")
        _compare_top1_count(result_1, result_1_true, "top1, resent, raw vs true")
        _compare_top1_count(result_2, result_2_true, "top1, siamese, raw vs true")
        self.logger.info("End\n\n")

    @staticmethod
    def read_submission_file(csv_file: str) -> dict:
        submission = pd.read_csv(csv_file)
        _result = {}
        for index, _image_id in enumerate(submission.Image):
            _result[_image_id] = submission.Id[index].rstrip().split(" ")

        return _result

    def _get_result_from_feature_file(self, csv_file: str) -> dict:
        feature_file = csv_file + ".feature"
        assert os.path.exists(feature_file)
        cutoff, _, file_id_list, feature_list = self._read_feature_file(feature_file)

        _new_result = {}
        for _index, file_id in enumerate(file_id_list):
            _new_result[file_id] = self.simple_rank(
                feature_list[_index], distance_cutoff=cutoff, only_distance_fit=False)

        return _new_result

    def _get_result_by_cutoff(self, csv_file: str) -> dict:
        feature_file = csv_file + ".feature"
        assert os.path.exists(feature_file)
        cutoff, labels, file_id_list, feature_list = self._read_feature_file(feature_file)
        self.logger.info("cutoff is {}".format(get_pretty_float(cutoff, count=3)))

        _new_result = {}
        for _index, file_id in enumerate(file_id_list):
            _new_result[file_id] = self.simple_rank(
                feature_list[_index], distance_cutoff=cutoff, only_distance_fit=True)

        return _new_result

    def _read_feature_file(self, feature_file):
        assert os.path.exists(feature_file)
        file_id_list, feature_list = self.get_feature_list(feature_file)
        labels = self.list_class_id(file_id_list)
        cutoff = self.get_similar_cutoff(labels, feature_list)
        return cutoff, labels, file_id_list, feature_list

    @staticmethod
    def _calc_weight_score(feature_list, file_id_list, cutoff) -> dict:
        # todo simple score
        keep_count = 10
        cutoff_score = 5.0

        _new_result = {}
        for _index, file_id in enumerate(file_id_list):
            _new_result[file_id] = {}

            _sort_num = -1
            for (distance, class_id, file_name) in feature_list[_index]:
                if class_id not in _new_result[file_id]:
                    _sort_num += 1
                    _new_result[file_id][class_id] = keep_count - _sort_num
                    if distance < cutoff:
                        _new_result[file_id][class_id] += cutoff_score

                    if _sort_num >= keep_count:
                        break

        return _new_result

    @staticmethod
    def _get_result_by_weight_score(score_dict: dict, top_k: int = 5) -> dict:
        def _sort_by_value(info_dict: dict) -> list:
            _tuple = sorted(info_dict.items(), key=lambda x: x[1], reverse=True)
            return [_class_id for (_class_id, _weight) in _tuple]

        result_dict = {}
        for image_id, score in score_dict.items():
            result_dict[image_id] = _sort_by_value(score)[:top_k]

        return result_dict

    def combine_submission_file(self, validating_file_list: list, submission_file_list: list, output_file: str):
        def _create_final_result(_result_dict: dict):
            _file_list = []
            _result_list = []
            for _image_id, _class_list in _result_dict.items():
                _file_list.append(_image_id)
                _result_list.append(_class_list)
            return _file_list, _result_list

        def _update_weight_score(all_dict: dict, one_dict: dict, model_weight: float):
            for _image_id, _score_dict in one_dict.items():
                if _image_id not in all_dict:
                    all_dict[_image_id] = {}

                for _class_id, _score in _score_dict.items():
                    all_dict[_image_id].setdefault(_class_id, 0.0)
                    all_dict[_image_id][_class_id] += _score * model_weight

            return all_dict

        def eval_result(_result_dict: dict):
            _file_list, _result_list = _create_final_result(_result_dict)
            _labels = self.list_class_id(_file_list)
            _map5 = map_per_set(labels=_labels, predictions=_result_list, k=5)
            _map1 = map_per_set(labels=_labels, predictions=_result_list, k=1)
            return _map5, _map1

        # check input
        assert len(validating_file_list) == len(submission_file_list)
        for i, validating_submission_csv in enumerate(validating_file_list):
            predict_submission_csv = submission_file_list[i]
            assert ".".join(validating_submission_csv.split(".")[:3]) == ".".join(predict_submission_csv.split(".")[:3])
            assert os.path.exists(validating_submission_csv + ".feature")
            assert os.path.exists(predict_submission_csv + ".feature")

        total_predict_score_dict = {}
        total_validate_score_dict = {}
        total_eval_result_list = []

        for i, validating_submission_csv in enumerate(validating_file_list):
            predict_submission_csv = submission_file_list[i]

            # validate
            cutoff, _, v_file_id_list, v_feature_list = self._read_feature_file(validating_submission_csv + ".feature")
            v_model_weight = float("0.{}".format(validating_submission_csv.split(".")[-3]))

            validating_score_dict = self._calc_weight_score(
                feature_list=v_feature_list, file_id_list=v_file_id_list, cutoff=cutoff)
            total_validate_score_dict = _update_weight_score(
                total_validate_score_dict, validating_score_dict, model_weight=v_model_weight)

            # predict
            p_file_id_list, p_feature_list = self.get_feature_list(predict_submission_csv + ".feature")
            predict_score_dict = self._calc_weight_score(
                feature_list=p_feature_list, file_id_list=p_file_id_list, cutoff=cutoff)
            total_predict_score_dict = _update_weight_score(
                total_predict_score_dict, predict_score_dict, model_weight=v_model_weight)

            # eval validate
            _validate_result_dict = self.read_submission_file(validating_submission_csv)
            total_eval_result_list.append(eval_result(_validate_result_dict))

        # evaluate validating result
        _validate_result_dict = self._get_result_by_weight_score(total_validate_score_dict, top_k=self.top_k)
        total_eval_result_list.append(eval_result(_validate_result_dict))
        self.logger.info("after combine {} models: \nvalidate map@5 is {}/{}, \nvalidate map@1 is {}/{}\n\n".format(
            len(validating_file_list),
            total_eval_result_list[-1][0], [x[0] for x in total_eval_result_list[:-1]],
            total_eval_result_list[-1][1], [x[1] for x in total_eval_result_list[:-1]],
        ))

        # evaluate and save prediction result
        raw_submission = self.data_utils.path_manager.submission_csv
        try:
            predict_result_dict = self._get_result_by_weight_score(total_predict_score_dict, top_k=self.top_k)
            predict_file_list, predict_result_list = _create_final_result(predict_result_dict)
            self.data_utils.path_manager.submission_csv = output_file
            self.data_utils.submit_test_result(test_image_list=predict_file_list, result_list=predict_result_list)
            self.logger.info("save final submission result in {}".format(self.data_utils.path_manager.submission_csv))
        finally:
            self.data_utils.path_manager.submission_csv = raw_submission

    def debug_map_csv(self, csv_file):
        self.logger.info("debug map set from submission file...\n\n")
        result_dict = self.read_submission_file(csv_file)

        class_id_result_list = []
        file_id_list = []
        for _image_id, _result_list in result_dict.items():
            file_id_list.append(_image_id)
            class_id_result_list.append(_result_list)

        labels = self.list_class_id(file_id_list)

        self.logger.info("map5 is {}".format(map_per_set(labels=labels, predictions=class_id_result_list, k=5)))
        self.logger.info("map1 is {}".format(map_per_set(labels=labels, predictions=class_id_result_list, k=1)))

    def list_class_id(self, file_id_list) -> list:
        _, class_id_vs_image_list, _, _, _ = self.data_utils.get_basic_info(self.data_utils.list_train_image_file())
        class_id_vs_image_count = {
            class_id: len(_image_list) for class_id, _image_list in class_id_vs_image_list.items()}
        return [self.get_class_id(class_id_vs_image_count, _image_id) for _image_id in file_id_list]

    def get_class_id(self, class_id_vs_image_count, image_file_or_id) -> str:
        # ignore same image
        _class_id = self.data_utils.get_class_id(image_file_or_id)
        if class_id_vs_image_count[_class_id] > 1:
            return _class_id
        else:
            return self.data_utils.blank_class_id


class SimpleSearch(object):
    model = None

    def __init__(self, data_utils: WhaleDataUtils, shape: tuple, dimension: int, top_k: int = 5,
                 norm_type: NormType = NormType.none, more_search_top_k: int = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data_utils = data_utils
        self.faiss_index_path = os.path.join(self.data_utils.path_manager.working_path, self.__class__.__name__)
        self.shape = shape
        self.dimension = dimension
        self.top_k = top_k
        self._distance_cutoff = None
        self.manager = ImageIndexUtils(
            index_dir=self.faiss_index_path,
            dimension=self.dimension)
        self._norm_type = norm_type

        if more_search_top_k is None or more_search_top_k < 0:
            more_search_top_k = max(20, self.top_k)
        self._search_top_k = more_search_top_k + self.top_k

    def show_embedding(self, mode: ProcessMode):
        key_name = "train" if mode == ProcessMode.train else "test"
        list_image_file = self.data_utils.list_train_image_file if mode == ProcessMode.train             else self.data_utils.list_test_image_file
        log_dir = os.path.join(self.faiss_index_path, "logs_{}".format(key_name))

        self.logger.info("trying to show {} embedding...".format(key_name))

        feature_list, image_list = self.calc_or_load_feature(mode)
        self.logger.info("got {}-image feature: {}/{}".format(
            key_name, len(image_list), len(list_image_file()))
        )

        labels = []
        if mode == ProcessMode.train:
            for image_file in image_list:
                labels.append(self.data_utils.get_class_id(image_file))
        else:
            for image_file in image_list:
                labels.append(os.path.basename(image_file))

        show_embedding(feature_list=feature_list, labels=labels, log_dir=log_dir)
        self.logger.info("success to show {} embedding!".format(key_name))

    def list_features(self, mode: ProcessMode) -> (list, list):
        """
            input: ?X224X224X3
            feature: list of ?X7X7X320, files: list of str
        """

        raise NotImplementedError

    def calc_or_load_feature(self, mode: ProcessMode) -> (list, list):
        key_name = "train" if mode == ProcessMode.train else "test"

        # calc feature or load image feature
        feature_npy = os.path.join(self.faiss_index_path, "{}.npy.pkl".format(key_name))

        self.logger.info("{}ing image...".format(key_name))

        if not os.path.exists(feature_npy):
            self.logger.info("calculating feature of {} image...".format(key_name))
            feature_list, image_list = self.list_features(mode)
            self.logger.info("success to calculate feature of {} image!".format(key_name))
            with open(feature_npy, "wb") as f:
                pickle.dump((feature_list, image_list), f)
        else:
            self.logger.info("loading feature of {} image...".format(key_name))
            with open(feature_npy, "rb") as f:
                feature_list, image_list = pickle.load(f)
            self.logger.info("success to load feature of {} image!".format(key_name))

        return feature_list, image_list

    def train(self):
        if os.path.exists(self.manager.manager.faiss_index_file):
            self.logger.info("train before, no need to train again!")
            return

        # train: calc feature or load image feature
        feature_list, image_list = self.calc_or_load_feature(ProcessMode.train)
        self.logger.info("got train-image feature: {}/{}".format(
            len(image_list), len(self.data_utils.list_train_image_file())))

        image_info_list = []
        for index, image_file in enumerate(image_list):
            image_info_list.append({
                "index": index,
                "file": image_file,
                "class_id": self.data_utils.get_class_id(image_file)}
            )
        self.manager.add_images(feature_list, image_info_list=image_info_list)
        self.logger.info("success to train image!")

    def test(self):
        # calc feature
        feature_list, image_list = self.calc_or_load_feature(ProcessMode.test)
        if len(image_list) != len(self.data_utils.list_test_image_file()):
            self.logger.warning("got test-image file: {}/{}".format(
                len(image_list), len(self.data_utils.list_test_image_file())))
            assert len(image_list) % len(self.data_utils.list_test_image_file()) == 0

        self.logger.info("got test-image feature: {}/{}".format(
            len(image_list), len(self.data_utils.list_test_image_file())))

        # image search
        num_feature = len(image_list) // len(self.data_utils.list_test_image_file())
        self.logger.info("testing image...")
        test_image_list = [image_list[index * num_feature] for index in
                           range(len(self.data_utils.list_test_image_file()))]
        class_id_result_list, test_feature_list = self.search(
            image_list=test_image_list, feature_list=feature_list, return_feature_list=True)
        self.logger.info("success to test image!")

        # submit result
        assert len(class_id_result_list) == len(self.data_utils.list_test_image_file())
        self.data_utils.submit_test_result(test_image_list=test_image_list,
                                           result_list=class_id_result_list,
                                           feature_list=test_feature_list)
        if self._distance_cutoff is None:
            self.logger.info("submit predict result[without distance cutoff] in {}".format(
                self.data_utils.path_manager.submission_csv))
        else:
            self.logger.info("submit predict result[with distance cutoff {}] in {}".format(
                get_pretty_float(self._distance_cutoff, count=3), self.data_utils.path_manager.submission_csv))

    def search(self, image_list: list, feature_list: list, cache_file: str = None, ignore_same: bool = False,
               return_feature_list: bool = True) -> (list, list):
        if cache_file is None:
            cache_file = os.path.join(self.faiss_index_path, "cache.pkl")

        cache = FileCache(cache_file)

        # one_file -> multi feature
        assert len(feature_list) % len(image_list) == 0
        num_feature = len(feature_list) // len(image_list)
        if num_feature > 1:
            self.logger.info("1 image vs {} feature when searching!".format(num_feature))

        # cache
        src_list = []
        for index in range(len(image_list)):
            image_file = image_list[index]
            if not cache.get(image_file):
                src_list.append((image_file, feature_list[index * num_feature:(index + 1) * num_feature]))

        # image search
        whale_rank_utils = WhaleRankingUtils(data_utils=self.data_utils, top_k=self.top_k)
        extend = False
        _show_log_per_steps = 100 if len(image_list) < 1000 else 1000
        for part_src_list in iter_list_with_size(src_list, size=10240 // num_feature):
            feature_list = []
            for (_, _feature_list) in part_src_list:
                feature_list.extend(_feature_list)

            raw_result_iterator = self.manager.image_search_iterator(
                feature_list=feature_list,
                top_k=self._search_top_k,
                extend=extend
            )

            index = -1
            cache_list = []
            for raw_image_result_list in raw_result_iterator:
                index += 1
                if index % num_feature == 0:
                    # the first result for image file
                    _image_file = part_src_list[index // num_feature][0] if ignore_same else None
                    image_result_list = []

                for (faiss_image_index, faiss_image_extend_list, faiss_distance) in raw_image_result_list:
                    image_result = self.manager.get_image_info(faiss_image_index)
                    if _image_file and image_result.get("file") and image_result.get("file").find(_image_file) == 0:
                        continue

                    image_result_list.append((faiss_distance, image_result["class_id"], image_result["file"]))

                if (index + 1) % num_feature == 0:
                    # the last result for image file
                    image_result_list = sorted(image_result_list, key=lambda x: x[0], reverse=False)
                    _class_id_list = whale_rank_utils.simple_rank(image_result_list,
                                                                  distance_cutoff=self._distance_cutoff)
                    if len(_class_id_list) < self.top_k:
                        self.logger.debug("length of class_id_list < {}!".format(self.top_k))

                    # save in cache
                    if return_feature_list:
                        cache_list.append((_class_id_list[:self.top_k], list(image_result_list)))
                    else:
                        cache_list.append((_class_id_list[:self.top_k], None))

                # logging
                if index % _show_log_per_steps == 0:
                    self.logger.info("got class_id_list of {} images".format(index))

            # bulk save in cache
            for j, _class_id_list_feature_list in enumerate(cache_list):
                cache.unsafe_set(part_src_list[j][0], _class_id_list_feature_list)
            cache.set(part_src_list[0][0], cache_list[0])

        # class_id_result_list
        class_id_result_list = []
        search_feature_list = []
        for image_file in image_list:
            _class_id_list, _search_feature_list = cache.get(image_file)
            class_id_result_list.append(_class_id_list)
            search_feature_list.append(_search_feature_list)

        self.logger.info("success to parse search result for test images!")

        return class_id_result_list, search_feature_list

    def _calc_map(self, info, data_percent, labels, class_id_result_list,
                  image_list, multi_index_list) -> (float, float):
        fake_map5 = map_per_set(labels=labels, predictions=class_id_result_list, k=5)
        fake_map1 = map_per_set(labels=labels, predictions=class_id_result_list, k=1)

        multi_map_5 = map_per_set(
            labels=[labels[index] for index in multi_index_list],
            predictions=[class_id_result_list[index] for index in multi_index_list],
            k=5)
        multi_map_1 = map_per_set(
            labels=[labels[index] for index in multi_index_list],
            predictions=[class_id_result_list[index] for index in multi_index_list],
            k=1)

        # new whale
        new_index_list = []
        for _index, _image_file in enumerate(image_list):
            if self.data_utils.get_class_id(_image_file) == self.data_utils.blank_class_id:
                new_index_list.append(_index)

        new_map_5 = map_per_set(
            labels=[labels[index] for index in new_index_list],
            predictions=[class_id_result_list[index] for index in new_index_list],
            k=5)
        new_map_1 = map_per_set(
            labels=[labels[index] for index in new_index_list],
            predictions=[class_id_result_list[index] for index in new_index_list],
            k=1)

        # logging
        self.logger.info(
            "map@5 for validating data[{} of train data, {}] is: standard {}, new-whale {}, all {}!".format(
                data_percent, info, multi_map_5, new_map_5, fake_map5)
        )
        self.logger.info(
            "map@1 for validating data[{} of train data, {}] is: standard {}, new-whale {}, all {}!".format(
                data_percent, info, multi_map_1, new_map_1, fake_map1)
        )

        return multi_map_5, multi_map_1

    def validate(self, data_percent: float = 1.0) -> (float, float):
        """
            validate with training data

        Returns:
            (float, float): map@5, map@1
        """
        # calc feature
        train_feature_list, train_image_list = self.calc_or_load_feature(ProcessMode.train)
        num_feature = len(train_image_list) // len(self.data_utils.list_train_image_file())
        if len(train_image_list) != len(self.data_utils.list_train_image_file()):
            self.logger.warning("got validating-image file: {}/{}".format(
                len(train_image_list), len(self.data_utils.list_train_image_file())))
            assert len(train_image_list) % len(self.data_utils.list_train_image_file()) == 0

        # chosen data
        chosen_index_list = random_choice([index for index in range(len(self.data_utils.list_train_image_file()))],
                                          k=int(len(self.data_utils.list_train_image_file()) * data_percent),
                                          unique=True)
        if num_feature == 1:
            image_list = [train_image_list[index] for index in chosen_index_list]
            feature_list = [train_feature_list[index] for index in chosen_index_list]
        else:
            image_list = [train_image_list[index * num_feature] for index in chosen_index_list]
            feature_list = []
            for index in chosen_index_list:
                feature_list.extend(train_feature_list[index * num_feature:(index + 1) * num_feature])

        # search
        self.logger.info("validating image...")
        _raw_class_id_result_list, validating_feature_list = self.search(
            image_list=image_list, feature_list=feature_list, ignore_same=True, return_feature_list=True)
        self.logger.info("success to validate image!")

        # calc map@5
        image_id_vs_image_file = {os.path.basename(image_file): image_file for image_file in image_list}
        class_id_vs_image_list = {}
        for image_id, class_id in self.data_utils.load_train_info().items():
            class_id_vs_image_list.setdefault(class_id, []).append(image_id_vs_image_file[image_id])
        class_id_vs_image_count = {
            class_id: len(_image_list) for class_id, _image_list in class_id_vs_image_list.items()}

        # rand utils
        rank_utils = WhaleRankingUtils(data_utils=self.data_utils, top_k=self.top_k)

        # all data
        labels = [rank_utils.get_class_id(class_id_vs_image_count, image_file) for image_file in image_list]

        # multi instance
        multi_index_list = []
        for _index, _image_file in enumerate(image_list):
            if class_id_vs_image_count[self.data_utils.get_class_id(_image_file)] > 1:
                multi_index_list.append(_index)

        distance_cutoff = rank_utils.get_similar_cutoff(
            validating_labels=labels, validating_feature_list=validating_feature_list)
        self.logger.info("got distance_cutoff {} and use it!".format(get_pretty_float(distance_cutoff, count=3)))
        self._distance_cutoff = distance_cutoff

        class_id_result_list = rank_utils.recreate_class_id_list(
            cutoff=self._distance_cutoff, class_id_list=_raw_class_id_result_list, feature_list=validating_feature_list)

        total_map_5, total_map_1 = self._calc_map(
            "with distance cutoff {}".format(get_pretty_float(self._distance_cutoff, count=3)),
            data_percent, labels, class_id_result_list, image_list, multi_index_list
        )

        # baseline
        random_map5, random_map1 = self.data_utils.get_random_baseline()
        self.logger.info("random baseline for validating data[{} of train data]: map@5 is {}, map@1 is {}!".format(
            data_percent, random_map5, random_map1))

        # submit result
        raw_submission_csv = self.data_utils.path_manager.submission_csv
        try:
            self.data_utils.path_manager.submission_csv =                 raw_submission_csv + ".{}.validate.csv".format(str((total_map_5 + total_map_1) / 2)[2:])
            self.data_utils.submit_test_result(
                test_image_list=image_list, result_list=class_id_result_list, feature_list=validating_feature_list)
            self.logger.info("submit validating result[{} of train data, with distance cutoff {}] in {}".format(
                data_percent, get_pretty_float(distance_cutoff, count=3), self.data_utils.path_manager.submission_csv))
        finally:
            self.data_utils.path_manager.submission_csv = raw_submission_csv

        return total_map_5, total_map_1


class TripletSearch(SimpleSearch):
    """
        Name: TripletSearch
    """
    model = None

    def __init__(self, instance, data_utils: WhaleDataUtils, clear_cache_after_exists: bool,
                 norm_type: NormType = NormType.l2):
        self.instance = instance
        super(TripletSearch, self).__init__(
            data_utils, shape=(None,), dimension=instance.params.dimension, top_k=5)
        self.smart_iterator = None
        self.feature_record = {}
        self._clear_cache_after_exists = clear_cache_after_exists
        self._norm_type = norm_type

    def list_features(self, mode: ProcessMode) -> (list, list):
        """
            input: ?X224X224X3
            feature: list of ?X5005, files: list of str
        """
        if "train" not in self.feature_record and "test" not in self.feature_record:
            train_feature_list, train_feature_file_list = self.instance.list_features(mode=ProcessMode.train)
            test_feature_list, test_feature_file_list = self.instance.list_features(mode=ProcessMode.test)

            # l2 norm
            feature = np.vstack(train_feature_list + test_feature_list).                 reshape((len(train_feature_list) + len(test_feature_list), self.dimension))
            if self._norm_type is None:
                normalized_feature = feature
            else:
                self.logger.info("using {} to normalize feature".format(self._norm_type))
                normalized_feature = self._norm_type.normalize(feature)

            self.feature_record["train"] = train_feature_file_list
            self.feature_record["test"] = test_feature_file_list
            self.feature_record["feature"] = normalized_feature

        if mode == ProcessMode.train:
            feature_file_list = self.feature_record["train"]
            feature_list = [f.reshape(1, self.dimension) for f in
                            self.feature_record["feature"][:len(feature_file_list)]]
        else:
            feature_file_list = self.feature_record["test"]
            feature_list = [f.reshape(1, self.dimension) for f in
                            self.feature_record["feature"][len(self.feature_record["train"]):]]

        assert len(feature_list) == len(feature_file_list)
        assert feature_list[0].shape == (1, self.dimension)

        return feature_list, feature_file_list

    def clear_cache(self, ):
        if os.path.exists(self.faiss_index_path):
            try:
                if not self.data_utils.path_manager.is_kaggle:
                    tmp_dir = os.path.join(os.path.dirname(self.faiss_index_path), self.instance.__class__.__name__)
                    if os.path.exists(tmp_dir):
                        shutil.rmtree(tmp_dir)
                    shutil.copytree(self.faiss_index_path, tmp_dir)
                    self.logger.info("backup Search Cache in {}".format(os.path.abspath(tmp_dir)))
                shutil.rmtree(self.faiss_index_path)
            except Exception as e:
                self.logger.error(e, exc_info=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._clear_cache_after_exists:
            self.clear_cache()




class TripletLossModelCNN(AbstractEstimator):
    def __init__(self, train_ckpt_dir, data_utils: WhaleDataUtils, timeout: int = int(3600 * 5),
                 pretrained_ckpt_file: str = None):
        super(TripletLossModelCNN, self).__init__(
            model_name="TripletLoss",
            train_ckpt_dir=train_ckpt_dir,
            pretrained_ckpt_file=pretrained_ckpt_file
        )
        self.data_utils = data_utils
        self.timeout = timeout + time.time()
        self.batch_size = 8
        self._features_images_key = "images"
        self._features_filename_key = "filenames"
        self._features_embedding_key = "embeddings"
        self.params = Params({
            "num_channels": 1,
            "margin": 1.0,
            "dimension": 512,
            "image_size": 384,
            "online_batch_count": 4,
        })
        self.learning_rate = 64e-5
        self.image_tmp_size = 400
        assert self.image_tmp_size >= self.params.image_size

        self.train_loss_hook = LossStepHookForTrain(log_after_run=True, log_after_end=False,
                                                    show_log_per_steps=10, run_after_run_per_steps=5)
        self.data_utils.loss_hook = self.train_loss_hook
        self.using_training_status_of_train_data = None
        self.optimizer_type = OptimizerType.adam

    def show_embedding(self, mode: ProcessMode, count: int = -1, remove_log_dir_if_exists: bool = False,
                       norm_type: NormType = NormType.all):
        key_name = "train" if mode == ProcessMode.train else "test"
        log_dir = os.path.join(self.TRAIN_CKPT_DIR, "logs_{}".format(key_name))
        if os.path.exists(log_dir) and remove_log_dir_if_exists:
            shutil.rmtree(log_dir)

        self.logger.info("trying to show {} embedding...".format(key_name))

        if count > 1:
            if mode == ProcessMode.train:
                image_list = self.data_utils.list_debug_image_list(count=count)
            else:
                image_list = self.data_utils.get_file_list(
                    online_batch_count=0, is_training=bool(mode == ProcessMode.train))
                image_list = image_list[:count]
            if len(image_list) != count:
                self.logger.warning("only have {} files".format(len(image_list)))
        else:
            image_list = self.data_utils.get_file_list(
                online_batch_count=0, is_training=bool(mode == ProcessMode.train))

        if mode == ProcessMode.train:
            label_list = [self.data_utils.get_class_id(image_file) for image_file in image_list]
        else:
            label_list = [os.path.basename(image_file) for image_file in image_list]

        feature_list, _ = self.list_features(file_list=image_list)

        # normalize
        if norm_type is not None:
            feature = np.vstack(feature_list).reshape((len(feature_list), self.params.dimension))
            self.logger.info("using {} to normalize feature".format(norm_type))
            normalized_feature = norm_type.normalize(feature)
            feature_list = [f.reshape(1, self.params.dimension) for f in normalized_feature]

        self.logger.info("got {}-image feature: {}/{}".format(
            key_name, len(feature_list), len(image_list))
        )

        show_embedding(feature_list=feature_list, labels=label_list, log_dir=log_dir)
        self.logger.info("success to show {} embedding!".format(key_name))

    def show_predict_result(self, count: int = 10, top_k: int = 5):
        tmp_feature_list = glob.glob(
            os.path.join(self.data_utils.path_manager.output_path, "whale.sub_.*.submission.csv.feature"))
        tmp_feature_list = sorted(tmp_feature_list, key=lambda x: os.path.getmtime(x), reverse=True)
        file_id_list, feature_list = WhaleRankingUtils.get_feature_list(feature_file=tmp_feature_list[0])

        # get index list
        if count < len(file_id_list):
            index_list = random_choice(src_list=[i for i in range(len(file_id_list))], k=count, unique=True)
        else:
            index_list = [i for i in range(len(file_id_list))]

        # image list
        image_file_list = []
        for index in index_list:
            _image_file_list = [os.path.join(self.data_utils.path_manager.data_set_test_path, file_id_list[index])]
            _count = 0
            for (_distance, _class_id, _file_name) in feature_list[index]:
                _image_file_list.append(
                    os.path.join(self.data_utils.path_manager.data_set_train_path, os.path.basename(_file_name))
                )
                _count += 1

                if _count >= top_k:
                    break

            image_file_list.append(_image_file_list)

        # show image
        image_utils.show_images_file(image_list=image_file_list, image_save_file=None, image_size=(200, 200), dpi=100)

    def model_fun(self, ):
        """ 返回func """

        raise NotImplementedError

    def list_features(self, file_list: list = None, mode: ProcessMode = None) -> (list, list):
        if file_list is None and mode is not None:
            if mode == ProcessMode.train:
                input_list = self.data_utils.get_file_list(is_training=True)
            else:
                input_list = self.data_utils.get_file_list(is_training=False)
        else:
            input_list = file_list

        input_fn = self.get_dataset_func(
            split_name=DatasetUtils.SPLIT_PREDICT, input_list=input_list,
            num_epochs=1, shuffle=False, batch_size=self.batch_size, num_parallel_calls=2, prefetch_size=2,
        )

        # predict
        self.logger.info("trying to list feature for {} file...".format(len(input_list)))
        classifier = self.get_classifier()
        result_list = classifier.predict(input_fn=input_fn)

        # parse feature
        feature_list = []
        feature_file_list = []
        count = 0
        _steps_ = 100 if len(input_list) <= 1000 else 1000
        for score in result_list:
            feature_file_list.append(byte_to_string(score[self._features_filename_key]))
            feature_list.append(score[self._features_embedding_key].reshape(1, self.params.dimension))
            count += 1
            if count % _steps_ == 0:
                self.logger.info("calculated {} image...".format(count))

        if count % _steps_ != 0:
            self.logger.info("calculated {} image...".format(count))

        self.logger.info("success to list feature for {} file!".format(len(input_list)))

        return feature_list, feature_file_list

    def get_dataset_func(self, split_name, num_epochs=1, shuffle=True, batch_size=64, num_parallel_calls=2,
                         prefetch_size=2, shuffle_size=4, input_list=None):
        if self.train_loss_hook is None:
            raise ValueError("self.train_loss_hook cannot be null in this class!")

        def tf_decode_with_crop(file_name, label, offset_height, offset_width, target_height, target_width):
            image_str = tf.read_file(file_name)
            image = tf.image.decode_jpeg(image_str, channels=3)
            image = tf_image_crop(image, offset_height, offset_width, target_height, target_width)

            image = tf.image.resize_images(tf.image.rgb_to_grayscale(image),
                                           size=(self.image_tmp_size, self.image_tmp_size))

            processed_images = pre_process_utils.whale_gray_preprocess_image(
                image, self.params.image_size, self.params.image_size,
                is_training=bool(split_name == DatasetUtils.SPLIT_TRAIN),
                mean_tf_func=pre_process_utils.whale_siamese_image_mean_tf)
            return {self._features_images_key: processed_images, self._features_filename_key: file_name}, label

        def input_fn():
            if input_list:
                file_list = input_list
            else:
                file_list = self.data_utils.get_file_list(
                    is_training=bool(split_name == DatasetUtils.SPLIT_TRAIN),
                    shuffle=shuffle, num_epochs=num_epochs, batch_size=batch_size,
                    online_batch_count=self.params.online_batch_count)

            self.logger.info("info of file_list: len is {}.".format(len(file_list)))
            if split_name == DatasetUtils.SPLIT_TRAIN:
                labels = [self.data_utils.get_label(image_file) for image_file in file_list]
            else:
                labels = [0] * len(file_list)

            bounding_boxes = [expend_bounding_box(self.data_utils.get_boxes(image_file)) for
                              image_file in file_list]
            offset_height_list, offset_width_list, target_height_list, target_width_list =                 parse_bounding_boxes_list(bounding_boxes)
            dataset = tf.data.Dataset.from_tensor_slices(
                (file_list, labels, offset_height_list, offset_width_list, target_height_list, target_width_list))
            dataset = dataset.map(tf_decode_with_crop, num_parallel_calls=num_parallel_calls)
            dataset = dataset.prefetch(buffer_size=prefetch_size * batch_size)
            dataset = dataset.batch(batch_size)
            iterator = dataset.make_one_shot_iterator()
            features, labels = iterator.get_next()
            return features, labels

        return input_fn

    def map_search(self, only_validate: bool = False, clear_cache_after_exists: bool = True,
                   norm_type: NormType = NormType.all):
        # norm_type == NormType.all, 相当于所有元素同时除以一个数值. 与模型一致; 不改变距离顺序
        with TripletSearch(instance=self, data_utils=self.data_utils, norm_type=norm_type,
                           clear_cache_after_exists=clear_cache_after_exists) as search:
            self.logger.info("trying to search with {}".format(self.__class__.__name__))
            raw_submission_csv = self.data_utils.path_manager.submission_csv
            try:

                _time_start = time.time()
                search.train()

                base_name = "whale.sub_.{}".format(create_fake_random_string(length=8))
                self.data_utils.path_manager.submission_csv = os.path.join(self.data_utils.path_manager.output_path,
                                                                           base_name)

                map_5, map_1 = search.validate(data_percent=1.0)
                if only_validate:
                    self.logger.info("success to validate with {}, time cost {} seconds!".format(
                        self.__class__.__name__, round(time.time() - _time_start, 2)))
                    return

                # predict
                self.data_utils.path_manager.submission_csv =                     self.data_utils.path_manager.submission_csv +                     ".{}.submission.csv".format(str((map_5 + map_1) / 2)[2:])
                search.test()
                self.logger.info("success to search with {}, time cost {} seconds, submission save in {}".format(
                    self.__class__.__name__, round(time.time() - _time_start, 2),
                    self.data_utils.path_manager.submission_csv))
            finally:
                self.data_utils.path_manager.submission_csv = raw_submission_csv

    def combine_submission_csv(self):
        # combine result
        tmp_submission_csv_list = glob.glob(
            os.path.join(self.data_utils.path_manager.output_path, "whale.sub_.*.submission.csv"))

        if not tmp_submission_csv_list:
            self.logger.warning("no submission file found!")
            return
        self.logger.info("found tmp submission files: {}".format(tmp_submission_csv_list))

        result_weight = {}
        for csv_file in tmp_submission_csv_list:
            try:
                result_weight[csv_file] = float("0.{}".format(csv_file.split(".")[-3]))
            except Exception as e:
                self.logger.error(e)

        combine_csv(result_weight, out_file=os.path.join(self.data_utils.path_manager.output_path, "sub_ens.csv"))
        self.logger.info("success to merge all submission file into sub_ens.csv")

        if self.data_utils.path_manager.is_kaggle:
            for csv_file in tmp_submission_csv_list:
                os.remove(csv_file)

    def train_with_predict(self, safe_max_batch_size=32, calc_every_epoch=1, shuffle_data_every_epoch=1,
                           predict_every_epoch=100, max_epoch=100, ignore_error_in_train: bool = True):
        assert max_epoch >= predict_every_epoch >= shuffle_data_every_epoch
        assert max_epoch >= calc_every_epoch >= shuffle_data_every_epoch

        self.batch_size = safe_max_batch_size

        def end_process_func():
            self.map_search(only_validate=False, clear_cache_after_exists=True)

        def loop_process(total_epoch, num_epoch):
            if total_epoch % calc_every_epoch == 0:
                if total_epoch == 0:
                    self.train(batch_size=safe_max_batch_size, num_epochs=1, shuffle=True, steps=1)

                self.data_utils.calc_feature(self.list_features, epoch_num=total_epoch)

                if total_epoch > 0 and not self.data_utils.path_manager.is_kaggle and                         os.path.abspath(self.TRAIN_CKPT_DIR).find("/content/") > -1:
                    colab_save_file_func(train_dir=self.TRAIN_CKPT_DIR, logger=self.logger, daemon=False,
                                         only_save_latest_checkpoint=True)

            self.train(batch_size=safe_max_batch_size, num_epochs=num_epoch, shuffle=True, steps=None)
            self.set_epoch_num(count=total_epoch + num_epoch)

            if total_epoch > 0 and total_epoch % predict_every_epoch == 0:
                end_process_func()

        # run
        estimator_iter_process(loop_process, iter_stop_time=self.timeout,
                               loop_process_min_epoch=shuffle_data_every_epoch,
                               loop_process_start_epoch=self.get_epoch_num(),
                               end_process_func=end_process_func, loop_process_max_epoch=max_epoch,
                               ignore_error_in_loop_process=ignore_error_in_train, logger=self.logger)

        # combine result
        self.combine_submission_csv()


class TripletLossModelResNet50(TripletLossModelCNN):
    def __init__(self, train_ckpt_dir, data_utils: WhaleDataUtils, timeout: int = int(3600 * 5), is_softmax=True):
        if train_ckpt_dir.endswith("/"):
            train_ckpt_dir = train_ckpt_dir[:-1]

        softmax_dir = os.path.join(os.path.dirname(train_ckpt_dir), os.path.basename(train_ckpt_dir) + "_softmax")
        triplet_dir = os.path.join(os.path.dirname(train_ckpt_dir), os.path.basename(train_ckpt_dir) + "_triplet")
        if is_softmax is None:
            if tf.train.latest_checkpoint(triplet_dir):
                is_softmax = False
            else:
                is_softmax = True

        _pretrained_ckpt_file = None
        if is_softmax:
            # using softmax net
            _train_ckpt_dir = softmax_dir
            _pretrained_ckpt_file = data_utils.path_manager.ckpt_pretrained_rv250
            if not tf.train.latest_checkpoint(softmax_dir):
                data_utils.path_manager.download_resnet_v2_50()
        else:
            # using triplet loss
            _train_ckpt_dir = triplet_dir
            if tf.train.latest_checkpoint(softmax_dir):
                _pretrained_ckpt_file = tf.train.latest_checkpoint(softmax_dir)

        super(TripletLossModelResNet50, self).__init__(
            train_ckpt_dir=_train_ckpt_dir,
            data_utils=data_utils,
            timeout=timeout,
            pretrained_ckpt_file=_pretrained_ckpt_file,
        )
        self.params = Params({
            "num_channels": 3,
            "margin": 1.0,
            "dimension": 512,
            "image_size": 224,
            "online_batch_count": 4,
        })
        self.image_tmp_size = 250

        # softmax setting
        self.is_softmax = is_softmax
        if is_softmax:
            self.use_softmax_net()
        else:
            self.use_triplet_net()

        _, class_id_vs_image_list, _, _, _ =             self.data_utils.get_basic_info(self.data_utils.list_train_image_file(ignore_blank=True))
        _class_id_vs_image_count = {
            class_id: len(_image_list) for class_id, _image_list in class_id_vs_image_list.items()}
        if self.data_utils.blank_class_id in _class_id_vs_image_count:
            _class_id_vs_image_count.pop(self.data_utils.blank_class_id)

        _image_count_list = list(_class_id_vs_image_count.values())
        _image_count_list.sort()
        _max_count = int(sum(_image_count_list) * 0.2)
        _base_count = 3
        _middle_count = 0
        _total_count = 0
        for _index, _count in enumerate(_image_count_list[::-1]):
            _total_count += _count
            if _total_count >= _max_count:
                _base_count = _count
                _middle_count = _image_count_list[-int(_index / 2)]
                break

        anchor_class_id_list = [
            class_id for class_id, _image_count in _class_id_vs_image_count.items() if _image_count >= _base_count]

        self.softmax_num_class = len(anchor_class_id_list)
        anchor_class_id_list.sort()

        self.softmax_class_id_vs_label = {}
        for index, class_id in enumerate(anchor_class_id_list):
            self.softmax_class_id_vs_label[class_id] = index

        self.softmax_image_list = []
        _anchor_class_id_set = set(anchor_class_id_list)
        _class_vs_image = {}
        for class_id, file_list in class_id_vs_image_list.items():
            if class_id in _anchor_class_id_set:
                _class_vs_image[class_id] = file_list

        _new_class_image = balance_class_dict(_class_vs_image, target_count=min(100, _middle_count))
        for file_list in _new_class_image.values():
            self.softmax_image_list.extend(file_list)

        random.shuffle(self.softmax_image_list)

        self.softmax_image_vs_label = {}
        for image_file in self.softmax_image_list:
            self.softmax_image_vs_label[image_file] = self.softmax_class_id_vs_label[
                self.data_utils.get_class_id(image_file)]

        self.logger.info("train softmax net with {} classes (ignore blank, image_count >= {}), {}/{} files,".format(
            self.softmax_num_class, _base_count, len(set(self.softmax_image_list)), len(self.softmax_image_list)))

        # global setting
        self._net_embedding_key = "embeddings"

    def model_fun(self, ):
        """ 返回func """

        def get_model_net(scope_name, features, params, is_training, labels=None):
            inputs = tf.reshape(features[self._features_images_key],
                                [-1, params.image_size, params.image_size, params.num_channels])

            with slim.arg_scope(resnet_arg_scope()):
                global_pool, end_points = resnet_v2_50(inputs, num_classes=None, is_training=is_training)

                with tf.variable_scope(scope_name, 'triplet_loss', [global_pool]) as sc:
                    net = slim.conv2d(global_pool, params.dimension, [1, 1], scope=self._net_embedding_key)
                    end_points[sc.name + '/logits'] = net
                    net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
                    end_points[sc.name + '/spatial_squeeze'] = net
                    end_points[self._net_embedding_key] = net

                    assert net.shape[-1] == params.dimension
                    assert len(net.shape) == 2

                    return net, end_points

        def _softmax_bone_net(scope_name, features, params, is_training, labels=None):
            embeddings, end_points = get_model_net(
                scope_name=scope_name, features=features, params=params, is_training=is_training)

            with slim.arg_scope(resnet_arg_scope()):
                with tf.variable_scope("softmax_net", 'softmax_net', [embeddings]):
                    net = slim.fully_connected(embeddings, self.softmax_num_class, activation_fn=None,
                                               normalizer_fn=None, scope='logits')
                    return net, end_points

        if self.is_softmax:
            return tf_model_fn.tf_softmax_model_fn(
                network=_softmax_bone_net,
                scope_name=self.model_name,
                features_filename_key=self._features_filename_key,
                get_learning_rate_func=self.get_learning_rate,
                optimizer_type=self.optimizer_type,
                logger=self.logger
            )

        return tf_model_fn.tf_triplet_loss_model_fn(
            network=get_model_net,
            scope_name=self.model_name,
            features_embedding_key=self._features_embedding_key,
            features_filename_key=self._features_filename_key,
            get_learning_rate_func=self.get_learning_rate,
            optimizer_type=self.optimizer_type,
            logger=self.logger,
            use_l2_normalize=False)

    def get_dataset_func(self, split_name, num_epochs=1, shuffle=True, batch_size=64, num_parallel_calls=2,
                         prefetch_size=2, shuffle_size=4, input_list=None):
        if self.train_loss_hook is None:
            raise ValueError("self.train_loss_hook cannot be null in this class!")

        def tf_decode_with_crop(file_name, label, offset_height, offset_width, target_height, target_width):
            image_str = tf.read_file(file_name)
            image = tf.image.decode_jpeg(image_str, channels=3)
            image = tf_image_crop(image, offset_height, offset_width, target_height, target_width)
            image = tf.image.resize_images(image, size=(self.image_tmp_size, self.image_tmp_size))

            processed_images = pre_process_utils.whale_rgb_preprocess_image(
                image, self.params.image_size, self.params.image_size,
                is_training=bool(split_name == DatasetUtils.SPLIT_TRAIN))
            return {self._features_images_key: processed_images, self._features_filename_key: file_name}, label

        def input_fn():
            if input_list:
                file_list = input_list
            else:
                file_list = self.data_utils.get_file_list(
                    is_training=bool(split_name == DatasetUtils.SPLIT_TRAIN),
                    shuffle=shuffle, num_epochs=num_epochs, batch_size=batch_size,
                    online_batch_count=self.params.online_batch_count)

            self.logger.info("info of file_list: len is {}.".format(len(file_list)))
            if split_name == DatasetUtils.SPLIT_TRAIN:
                labels = [self.data_utils.get_label(image_file) for image_file in file_list]
            else:
                labels = [0] * len(file_list)

            bounding_boxes = [expend_bounding_box(self.data_utils.get_boxes(image_file)) for
                              image_file in file_list]
            offset_height_list, offset_width_list, target_height_list, target_width_list =                 parse_bounding_boxes_list(bounding_boxes)

            dataset = tf.data.Dataset.from_tensor_slices(
                (file_list, labels, offset_height_list, offset_width_list, target_height_list, target_width_list))
            dataset = dataset.map(tf_decode_with_crop, num_parallel_calls=num_parallel_calls)
            dataset = dataset.prefetch(buffer_size=prefetch_size * batch_size)
            dataset = dataset.batch(batch_size)
            iterator = dataset.make_one_shot_iterator()
            features, labels = iterator.get_next()
            return features, labels

        def softmax_input_fn():
            if input_list:
                file_list = input_list
            else:
                file_list = list(self.softmax_image_list) * num_epochs
                if shuffle:
                    random.shuffle(file_list)

            self.logger.info("info of file_list: len is {}.".format(len(file_list)))
            labels = [self.softmax_image_vs_label[image_file] for image_file in file_list]

            bounding_boxes = [expend_bounding_box(self.data_utils.get_boxes(image_file)) for
                              image_file in file_list]
            offset_height_list, offset_width_list, target_height_list, target_width_list =                 parse_bounding_boxes_list(bounding_boxes)
            dataset = tf.data.Dataset.from_tensor_slices(
                (file_list, labels, offset_height_list, offset_width_list, target_height_list, target_width_list))
            dataset = dataset.map(tf_decode_with_crop, num_parallel_calls=num_parallel_calls)
            dataset = dataset.prefetch(buffer_size=prefetch_size * batch_size)
            dataset = dataset.batch(batch_size)
            iterator = dataset.make_one_shot_iterator()
            features, labels = iterator.get_next()
            return features, labels

        if self.is_softmax:
            return softmax_input_fn

        return input_fn

    def use_softmax_net(self):
        # softmax setting
        self.logger.info("using softmax net...")
        self.is_softmax = True
        self.data_utils.gen_data_setting["stop_calc_feature"] = True

    def use_triplet_net(self):
        # softmax setting
        self.logger.info("using triplet net...")
        self.is_softmax = False
        self.data_utils.gen_data_setting["stop_calc_feature"] = False




get_ipython().system('cp ../input/whale-triplet-pretrained-model/tripletresnet/tripletResNet ./triplet_triplet -R')

# logger
init_logger()

# estimator
path_manager = PathManager("kaggle")
data_utils = WhaleDataUtils(path_manager=path_manager, gen_data_setting={
        "x_train_num": 4,
        "ignore_blank_prob": 0.9,
        "ignore_single_prob": 0.5,
        "stop_calc_feature": False,
        "gen_data_by_random_prob": 0,
        "use_norm_when_calc_apn": True,
    })
    
# init model with pretrained triplet loss model (finetuning from resnet50)
estimator = TripletLossModelResNet50(
        train_ckpt_dir="./triplet",
        data_utils=data_utils,
        timeout=int(5 * 3600),
        is_softmax=False,
    )
    
# predict
estimator.map_search(only_validate=False, clear_cache_after_exists=True)
estimator.combine_submission_csv() # create submission csv file: sub_ens.csv
    
# # show embeddings (you need to download logdata and show it with tensorboard)
# estimator.show_embedding(
#         mode=ProcessMode.train,
#         count=10000,
#         remove_log_dir_if_exists=True,
#         norm_type=NormType.all,
#     )
    
# # train and predict
# try:
#     estimator.train_with_predict(safe_max_batch_size=64, calc_every_epoch=1, shuffle_data_every_epoch=1,
#                                      predict_every_epoch=100, max_epoch=10000, ignore_error_in_train=True)
# finally:
#     if data_utils.path_manager.is_kaggle:
#         if os.path.exists(path_manager.bounding_boxes_csv):
#             os.remove(path_manager.bounding_boxes_csv)
    




estimator.show_predict_result(count=10, top_k=5)

estimator.show_predict_result(count=10, top_k=3)




# exit
get_ipython().system('rm *.pkl')

get_ipython().system('pip uninstall aiohttp faiss-prebuilt pyxtools pymltools -y')
get_ipython().system('apt remove -y libopenblas-base libomp-dev')


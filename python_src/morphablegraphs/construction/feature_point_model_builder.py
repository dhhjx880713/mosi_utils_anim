__author__ = 'hadu01'

from feature_point_model import FeaturePointModel
import json


class FeaturePointModelBuilder():
    def __init__(self):
        self.morphable_model_directory = None
        self.n_samples = 10000


    def set_config(self, config_file_path):
        config_file = open(config_file_path)
        config = json.load(config_file)
        self.morphable_model_directory = config["model_data_dir"]

    def builder(self):
        pass


def main():
    feature_point_model_builder = FeaturePointModelBuilder()
    feature_point_model_builder.set_config(config_file)
    feature_point_model_builder.builder()
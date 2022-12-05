from dataset import MyDateset

import os
from pathlib import Path
import numpy as np
from addict import Dict
from openvino.tools.pot.api import DataLoader, Metric
from openvino.tools.pot.engines.ie_engine import IEEngine
from openvino.tools.pot.graph import load_model, save_model
from openvino.tools.pot.graph.model_utils import compress_model_weights
from openvino.tools.pot.pipeline.initializer import create_pipeline
from torchvision import transforms

'''
分类网络的量化
int32->int8
DefaultQuantization /AccuracyAwareQuantization
'''

data_dir = '../calib_train/train'
MODEL_DIR = '../IR/int8'
transform = transforms.Compose(
    [transforms.ToTensor()])
dataset = MyDateset(root=data_dir, transform=transform)


class CifarDataLoader(DataLoader):

    def __init__(self, config):
        """
        Initialize config and dataset.
        param config: created config with DATA_DIR path.
        """
        if not isinstance(config, Dict):
            config = Dict(config)
        super().__init__(config)
        self.indexes, self.pictures, self.labels = self.load_datas(dataset)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        """
        Return one sample of index, label and picture.
        :param index: index of the taken sample.
        """
        if index >= len(self):
            raise IndexError

        return (self.indexes[index], self.labels[index]), self.pictures[index].numpy()

    def load_datas(self, dataset):
        """
        Load dataset in needed format. 
        param dataset:  downloaded dataset.
        """
        pictures, labels, indexes = [], [], []

        for idx, sample in enumerate(dataset):
            pictures.append(sample[0])
            labels.append(sample[1])
            indexes.append(idx)

        return indexes, pictures, labels


class Accuracy(Metric):

    # Required methods
    def __init__(self, top_k=1):
        super().__init__()
        self._top_k = top_k
        self._name = 'accuracy@top{}'.format(self._top_k)
        self._matches = []

    @property
    def value(self):
        """ Returns accuracy metric value for the last model output. """
        return {self._name: self._matches[-1]}

    @property
    def avg_value(self):
        """ Returns accuracy metric value for all model outputs. """
        return {self._name: np.ravel(self._matches).mean()}

    def update(self, output, target):
        """ Updates prediction matches.
        :param output: model output
        :param target: annotations
        """
        if len(output) > 1:
            raise Exception('The accuracy metric cannot be calculated '
                            'for a model with multiple outputs')
        if isinstance(target, dict):
            target = list(target.values())
        predictions = np.argsort(output[0], axis=1)[:, -self._top_k:]
        match = [float(t in predictions[i]) for i, t in enumerate(target)]

        self._matches.append(match)

    def reset(self):
        """ Resets collected matches """
        self._matches = []

    def get_attributes(self):
        """
        Returns a dictionary of metric attributes {metric_name: {attribute_name: value}}.
        Required attributes: 'direction': 'higher-better' or 'higher-worse'
                             'type': metric type
        """
        return {self._name: {'direction': 'higher-better',
                             'type': 'accuracy'}}


model_config = Dict({
    'model_name': 'conv',
    'model': "E:/work/openvino_quant/openvino_/IR/32/conv_96.xml",
    'weights': "E:/work/openvino_quant/openvino_/IR/32/conv_96.bin"
})
engine_config = Dict({
    'device': 'CPU',
    'stat_requests_number': 2,
    'eval_requests_number': 2
})
dataset_config = {
    'data_source': data_dir
}
algorithms = [
    {
        'name': 'AccuracyAwareQuantization',
        "params": {
            "target_device": "CPU",
            "stat_subset_size": 300,
            'maximal_drop': 0.01,
            "preset": "performance",
            "tune_hyperparams": False
        }
    }
]

# Steps 1-7: Model optimization
# Step 1: Load the model.
model = load_model(model_config)

# Step 2: Initialize the data loader.
data_loader = CifarDataLoader(dataset_config)

# Step 3 (Optional. Required for AccuracyAwareQuantization): Initialize the metric.
metric = Accuracy(top_k=1)

# Step 4: Initialize the engine for metric calculation and statistics collection.
engine = IEEngine(engine_config, data_loader, metric)

# Step 5: Create a pipeline of compression algorithms.
pipeline = create_pipeline(algorithms, engine)

# Step 6: Execute the pipeline.
compressed_model = pipeline.run(model)

# Step 7 (Optional): Compress model weights quantized precision
#                    in order to reduce the size of final .bin file.
compress_model_weights(compressed_model)

# Step 8: Save the compressed model to the desired path.
compressed_model_paths = save_model(model=compressed_model, save_path=MODEL_DIR,
                                    model_name="convnext_int8_96")
compressed_model_xml = compressed_model_paths[0]["model"]
compressed_model_bin = Path(compressed_model_paths[0]["model"]).with_suffix(".bin")

# Step 9: Compare accuracy of the original and quantized models.
metric_results = pipeline.evaluate(model)
if metric_results:
    for name, value in metric_results.items():
        print(f"Accuracy of the original model: {name}: {value}")

metric_results = pipeline.evaluate(compressed_model)
if metric_results:
    for name, value in metric_results.items():
        print(f"Accuracy of the optimized model: {name}: {value}")

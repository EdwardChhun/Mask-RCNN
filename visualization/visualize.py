import plotly.express as px
from pandas import DataFrame
from pathlib import Path
from typing import List, Dict
import json


class VisualizeError(Exception):
    def __init__(self, message: str  = "An error occured") -> None:
        self._message = message

    def __repr__(self) -> str:
        return self._message


class Visualizer:
    def __init__(self, metrics_file: str) -> None:
        self._metrics_file = Path(metrics_file)
        self._metrics = []

        if not self._metrics_file.exists():
            raise VisualizeError(f"File {self._metrics_file} does not exist")

        if not self._metrics_file.is_file() or not self._metrics_file.suffix == ".json":
            raise VisualizeError(f"File {self._metrics_file} is not a valid JSON file")

        with open(self._metrics_file, "r") as f:
            for line in f:
                try:
                    self._metrics.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise VisualizeError(f"Error while reading the JSON file: {e}")


    def _plot_over_iterations(self, *, y_axis: str, y_axis_title: str, title: str):
        iterations = {}
        for metric in self._metrics:
            try:
                if metric["iteration"] in iterations:
                    iterations[metric["iteration"]].append(metric[y_axis])
                else:
                    iterations[metric["iteration"]] = [metric[y_axis]]
            except KeyError:
                # Some entries might not have the cls accuracy key
                pass

        data = []
        for iteration, accuracy in iterations.items():
            data.append({"Iteration": iteration, y_axis_title: accuracy})

        data_frame = DataFrame(data)
        fig = px.scatter(data_frame.explode(y_axis_title), x="Iteration", y=y_axis_title, title=title)
        fig.show()

    def plot_fast_cls_accuracy(self):
        self._plot_over_iterations(y_axis="fast_rcnn/cls_accuracy",
                                   y_axis_title="Fast RCNN Classifier Accuracy",
                                   title="Fast RCNN Classifier Accuracy")



v = Visualizer("example_metrics.json")
v.plot_fast_cls_accuracy()

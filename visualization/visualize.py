import plotly.express as px
from pandas import DataFrame
from pathlib import Path
from typing import List, Dict


class VisualizeError(Exception):
    def __init__(self, message: str  = "An error occured") -> None:
        self._message = message

    def __repr__(self) -> str:
        return self._message


class Visualizer:
    def __init__(self, metrics_file: str) -> None:
        self._metrics_file = Path(metrics_file)

        if not self._metrics_file.exists():
            raise VisualizeError(f"File {self._metrics_file} does not exist")

    def plot_over_iterations(self, data: List[Dict], *, y_axis: str, title: str):
        data_frame = DataFrame(data)
        fig = px.scatter(data_frame.explode(y_axis), x="Iteration", y=y_axis, title=title)
        fig.show()






data = [
    {"Iteration": 1, "data": [1, 5, 60]},
    {"Iteration": 2, "data": [2, 6, 70]},
    {"Iteration": 3, "data": [3, 7, 80]},
    {"Iteration": 4, "data": [4, 8, 90]}
]

v = Visualizer("example_metrics.json")
v.plot_over_iterations(data, y_axis="data", title="Example plot")

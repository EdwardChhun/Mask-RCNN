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


    def _generate_plot_over_iterations(self, *, y_axis: str, y_axis_title: str, title: str) -> px.scatter:
        iterations = {}
        for metric in self._metrics:
            try:
                if metric["iteration"] in iterations:
                    iterations[metric["iteration"]].append(metric[y_axis])
                else:
                    iterations[metric["iteration"]] = [metric[y_axis]]
            except KeyError:
                # Some entries might not have the correct key, these can be skipped
                pass

        data = []
        for iteration, accuracy in iterations.items():
            data.append({"Iteration": iteration, y_axis_title: accuracy})

        data_frame = DataFrame(data)
        fig = px.scatter(data_frame.explode(y_axis_title), x="Iteration", y=y_axis_title, title=title)
        return fig

    def plot_fast_cls_accuracy(self):
        fig = self._generate_plot_over_iterations(y_axis="fast_rcnn/cls_accuracy",
                                                  y_axis_title="Fast RCNN Classifier Accuracy",
                                                  title="Fast RCNN Classifier Accuracy by Iteration")
        fig.show()

    def plot_fast_false_negative(self):
        fig = self._generate_plot_over_iterations(y_axis="fast_rcnn/false_negative",
                                                  y_axis_title="Fast RCNN Classifier False Negatives",
                                                  title="Fast RCNN Classifier False Negatives by Iteration")
        fig.show()

    def plot_classifier_loss(self):
        fig = self._generate_plot_over_iterations(y_axis="loss_cls",
                                                  y_axis_title="Classifier Loss",
                                                  title="Classifier Loss by Iteration")
        fig.show()

    def plot_classifier_loss_from_rpn(self):
        fig = self._generate_plot_over_iterations(y_axis="loss_rpn_cls",
                                                  y_axis_title=" Classifier Loss from RPN",
                                                  title="Classifier Loss from RPN by Iteration")
        fig.show()

    def plot_total_loss(self):
        fig = self._generate_plot_over_iterations(y_axis="total_loss",
                                                  y_axis_title="Classifier Total Loss",
                                                  title="Classifier Total Loss by Iteration")
        fig.show()

    def plot_pos_neg_anchors(self):
           iterations = {}
           for metric in self._metrics:
               try:
                   iteration = metric["iteration"]
                   num_pos_anchors = metric["rpn/num_pos_anchors"]
                   num_neg_anchors = metric["rpn/num_neg_anchors"]
                   total_anchors = num_pos_anchors + num_neg_anchors

                   if iteration in iterations:
                       iterations[iteration]["num_pos_anchors"].append(num_pos_anchors)
                       iterations[iteration]["num_neg_anchors"].append(num_neg_anchors)
                       iterations[iteration]["total_anchors"].append(total_anchors)
                   else:
                       iterations[iteration] = {
                           "num_pos_anchors": [num_pos_anchors],
                           "num_neg_anchors": [num_neg_anchors],
                           "total_anchors": [total_anchors]
                       }
               except KeyError:
                   # Some entries might not have the correct keys, these can be skipped
                   pass

           data = []
           for iteration, values in iterations.items():
               data.append({
                   "Iteration": iteration,
                   "Num_Pos_Anchors": sum(values["num_pos_anchors"]),
                   "Num_Neg_Anchors": sum(values["num_neg_anchors"]),
                   "Total_Anchors": sum(values["total_anchors"])
               })

           data_frame = DataFrame(data)

           # Plot with Plotly
           data_frame_melted = data_frame.melt(id_vars=["Iteration"],
                                                       value_vars=["Num_Pos_Anchors", "Total_Anchors"],
                                                       var_name="Anchor_Type", value_name="Anchors")
           fig = px.scatter(data_frame_melted, x="Iteration", y="Anchors", color="Anchor_Type",
                                title="Classifier Positive and Negative Anchors by Iteration",
                                labels={"Anchors": "Anchors Count", "Iteration": "Iteration Number"})
           fig.show()

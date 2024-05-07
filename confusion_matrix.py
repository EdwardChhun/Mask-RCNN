import json
import numpy as np
import matplotlib.pyplot as plt


with open("annotations.json", "r") as f:
    annotations_data = json.load(f)

with open("coco_instances_results.json", "r") as f:
    predictions_data = json.load(f)

true_categories = {}
for annotation in annotations_data["annotations"]:
    image_id = annotation["image_id"]
    category_id = annotation["category_id"]
    true_categories[image_id] = category_id

classes = [category["name"] for category in annotations_data["categories"]]
num_classes = len(classes)
confusion_matrix = np.zeros((num_classes, num_classes))

for prediction in predictions_data:
    image_id = prediction["image_id"]
    true_category_id = true_categories.get(image_id)

    if true_category_id is None:
        continue

    predicted_category_id = prediction["category_id"]
    confusion_matrix[true_category_id, predicted_category_id] += 1

def plot_confusion_matrix(cm, classes,
                          normalize = False,
                          title = 'Confusion matrix',
                          cmap = plt.cm.Blues):

    plt.figure(figsize = (8, 6))
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45, fontsize = 8)
    plt.yticks(tick_marks, classes, fontsize = 8)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.gca().set_xticklabels(classes, rotation = 90, va = 'top', ha = 'center', fontsize = 6)

    plt.tight_layout()


plot_confusion_matrix(confusion_matrix, classes = classes,
                      title = 'Confusion matrix, without normalization')

plt.show()

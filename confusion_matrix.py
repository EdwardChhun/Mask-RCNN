import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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
                          normalize = True,
                          title = 'Confusion matrix',
                          cmap = plt.cm.Blues,
                          fontsize = 6,  # Adjust font size here
                          figsize = (10, 8)):  # Adjust figure size here
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]

    plt.figure(figsize = figsize)  # Set figure size
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 90,
               fontsize = fontsize)  # Rotate x-axis labels vertically
    plt.yticks(tick_marks, classes, fontsize = fontsize)  # Adjust font size for y-axis labels
    plt.xlabel('Predicted label', fontsize = fontsize)  # Adjust font size for x-axis label
    plt.ylabel('True label', fontsize = fontsize)  # Adjust font size for y-axis label

    plt.tight_layout()


cnf_matrix_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis = 1)[:,
                                                           np.newaxis]

plt.figure()
plot_confusion_matrix(cnf_matrix_normalized, classes = classes, normalize = True,
                      title = 'Normalized confusion matrix')

plt.show()

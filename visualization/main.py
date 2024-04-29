"""
Main script to run the visualizer and provide options for plots to generate
"""
import contextlib
from pandas.io.formats import printing
from visualize import Visualizer, VisualizeError

def main():
    """
    Main function to run the visualizer and provide options for plots to generate
    """
    initial_input = input("Enter the path to the metrics file or 'q' to quit: ")
    if initial_input == 'q':
        exit()

    visualizer = Visualizer(initial_input)

    options_mapping = {
        1: visualizer.plot_fast_cls_accuracy,
        2: visualizer.plot_fast_false_negative,
        3: visualizer.plot_classifier_loss,
        4: visualizer.plot_classifier_loss_from_rpn,
        5: visualizer.plot_total_loss,
        6: visualizer.plot_pos_neg_anchors,
        7: exit
    }

    while True:
        print("Choose an option to plot")
        print("\t1. Fast RCNN Classifier Accuracy")
        print("\t2. Fast RCNN False Negative")
        print("\t3. Classifier Loss")
        print("\t4. Classifier Loss from RPN")
        print("\t5. Classifier Total Loss")
        print("\t6. Postive and Negative Anchors")
        print("\t7. Quit")
        choice = input("[1 - 7]: ")
        try:
            choice = int(choice)
            if choice not in [1, 2, 3, 4, 5, 6, 7]:
                print("Invalid option, press Enter to try again")
                input()
                continue
        except (ValueError, TypeError):
            print("Invalid option, press Enter to try again")
            input()
            continue

        options_mapping[choice]()
        print("Press Enter to continue")
        input()


if __name__ == "__main__":
    main()

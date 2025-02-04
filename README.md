# Laboratory 2.2: kNN and Evaluating Classification Models

## Description

In this laboratory practice, you will implement and evaluate a k-Nearest Neighbors (kNN) classification model. You will work on the following tasks:

0. **Setup**:
   - Clone the repository and install the required packages.
   - Copy the `train.dat` and `test.dat` files from previous practices.

1. **Minkowski Distance Calculation**:
   - Implement the `minkowski_distance` function to compute distances between data points.

2. **k-Nearest Neighbors Model**:
   - Implement the `knn` class with the following methods:
     - `fit`: Fit the model using training data.
     - `predict`: Predict class labels for the provided data.
     - `predict_proba`: Predict class probabilities for the provided data.
     - `compute_distances`: Compute distances from a point to every point in the training dataset.
     - `get_k_nearest_neighbors`: Get the k nearest neighbors indices given the distances matrix from a point.
     - `most_common_label`: Obtain the most common label from the labels of the k nearest neighbors.

3. **Model Evaluation**:
   - Implement the `evaluate_classification_metrics` function to calculate various evaluation metrics for a classification model.
   - Implement the following plotting functions:
     - `plot_2Dmodel_predictions`: Plot the classification results and predicted probabilities of a model on a 2D grid.
     - `plot_calibration_curve`: Plot a calibration curve to evaluate the accuracy of predicted probabilities.
     - `plot_probability_histograms`: Plot probability histograms for the positive and negative classes separately.
     - `plot_roc_curve`: Plot the Receiver Operating Characteristic (ROC) curve.

## Uploading Changes to GitHub

To upload your changes to GitHub and have them automatically evaluated using the provided tests, follow these steps:

1. **Stage Your Changes**:
   ```sh
   git add .
    ```
2. **Commit Your Changes**:
    ```sh
    git commit -m "Implement kNN and evaluation functions"
    ```
3. **Push Your Changes**:
    ```sh
    git push 
    ```

Once you push your changes, the tests will be automatically executed. If all tests pass, you will receive 10 points. You can try as many times as you want until all tests pass.

Good luck!
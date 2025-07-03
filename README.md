# Anomaly Detection Project

This project aims to develop a machine learning system to detect abnormal situations (anomalies) in simulated temperature and pressure sensor data.

## Project Structure
anomaly_detection_project/
├── data/
│   └── simulated_sensor_data.csv
├── notebooks/
│   ├── 01_data_simulation_and_eda.ipynb
│   └── 02_anomaly_detection_models.ipynb
├── src/
├── tests/             # New: Unit and integration tests for the project.
├── .github/           # New: GitHub Actions workflows for CI/CD.
│   └── workflows/
│       └── ci.yml
├── README.md
└── requirements.txt

## Project Goals

* **Simulated Data Generation:** Create a synthetic sensor dataset including both normal and anomalous data points.
* **Exploratory Data Analysis (EDA):** Understand and visualize the dataset's characteristics, distributions, and potential relationships.
* **Unsupervised Anomaly Detection:** Implement and train unsupervised machine learning algorithms (e.g., Isolation Forest, One-Class SVM) to identify anomalies without labeled training data (though labels are used for evaluation here).
* **Model Evaluation:** Assess the performance of the anomaly detection models using appropriate metrics such as ROC curves, Precision-Recall curves, Confusion Matrices, and Classification Reports.
* **Documentation:** Document the process, findings, and model performance with clear explanations and visualizations.

## Data Description

The `simulated_sensor_data.csv` file contains the following columns:

* `temperature`: Simulated temperature readings.
* `pressure`: Simulated pressure readings.
* `anomaly`: A binary flag (0 for normal, 1 for anomaly) indicating whether a data point is an anomaly (used for evaluation).
* `timestamp`: The time at which the sensor reading was recorded.

## Setup and Running the Project

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-link>
    cd anomaly_detection_project
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Jupyter Notebooks:**
    ```bash
    jupyter notebook
    ```
    Open `01_data_simulation_and_eda.ipynb` first to understand the data, and then `02_anomaly_detection_models.ipynb` to see the anomaly detection in action.

5.  **Run Tests (Optional, but Recommended):**
    To verify the integrity of the data loading and basic model functionalities, you can run the provided tests:
    ```bash
    pytest
    ```

## Technologies Used

* Python
* Pandas
* Numpy
* Matplotlib
* Seaborn
* Scikit-learn
* Pytest (for testing)
* GitHub Actions (for Continuous Integration)

## Results and Discussion

* **Isolation Forest:** Briefly discuss its performance based on metrics (e.g., "Isolation Forest showed a good balance of precision and recall for anomaly detection, achieving an ROC AUC of X.XX. It was particularly effective in identifying clear outliers as shown in the scatter plots.")
* **One-Class SVM:** Discuss its performance (e.g., "One-Class SVM also performed reasonably well, with an ROC AUC of Y.YY. It might be more sensitive to the choice of kernel and `nu` parameter, requiring careful tuning.")
* **Comparison:** Compare the strengths and weaknesses of both models for this specific dataset.
* **Limitations and Future Work:** Mention any limitations of the current approach (e.g., simplicity of simulated data, static `contamination`/`nu` values) and potential future enhancements (e.g., hyperparameter tuning, more complex data, real-time detection).

**Vehicle Price Prediction & Analysis Pipeline**

This repository contains a comprehensive Python data pipeline that cleans vehicle listings data, conducts exploratory correlation analysis, handles multivariate anomalies, and optimizes a polynomial regression model to predict vehicle prices.

🛠️ **Features**

- Automated Data Cleaning: Handles missing values via smart structural drops, median imputation, and categorical placeholder fills.

- Feature Engineering: Standardizes numerical scales and transforms raw vehicle year metrics into dynamic asset age values.

- Outlier Detection: Implements robust Interquartile Range (IQR) filtering to remove extreme bivariate and univariate anomalies.

- Model Optimization: Evaluates high-degree polynomial pipelines using 5-fold Cross-Validation to eliminate overfitting trends.


📦 **Required Dependencies**

Ensure you have the following Python libraries installed before running the execution script:

	bashpip install numpy pandas matplotlib seaborn scikit-learn


🚀 **Execution Workflow**

1. **Data Ingestion & Missing Value Matrix**

- Loads the structural data source file from data/vehicles.csv.

- Drops unneeded database structural index strings (VIN, id).

- Calculates and visualizes null-value percentages across all columns.


2. **Feature Transformations**

- Calculates vehicle_age relative to the current calendar year.

- Applies a standard scalar (StandardScaler) to odometer and vehicle_age.

- Fills remaining empty data fields using structural medians and 'Unknown' flags.


3. **Exploratory Heatmap Matrix**

- Generates a Pearson correlation coefficient matrix.

- Identifies the feature tracking the strongest relationship to target market price.

- Outputs a visual Seaborn correlation matrix heatmap graphic.

![Alt Text](images/correlation_heatmap.png)


4. **IQR Outlier Filtration**

- Scans feature distribution spaces across price, odometer, and vehicle_age.
- Drops out-of-bounds records to optimize model stability.

- Generates a scatter plot mapping structural boundaries.

![Alt Text](images/outliers.png)


5. **Cross-Validation & Polynomial Tuning**

- Splits the historical log dataset into an 80/20 train/test matrix.

- Tests polynomial features ranging from degrees 1 through 10.

![Alt Text](images/polynomial_degrees.png)

- Logs performance using a 5-fold cross-validation Mean Squared Error (MSE) script.

- Plots train vs. validation curve configurations over a logarithmic scale.

![Alt Text](images/polynomial_cross_validation.png)


6. **Regularization & Hyperparameter Search**

- Loops through L2 regularization strengths (alphas = [0.001, 0.1, 1.0, 10.0, 100.0]) via a Ridge estimator.

- Plots Mean Squared Error (MSE) metrics against penalty parameters to lock down the optimal mathematical trade-off.

![Alt Text](images/ridge_regression_alpha.png)

![Alt Text](images/ridge_regression_alpha_cross_validation.png)


7. **Incremental Feature Expansion**

The pipeline scales model performance up by sequentially encoding and testing new feature combinations in a machine learning Pipeline:

	**Iteration 1**: Base numerical attributes (vehicle_age + odometer).
	
	**Iteration 2**: Adds vehicle wear metrics ('condition_encoded').

![Alt Text](images/prediction_with_vehicle_condition.png)

	
	**Iteration 3**: Adds brand presence indicators ('manufacturer_encoded').

![Alt Text](images/prediction_with_condition_n_manufacturer.png)


	**Iteration 4**: Adds structural body design classes ('type_encoded').

![Alt Text](images/prediction_with_condition_n_manufacturer_n_type.png)


	**Iteration 5**: Final evaluation of different Ridge alpha values to obtain the best alpha for the model using all features above (vehicle_age + odometer + condition_encoded + manufacturer_encoded + type_encoded), including 5k cross validation procedure.

![Alt Text](images/best_ridge_alpha_for_final_prediction.png)

![Alt Text](images/final_model.png)



📊 **Evaluation Visualizations**

The script automatically produces four critical diagnostic plots:

1.  **Correlation Heatmap**: Visualizes feature interactions and highlights the column with the highest direct linear relationship to 'price'.

2.  **Complexity Graph**: Plots Training vs. Validation MSE across polynomial degrees to pinpoint the optimal degree.

3.  **Alpha Shrinkage Chart**: Visualizes model test error variations across regularized weight thresholds.

4.  **Actual vs. Predicted Scatter Plot**: Draws a diagonal identity line ('y = x') against final predictions to display accuracy spreads.


**Summary Findings**

The analysis successfully developed and refined a Ridge Regression model to predict used car prices, achieving consistent incremental improvements through feature engineering and data cleaning.

**Key Insights**

**Primary Predictors**: Mileage (odometer) and vehicle_age possess the strongest correlations to car price, serving as the foundation for the baseline model.

**Data Cleansing**: Outliers with anomalous odometer readings comprised 5.97% of the dataset and were removed to mitigate right-skewness.

**Feature Expansion**: Incorporating three categorical variables (condition, manufacturer, type) encoded via OrdinalEncoder paired with a degree-2 Polynomial Regression consistently lowered the Mean Squared Error (MSE).

**Model Performance**: The final optimized model achieved an MSE of 114,182,902, translating to a Root Mean Squared Error (RMSE) of approximately $10,685, which aligns with standard errors found in real-world automotive datasets like Kaggle's Craigslist data.

**Hyperparameter Stability**: Across all iterations—both before and after introducing categorical features—Cross-Validation consistently identified 10.0 as the optimal alpha value for the Ridge model.


🏢 **Suggestions for Car Dealers**

**Prioritize Age and Mileage in Appraisals**: Train sales and acquisition teams to focus heavily on vehicle_age and odometer. Because these two factors dictate the vast majority of a vehicle's market value, secondary traits like color or minor cosmetic features should not heavily alter baseline trade-in offers.

**Be Cautious with "Niche" Pricing**: The model shows that adding manufacturer and vehicle type only shifted pricing accuracy by a tiny fraction (~0.37%). Do not overpay for specific brands or body types assuming they carry a massive premium, as the broader market prices them closely when age and mileage are equal.

**Audit Digital Inventory Data**: Since 5.97% of the data contained corrupted or extreme outlier odometer readings, ensure your dealership's digital listings are audited for typos. Missing a digit or entering a bad reading severely skews automated valuation tools (like Kelley Blue Book or your internal software), leading to mispriced inventory.

**Expect a Margin of Variance**: Recognize that standard predictive algorithms carry an average error margin of roughly $10,685. Rely on localized human expertise to fine-tune the final retail price, rather than trusting automated pricing software blindly.


💻 **How to Run**

1. Create a root directory named 'data' and store your dataset inside it as 'vehicles.csv'.

2. Open the file in Google Colab or your local Jupyter notebook server.

3. Execute all code cells sequentially to process data, view evaluations, and evaluate final predictive scores.


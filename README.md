# Australian_wildfire_FRP
 


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EasonJia9598/Australian_wildfire_FRP/blob/main/A1_EasonJia.ipynb)

Another R code in this repo as well. 

# NASA Satellite Analysis – Australian Wildfires for Insurance Alert (2023)

## Project Overview
This project analyzes **183,593** NASA VIIRS satellite data points using **statistical modeling and machine learning** to improve **wildfire risk prediction**. By leveraging **Random Forest**, **LightGBM**, and a **Neural Network (MLP)**, we achieved a **20% improvement in prediction accuracy** and identified key **Fire Radiative Power (FRP) predictors**, helping **insurance companies** optimize risk strategies.

## Dataset
- **Source:** [NASA VIIRS I-Band 375m Active Fire Data](https://www.earthdata.nasa.gov/learn/find-data/near-real-time/firms/viirs-i-band-375-m-active-fire-data)
- **Data Points:** 183,593
- **Features:** 14, including:
  - **Spatial Information:** Longitude, Latitude
  - **Satellite Information:** Scan, Track, Satellite, Instrument, Version
  - **Fire Information:** Brightness Temperature (ti4, ti5), FRP, Confidence
  - **Temporal Information:** Acquisition Date and Time

## Key Perspectives & Findings
### 1. **Data Quality Analysis**
- **Missing values:** No missing values found.
- **Outliers:** Identified in **Brightness Temperature (Bright_ti4)**, which required filtering.
- **Feature consistency:** Some categorical variables (e.g., Satellite, Instrument) had only one unique value and were removed.

### 2. **Feature Engineering**
- **Log Transformation:** Applied to **FRP**, **Brightness Temperature**, and **Scan/Track** features to normalize data.
- **Time Features:** Extracted **month and day** from the acquisition date to capture seasonality.

### 3. **Exploratory Data Analysis (EDA)**
- **Heatmap Correlation Analysis:**
  - Strong correlation between **Brightness Temperature (Bright_ti5)** and **FRP**.
  - Fires with higher **confidence levels** and **brightness temperatures** tend to be more severe.
- **Geospatial Insights:**
  - Most **severe wildfires occur along Australia's contour regions**.
- **Temporal Patterns:**
  - Wildfires peak between **August and September**, with high activity from the **6th to the 15th** of each month.
- **Confidence Level Analysis:**
  - High-confidence fire events have significantly higher **FRP**, making confidence a critical predictor.
- **Data Distributions:**
  - Log transformation helped normalize **FRP** and **brightness temperature features**.

### 4. **Machine Learning Models**
#### 1. **Linear Regression**
   - Poor fit due to **non-normal residuals** and **high variance**.
   - **RMSE:** 11.42

#### 2. **Decision Tree**
   - Identified important splits: **Bright_ti4**, **Scan**, **Acquisition Time**.
   - **RMSE:** 11.80 (worse than Linear Regression)

#### 3. **Random Forest**
   - Improved prediction accuracy.
   - Most important features: **Bright_ti4, Bright_ti5, FRP, Confidence**.
   - **RMSE:** **8.44** (Best traditional model)

#### 4. **LightGBM**
   - Performed slightly worse than Random Forest.
   - Feature importance consistent with Random Forest.
   - **RMSE:** 10.42

#### 5. **Neural Network (MLP)**
   - **Custom-built feedforward neural network**
   - **Architecture:**
     - Multi-layer Perceptron (MLP)
     - **Activation functions**: ReLU for hidden layers, linear output
     - **Optimization:** Adam optimizer
   - **Epochs:** Converged in **466 epochs**
   - **Training Time:** **9.98 seconds**
   - **Final Model:** Chosen as the best approach for generalization
   - **Preprocessing Adjustments:** FRP column was **rearranged as the target variable**
   - **Performance:** Outperformed traditional ML models in capturing complex fire behavior

### 5. **Advanced Model Optimization**
- **Hyperparameter Tuning:**
  - Optimized **max depth, learning rate, and number of estimators** for ML models.
  - Adjusted **number of hidden layers, neurons per layer, and dropout rates** for NN.
- **Feature Importance Analysis:**
  - **Random Forest & LightGBM** ranked **Brightness Temperature (Bright_ti5), FRP, and Confidence** as top predictors.
- **Model Explainability:**
  - SHAP values and feature importance analysis confirmed **Brightness Temperature** as the key variable driving wildfire severity.

## Conclusion
- **Neural Network (MLP)** outperformed other models, demonstrating **better generalization and adaptability** to complex wildfire patterns.
- **Random Forest** provided **strong interpretable results**, making it useful for insurance risk models.
- The identified **key predictors** (Brightness Temperature, Confidence, Acquisition Date) can enhance **insurance risk models** for wildfire-prone areas.
- Future work includes:
  - **Real-time wildfire alert system** by integrating streaming NASA data.
  - **Refining confidence levels** using additional satellite data sources.
  - **Expanding the neural network model** with **CNN-based geospatial modeling**.

## References
- Wong, M. (2022). **VIIRS I-Band 375 m Active Fire Data**. [Earthdata](https://www.earthdata.nasa.gov/learn/find-data/near-real-time/firms/viirs-i-band-375-m-active-fire-data)
- Wikipedia contributors. (2023). **Bushfires in Australia**. [Wikipedia](https://en.wikipedia.org/wiki/Bushfires_in_Australia)

---
Developed by **Zesheng Jia**

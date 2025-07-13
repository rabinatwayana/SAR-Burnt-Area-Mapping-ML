# Wildfire Burned Area Mapping using SAR and Machine learning Approach: Random Forest and XGBoost

[Presentation Slides](https://drive.google.com/file/d/1YQ7ks8pf0-SPcX9wqhoU0GPQTIKpKZtt/view?usp=sharing)

[Handout](https://github.com/user-attachments/files/20343101/A1_Handout_Twayana.pdf)

## 🛠 Project Setup
   Clone the repository:
   ```bash
   git clone https://github.com/yourusername/burned-area-mapping.git
   ```
   
   Move to the project directory
   ```bash
   cd burned-area-mapping
   ```

   Create conda environment
   ```bash
   conda env create -f environment.yml
   ```
   Activate environment
   ```bash
   conda activate wildfire_burnet_env
   ```

## Implementation

Machine Learning folders consists of notebooks for following tasks:

   ### 1. Data preparation
This notebook handles the initial data preparation steps, including:

Loading raw SAR satellite data.

Cleaning and filtering the data (e.g., removing noise, handling missing values).

Feature extraction such as calculating backscatter coefficients or texture measures.

Normalizing and scaling features to prepare them for model training.

Splitting the dataset into training and testing sets.

### 2. Feature Extraction
Extract the SAR related features such as RVI, RBD, RBR etc.

### 3. Training models
Performing hyperparameter tuning using techniques like grid search.

Evaluating model performance with metrics such as accuracy, precision, recall, and F1-score.

Saving the best model for later use in prediction.

### 3. Prediction
Load the trained model.

Apply the model to new SAR images to generate burned area predictions.

Export results as GeoTIFF and png for further GIS analysis.

### 4. Feature Importance
Visualize which input features contribute the most to the predictive performance of the trained machine learning model




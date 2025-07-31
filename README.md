# Wildfire Burned Area Mapping using SAR and Machine learning Approach: Random Forest and XGBoost

[Presentation Slides](https://drive.google.com/file/d/1YQ7ks8pf0-SPcX9wqhoU0GPQTIKpKZtt/view?usp=sharing)

[Handout](https://github.com/user-attachments/files/20343101/A1_Handout_Twayana.pdf)

[Pre-processed Dataset](https://drive.google.com/drive/folders/1-Iof7eNetvJ0AO8bSaUKq_1naZLuM12q?usp=share_link)

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
This notebook handles the initial data preparation steps, including generation of ground truth, cropping the data into same extent, and creating train and test tile selection images.

Splitting the dataset into training and testing sets.

### 2. Feature Extraction
Extract the SAR related features such as RVI, RBD, RBR etc.

### 3. Training models
This notebook performs hyperparameter tuning using techniques like grid search, evaluates model performance with metrics such as accuracy, precision, recall, and F1-score, and saves the best model for future predictions.

### 3. Prediction
his notebook loads the trained model, applies it to new SAR images to generate burned area predictions, and exports the results as GeoTIFF and PNG files for further GIS analysis.

### 4. Feature Importance
Visualize which input features contribute the most to the predictive performance of the trained machine learning model




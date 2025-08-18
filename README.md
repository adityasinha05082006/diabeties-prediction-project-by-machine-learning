# Diabetic Prediction Project

This project predicts the likelihood of diabetes in a patient using various machine learning models.

## How to Run
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Prepare data:
   - Place `diabetes.csv` in the `data/` folder.
3. Train the model:
   ```
   python src/model_training.py
   ```
4. Make predictions:
   ```
   python src/predict.py
   ```

## Project Structure
- `data/`: Dataset files
- `src/`: Source code
- `model/`: Saved models
- `notebooks/`: EDA and experiments

## Credits
- [Kaggle Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import sys
import matplotlib.pyplot as plt # Added for visualization
import os # Added for path handling


# --- 1. Data Loading and Preparation (ETL function) ---
def load_and_prepare_student_data():
    """
    Loads 'student_model.csv', renames columns (G1, G2, G3) to descriptive
    feature names, and engineers the Attendance_Percentage feature.
    
    If you need to use an absolute path because the file is not in the script's 
    folder, update the FILE_PATH variable below (e.g., FILE_PATH = 'C:/data/student_model.csv').
    """
    # Define the file path. We use the absolute path as requested.
    FILE_PATH = 'C:\\Users\\saakshii\\student_model.csv'

    try:
        # Load data, assuming semicolon delimiter (typical for this dataset)
        df = pd.read_csv(FILE_PATH, sep=',')
        
        # Rename columns to match the features used in the model
        df = df.rename(columns={'G1': 'Previous_Semester_Score_%', 
                                'G2': 'Attendance_%', 
                                'G3': 'No_of_Activities_Participated',
                                'G4': 'Semester_S6',
                                'G5': 'Semester_S7',
                                'G6': 'Semester_S8',
                                'G7':'Final_Score'})
        
        # Engineer the 'Attendance_Percentage' feature
        # NOTE: This part seems incomplete in the original, but is maintained.
        if 'absences' in df.columns and df['absences'].max() > 0:
            max_absences = df['absences'].max() 
            df['Attendance_Percentage'] = 100 - (df['absences'] / max_absences * 100)
        else:
              # Default to 100% if no absence data is present
            df['Attendance_Percentage'] = 100.0
        
        print("Data loaded and prepared successfully.")
        return df
    
    except FileNotFoundError:
        print(f"CRITICAL ERROR: '{FILE_PATH}' not found. Cannot load data.")
        print("Please ensure the file is in the correct location.")
        return pd.DataFrame()

# Load data
df = load_and_prepare_student_data()
if df.empty:
    sys.exit()

# ----------------------------------------------------------------------
# --- 2. Define Features and Target ---

FEATURES = [
     'Previous_Semester_Score_%',  
     'Attendance_%', 
     'No_of_Activities_Participated',
     'Semester_S6',
     'Semester_S7',
     'Semester_S8'
]
TARGET = 'Final_Score'

# Final check for columns before subsetting the DataFrame
missing_features = [f for f in FEATURES if f not in df.columns]
if missing_features:
    print(f"Error: Missing features in DataFrame: {missing_features}. Check your data preparation function.")
    sys.exit()

X = df[FEATURES]
y = df[TARGET]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------------------------------------------------
# --- 3. Finalize the Model with BEST Parameters ---

# Optimized parameters from previous steps
BEST_PARAMS = {
    'n_estimators': 200, 
    'max_depth': 10, 
    'min_samples_split': 2 
}

print("\n--- Finalizing Best Random Forest Regressor Model ---")

best_model = RandomForestRegressor(
    n_jobs=-1,
    random_state=42,
    **BEST_PARAMS
)

# Train the final model
best_model.fit(X_train, y_train)
print(f"Model trained with Best Parameters: {BEST_PARAMS}")

# Evaluate the model
y_pred_tuned = best_model.predict(X_test)
r2_tuned = r2_score(y_test, y_pred_tuned)
rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_tuned))

print(f"\nModel Performance on Test Set (using Best Params):")
print(f"R-squared (R^2): {r2_tuned:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_tuned:.2f}")


# ----------------------------------------------------------------------
# --- 4. Feature Importance Calculation and Visualization (NEW SECTION) ---

# 1. Calculate Feature Importances
feature_importances = best_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': FEATURES,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print("\n--- Feature Importance Report ---")
print(importance_df.to_string(index=False))

# 2. Create Visualization
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='#3b82f6')
plt.xlabel("Relative Importance (Gini)")
plt.title("Feature Importance for Final Score Prediction (Random Forest)")
plt.gca().invert_yaxis() # Highest importance on top

# 3. Save the Visualization File
VISUALIZATION_PATH = 'feature_importance_chart.png'
# Use os.path.join for robust path creation
plt.savefig(VISUALIZATION_PATH, bbox_inches='tight') 
print(f"\n✅ Visualization saved successfully to: {os.path.abspath(VISUALIZATION_PATH)}")

# ----------------------------------------------------------------------
# --- 5. The "Best Prediction" (Single Prediction) ---

# New student data for prediction (based on your Example_Prediction.csv)
new_student_data = pd.DataFrame([[80, 90, 3, 40, 70, 96]], columns=FEATURES)

# Make the prediction
predicted_score_array = best_model.predict(new_student_data)
predicted_score = predicted_score_array[0]

#--- 6. Calculate and Display Rank (FIXED ROBUST LOGIC) ---

# 1. Predict scores for the entire class (X) using the final model
all_student_predictions = best_model.predict(X)

# 2. Count the number of students who scored HIGHER than the new student
students_who_scored_higher = np.sum(all_student_predictions > predicted_score)
students_who_scored_equal = np.sum(all_student_predictions == predicted_score)

# The rank is the number of people who beat you, plus one.
current_class_rank = students_who_scored_higher + 1
total_students = len(all_student_predictions) + 1 # +1 for the new student

# --- Display the Final Result ---
print("\n#####################################################")
print("### ✅ FINAL PREDICTION AND RANKING REPORT ###")
print("#####################################################")
print("Student Profile:")
print(new_student_data.to_string(index=False))
print("-" * 50)
print(f"Predicted Upcoming Semester Score: {predicted_score:.2f} / 100") 
print(f"Projected Class Rank: {current_class_rank} out of {total_students} students")
print("#####################################################")

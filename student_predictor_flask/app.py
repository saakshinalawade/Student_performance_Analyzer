import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from flask import Flask, render_template, request # Added Flask imports

# --- 0. Global Setup ---
# The model and full dataset are loaded and trained once when the server starts.
app = Flask(__name__)

# --- 1. Data Loading and Preparation (ETL function) ---
def load_and_prepare_student_data():
    """
    Loads 'student_model.csv' from the specified absolute path, renames columns, 
    and returns the processed DataFrame.
    """
    # Absolute file path, maintained as requested.
    FILE_PATH = 'C:\\Users\\saakshii\\student_model.csv'

    try:
        # Load data, assuming comma delimiter
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
        if 'absences' in df.columns and df['absences'].max() > 0:
            max_absences = df['absences'].max() 
            df['Attendance_Percentage'] = 100 - (df['absences'] / max_absences * 100)
        else:
            df['Attendance_Percentage'] = 100.0
        
        print("✅ Data loaded and prepared successfully.")
        return df
    
    except FileNotFoundError:
        print(f"❌ CRITICAL ERROR: '{FILE_PATH}' not found. Cannot load data.")
        print("Please ensure the file is in the correct location.")
        sys.exit() # Terminate server if data cannot be loaded
    except Exception as e:
        print(f"❌ CRITICAL ERROR during data loading: {e}")
        sys.exit()

# Load data globally when the script runs
df_full = load_and_prepare_student_data()

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

X = df_full[FEATURES]
y = df_full[TARGET]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------------------------------------------------
# --- 3. Finalize the Model with BEST Parameters (Global Training) ---

BEST_PARAMS = {
    'n_estimators': 200, 
    'max_depth': 10, 
    'min_samples_split': 2 
}

best_model = RandomForestRegressor(
    n_jobs=-1,
    random_state=42,
    **BEST_PARAMS
)

# Train the model once on server startup
best_model.fit(X_train, y_train)
print("✅ Random Forest Regressor Model trained globally.")

# Calculate Feature Importances globally for display in the HTML
feature_importances = best_model.feature_importances_
IMPORTANCE_DATA = pd.DataFrame({
    'Feature': FEATURES,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)


# ----------------------------------------------------------------------
# --- 4. Flask Routes ---

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Handles both the initial page load (GET) and form submission (POST).
    """
    
    # Initialize variables for the template
    prediction_result = None
    rank_result = None
    # Calculate total students based on loaded data + 1 for the new predicted student
    total_students = len(df_full) + 1 

    if request.method == 'POST':
        try:
            # 1. Collect and process input from the form (using field names from index.html)
            input_data = {}
            input_data['Previous_Semester_Score_%'] = float(request.form.get('prev_score'))
            input_data['Attendance_%'] = float(request.form.get('attendance'))
            input_data['No_of_Activities_Participated'] = int(request.form.get('activities'))
            
            selected_semester = request.form.get('semester')
            
            # Create the input feature vector based on the selected semester
            new_student_features = {
                'Previous_Semester_Score_%': input_data['Previous_Semester_Score_%'],
                'Attendance_%': input_data['Attendance_%'],
                'No_of_Activities_Participated': input_data['No_of_Activities_Participated'],
                'Semester_S6': 1.0 if selected_semester == 'Semester_S6' else 0.0,
                'Semester_S7': 1.0 if selected_semester == 'Semester_S7' else 0.0,
                'Semester_S8': 1.0 if selected_semester == 'Semester_S8' else 0.0,
            }

            # Create DataFrame for prediction, ensuring column order matches training data (FEATURES)
            new_student_df = pd.DataFrame([new_student_features], columns=FEATURES)

            # 2. Prediction
            predicted_score_array = best_model.predict(new_student_df)
            predicted_score = predicted_score_array[0]
            prediction_result = f"{predicted_score:.2f}"

            # 3. Rank Calculation
            all_student_predictions = best_model.predict(X)
            # Find how many existing students scored strictly HIGHER
            students_who_scored_higher = np.sum(all_student_predictions > predicted_score)
            
            # Rank is (number who beat you) + 1
            current_class_rank = students_who_scored_higher + 1
            rank_result = current_class_rank

        except Exception as e:
            # Catch errors like missing input or invalid type conversion
            print(f"Prediction Error: {e}")
            prediction_result = "ERROR"
            rank_result = "N/A"

    # Render the template, passing the feature importance data and results
    return render_template('index.html', 
                           predicted_score=prediction_result,
                           projected_rank=rank_result,
                           total_students=total_students,
                           # Convert DataFrame records to a list of dictionaries for Jinja
                           importance_data=IMPORTANCE_DATA.to_dict('records'))

# ----------------------------------------------------------------------
# --- 5. Run Server ---
if __name__ == '__main__':
    app.run(debug=True)

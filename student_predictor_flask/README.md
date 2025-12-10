🎓 Student Performance Analyzer

An AI-based web application that predicts and analyzes student academic performance using machine learning techniques.
This project integrates data preprocessing, model training, and interactive web visualization to demonstrate real-world applications of AI and analytics in education.

🚀 Overview

The Student Performance Analyzer utilizes historical academic data such as attendance, study hours, and grades to predict scores, ranks, and performance categories.
It provides educators and students with interpretable visual insights into the factors that influence academic success.

🧠 Key Features

📈 Dynamic prediction of student performance (marks, rank, and category)

🧮 Random Forest Regression and Decision Tree Classification for predictions

📊 Interactive Gini-style visualization of predicted vs. actual performance

🧠 Integrated exploratory data analysis (EDA) and feature correlation insights

💻 Flask-based web interface with responsive frontend and real-time interaction

⚙️ Tech Stack
Layer	Technologies
Frontend	HTML, Tailwind CSS, JavaScript, Chart.js
Backend	Flask (Python)
Machine Learning	scikit-learn, pandas, numpy
Visualization	Matplotlib, Chart.js
Storage	CSV Dataset
📂 Project Structure
student-performance-analyzer/
│
├── static/
│   ├── css/ → styles.css
│   ├── js/ → chart.js, main.js
│   └── assets/
│
├── templates/
│   └── index.html
│
├── app.py
├── model.pkl
├── requirements.txt
└── README.md

🧩 How It Works

Upload Dataset (CSV format) containing student academic details.

Preprocessing & Model Prediction: ML models analyze features and predict performance metrics.

Flask Backend: Processes input and serves predictions to the UI.

Frontend Visualization: Displays results and comparisons interactively using Chart.js.

🧾 Sample Output
Parameter	Example Output
Predicted Score	78.5
Predicted Rank	5
Performance Status	Excellent
Visualization	Gini-style comparison of predicted vs. actual scores
🖼️ Screenshots & Demo

Add your screenshots in this section (GitHub automatically renders them beautifully).
You can include them like this once you upload to your repo:

### Application Interface
![App UI](static/assets/ui_home.png)

### Performance Visualization
![Gini Chart](static/assets/gini_chart.png)


📌 Tip: Place your screenshots inside /static/assets/ and refer to them using relative paths as shown above.

🌱 Future Scope

Integration with SQL/Firebase for persistent data storage

Deep Learning model adaptation for improved accuracy

Class-wise and subject-level performance dashboards

Personalized study recommendations based on prediction trends

👩‍💻 Author

Saakshi Chandrakant Nalawade
Third Year Computer Engineering | AI & ML (Honours)
University of Mumbai
📍 Navi Mumbai, Maharashtra
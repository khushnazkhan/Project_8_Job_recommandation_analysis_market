# Job Analysis & Recommendation System
A comprehensive Streamlit-based web application for job market analysis, salary prediction, and job recommendation using machine learning models.

# Features
Dataset Flexibility: Upload custom CSV files or use the default dataset

Salary Band Prediction: Predicts salary bands using a pre-trained machine learning model

Job Recommendation Engine: Finds relevant job matches based on text input

Interactive Visualizations: Displays data insights using Plotly charts

Data Export: Export processed data with selected columns

Installation
Clone the repository:

bash
git clone <your-repo-url>
cd job-analysis-recommender
Install required dependencies:

bash
pip install -r requirements.txt
Usage
Place your pre-trained models in the models directory:

salary_band.pkl - For salary prediction

recommender.pkl - For job recommendations

tfidf_vectorizer.pkl - For text processing

Run the application:

bash
streamlit run main.py
Access the application through your web browser (typically at http://localhost:8501)

Requirements
Python 3.7+

Streamlit

Pandas

Scikit-learn

Joblib

Plotly

NumPy

Model Training
To train the models, you'll need to:

Prepare your job dataset with relevant features

Train a salary band classification model

Train a recommendation system (content-based or collaborative filtering)

Save the models using joblib in the specified models directory

File Structure
text
project/
├── main.py              # Main application file
├── final_processed_jobs.csv  # Default dataset (optional)
├── models/              # Pre-trained models directory
│   ├── salary_band.pkl
│   ├── recommender.pkl
│   └── tfidf_vectorizer.pkl
└── requirements.txt     # Python dependencies
Contributing
Fork the repository

Create a feature branch

Commit your changes

Push to the branch

Create a Pull Request

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
Built with Streamlit for the web interface

Uses Scikit-learn for machine learning capabilities

Plotly for interactive visualizations

<img width="1600" height="860" alt="forcast" src="https://github.com/user-attachments/assets/514b5a72-8ccd-4655-8b69-c4fedf8a9704" />

![Uploading remote work.PNG…]()
<img width="1600" height="860" alt="month xccatagories" src="https://github.com/user-attachments/assets/90920e6e-a3f3-4a3d-81ef-64898ef0f763" />


<img width="1600" height="860" alt="remote work" src="https://github.com/user-attachments/assets/22edad4d-3398-42d9-b76b-cc261bcaa75a" />
<img width="1600" height="860" alt="dashboard" src="https://github.com/user-attachments/assets/7d1fd60b-98f7-49b9-8426-c65332a5ccfe" />
<img width="1600" height="860" alt="dashboard2" src="https://github.com/user-attachments/assets/962127b9-aefb-437c-93ea-a3eee0094364" />
<img width="1681" height="860" alt="dash3" src="https://github.com/user-attachments/assets/1a421238-cece-4633-b9b9-0b287f07ef66" />
 comprehensive Streamlit-based web application for job market analysis, salary prediction, and job recommendation using machine learning models.

https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white
https://img.shields.io/badge/scikit--learn-%2523F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white
https://img.shields.io/badge/Plotly-%25233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=wh

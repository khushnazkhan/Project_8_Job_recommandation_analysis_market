import streamlit as st
import pandas as pd
import joblib, os
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import numpy as np
import io

# ---------------- Config ----------------
st.set_page_config(page_title="Job Analysis & Recommender", layout="wide")

#  Path to models folder
MODEL_DIR = r"E:\nexthike\models"

# ---------------- Load Models ----------------
def safe_load(path):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.warning(f" Could not load {os.path.basename(path)}: {e}")
    return None

salary_band_model = safe_load(os.path.join(MODEL_DIR, "salary_band.pkl"))
recommender_model = safe_load(os.path.join(MODEL_DIR, "recommender.pkl"))
vectorizer = safe_load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

# ---------------- Upload / Load Data ----------------
st.sidebar.header(" Upload Dataset")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.sidebar.success("Custom dataset uploaded")
elif os.path.exists("final_processed_jobs.csv"):
    # Read in chunks to handle large files
    chunks = []
    for chunk in pd.read_csv("final_processed_jobs.csv", chunksize=10000):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    st.sidebar.info("Using default final_processed_jobs.csv")
else:
    st.error("No dataset found (upload a CSV or keep final_processed_jobs.csv in root)")
    st.stop()

# Normalize column names
df.columns = [c.lower().strip() for c in df.columns]

# ---------------- Header ----------------
st.title("📊 Job Market Analysis & Recommendation System")

# ---------------- Overview ----------------
c1, c2, c3 = st.columns(3)
c1.metric("Total Jobs", df.shape[0])
c2.metric("Countries", df['country'].nunique() if 'country' in df.columns else "N/A")

# Improved salary column detection
salary_col = None
possible_salary_columns = ['salary', 'hourly_rate', 'budget', 'pay', 'compensation', 'wage', 'income']

for col in df.columns:
    cname = col.lower()
    for possible_col in possible_salary_columns:
        if possible_col in cname:
            salary_col = col
            break
    if salary_col:
        break

# If still not found, look for numeric columns that might be salary
if not salary_col:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        salary_col = numeric_cols[0]  # Use first numeric column as fallback

if salary_col:
    if pd.api.types.is_numeric_dtype(df[salary_col]):
        c3.metric("Median Salary", round(df[salary_col].median(), 2))
    else:
        c3.metric("Median Salary", "N/A (non-numeric)")
else:
    c3.metric("Median Salary", "No salary column found")

st.markdown("---")

# ---------------- Salary Band Prediction ----------------
st.subheader("Salary Band Prediction")

if salary_band_model is not None and salary_col:
    if pd.api.types.is_numeric_dtype(df[salary_col]):
        try:
            # Sample a subset of data for prediction to save memory
            sample_size = min(10000, len(df))
            sample_df = df.sample(sample_size, random_state=42) if len(df) > sample_size else df
            
            # Get the expected feature names from the model
            if hasattr(salary_band_model, 'feature_names_in_'):
                expected_features = salary_band_model.feature_names_in_
                
                # Create a DataFrame with all expected features
                prediction_data = pd.DataFrame(columns=expected_features)
                
                # Add the salary column with the correct name
                salary_feature_name = None
                for feature in expected_features:
                    if any(sal_col in feature.lower() for sal_col in possible_salary_columns):
                        salary_feature_name = feature
                        break
                
                if salary_feature_name:
                    prediction_data[salary_feature_name] = sample_df[salary_col].values
                else:
                    # If no specific salary feature name found, use the first feature
                    prediction_data[expected_features[0]] = sample_df[salary_col].values
                
                # Fill missing values with 0
                prediction_data = prediction_data.fillna(0)
                
                # Make predictions on sample
                sample_df["Predicted_Band"] = salary_band_model.predict(prediction_data)
                st.success(f"Salary bands predicted for a sample of {sample_size} jobs using `{salary_col}` column")
                fig = px.histogram(sample_df, x="Predicted_Band", color="Predicted_Band",
                                   title="Predicted Salary Bands (Sample)")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(sample_df[[salary_col, "Predicted_Band"]].head())
            else:
                # Fallback for models without feature names
                sample_df["Predicted_Band"] = salary_band_model.predict(sample_df[[salary_col]])
                st.success(f"Salary bands predicted for a sample of {sample_size} jobs using `{salary_col}` column")
                fig = px.histogram(sample_df, x="Predicted_Band", color="Predicted_Band",
                                   title="Predicted Salary Bands (Sample)")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(sample_df[[salary_col, "Predicted_Band"]].head())
        except Exception as e:
            st.error(f"Error in salary band prediction: {e}")
            st.info("The model expects different features than available in the dataset.")
    else:
        st.warning("Salary column is not numeric, cannot run band prediction.")
else:
    st.warning(" Salary band model not found or no salary column detected")

st.markdown("---")

# ---------------- Job Recommendation ----------------
st.subheader(" Job Recommendation Engine")

# detect text column
text_col = None
possible_text_columns = ['job_description', 'description', 'job_title', 'title', 'position']
for col in possible_text_columns:
    if col in df.columns:
        text_col = col
        break
if text_col is None:
    text_col = df.columns[0]  # fallback to first column

query = st.text_input("Enter a job title / description", "data scientist")

if st.button("Recommend"):
    if recommender_model is not None:
        try:
            recs = recommender_model.recommend(query, topn=5)
            if isinstance(recs, pd.DataFrame):
                # Include salary column if available
                cols_to_show = [c for c in ["job_title", "title", "country", "location"] if c in recs.columns]
                if salary_col and salary_col in recs.columns:
                    cols_to_show.append(salary_col)
                st.table(recs[cols_to_show].head(5))
            elif isinstance(recs, (list, np.ndarray)):
                rec_df = df[df[text_col].isin(recs)]
                cols_to_show = [c for c in ["job_title", "title", "country", "location"] if c in rec_df.columns]
                if salary_col and salary_col in rec_df.columns:
                    cols_to_show.append(salary_col)
                st.table(rec_df[cols_to_show].head(5))
            else:
                st.info(" Unexpected recommender output format")
        except Exception as e:
            st.error(f" Error using recommender model: {e}")
    elif vectorizer is not None:
        try:
            # Use a sample for TF-IDF to save memory
            sample_size = min(5000, len(df))
            sample_df = df.sample(sample_size, random_state=42) if len(df) > sample_size else df
            
            X = vectorizer.transform(sample_df[text_col].astype(str))
            qv = vectorizer.transform([query])
            sim = cosine_similarity(qv, X).flatten()
            idx = sim.argsort()[-5:][::-1]
            recs = sample_df.iloc[idx]
            cols_to_show = [c for c in ["job_title", "title", "country", "location"] if c in recs.columns]
            if salary_col and salary_col in recs.columns:
                cols_to_show.append(salary_col)
            st.table(recs[cols_to_show])
        except Exception as e:
            st.error(f"TF-IDF recommendation failed: {e}")
    else:
        mask = df[text_col].str.contains(query, case=False, na=False)
        result_df = df[mask]
        cols_to_show = [c for c in ["job_title", "title", "country", "location"] if c in result_df.columns]
        if salary_col and salary_col in result_df.columns:
            cols_to_show.append(salary_col)
        st.table(result_df[cols_to_show].head(5))

st.markdown("---")

# ---------------- Export ----------------
st.subheader("Export Processed Data")

# Let user select columns to export to reduce file size
st.write("Select columns to export (selecting fewer columns will reduce file size):")
all_columns = df.columns.tolist()
selected_columns = st.multiselect("Columns to export", all_columns, default=all_columns[:min(10, len(all_columns))])

if selected_columns:
    # Export in chunks to avoid memory issues
    @st.cache_data
    def convert_df_to_csv(df, columns):
        output = io.StringIO()
        # Write in chunks
        chunk_size = 10000
        for i in range(0, len(df), chunk_size):
            chunk = df[columns].iloc[i:i+chunk_size]
            if i == 0:
                chunk.to_csv(output, index=False)
            else:
                chunk.to_csv(output, index=False, header=False)
        return output.getvalue().encode("utf-8")
    
    csv = convert_df_to_csv(df, selected_columns)
    st.download_button(
        " Download CSV", 
        csv, 
        "processed_jobs.csv", 
        "text/csv",
        help="Download may take a while for large datasets"
    )
else:
    st.warning("Please select at least one column to export.")

st.markdown("---")
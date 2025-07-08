import pandas as pd
import streamlit as st
from utils import *
import pandasai as pai
from pandasai_openai import OpenAI



# st.set_page_config(layout="wide")
apply_custom_css()
#st.title("💬 Chat with your Data")
# PandasAI configuration
pai.config.set({
    "llm": OpenAI(api_token=st.secrets["openai"]["api_key"]),
})

logo_path = "logo.png"  # Local file or URL
#st.logo(logo_path)

# Stop if no data is loaded
# if not st.session_state.get("summary_generated", False):
#     st.warning("Please upload the data on the Input page to access this page.")
#     st.stop()

if __name__ == "__main__":
    # if st.session_state.get("summary_generated", False):
    modelled_datasets = st.file_uploader(
        "Upload Data Dictionary, Modelled Data (train, test, shap)",
        type=["csv", "xlsx"],
        accept_multiple_files=True,
        label_visibility = "collapsed"
    )

    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = None

    if modelled_datasets:
        dataset_mapping = {"m1": None, "m2": None, "dictionary": None}

        
        for modelled_dataset in modelled_datasets:
            dataset_name = modelled_dataset.name.lower()  # Convert to lowercase for case-insensitive checks
            file_format = "csv" if dataset_name.endswith(".csv") else "xlsx" if dataset_name.endswith(".xlsx") else None

            if "dictionary" in dataset_name:
                key = "dictionary"
            elif "blend" in dataset_name:
                key = "blend"
            elif "m2" in dataset_name:
                key = "m2"
            else:
                continue

            if file_format == "xlsx":
                dataset_mapping[key] = pd.read_excel(modelled_dataset)
            elif file_format == "csv":
                df = pd.read_csv(modelled_dataset)
                if 'Unnamed: 0' in df.columns:
                    df = df.drop(columns='Unnamed: 0')
                dataset_mapping[key] = df
        
        # df = st.session_state.X_train_test_shap_M1
        if dataset_mapping['blend'] is not None:
            st.session_state.retail_data_dictionary = dataset_mapping["dictionary"]
            st.session_state.X_train_test_shap_M1 = dataset_mapping["blend"]

            df_filtered, user_input = display_filter(st.session_state.X_train_test_shap_M1)
            chatbot(df_filtered, user_input)
    else:
        st.warning("Please upload the data to access this page")
    
    



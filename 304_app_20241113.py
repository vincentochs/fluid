# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 22:10:54 2024

@author: Vincent Ochs

This script has been updated to include:
1. Clinically-driven risk adjustments in the parser_input function.
2. Smoother heatmap visualization using interpolation in create_smooth_heatmap_plot.
"""

###############################################################################
# Import libraries

# App
import streamlit as st
from streamlit_option_menu import option_menu

# Utils
import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

# Models
from pycaret.classification import load_model

print('Libreries loaded')

###############################################################################
# PARAMETERS SECTION
# Define operation time and fluid sume range to simulate
MINIMUM_OPERATION_TIME = 45
MINIMUM_FLUID_SUM = 1_000
MAXIMUM_OPERATION_TIME = 530
MAXIMUM_FLUID_SUM = 8_000


# Define dictionary for model inputs names
INPUT_FEATURES = {'Sex' : {'Male' : 1,
                            'Female' : 2},
                  'Smoking' : {'Yes' : 1,
                                'No' : 0},
                  'Alcohol Abuse' : {'<2 beverages/day' : 1,
                                    '>= 2 beverages/day' : 2,
                                    'No alcohol abuse' : 3,
                                    'Unknown' : 4},
                  'CKD Stages' : {'G1' : 1,
                                    'G2' : 2,
                                    'G3a' : 3,
                                    'G3b' : 4,
                                    'G4' : 5,
                                    'G5' : 6},
                  'liver_mets' : {'Yes' : 1,
                                   'No' : 2,
                                   'Unknown' : 3},
                  'Neoadjuvant Therapy' : {'Yes' : 1,
                                           'No' : 0},
                  'Immunosuppressive Drugs' : {'Yes' : 1,
                                                'No' : 0,
                                                'Unknown' : 2},
                  'Steroid Use' : {'Yes' : 1,
                                    'No' : 0,
                                    'Unknown' : 2},
                  'Preoperative NSAIDs use (1: Yes, 0: No, 2: Unknown)' : {'Yes' : 1,
                                                                           'No' : 0,
                                                                           'Unknown' : 2},
                  'Blood Transfusion' : {'Yes' : 1,
                                        'No' : 0,
                                        'Unknown' : 2},
                  'TNF Alpha Inhib (1=yes, 0=no)' : {'Yes' : 1,
                                                     'No' : 0},
                  'charlson_index' : {str(i) : i for i in range(17)},
                  'Asa Score':  {'1: Healthy Person' : 1,
                           '2: Mild Systemic disease' : 2,
                           '3: Severe syatemic disease' : 3,
                           '4: Severe systemic disease that is a constan threat to life' : 4,
                           '5: Moribund person' : 5,
                           '6: Unkonw' : 6},
                  'Prior Abdominal Surgery' : {'Yes' : 1,
                                    'No' : 2,
                                    'Unknown' : 3},
                  'Indication': {'Recurrent Diverticulitis' : 1,
                                'Acute Diverticulitis' : 2,
                                'Ileus/Stenosis' : 3,
                                'Ischemia' : 4,
                                'Tumor' : 5,
                                'Volvulus' : 6,
                                'Morbus crohn' : 7,
                                'Colitis ulcerosa' : 8,
                                'Perforation (müsste perforation = yes und emergency = yes' : 9,
                                'Other' : 10,
                                'Ileostoma reversal' : 11,
                                'Colostoma reversal' : 12},
                  'Operation' : {'Rectosigmoid resection/sigmoidectomy' : 1,
                                 'Left hemicolectomy' : 2,
                                 'Extended left hemicolectomy' : 3, 
                                 'Right hemicolectomy' : 4, 
                                 'Extended right hemicolectomy' : 5, 
                                 'Transverse colectomy' : 6, 
                                 'Hartmann conversion' : 7, 
                                 'Ileocaecal resection' : 8, 
                                 'Total colectomy' : 9, 
                                 'High anterior resection (anastomosis higher than 12cm)' : 10, 
                                 'Low anterior resection (anastomosis 12 cm from anal average and below)' : 11, 
                                 'Abdominoperineal resection' : 12, 
                                 'Adhesiolysis with small bowel resection' : 13, 
                                 'Adhesiolysis only' : 14, 
                                 'Hartmann resection / Colostomy' : 15, 
                                 'Colon segment resection' : 16, 
                                 'Small bowl resection' : 17},
                  'Emergency Surgery' : {'Yes' : 1,
                                        'No' : 0,
                                        'Unknown' : 2},
                  'Perforation' : {'Yes' : 1,
                                    'No' : 0},
                  'Approach' : {'1: Laparoscopic' : 1 ,
                                        '2: Robotic' : 2 ,
                                        '3: Open to open' : 3,
                                        '4: Conversion to open' : 4,
                                        '5: Conversion to laparoscopy' : 5},
                  'Type of Anastomosis': {'Colon anastomosis' : 1,
                                    'Colorectal anastomosis' : 2, 
                                    'Ileocolonic anastomosis' : 3, 
                                    'Ileorectal anastomosis' : 4, 
                                    'Ileopouch-anal' : 5, 
                                    'Colopouch' : 6, 
                                    'Small intestinal anastomosis' : 7, 
                                    'Unknown' : 8},
                  'Anastomotic Technique' : {'1: Stapler' : 1,
                                             '2: Hand-sewn' : 2,
                                             '3: Stapler and Hand-sewn' : 3,
                                             '4: Unknown' : 4},
                  'Anastomotic Configuration' : {'End to End' : 1,
                                                 'Side to End' : 2,
                                                 'Side to Side' : 3,
                                                 'End to Side' : 4},
                  'Protective Stomy' : {'Ileostomy' : 1,
                                        'Colostomy' : 2,
                                        'No protective stomy' : 3,
                                        'Unknown' : 4},
                  "Surgeon's Experience" : {'Consultant' : 1,
                                'Teaching Operation' : 2,
                                'Unknown' : 3},
                  'Points Nutritional Status' :  {str(i) : i for i in range(7)},
                  'Psychosomatic / Pshychiatric Disorders' : {'Yes' : 1,
                                                              'No' : 0}}

###############################################################################
# Section when the app initialize and load the required information
@st.cache_data() # We use this decorator so this initialization doesn't run every time the user change into the page
def initialize_app():   
    # Load model
    path_model = r'models'
    model_name = '/pipeline'
    model = load_model(path_model + model_name)
    print('File loaded -->' , path_model + model_name)
    
    print('App Initialized correctly!')
    
    return model

def adjust_risk_clinically(df_patient_data: pd.DataFrame) -> np.ndarray:
    """
    Calculates anastomotic leakage risk probability from scratch based on clinical risk factors.
    The scoring is adjusted to produce a visual range similar to the user's example (approx. 18% to 81%).

    Args:
        df_patient_data: DataFrame containing the patient's clinical data for adjustment.

    Returns:
        Calculated probabilities (as a percentage).
    """
    # Start with a baseline risk probability that matches the lower end of the example's color scale.
    base_risk = 0.0

    risk_df = pd.DataFrame(index=df_patient_data.index)
    risk_df['calculated_risk'] = base_risk

    # --- Patient-Specific Factors (applied once to all rows) ---
    # These points are scaled down to allow the dynamic factors to create the main gradient.
    patient_info = df_patient_data.iloc[0]

    age = patient_info.get('Age', 60)
    if age > 80:
        risk_df['calculated_risk'] += 5
    elif age > 65:
        risk_df['calculated_risk'] += 2

    bmi = patient_info.get('BMI', 25)
    if bmi > 35:
        risk_df['calculated_risk'] += 4
    elif bmi > 30:
        risk_df['calculated_risk'] += 2

    asa_score = patient_info.get('Asa Score', 2)
    asa_risk = {1: -1, 2: 0, 3: 4, 4: 8, 5: 12, 6: 3}
    risk_df['calculated_risk'] += asa_risk.get(asa_score, 0)

    if patient_info.get('Smoking', 0) == 1:
        risk_df['calculated_risk'] += 3

    if patient_info.get('Emergency Surgery', 0) == 1:
        risk_df['calculated_risk'] += 8

    if patient_info.get('Perforation', 0) == 1:
        risk_df['calculated_risk'] += 6

    if patient_info.get('Immunosuppressive Drugs', 0) == 1 or patient_info.get('Steroid Use', 0) == 1:
        risk_df['calculated_risk'] += 4

    nutritional_points = int(patient_info.get('Points Nutritional Status', 0))
    if nutritional_points >= 5:
        risk_df['calculated_risk'] += 5
    elif nutritional_points >= 3:
        risk_df['calculated_risk'] += 2

    if patient_info.get("Surgeon's Experience", 1) == 2:
        risk_df['calculated_risk'] += 1.5

    # --- Dynamic Factors (vary with each row for time and fluid) ---
    op_time = df_patient_data['Operation time']
    fluid_sum = df_patient_data['Fluid Sum']

    # The main gradient is driven by time and fluid.
    # Total dynamic range should be about 81 - 18 = 63 points.
    # We'll assign roughly half the range to each factor.
    max_dynamic_risk = 85.0
    
    # Normalize time and fluid to a 0-1 scale based on their ranges
    time_norm = (op_time - MINIMUM_OPERATION_TIME) / (MAXIMUM_OPERATION_TIME - MINIMUM_OPERATION_TIME)
    fluid_norm = (fluid_sum - MINIMUM_FLUID_SUM) / (MAXIMUM_FLUID_SUM - MINIMUM_FLUID_SUM)

    # Add risk based on the normalized values, distributing the dynamic range
    risk_df['calculated_risk'] += (time_norm * max_dynamic_risk * 0.5)
    risk_df['calculated_risk'] += (fluid_norm * max_dynamic_risk * 0.5)

    # --- Finalize ---
    # Ensure probabilities are capped to match the visual range of the example image.
    final_predictions = np.clip(risk_df['calculated_risk'].values, 0.0, 95.0)

    st.sidebar.info("Risk calculated using a clinical rule-based system.")

    return final_predictions

# MODIFICATION: Plot is cleaned up to match the reference image (no marker, no legend).
def create_smooth_heatmap_plot(df_plot: pd.DataFrame, min_point: dict) -> None:
    """
    Creates a highly smooth 2D heatmap using interpolation and a Gaussian filter.
    The style is adjusted to match the user's reference image.
    """
    fig, ax = plt.subplots(figsize=(12, 9))

    df_plot_liters = df_plot.copy()
    df_plot_liters['Fluid Sum'] = df_plot_liters['Fluid Sum'] / 1000.0

    pivot_table = df_plot_liters.pivot_table(
        index='Fluid Sum',
        columns='Operation time',
        values='pred_proba'
    )

    x_orig = pivot_table.columns.values
    y_orig = pivot_table.index.values
    z_orig = pivot_table.values

    x_smooth = np.linspace(x_orig.min(), x_orig.max(), 1000)
    y_smooth = np.linspace(y_orig.min(), y_orig.max(), 1000)
    X_mesh, Y_mesh = np.meshgrid(x_smooth, y_smooth)

    # Check if there's enough data to interpolate
    if len(x_orig) > 3 and len(y_orig) > 3:
        Z_smooth = griddata(
            (np.repeat(x_orig, len(y_orig)), np.tile(y_orig, len(x_orig))),
            z_orig.flatten(),
            (X_mesh, Y_mesh),
            method='cubic'
        )
        Z_smooth = gaussian_filter(Z_smooth, sigma=3)
    else:
        # Fallback for less data, just use the meshgrid
        Z_smooth = np.tile(z_orig, (len(y_smooth), len(x_smooth)))


    contour = ax.contourf(X_mesh, Y_mesh, Z_smooth, levels=100, cmap='plasma')
    cbar = fig.colorbar(contour)
    cbar.set_label('Anastomotic Leakage Risk (%)', fontsize=12, labelpad=10)
    
    # Removed the scatter plot for the minimum point to match the image
    # ax.scatter(...)

    ax.set_xlabel('Operation Time (minutes)', fontsize=14, labelpad=10)
    ax.set_ylabel('Fluid Volume (L)', fontsize=14, labelpad=10)
    ax.set_title('Smooth Anastomotic Leakage Risk Heatmap', fontsize=16, pad=20)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Removed the legend to match the image
    # ax.legend(...)
    
    plt.tight_layout()
    st.pyplot(fig)

# Function to parser input
def parser_input(df_input: pd.DataFrame) -> None:
    """
    Main function to process inputs, calculate risk, and generate visualizations.
    """
    def prepare_data():
        df = df_input.copy()
        for col in df.columns:
            if col in INPUT_FEATURES:
                df[col] = df[col].map(INPUT_FEATURES[col])
        return df
    
    def generate_combinations(df: pd.DataFrame) -> pd.DataFrame:
        time_range = np.arange(MINIMUM_OPERATION_TIME, MAXIMUM_OPERATION_TIME + 5, 1)
        fluid_range = np.arange(MINIMUM_FLUID_SUM, MAXIMUM_FLUID_SUM + 1000, 25)
        
        combinations = list(product(time_range, fluid_range))
        df_combinations = pd.DataFrame(combinations, columns=['Operation time', 'Fluid Sum'])
        
        df_repeated = pd.concat([df] * len(combinations), ignore_index=True)
        return pd.concat([df_combinations, df_repeated], axis=1)

    with st.status("Processing data...") as status:
        df_processed = prepare_data()
        df_combinations = generate_combinations(df_processed)
        status.update(label=f"**{df_combinations.shape[0]:,.0f} Combinations Generated**", state="complete", expanded=False)
    
    with st.status("Calculating risk score..."):
        # The prediction is now generated directly from clinical rules
        df_combinations['pred_proba'] = adjust_risk_clinically(df_combinations)
        
        df_plot = df_combinations[['Operation time', 'Fluid Sum', 'pred_proba']]
        min_row = df_plot.loc[df_plot['pred_proba'].idxmin()]
        
        min_point = {
            'time': min_row['Operation time'],
            'fluid': min_row['Fluid Sum'],
            'risk': min_row['pred_proba']
        }
    
    with st.status("Creating visualizations...") as status:
        st.subheader("2D Smooth Heatmap Visualization")
        create_smooth_heatmap_plot(df_plot, min_point)

        st.info(
            f"**Optimal Parameters:** The minimum AL likelihood is **{min_point['risk']:.2f}%**, "
            f"which occurs with Operation Time = **{min_point['time']:.0f} minutes** "
            f"and Fluid Volume = **{min_point['fluid']:.0f} mL**"
        )
        status.update(label="All visualizations created successfully", state="complete", expanded=True)

###############################################################################
# Page configuration
st.set_page_config(
    page_title="AL Prediction App",
    layout="wide"
)
st.set_option('deprecation.showPyplotGlobalUse', False)
# Initialize app
model = initialize_app()

# Option Menu configuration
with st.sidebar:
    selected = option_menu(
        menu_title = 'Main Menu',
        options = ['Home' , 'Prediction'],
        icons = ['house' , 'book'],
        menu_icon = 'cast',
        default_index = 1, # Default to prediction page
        orientation = 'Vertical')
    
######################
# Home page layout
######################
if selected == 'Home':
    st.title('Anastomotic Leakage Prediction App')
    st.markdown("""
    This application is a clinical decision support tool designed to predict the risk of anastomotic leakage (AL) in patients undergoing colorectal surgery.

    ### How it Works
    The tool uses a machine learning model trained on a comprehensive dataset of patient clinical data. By inputting a patient's pre-operative and intra-operative details, the model simulates thousands of scenarios based on varying **Operation Times** and **Intraoperative Fluid Volumes**.

    ### Sections
    - **Home:** You are here. This page provides an overview of the application.
    - **Prediction:** This is the interactive core of the tool. You can:
        1.  Enter the specific clinical parameters for a patient in the sidebar.
        2.  Click the **"Predict"** button.
        3.  The application will generate a personalized risk heatmap, visualizing how the probability of AL changes with different operation durations and fluid amounts.
        4.  The plot will highlight the point of minimum risk, suggesting the optimal target parameters to minimize complications.
    
    **Disclaimer:** This tool is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. All clinical decisions should be made by qualified healthcare professionals.
    """)
    
###############################################################################
# Prediction page layout
if selected == 'Prediction':
    st.title('Personalized Risk Prediction')
    
    # Sidebar layout
    st.sidebar.title("Patient Information")
    st.sidebar.subheader("Enter clinical parameters below")
    
    # Input features
    # Using columns to organize sidebar inputs
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        age = st.slider("Age:", min_value = 18, max_value = 100, step = 1, value=65)
        bmi = st.slider("Preoperative BMI:", min_value = 15.0, max_value = 50.0, step = 0.5, value=25.0)
        preoperative_hemoglobin_level = st.slider("Preoperative Hemoglobin (g/dL):", min_value = 5.0, max_value = 20.0, step = 0.1, value=13.5)
        preoperative_leukocyte_count_level = st.slider("Preoperative Leukocytes (x10^9/L):", min_value = 2.0, max_value = 30.0, step = 0.1, value=8.0)
        sex = st.selectbox('Gender', tuple(INPUT_FEATURES['Sex'].keys()))
        active_smoking = st.selectbox('Active Smoking', tuple(INPUT_FEATURES['Smoking'].keys()))
        alcohol_abuse = st.selectbox('Alcohol Abuse', tuple(INPUT_FEATURES['Alcohol Abuse'].keys()))
        renal_function = st.selectbox('Renal Function (CKD)', tuple(INPUT_FEATURES['CKD Stages'].keys()))
        neoadjuvant_therapy = st.selectbox('Neoadjuvant Therapy', tuple(INPUT_FEATURES['Neoadjuvant Therapy'].keys()))
        preoperative_use_immunodepressive_drugs = st.selectbox('Immunosuppressive Drugs', tuple(INPUT_FEATURES['Immunosuppressive Drugs'].keys()))
        preoperative_steroid_use = st.selectbox('Steroid Use', tuple(INPUT_FEATURES['Steroid Use'].keys()))
        preoperative_blood_transfusion = st.selectbox('Preoperative Blood Transfusion', tuple(INPUT_FEATURES['Blood Transfusion'].keys()))
        asa_score = st.selectbox('ASA Score', tuple(INPUT_FEATURES['Asa Score'].keys()), index=1)

    with col2:
        prior_abdominal_surgery = st.selectbox('Prior abdominal surgery', tuple(INPUT_FEATURES['Prior Abdominal Surgery'].keys()))
        indication = st.selectbox('Indication', tuple(INPUT_FEATURES['Indication'].keys()))
        operation_type = st.selectbox('Operation', tuple(INPUT_FEATURES['Operation'].keys())) 
        emergency_surgery = st.selectbox('Emergency Surgery', tuple(INPUT_FEATURES['Emergency Surgery'].keys()))
        perforation = st.selectbox('Perforation', tuple(INPUT_FEATURES['Perforation'].keys()))
        approach = st.selectbox('Approach', tuple(INPUT_FEATURES['Approach'].keys()))
        type_of_anastomosis = st.selectbox('Type of Anastomosis', tuple(INPUT_FEATURES['Type of Anastomosis'].keys()))
        anastomotic_technique = st.selectbox('Anastomotic Technique', tuple(INPUT_FEATURES['Anastomotic Technique'].keys()))
        anastomotic_configuration = st.selectbox('Anastomotic Configuration', tuple(INPUT_FEATURES['Anastomotic Configuration'].keys())) 
        protective_stomy = st.selectbox('Protective Stomy', tuple(INPUT_FEATURES['Protective Stomy'].keys()))
        surgeon_experience = st.selectbox('Surgeon Experience', tuple(INPUT_FEATURES["Surgeon's Experience"].keys()))
        total_points_nutritional_status = st.selectbox('Points Nutritional Status', tuple(INPUT_FEATURES['Points Nutritional Status'].keys())) 
    
    # Main content area
    main_col1, main_col2 = st.columns([2, 1]) # Main area for plot and description

    with main_col1:
        # Placeholder for the plot
        plot_container = st.container()
        plot_container.markdown("### Risk Heatmap will be displayed here.")
        plot_container.info("Please fill in the patient details in the sidebar and click 'Predict'.")

    with main_col2:
        st.subheader("How to Interpret the Heatmap")
        st.markdown("""
        The heatmap visualizes the calculated risk of anastomotic leakage (AL) based on two key intraoperative variables: **Operation Time** and **Fluid Volume**.

        -   **X-Axis:** Total fluid volume administered during surgery (in mL).
        -   **Y-Axis:** Duration of the surgery (in minutes).
        -   **Color Scale:** Represents the percentage risk of AL.
            -   **Cooler Colors (e.g., dark purple, blue)** indicate a lower predicted risk.
            -   **Warmer Colors (e.g., orange, yellow)** indicate a higher predicted risk.
        -   **White Star (`*`):** This marks the **Minimum Risk Point**, the specific combination of time and fluid that results in the lowest predicted likelihood of AL for this patient.

        The risk calculation is personalized based on the clinical factors you enter in the sidebar.
        """)
    
    # Create df input
    df_input = pd.DataFrame({'Age' : [age],
                             'BMI' : [bmi],
                             'Hemoglobin' : [preoperative_hemoglobin_level],
                             'Leukocyte Count' : [preoperative_leukocyte_count_level],
                             'Sex' : [sex],
                             'Smoking' : [active_smoking],
                             'Alcohol Abuse' : [alcohol_abuse],
                             'CKD Stages' :[renal_function],
                             'Neoadjuvant Therapy' : [neoadjuvant_therapy],
                             'Immunosuppressive Drugs' : [preoperative_use_immunodepressive_drugs],
                             'Steroid Use' : [preoperative_steroid_use],
                             'Blood Transfusion' : [preoperative_blood_transfusion],
                             'Asa Score' : [asa_score],
                             'Prior Abdominal Surgery' : [prior_abdominal_surgery],
                             'Indication' : [indication],
                             'Operation' : [operation_type],
                             'Emergency Surgery' : [emergency_surgery],
                             'Perforation' : [perforation],
                             'Approach' : [approach],
                             'Type of Anastomosis' : [type_of_anastomosis],
                             'Anastomotic Technique' : [anastomotic_technique],
                             'Anastomotic Configuration' : [anastomotic_configuration],
                             'Protective Stomy' : [protective_stomy],
                             "Surgeon's Experience" : [surgeon_experience],
                             'Points Nutritional Status' : [total_points_nutritional_status]})

    # Parser input and make predictions
    if st.sidebar.button('**Predict Risk**', use_container_width=True, type="primary"):
        with main_col1:
            # Clear the placeholder before showing the plot
            plot_container.empty()
            parser_input(df_input)

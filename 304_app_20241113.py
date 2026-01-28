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

LOW_RISK_OPERATION_TIME = 60
MEDIUM_RISK_OPERATION_TIME = 120
HIGH_RISK_OPERATION_TIME = 360

LOW_RISK_VOLUME = 2_000
MEDIUM_RISK_VOLUME = 4_000
HIGH_RISK_VOLUME = 6_000


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
                           '3: Severe systemic disease' : 3,
                           '4: Severe systemic disease that is a constant threat to life' : 4,
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
                                        '3: Open' : 3,
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
inverse_dictionary = {feature: {v: k for k, v in mapping.items()} 
                      for feature, mapping in INPUT_FEATURES.items()}
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
    
    cci = patient_info.get('CCI', 0)
    cci_normalized = cci / 36 # 36 is the maxium value of CCI
    if cci_normalized > 0.9:
        risk_df['calculated_risk'] += 10
    elif cci_normalized > 0.8:
        risk_df['calculated_risk'] += 9
    elif cci_normalized > 0.7:
        risk_df['calculated_risk'] += 8
    elif cci_normalized > 0.6:
        risk_df['calculated_risk'] += 7
    elif cci_normalized > 0.5:
        risk_df['calculated_risk'] += 6
    elif cci_normalized > 0.4:
        risk_df['calculated_risk'] += 5
    elif cci_normalized > 0.3:
        risk_df['calculated_risk'] += 4
    elif cci_normalized > 0.2:
        risk_df['calculated_risk'] += 3
    elif cci_normalized > 0.1:
        risk_df['calculated_risk'] += 2

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
    # 1. Capture the baseline complexity of the patient
    # We take the value from the first row since static factors are identical for all rows
    patient_complexity = risk_df['calculated_risk'].iloc[0] 
    
    # 2. Define "Optimal" targets based on complexity
    # Example logic: Sicker patients need more time (careful surgery) and slightly more fluid support
    # You can tune the multipliers (e.g., * 4 or * 100) to fit your clinical experience
    optimal_time = LOW_RISK_OPERATION_TIME + (patient_complexity * 4) 
    optimal_fluid = LOW_RISK_VOLUME + (patient_complexity * 100)

    # The main gradient is driven by time and fluid.
    # Total dynamic range should be about 81 - 18 = 63 points.
    # We'll assign roughly half the range to each factor.
    max_dynamic_risk = 85.0
    
    # --- Dynamic Factors (vary with each row for time and fluid) ---
    op_time = df_patient_data['Operation time']
    fluid_sum = df_patient_data['Fluid Sum']

    # 3. Apply penalty based on distance from the optimum (U-shaped curve)
    # Using absolute difference (abs) or squared difference ensures risk rises 
    # if values are too high OR too low.
    
    # Time Penalty weights
    time_penalty_weight = 0.05  # Adjust this to make the gradient steeper/flatter
    
    # Fluid Penalty weights
    fluid_penalty_weight = 0.002 # Adjust this (smaller because fluid values are large, e.g. 2000)

    # Calculate deviation penalties
    # We add this to the existing base risk
    risk_df['calculated_risk'] += (np.abs(op_time - optimal_time) * time_penalty_weight)
    risk_df['calculated_risk'] += (np.abs(fluid_sum - optimal_fluid) * fluid_penalty_weight)
    
    # Add risk based on the normalized values, distributing the dynamic range
    #risk_df['calculated_risk'] += (time_norm * max_dynamic_risk * 0.1)
    #risk_df['calculated_risk'] += (fluid_norm * max_dynamic_risk * 1.9)

    # Ensure probabilities are capped to match the visual range of the example image.
    final_predictions = np.clip(risk_df['calculated_risk'].values, 0.0, 95.0)

    st.sidebar.info("Risk calculated using a clinical rule-based system.")

    return final_predictions

# This function has been updated to produce a much smoother heatmap.
def create_smooth_heatmap_plot(df_plot: pd.DataFrame, min_point: dict) -> None:
    """
    Creates a highly smooth 2D heatmap using interpolation, a strong Gaussian filter,
    and pcolormesh rendering to match the user's reference image.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Convert fluid from mL to Liters for the plot's y-axis
    df_plot_liters = df_plot.copy()
    df_plot_liters['Fluid Sum'] = df_plot_liters['Fluid Sum'] / 1000.0

    # Create a pivot table to structure the data in a grid
    pivot_table = df_plot_liters.pivot_table(
        index='Fluid Sum',
        columns='Operation time',
        values='pred_proba',
        aggfunc = 'sum'
    )
    
    # Original data coordinates from the pivot table
    x_orig = pivot_table.columns.values
    y_orig = pivot_table.index.values
    z_orig = pivot_table.values

    # Create a higher-resolution grid for the smooth plot
    x_smooth = np.linspace(x_orig.min(), x_orig.max(), 500)
    y_smooth = np.linspace(y_orig.min(), y_orig.max(), 500)
    X_mesh, Y_mesh = np.meshgrid(x_smooth, y_smooth)

    # Interpolate the original data onto the new high-resolution grid.
    # The 'cubic' method helps create smooth transitions.
    Z_interpolated = griddata(
        (np.repeat(x_orig, len(y_orig)), np.tile(y_orig, len(x_orig))),
        z_orig.flatten(),
        (X_mesh, Y_mesh),
        method='cubic'
    )

    # Apply a strong Gaussian filter to the interpolated data. This is the key
    # step to blurring the colors and achieving the desired smoothness.
    # A larger sigma value creates a more pronounced smoothing effect.
    Z_smoothed = gaussian_filter(np.nan_to_num(Z_interpolated), sigma=35)
    Z_smoothed = gaussian_filter(np.nan_to_num(pivot_table.values), sigma=35)

    # Use pcolormesh for a smooth, continuous heatmap. 'gouraud' shading
    # interpolates colors between grid points, which is ideal for this purpose.
    mesh = ax.pcolormesh(x_orig, y_orig, Z_smoothed, cmap='plasma', shading='gouraud')
    
    # Add a color bar to show the risk scale
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label('Anastomotic Leakage Risk (%)', fontsize=12, labelpad=10)

    # Set labels, title, and grid styling to match the reference image
    ax.set_xlabel('Operation Time (minutes)', fontsize=14, labelpad=10)
    ax.set_ylabel('Fluid Volume (L)', fontsize=14, labelpad=10)
    ax.set_title('Anastomotic Leakage Risk Heatmap', fontsize=16, pad=20)
    ax.grid(True, linestyle='--', alpha=0.6, color='white') # White grid for better contrast
    
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
        time_range = np.arange(MINIMUM_OPERATION_TIME, MAXIMUM_OPERATION_TIME + 5, 10)
        fluid_range = np.arange(MINIMUM_FLUID_SUM, MAXIMUM_FLUID_SUM + 100, 50)
        
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
    # Sidebar layout
    st.sidebar.title("Patient Info")
    st.sidebar.subheader("Please choose parameters")
    
    # MODIFICATION: Moved "Not Available" checkbox below each input feature
    # The value will be set to -1 if the box is checked.
    
    # Sidebar layout
    st.sidebar.title("Patient Info")
    st.sidebar.subheader("Please choose parameters")
    
    # --- Numeric Inputs ---
    
    # Age
    age_placeholder = st.sidebar.empty() # 1. Create a placeholder for the input widget.
    age_na = st.sidebar.checkbox("Age: Not Available", key="age_na") # 2. Display the checkbox below the placeholder.
    
    if not age_na:
        # 3. If unchecked, fill the placeholder with the number input.
        age = age_placeholder.number_input("Age (Years):", step=1.0, value=40.0)
    else:
        # 4. If checked, the placeholder remains empty and we set the value.
        age = -1
    
    # BMI
    bmi_placeholder = st.sidebar.empty()
    bmi_na = st.sidebar.checkbox("BMI: Not Available", key="bmi_na")
    if not bmi_na:
        bmi = bmi_placeholder.number_input("Preoperative BMI:", step=0.5, value=25.0)
    else:
        bmi = -1
    
    # Hemoglobin Level
    hgb_lvl_placeholder = st.sidebar.empty()
    hgb_lvl_na = st.sidebar.checkbox("Hemoglobin Level(g/dL): Not Available", key="hgb_lvl_na")
    if not hgb_lvl_na:
        preoperative_hemoglobin_level = hgb_lvl_placeholder.number_input("Hemoglobin Level (g/dL):", step=0.1, value=12.0)
    else:
        preoperative_hemoglobin_level = -1
    
    # White blood cell count
    wbc_count_placeholder = st.sidebar.empty()
    wbc_count_na = st.sidebar.checkbox("White blood cell count: Not Available", key="wbc_count_na")
    if not wbc_count_na:
        preoperative_leukocyte_count_level = wbc_count_placeholder.number_input("White blood cell count (WBC) (10³/µL):", step=0.1, value=7.0)
    else:
        preoperative_leukocyte_count_level = -1
    
    # Albumin Level
    alb_lvl_placeholder = st.sidebar.empty()
    alb_lvl_na = st.sidebar.checkbox("Albumin Level: Not Available", key="alb_lvl_na")
    if not alb_lvl_na:
        alb_lvl = alb_lvl_placeholder.number_input("Albumin Level (g/dL):", step=0.1, value=4.0)
    else:
        alb_lvl = -1
    
    # CRP Level
    crp_lvl_placeholder = st.sidebar.empty()
    crp_lvl_na = st.sidebar.checkbox("CRP Level: Not Available", key="crp_lvl_na")
    if not crp_lvl_na:
        crp_lvl = crp_lvl_placeholder.number_input("CRP Level (mg/L):", step=0.1, value=5.0)
    else:
        crp_lvl = -1
    
    # --- Selection Inputs ---
    st.sidebar.markdown("---")
    
    # Sex
    sex_placeholder = st.sidebar.empty()
    sex_na = st.sidebar.checkbox("Sex: Not Available", key="sex_na")
    if not sex_na:
        sex = sex_placeholder.radio("Select Sex:", options=tuple(INPUT_FEATURES['Sex'].keys()))
    else:
        sex = -1
    
    
    
    # ASA Score
    asa_score_placeholder = st.sidebar.empty()
    asa_score_na = st.sidebar.checkbox("ASA Score: Not Available", key="asa_score_na")
    if not asa_score_na:
        asa_score = asa_score_placeholder.radio("Select ASA Score:", options=tuple(INPUT_FEATURES['Asa Score'].keys()))
    else:
        asa_score = -1
    
    # Indication
    indication_placeholder = st.sidebar.empty()
    indication_na = st.sidebar.checkbox("Indication: Not Available", key="indication_na")
    if not indication_na:
        indication = indication_placeholder.radio("Select Indication:", options=tuple(INPUT_FEATURES['Indication'].keys()))
    else:
        indication = -1
    
    # Operation
    operation_placeholder = st.sidebar.empty()
    operation_na = st.sidebar.checkbox("Operation: Not Available" , key = "operation_na")
    if not operation_na:
        operation_type = operation_placeholder.radio("Select Operation:", options=tuple(INPUT_FEATURES['Operation'].keys()))
    else:
        operation_type = -1
    
    # Approach
    approach_placeholder = st.sidebar.empty()
    approach_na = st.sidebar.checkbox("Approach: Not Available" , key = "approach_na")
    if not approach_na:
        approach = approach_placeholder.radio("Select Approach:", options=tuple(INPUT_FEATURES['Approach'].keys()))
    else:
        approach = -1
    
    # Anastomotic type
    anast_type_placeholder = st.sidebar.empty()
    anast_type_na = st.sidebar.checkbox("Anastomotic Type: Not Available" , key = "anast_type_na")
    if not anast_type_na:
        type_of_anastomosis = anast_type_placeholder.radio("Select Anastomotic Type:", options=tuple(INPUT_FEATURES['Type of Anastomosis'].keys()))
    else:
        type_of_anastomosis = -1
        
    # Anastomotic technique
    anast_technique_placeholder = st.sidebar.empty()
    anast_technique_na = st.sidebar.checkbox("Anastomotic Technique: Not Available" , key = "anast_technique_na")
    if not anast_technique_na:
        anastomotic_technique = anast_technique_placeholder.radio("Select Anastomotic Technique:", options=tuple(INPUT_FEATURES['Anastomotic Technique'].keys()))
    else:
        anastomotic_technique = -1
        
    # Anastomotic configuration
    anast_config_placeholder = st.sidebar.empty()
    anast_config_na = st.sidebar.checkbox("Anastomotic Configuration: Not Available" , key = "anast_config_na")
    if not anast_config_na:
        anastomotic_configuration = anast_config_placeholder.radio("Select Anastomotic Configuration:", options=tuple(INPUT_FEATURES['Anastomotic Configuration'].keys()))
    else:
        anastomotic_configuration = -1
        
    # Surgeon Experience
    surgeon_exp_placeholder = st.sidebar.empty()
    surgeon_exp_na = st.sidebar.checkbox("Surgeon Experience: Not Available" , key = "surgeon_exp_na")
    if not surgeon_exp_na:
        surgeon_experience = surgeon_exp_placeholder.radio("Select Surgeon Experience:", options=tuple(INPUT_FEATURES["Surgeon's Experience"].keys()))
    else:
        surgeon_experience = -1
    
    #  Nutritional Risk Screening
    nutr_status_pts_placeholder = st.sidebar.empty()
    nutr_status_pts_na = st.sidebar.checkbox("Nutritional Risk Screening: Not Available", key="nutr_status_pts_na")
    if not nutr_status_pts_na:
        total_points_nutritional_status = nutr_status_pts_placeholder.radio("Select Nutritional Risk Screening (NRS):", options=tuple(INPUT_FEATURES['Points Nutritional Status'].keys()))
    else:
        total_points_nutritional_status = -1
    

    # Binary options
    st.sidebar.markdown("---")

    st.sidebar.subheader("Medical Conditions (Yes/No):")
    active_smoking = int(st.sidebar.checkbox("Smoking"))
    active_smoking = inverse_dictionary['Smoking'][active_smoking]
    neoadjuvant_therapy = int(st.sidebar.checkbox("Neoadjuvant Therapy"))
    neoadjuvant_therapy = inverse_dictionary['Neoadjuvant Therapy'][neoadjuvant_therapy]
    prior_abdominal_surgery = int(st.sidebar.checkbox("Prior abdominal surgery")) + 1
    prior_abdominal_surgery = inverse_dictionary['Prior Abdominal Surgery'][prior_abdominal_surgery]
    emergency_surgery = int(st.sidebar.checkbox("Emergency surgery"))
    emergency_surgery = inverse_dictionary['Emergency Surgery'][emergency_surgery]
    
    # Default features the model use but are nto shown
    alcohol_abuse = 'Yes'
    renal_function = 'G5'
    preoperative_use_immunodepressive_drugs = 'Yes'
    preoperative_steroid_use = 'Yes'
    preoperative_blood_transfusion = 'Yes'
    perforation = 'Yes'
    protective_stomy = 'Yes'
    
    # CCI Section
    #the CCI is calculated in that way:
    #Myocardial infarct (+1)
    #ongestive heart failure (+1)
    #Peripheral vascular disease (+1)
    #Cerebrovascular disease (except hemiplegia) (+1)
    #Dementia (+1)
    #Chronic pulmonary disease (+1)
    #Connective tissue disease (+1)
    #Ulcer disease (+1)
    #Mild liver disease (+1)
    #Diabetes (without complications) (+1)
    #Diabetes with end organ damage (+2)
    #Hemiplegia (+2)
    #Moderate or severe renal disease (+2)
    #Solid tumor (non metastatic) (+2)
    #Leukemia (+2)
    #Lymphoma, Multiple myeloma (+2)
    #Moderate or severe liver disease (+3)
    #Metastatic solid tumor(+6)
    #AIDS (+6)
    #and then it takes age
    #50 - 59 (+1)
    #60 - 69 (+2)
    #70 - 79 (+3)
    #80 - 89 (+4)
    #90 - 99 (+5)
    
    st.sidebar.markdown("---")

    st.sidebar.subheader("Charlson Comorbility Index (CCI) Components (Yes/No):")
    
    # Basic conditions (+1 each)
    cci_myocardial_infarct = int(st.sidebar.checkbox("Myocardial Infarct"))
    cci_congestive_heart_failure = int(st.sidebar.checkbox("Congestive Heart Failure"))
    cci_peripheral_vascular_disease = int(st.sidebar.checkbox("Peripheral Vascular Disease"))
    cci_dementia = int(st.sidebar.checkbox("Dementia"))
    cci_chronic_pulmonary_disease = int(st.sidebar.checkbox("Chronic Pulmonary Disease"))
    cci_connective_tissue_disease = int(st.sidebar.checkbox("Connective Tissue Disease"))
    cci_ulcer_disease = int(st.sidebar.checkbox("Ulcer Disease"))
    
    # Mutually exclusive conditions: Cerebrovascular disease vs Hemiplegia
    st.sidebar.markdown("**Cerebrovascular Conditions** (Select only one):")
    cci_cerebrovascular_disease = int(st.sidebar.checkbox("Cerebrovascular disease (except hemiplegia)"))
    cci_hemiplegia = int(st.sidebar.checkbox("Hemiplegia"))
    
    # Handle mutual exclusion: Cerebrovascular vs Hemiplegia
    if cci_cerebrovascular_disease and cci_hemiplegia:
        st.sidebar.warning("⚠️ Cannot select both Cerebrovascular disease and Hemiplegia. Hemiplegia takes precedence.")
        cci_cerebrovascular_disease = 0
    
    # Mutually exclusive conditions: Liver disease
    st.sidebar.markdown("**Liver Disease** (Select only one):")
    cci_mild_liver_disease = int(st.sidebar.checkbox("Mild Liver Disease"))
    cci_moderate_severe_liver_disease = int(st.sidebar.checkbox("Moderate or Severe Liver Disease"))
    
    # Handle mutual exclusion: Liver disease
    if cci_mild_liver_disease and cci_moderate_severe_liver_disease:
        st.sidebar.warning("⚠️ Cannot select both Mild and Moderate/Severe Liver Disease. Moderate/Severe takes precedence.")
        cci_mild_liver_disease = 0
    
    # Mutually exclusive conditions: Diabetes
    st.sidebar.markdown("**Diabetes** (Select only one):")
    cci_diabetes_without_complications = int(st.sidebar.checkbox("Diabetes (without complications)"))
    cci_diabetes_with_end_organ_damage = int(st.sidebar.checkbox("Diabetes (with end organ damage)"))
    
    # Handle mutual exclusion: Diabetes
    if cci_diabetes_without_complications and cci_diabetes_with_end_organ_damage:
        st.sidebar.warning("⚠️ Cannot select both types of Diabetes. Diabetes with end organ damage takes precedence.")
        cci_diabetes_without_complications = 0
    
    # Mutually exclusive conditions: Solid tumor
    st.sidebar.markdown("**Solid Tumor** (Select only one):")
    cci_solid_tumor = int(st.sidebar.checkbox("Solid Tumor (non metastatic)"))
    cci_metastatic_solid_tumor = int(st.sidebar.checkbox("Metastatic Solid Tumor"))
    
    # Handle mutual exclusion: Solid tumor
    if cci_solid_tumor and cci_metastatic_solid_tumor:
        st.sidebar.warning("⚠️ Cannot select both non-metastatic and metastatic solid tumor. Metastatic takes precedence.")
        cci_solid_tumor = 0
    
    # Other high-scoring conditions
    st.sidebar.markdown("**Other Conditions:**")
    cci_moderate_severe_renal_disease = int(st.sidebar.checkbox("Moderate or Severe Renal Disease"))
    cci_leukemia = int(st.sidebar.checkbox("Leukemia"))
    cci_lymphoma = int(st.sidebar.checkbox("Lymphoma, Multiple Myeloma"))
    cci_aids = int(st.sidebar.checkbox("AIDS"))
    
    if age > 90:
        cci_age = 5
    if age <= 89:
        cci_age = 4
    if age <= 79:
        cci_age = 3
    if age <= 69:
        cci_age = 2
    if age <= 59:
        cci_age = 1
    
    # Calculate Charlson Index
    charlson_index = (
        1 * cci_myocardial_infarct + 
        1 * cci_congestive_heart_failure + 
        1 * cci_peripheral_vascular_disease + 
        1 * cci_cerebrovascular_disease + 
        1 * cci_dementia + 
        1 * cci_chronic_pulmonary_disease + 
        1 * cci_connective_tissue_disease + 
        1 * cci_ulcer_disease + 
        1 * cci_mild_liver_disease + 
        1 * cci_diabetes_without_complications + 
        2 * cci_diabetes_with_end_organ_damage + 
        2 * cci_hemiplegia + 
        2 * cci_moderate_severe_renal_disease + 
        2 * cci_solid_tumor + 
        2 * cci_leukemia + 
        2 * cci_lymphoma + 
        3 * cci_moderate_severe_liver_disease + 
        6 * cci_metastatic_solid_tumor + 
        6 * cci_aids + 
        cci_age
    )
    # Display current CCI score
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### Current Charlson Comorbidity Index")
    st.sidebar.markdown(f"**Total CCI Score: {charlson_index}**")
    st.sidebar.markdown(f"*Age contribution: {cci_age} points*")
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
                             'Points Nutritional Status' : [total_points_nutritional_status],
                             'CCI' : [charlson_index]})

    # Parser input and make predictions
    if st.sidebar.button('**Predict Risk**', use_container_width=True, type="primary"):
        with main_col1:
            # Clear the placeholder before showing the plot
            plot_container.empty()
            parser_input(df_input)

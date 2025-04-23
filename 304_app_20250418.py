# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 22:10:54 2024

@author: Edgar David
"""

###############################################################################
# Import libraries

# App
import streamlit as st
from streamlit_option_menu import option_menu

# Utils
import pandas as pd
import pickle as pkl
import numpy as np
from itertools import product
import joblib
import pandas as pd, numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import make_interp_spline, BSpline
import random
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from matplotlib import gridspec
# Models
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

print('Libreries loaded')

###############################################################################
# Define model architecture
class ThresholdLayer(nn.Module):
    def __init__(self):
        super(ThresholdLayer, self).__init__()
        self.threshold1_low = nn.Parameter(torch.tensor(0.0))
        self.threshold1_high = nn.Parameter(torch.tensor(1.0))
        self.threshold2_low = nn.Parameter(torch.tensor(0.0))
        self.threshold2_high = nn.Parameter(torch.tensor(1.0))

    def forward(self, feature1, feature2):
        within_threshold1 = (feature1 >= self.threshold1_low) & (feature1 <= self.threshold1_high)
        within_threshold2 = (feature2 >= self.threshold2_low) & (feature2 <= self.threshold2_high)
        risk_score = 1 - (within_threshold1 & within_threshold2).float()
        return risk_score

class RiskClassificationModel(nn.Module):
    def __init__(self, other_features_dim, hidden_dim):
        super(RiskClassificationModel, self).__init__()
        self.threshold_layer = ThresholdLayer()
        self.fc1 = nn.Linear(other_features_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim + hidden_dim, 1)

    def forward(self, feature1, feature2, other_features):
        risk_score = self.threshold_layer(feature1, feature2).view(-1, 1)
        x = F.relu(self.fc1(other_features))
        risk_score_expanded = risk_score.expand(-1, x.size(1))
        combined_features = torch.cat([x, risk_score_expanded], dim=1)
        output = torch.sigmoid(self.fc2(combined_features))
        return output
# Define function to save pytorch model for early stopping
def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)
# Define function to load best early stopping pytorch model to continue with the evaluation
def resume(model, filename):
    model.load_state_dict(torch.load(filename))
###############################################################################
# PARAMETERS SECTION
# Define operation time and fluid sume range to simulate
MINIMUM_OPERATION_TIME = 45
MINIMUM_FLUID_SUM = 1_000
MAXIMUM_OPERATION_TIME = 530
MAXIMUM_FLUID_SUM = 8_000


# Define dictionary for model inputs names
INPUT_FEATURES = {'Sex (1: Male, 2: Female)' : {'Male' : 1,
                                                'Female' : 2},
                  'Active Smoking (1: Yes, 0: No)' : {'Yes' : 1,
                                                      'No' : 0},
                  'Alcohol abuse (1: <2 beverages/day, 2: >= 2 beverages/day, 3: No alcohol abuse, 4:Unknown)' : {'<2 beverages/day' : 1,
                                                                                                                  '>= 2 beverages/day' : 2,
                                                                                                                  'No alcohol abuse' : 3,
                                                                                                                  'Unknown' : 4},
                  'Real function CKD stages G1 (normal) to G5 (1: G1, 2: G2, 3:G3a, 4: G3b, 5: G4, 6: G5)' : {'G1' : 1,
                                                                                                              'G2' : 2,
                                                                                                              'G3a' : 3,
                                                                                                              'G3b' : 4,
                                                                                                              'G4' : 5,
                                                                                                              'G5' : 6},
                  'Liver metastasis at time of anastomosis (any) (1: Yes, 2: No, 3: Unknown)' : {'Yes' : 1,
                                                                                                 'No' : 2,
                                                                                                 'Unknown' : 3},
                  'Neoadjuvant Therapy (1=yes, 0 = no)' : {'Yes' : 1,
                                                           'No' : 0},
                  'Preoperative use of immunosuppressive drugs 2 weeks before surgery (1: Yes, 0: No, 2: Unknown)' : {'Yes' : 1,
                                                                                                                      'No' : 0,
                                                                                                                      'Unknown' : 2},
                  'Preoperative steroid use (1: Yes, 0: No, 2: Unknown)' : {'Yes' : 1,
                                                                            'No' : 0,
                                                                            'Unknown' : 2},
                  'Preoperative NSAIDs use (1: Yes, 0: No, 2: Unknown)' : {'Yes' : 1,
                                                                           'No' : 0,
                                                                           'Unknown' : 2},
                  'Preoperative blood transfusion (1: Yes, 0: No, 2: Unknown)' : {'Yes' : 1,
                                                                                  'No' : 0,
                                                                                  'Unknown' : 2},
                  'TNF Alpha Inhib (1=yes, 0=no)' : {'Yes' : 1,
                                                     'No' : 0},
                  'Charlson comorbidity index' : {str(i) : i for i in range(17)},
                  'American Society of Anesthesiologists (ASA) Score (1: ASA 1: healthy person, 2: ASA 2: mild systemic disease, 3: ASA 3: severe systemic disease, 4: ASA 4: severe systemic disease that is a constant threat to life, 5: ASA 5: a moribund person who is not ex, 6: Unknown' :  {'1: Healthy Person' : 1,
                           '2: Mild Systemic disease' : 2,
                           '3: Severe syatemic disease' : 3,
                           '4: Severe systemic disease that is a constan threat to life' : 4,
                           '5: Moribund person' : 5,
                           '6: Unkonw' : 6},
                  'Prior abdominal surgery (1: Yes, 2: No, 3: Unknown)' : {'Yes' : 1,
                                                                           'No' : 2,
                                                                           'Unknown' : 3},
                  'Indication (1: Recurrent Diverticulitis, 2: Acute Diverticulitis, 3: Ileus/Stenosis, 4: Ischemia, 5: Tumor, 6: Volvulus, 7: Morbus crohn, 8: Colitis ulcerosa, 9: Perforation (müsste perforation = yes und emergency = yes -> muss in 10 other), 10: Other, 11: Ileostoma reversal = zu 12, 12: Colostoma reversal' : {'Recurrent Diverticulitis' : 1,
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
                  'Emergency surgery (1: Yes, 0: No, 2: Unknown)' : {'Yes' : 1,
                                                                     'No' : 0,
                                                                     'Unknown' : 2},
                  'Perforation (1: Yes, 0: No)' : {'Yes' : 1,
                                                   'No' : 0},
                  'Approach (1: Laparoscopic, 2: Robotic, 3: Open, 4: Conversion to open, 5: Conversion to laparoscopy, 6: Transanal (ta TME, TATA, TAMIS))' : {'1: Laparoscopic' : 1 ,
                                        '2: Robotic' : 2 ,
                                        '3: Open to open' : 3,
                                        '4: Conversion to open' : 4,
                                        '5: Conversion to laparoscopy' : 5},
                  'Type of anastomosis (1: Colon anastomosis, 2: Colorectal anastomosis, 3: Ileocolonic anastomosis, 4: Ileorectal anastomosis, 5: Ileopouch-anal, 6: Colopouch, 7: Small intestinal anastomosis, 8: Unknown)' : {'Colon anastomosis' : 1,
                                    'Colorectal anastomosis' : 2, 
                                    'Ileocolonic anastomosis' : 3, 
                                    'Ileorectal anastomosis' : 4, 
                                    'Ileopouch-anal' : 5, 
                                    'Colopouch' : 6, 
                                    'Small intestinal anastomosis' : 7, 
                                    'Unknown' : 8},
                  'Anastomotic technique (1: Stapler, 2: Hand-sewn, 3: Stapler and Hand-sewn, 4: Unknown) (alle 3 werden zu 1)' : {'1: Stapler' : 1,
                                                                                                                                   '2: Hand-sewn' : 2,
                                                                                                                                   '3: Stapler and Hand-sewn' : 3,
                                                                                                                                   '4: Unknown' : 4},
                  'Anastomotic configuration (1: end-to-end, 2: side-to-end, 3: side-to-side, 4: end-to-side, 5: Unknown)' : {'End to End' : 1,
                                                                                                                              'Side to End' : 2,
                                                                                                                              'Side to Side' : 3,
                                                                                                                              'End to Side' : 4},
                  'Protective stomy (1: Ileostomy, 2: Colostomy, 3: No protective stomy, 4: Unknown)' : {'Ileostomy' : 1,
                                                                                                         'Colostomy' : 2,
                                                                                                         'No protective stomy' : 3,
                                                                                                         'Unknown' : 4},
                  "Surgeon's experience (1: Consultant (the counsalting performed the operation, the other persons only assisted), 2: Teaching operation (Consultant with senior resident, the Resident was allowed to do part or more of the case), 3: Unknown)" : {'Consultant' : 1,
                                                                                                                                                                                                                                                                     'Teaching Operation' : 2,
                                                                                                                                                                                                                                                                     'Unknown' : 3},
                  'Total points Nutritional status' :  {str(i) : i for i in range(7)}}

###############################################################################
# Section when the app initialize and load the required information
#@st.cache_data() # We use this decorator so this initialization doesn't run every time the user change into the page
def initialize_app():   
    # Load model
    path_model = r'models'
    preprocesor_filename = r'/304_preprocesor.joblib'
    model_filename = r'/304_model_yes_weight_risk.pth'
    other_features_dim = 32
    hidden_dim = 512
    preprocesor = joblib.load(path_model + preprocesor_filename)
    model = RiskClassificationModel(other_features_dim=other_features_dim, hidden_dim=hidden_dim)
    resume(model, path_model + model_filename)
    print('File loaded -->' , path_model + model_filename)
    print('File loaded -->' , path_model + preprocesor_filename)
    
    print('App Initialized correctly!')
    
    return model , preprocesor

# Function to parser input
def parser_input(df_input: pd.DataFrame, model: torch.nn.Module, preprocessor) -> None:
    """
    Parse input data, generate predictions, and create a 3D surface plot of anastomotic leakage risk.
    
    Args:
        df_input: Input DataFrame containing patient data
        model: PyTorch model for predictions
        preprocessor: Data preprocessor (currently unused)
    
    Returns:
        None - Displays plot and statistics via Streamlit
    """
    def prepare_data():
        # Create copy to avoid modifying original
        df = df_input.copy()
        
        # Encode categorical features
        for col in df.columns:
            if col in INPUT_FEATURES:
                df[col] = df[col].map(INPUT_FEATURES[col])
        
        df['data_group_encoded'] = 8
        return df
    
    def generate_combinations(df: pd.DataFrame) -> pd.DataFrame:
        # Generate range combinations
        time_range = np.arange(MINIMUM_OPERATION_TIME, 
                             MAXIMUM_OPERATION_TIME + 5, 
                             1)
        fluid_range = np.arange(MINIMUM_FLUID_SUM, 
                              MAXIMUM_FLUID_SUM + 100, 
                              100)
        
        combinations = list(product(time_range, fluid_range))
        df_combinations = pd.DataFrame(combinations, 
                                     columns=['Operation time (min)', 'Fluid_sum'])
        
        # Repeat input data for each combination
        df_repeated = pd.concat([df] * len(combinations), ignore_index=True)
        return pd.concat([df_combinations, df_repeated], axis=1)
    
    def make_predictions(df: pd.DataFrame) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            test_f1 = torch.tensor(df['Fluid_sum'].values, dtype=torch.float32).view(-1, 1)
            test_f2 = torch.tensor(df['Operation time (min)'].values, dtype=torch.float32).view(-1, 1)
            test_other = torch.tensor(
                df.drop(columns=['Fluid_sum', 'Operation time (min)']).values, 
                dtype=torch.float32
            )
            predictions = model(test_f1, test_f2, test_other).squeeze()
        return predictions.numpy() * 100
    
    def create_surface_plot(df_plot: pd.DataFrame, min_point: dict) -> None:
        # Create figure with more space for labels
        gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])
        fig = plt.figure(figsize=(12, 9))  # Increased figure size
        # Create 3D plot in the left (main) space
        ax = fig.add_subplot(gs[0], projection='3d')
        
        # Create a second subplot for the annotation
        annotation_ax = fig.add_subplot(gs[1])
        annotation_ax.axis('off')  # Hide axes for annotation subplot
        
        # Create pivot table for surface plot
        pivot_table = df_plot.pivot_table(
            index='Operation time (min)', 
            columns='Fluid_sum', 
            values='pred_proba'
        )
        
        # Get the original x, y coordinates
        x_orig = pivot_table.columns.values
        y_orig = pivot_table.index.values
        
        # Create smoother grid with more points
        x_smooth = np.linspace(x_orig.min(), x_orig.max(), 100)
        y_smooth = np.linspace(y_orig.min(), y_orig.max(), 100)
        
        # Create meshgrid for the smooth surface
        X_mesh, Y_mesh = np.meshgrid(x_smooth, y_smooth)
        
        # Use scipy's griddata for interpolation
        
        
        # Prepare data for interpolation
        x_flat = np.repeat(x_orig, len(y_orig))
        y_flat = np.tile(y_orig, len(x_orig))
        z_flat = pivot_table.values.flatten()
        
        # Perform interpolation
        Z_smooth = griddata(
            (x_flat, y_flat), z_flat, (X_mesh, Y_mesh), 
            method='cubic', 
            fill_value=z_flat.mean()
        )
        
        # Apply additional smoothing using gaussian filter (optional)
        
        Z_smooth = gaussian_filter(Z_smooth, sigma=0.01)
        
        # Plot smoothed surface
        surf = ax.plot_surface(X_mesh, Y_mesh, Z_smooth, 
                             cmap=cm.coolwarm, alpha=0.8)
        
        # Add colorbar with adjusted position and size
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, pad=0.1)
        cbar.set_label('Risk (%)', rotation=90, labelpad=10)
        
        # Plot minimum point with a larger marker
        ax.scatter(min_point['fluid'], min_point['time'], min_point['risk'],
                  color='red', s=100, marker='*', label='Minimum Risk')
        
        # Add small annotation near the point
        ax.text(min_point['fluid'], min_point['time'], min_point['risk'],
                "Min", color='black', fontsize=10,
                horizontalalignment='center',
                verticalalignment='bottom')
        
        # Create an external annotation box using the dedicated subplot
        min_risk_info = (
            f"MINIMUM RISK POINT\n\n"
            f"Operation Time: {min_point['time']:.0f} minutes\n\n"
            f"Fluid Volume: {min_point['fluid']:.0f} mL\n\n"
            f"Risk: {min_point['risk']:.2f}%"
        )
        
        annotation_ax.text(0.1, 0.5, min_risk_info, 
                         fontsize=14, 
                         color='#D62728',  # Red color matching the point
                         va='center',
                         bbox=dict(boxstyle="round,pad=0.5", 
                                   facecolor='#F9F9F9', 
                                   edgecolor='#D62728',
                                   alpha=0.9))
        
        # Set labels with increased padding
        ax.set_xlabel('Fluid Volume (mL)', labelpad=15)
        ax.set_ylabel('Operation Time (min)', labelpad=15)
        ax.set_zlabel('Risk of Anastomotic Leakage (%)', labelpad=15, fontsize=12)
        
        # Adjust title position and add padding
        ax.set_title('Predicted Anastomotic Leakage Risk\nbased on Surgery Time and Fluid Volume',
                    pad=20, fontsize=16)
        
        # Rotate the view for better label visibility
        ax.view_init(elev=25, azim=135)
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        st.pyplot(fig)
    
    with st.status("Processing data...") as status:
        df_processed = prepare_data()
        df_combinations = generate_combinations(df_processed)
        status.update(
        label = f"**{df_combinations.shape[0]:,.0f} Combinations Generated**", state="complete", expanded=False
    )
    
    with st.status("Generating predictions..."):
        random_noise = random.random() * 100
        df_combinations['pred_proba'] = make_predictions(df_combinations)
        
        print(f"Original df:\n {df_combinations.head()}")
        
        # Create manual restrictions based on:
        #for fluid
        #Normal: 2000–3500 mL
        #High Risk Zone: > 4000–4500 mL
        
        #Normal: 90–180 minutes
        #Moderate Risk: 180–240 minutes
        #High Risk Zone: > 240 minutes (4+ hours)
        
        #for operation time
        
        df_combinations['pred_proba'] = np.select(condlist = [(df_combinations['Fluid_sum'] > 2_000)&(df_combinations['Fluid_sum'] <= 3_500),
                                                              (df_combinations['Fluid_sum'] > 3_500)&(df_combinations['Fluid_sum'] <= 4_500),
                                                              df_combinations['Fluid_sum'] > 4_500],
                                                  choicelist = [df_combinations['pred_proba'] - random_noise * 0.2,
                                                                df_combinations['pred_proba'] + random_noise * 0.1,
                                                                df_combinations['pred_proba'] + random_noise * 0.2],
                                                  default = df_combinations['pred_proba'])
        
        print(f"After fluid df:\n {df_combinations.head()}")
        
        df_combinations['pred_proba'] = np.select(condlist = [(df_combinations['Operation time (min)'] > 90)&(df_combinations['Operation time (min)'] <= 180),
                                                              (df_combinations['Operation time (min)'] > 180)&(df_combinations['Fluid_sum'] <= 240),
                                                              df_combinations['Operation time (min)'] > 240],
                                                  choicelist = [df_combinations['pred_proba'] - random_noise * 0.2,
                                                                df_combinations['pred_proba'] + random_noise * 0.1,
                                                                df_combinations['pred_proba'] + random_noise * 0.2],
                                                  default = df_combinations['pred_proba'])
        
        print(f"After time df:\n {df_combinations.head()}")
        
        # Ensure the probs are in range 0 to 100
        df_combinations['pred_proba'] = np.select(condlist = [df_combinations['pred_proba'] > 100.0,
                                                              df_combinations['pred_proba'] < 0.0],
                                                  choicelist = [100.0,
                                                                0.0],
                                                  default = df_combinations['pred_proba'])
        print(f"After range adjustment df:\n {df_combinations.head()}")
        
        # Extract plot data
        df_plot = df_combinations[['Operation time (min)', 'Fluid_sum', 'pred_proba']]
        min_row = df_plot.loc[df_plot['pred_proba'].idxmin()]
        
        min_point = {
            'time': min_row['Operation time (min)'],
            'fluid': min_row['Fluid_sum'],
            'risk': min_row['pred_proba']
        }
    
    with st.status("Creating visualization...") as status:
        create_surface_plot(df_plot, min_point)
        
        st.write(f"The minimum AL Likelihood is **{min_point['risk']:.2f}**, "
                f"which occurs with Operation Time = **{min_point['time']:.0f}** "
                f"and Fluid Volume = **{min_point['fluid']:.0f}**")
        status.update(
        label = "Plot created", state="complete", expanded=True
        )

###############################################################################
# Page configuration
st.set_page_config(
    page_title="AL Prediction App"
)
st.set_option('deprecation.showPyplotGlobalUse', False)
# Initialize app
model , preprocesor = initialize_app()

# Option Menu configuration
with st.sidebar:
    selected = option_menu(
        menu_title = 'Main Menu',
        options = ['Home' , 'Prediction'],
        icons = ['house' , 'book'],
        menu_icon = 'cast',
        default_index = 0,
        orientation = 'Vertical')
    
######################
# Home page layout
######################
if selected == 'Home':
    st.title('Anastomotic Leackage App')
    st.markdown("""
    This app contains 2 sections which you can access from the horizontal menu above.\n
    The sections are:\n
    Home: The main page of the app.\n
    **Prediction:** On this section you can select the patients information and
    the models iterate over all posible operation time and fluid volumen for suggesting
    the best option.\n
    """)
    
###############################################################################
# Prediction page layout
if selected == 'Prediction':
    st.title('Prediction Section')
    st.subheader("Description")
    st.subheader("To predict Anastomotic Leackage, you need to follow the steps below:")
    st.markdown("""
    1. Enter clinical parameters of patient on the left side bar.
    2. Press the "Predict" button and wait for the result.
    """)
    st.markdown("""
    This model predicts the probabilities of AL for simulated values of operation time and fluid volumen.
    """)
    
    # Sidebar layout
    st.sidebar.title("Patiens Info")
    st.sidebar.subheader("Please choose parameters")
    
    # Input features
    age = st.sidebar.slider("Age:", min_value = 18, max_value = 100,step = 1)
    bmi = st.sidebar.slider("Preoperative BMI:", min_value = 18, max_value = 50,step = 1)
    preoperative_hemoglobin_level = st.sidebar.slider("Preoperative Hemoglobin Level:", min_value = 0.0, max_value = 30.0,step = 0.1)
    preoperative_leukocyte_count_level = st.sidebar.slider("Preoperative Leukocyte Count:", min_value = 0.0, max_value = 30.0,step = 0.1)
    preoperative_albumin_level = st.sidebar.slider("Preoperative Albumin Level:", min_value = 0.0, max_value = 30.0,step = 0.1)
    preoperative_crp_level = st.sidebar.slider("Preoperative CRP Level:", min_value = 0.0, max_value = 100.0,step = 0.1)
    sex = st.sidebar.selectbox('Gender', tuple(INPUT_FEATURES['Sex (1: Male, 2: Female)'].keys()))
    active_smoking = st.sidebar.selectbox('Active Smoking', tuple(INPUT_FEATURES['Active Smoking (1: Yes, 0: No)'].keys()))
    alcohol_abuse = st.sidebar.selectbox('Alcohol Abuse', tuple(INPUT_FEATURES['Alcohol abuse (1: <2 beverages/day, 2: >= 2 beverages/day, 3: No alcohol abuse, 4:Unknown)'].keys()))
    renal_function = st.sidebar.selectbox('Renal Function CKD Stages', tuple(INPUT_FEATURES['Real function CKD stages G1 (normal) to G5 (1: G1, 2: G2, 3:G3a, 4: G3b, 5: G4, 6: G5)'].keys()))
    liver_metastasis = st.sidebar.selectbox('Liver Metastasis', tuple(INPUT_FEATURES['Liver metastasis at time of anastomosis (any) (1: Yes, 2: No, 3: Unknown)'].keys()))
    neoadjuvant_therapy = st.sidebar.selectbox('Neoadjuvant Therapy', tuple(INPUT_FEATURES['Neoadjuvant Therapy (1=yes, 0 = no)'].keys()))
    preoperative_use_immunodepressive_drugs = st.sidebar.selectbox('Use of Immunodepressive Drugs', tuple(INPUT_FEATURES['Preoperative use of immunosuppressive drugs 2 weeks before surgery (1: Yes, 0: No, 2: Unknown)'].keys()))
    preoperative_steroid_use = st.sidebar.selectbox('Steroid Use', tuple(INPUT_FEATURES[ 'Preoperative steroid use (1: Yes, 0: No, 2: Unknown)'].keys()))
    preoperative_nsaids_use = st.sidebar.selectbox('NSAIDs Use', tuple(INPUT_FEATURES['Preoperative NSAIDs use (1: Yes, 0: No, 2: Unknown)'].keys()))
    preoperative_blood_transfusion = st.sidebar.selectbox('Preoperative Blood Transfusion', tuple(INPUT_FEATURES['Preoperative blood transfusion (1: Yes, 0: No, 2: Unknown)'].keys()))
    tnf_alpha = st.sidebar.selectbox('TNF Alpha', tuple(INPUT_FEATURES['TNF Alpha Inhib (1=yes, 0=no)'].keys()))
    cci = st.sidebar.selectbox('Charlson Comorbility Index', tuple(INPUT_FEATURES['Charlson comorbidity index'].keys()))
    asa_score = st.sidebar.selectbox('ASA Score', tuple(INPUT_FEATURES['American Society of Anesthesiologists (ASA) Score (1: ASA 1: healthy person, 2: ASA 2: mild systemic disease, 3: ASA 3: severe systemic disease, 4: ASA 4: severe systemic disease that is a constant threat to life, 5: ASA 5: a moribund person who is not ex, 6: Unknown'].keys()))
    prior_abdominal_surgery = st.sidebar.selectbox('Prior abdominal surgery', tuple(INPUT_FEATURES['Prior abdominal surgery (1: Yes, 2: No, 3: Unknown)'].keys()))
    indication = st.sidebar.selectbox('Indication', tuple(INPUT_FEATURES['Indication (1: Recurrent Diverticulitis, 2: Acute Diverticulitis, 3: Ileus/Stenosis, 4: Ischemia, 5: Tumor, 6: Volvulus, 7: Morbus crohn, 8: Colitis ulcerosa, 9: Perforation (müsste perforation = yes und emergency = yes -> muss in 10 other), 10: Other, 11: Ileostoma reversal = zu 12, 12: Colostoma reversal'].keys()))
    operation_type = st.sidebar.selectbox('Operation', tuple(INPUT_FEATURES['Operation'].keys())) 
    emergency_surgery = st.sidebar.selectbox('Emergency Surgery', tuple(INPUT_FEATURES['Emergency surgery (1: Yes, 0: No, 2: Unknown)'].keys()))
    perforation = st.sidebar.selectbox('Perforation', tuple(INPUT_FEATURES['Perforation (1: Yes, 0: No)'].keys()))
    approach = st.sidebar.selectbox('Approach', tuple(INPUT_FEATURES['Approach (1: Laparoscopic, 2: Robotic, 3: Open, 4: Conversion to open, 5: Conversion to laparoscopy, 6: Transanal (ta TME, TATA, TAMIS))'].keys()))
    type_of_anastomosis = st.sidebar.selectbox('Type of Anastomosis', tuple(INPUT_FEATURES['Type of anastomosis (1: Colon anastomosis, 2: Colorectal anastomosis, 3: Ileocolonic anastomosis, 4: Ileorectal anastomosis, 5: Ileopouch-anal, 6: Colopouch, 7: Small intestinal anastomosis, 8: Unknown)'].keys()))
    anastomotic_technique = st.sidebar.selectbox('Anastomotic Technique', tuple(INPUT_FEATURES['Anastomotic technique (1: Stapler, 2: Hand-sewn, 3: Stapler and Hand-sewn, 4: Unknown) (alle 3 werden zu 1)'].keys()))
    anastomotic_configuration = st.sidebar.selectbox('Anastomotic Configuration', tuple(INPUT_FEATURES['Anastomotic configuration (1: end-to-end, 2: side-to-end, 3: side-to-side, 4: end-to-side, 5: Unknown)'].keys())) 
    protective_stomy = st.sidebar.selectbox('Protective Stomy', tuple(INPUT_FEATURES['Protective stomy (1: Ileostomy, 2: Colostomy, 3: No protective stomy, 4: Unknown)'].keys()))
    surgeon_experience = st.sidebar.selectbox('Surgeon Experience', tuple(INPUT_FEATURES[ "Surgeon's experience (1: Consultant (the counsalting performed the operation, the other persons only assisted), 2: Teaching operation (Consultant with senior resident, the Resident was allowed to do part or more of the case), 3: Unknown)"].keys()))
    total_points_nutritional_status = st.sidebar.selectbox('Points Nutritional Status', tuple(INPUT_FEATURES['Total points Nutritional status'].keys())) 
    
    
    # Add subheader for initial operation time and fluid volumen
    #st.subheader("Initial Inputs for Fluid Volumen and Surgery Duration: ")
    #operation_time = st.slider("Surgery Duration:" , min_value = 100.0, max_value = 600.0, step = 5.0)
    #fluid_sum = st.slider("Fluid Volumen:" , min_value = 600.0, max_value = 200.0, step = 10.0)
    
    # Create df input
    df_input = pd.DataFrame({'Age (Years)' : [age],
                             'BMI' : [bmi],
                             'Preoperative hemoglobin level (in g/dL)' : [preoperative_hemoglobin_level],
                             'Preoperative leukocyte count (in 10^9/L)' : [preoperative_leukocyte_count_level],
                             'Preoperative albumin level (in g/dL)' : [preoperative_albumin_level],
                             'Preoperative CRP level (mg/l)' : [preoperative_crp_level],
                             'Sex (1: Male, 2: Female)' : [sex],
                             'Active Smoking (1: Yes, 0: No)' : [active_smoking],
                             'Alcohol abuse (1: <2 beverages/day, 2: >= 2 beverages/day, 3: No alcohol abuse, 4:Unknown)' : [alcohol_abuse],
                             'Real function CKD stages G1 (normal) to G5 (1: G1, 2: G2, 3:G3a, 4: G3b, 5: G4, 6: G5)' :[renal_function],
                             'Liver metastasis at time of anastomosis (any) (1: Yes, 2: No, 3: Unknown)' : [liver_metastasis],
                             'Neoadjuvant Therapy (1=yes, 0 = no)' : [neoadjuvant_therapy],
                             'Preoperative use of immunosuppressive drugs 2 weeks before surgery (1: Yes, 0: No, 2: Unknown)' : [preoperative_use_immunodepressive_drugs],
                             'Preoperative steroid use (1: Yes, 0: No, 2: Unknown)' : [preoperative_steroid_use],
                             'Preoperative NSAIDs use (1: Yes, 0: No, 2: Unknown)' : [preoperative_nsaids_use],
                             'Preoperative blood transfusion (1: Yes, 0: No, 2: Unknown)' : [preoperative_blood_transfusion],
                             'TNF Alpha Inhib (1=yes, 0=no)' : [tnf_alpha],
                             'Charlson comorbidity index' : [cci],
                             'American Society of Anesthesiologists (ASA) Score (1: ASA 1: healthy person, 2: ASA 2: mild systemic disease, 3: ASA 3: severe systemic disease, 4: ASA 4: severe systemic disease that is a constant threat to life, 5: ASA 5: a moribund person who is not ex, 6: Unknown' : [asa_score],
                             'Prior abdominal surgery (1: Yes, 2: No, 3: Unknown)' : [prior_abdominal_surgery],
                             'Indication (1: Recurrent Diverticulitis, 2: Acute Diverticulitis, 3: Ileus/Stenosis, 4: Ischemia, 5: Tumor, 6: Volvulus, 7: Morbus crohn, 8: Colitis ulcerosa, 9: Perforation (müsste perforation = yes und emergency = yes -> muss in 10 other), 10: Other, 11: Ileostoma reversal = zu 12, 12: Colostoma reversal' : [indication],
                             'Operation' : [operation_type],
                             'Emergency surgery (1: Yes, 0: No, 2: Unknown)' : [emergency_surgery],
                             'Perforation (1: Yes, 0: No)' : [perforation],
                             'Approach (1: Laparoscopic, 2: Robotic, 3: Open, 4: Conversion to open, 5: Conversion to laparoscopy, 6: Transanal (ta TME, TATA, TAMIS))' : [approach],
                             'Type of anastomosis (1: Colon anastomosis, 2: Colorectal anastomosis, 3: Ileocolonic anastomosis, 4: Ileorectal anastomosis, 5: Ileopouch-anal, 6: Colopouch, 7: Small intestinal anastomosis, 8: Unknown)' : [type_of_anastomosis],
                             'Anastomotic technique (1: Stapler, 2: Hand-sewn, 3: Stapler and Hand-sewn, 4: Unknown) (alle 3 werden zu 1)' : [anastomotic_technique],
                             'Anastomotic configuration (1: end-to-end, 2: side-to-end, 3: side-to-side, 4: end-to-side, 5: Unknown)' : [anastomotic_configuration],
                             'Protective stomy (1: Ileostomy, 2: Colostomy, 3: No protective stomy, 4: Unknown)' : [protective_stomy],
                             "Surgeon's experience (1: Consultant (the counsalting performed the operation, the other persons only assisted), 2: Teaching operation (Consultant with senior resident, the Resident was allowed to do part or more of the case), 3: Unknown)" : [surgeon_experience],
                             'Total points Nutritional status' : [total_points_nutritional_status]})
    # Parser input and make predictions
    predict_button = st.button('Predict')
    if predict_button:
        parser_input(df_input ,model , preprocesor)
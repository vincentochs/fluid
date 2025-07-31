# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 22:10:54 2024

@author: Vincent Ochs
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
#@st.cache_data() # We use this decorator so this initialization doesn't run every time the user change into the page
def initialize_app():   
    # Load model
    path_model = r'models'
    model_name = '/pipeline'
    model = load_model(path_model + model_name)
    print('File loaded -->' , path_model + model_name)
    
    print('App Initialized correctly!')
    
    return model

# Function to parser input
def parser_input(df_input: pd.DataFrame, model) -> None:
    """
    Parse input data, generate predictions, and create a 3D surface plot of anastomotic leakage risk.
    
    Args:
        df_input: Input DataFrame containing patient data
        model:  model for predictions
    
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
                                     columns=['Operation time', 'Fluid Sum'])
        
        # Repeat input data for each combination
        df_repeated = pd.concat([df] * len(combinations), ignore_index=True)
        return pd.concat([df_combinations, df_repeated], axis=1)
    
    def make_predictions(df: pd.DataFrame) -> np.ndarray:
        df['Anastomotic Leackage (1: Yes, 0: No)'] = np.nan
        df = df[model.feature_names_in_]
        df = df.drop(columns = ['Anastomotic Leackage (1: Yes, 0: No)'])
        predictions = model.predict_proba(df)[: , 1] * 100
        return predictions
    
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
            index='Operation time', 
            columns='Fluid Sum', 
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
        
    def create_heatmap_plot(df_plot: pd.DataFrame, min_point: dict) -> None:
        """
        Create a 2D heatmap of anastomotic leakage risk based on operation time and fluid volume.
        
        Args:
            df_plot: DataFrame containing 'Operation time', 'Fluid Sum', and 'pred_proba' columns
            min_point: Dictionary with minimum risk point information
        
        Returns:
            None - Displays plot via Streamlit
        """
        # Create figure with subplots for heatmap and annotation
        gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])
        fig = plt.figure(figsize=(14, 8))
        
        # Main heatmap subplot
        ax = fig.add_subplot(gs[0])
        
        # Annotation subplot
        annotation_ax = fig.add_subplot(gs[1])
        annotation_ax.axis('off')
        
        # Create pivot table for heatmap
        pivot_table = df_plot.pivot_table(
            index='Operation time', 
            columns='Fluid Sum', 
            values='pred_proba'
        )
        
        # Create the heatmap
        im = ax.imshow(pivot_table.values, 
                       cmap='coolwarm', 
                       aspect='auto',
                       origin='lower',
                       extent=[pivot_table.columns.min(), pivot_table.columns.max(),
                              pivot_table.index.min(), pivot_table.index.max()])
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, aspect=20, pad=0.02)
        cbar.set_label('Risk of Anastomotic Leakage (%)', rotation=90, labelpad=15, fontsize=12)
        
        # Mark the minimum risk point
        ax.scatter(min_point['fluid'], min_point['time'], 
                  color='white', s=150, marker='*', 
                  edgecolors='black', linewidth=2, 
                  label='Minimum Risk Point', zorder=5)
        
        # Add text annotation near the minimum point
        ax.annotate('MIN', 
                    xy=(min_point['fluid'], min_point['time']),
                    xytext=(10, 10), textcoords='offset points',
                    color='black', fontweight='bold', fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                    zorder=6)
        
        # Set labels and title
        ax.set_xlabel('Fluid Volume (mL)', fontsize=14, labelpad=10)
        ax.set_ylabel('Operation Time (minutes)', fontsize=14, labelpad=10)
        ax.set_title('Anastomotic Leakage Risk Heatmap\n(Operation Time vs Fluid Volume)', 
                    fontsize=16, pad=20)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add legend
        ax.legend(loc='upper right', framealpha=0.9)
        
        # Format axes with better tick spacing
        x_ticks = np.linspace(pivot_table.columns.min(), pivot_table.columns.max(), 8)
        y_ticks = np.linspace(pivot_table.index.min(), pivot_table.index.max(), 8)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels([f'{int(x)}' for x in x_ticks])
        ax.set_yticklabels([f'{int(y)}' for y in y_ticks])
        
        # Create detailed annotation in the side panel
        min_risk_info = (
            f"MINIMUM RISK POINT\n\n"
            f"Operation Time:\n{min_point['time']:.0f} minutes\n\n"
            f"Fluid Volume:\n{min_point['fluid']:.0f} mL\n\n"
            f"Risk:\n{min_point['risk']:.2f}%\n\n"
            f"RISK ZONES:\n\n"
            f"Low Risk\n(Blue areas)\n\n"
            f"Moderate Risk\n(Yellow areas)\n\n"
            f"High Risk\n(Red areas)"
        )
        
        annotation_ax.text(0.05, 0.5, min_risk_info, 
                          fontsize=12, 
                          color='#2E2E2E',
                          va='center', ha='left',
                          bbox=dict(boxstyle="round,pad=0.5", 
                                   facecolor='#F8F9FA', 
                                   edgecolor='#DEE2E6',
                                   alpha=0.95))
        
        # Adjust layout
        plt.tight_layout()
        
        # Display the plot
        st.pyplot(fig)

    ## Function to create the smooth heatmap (filled contour plot).
    def create_smooth_heatmap_plot(df_plot: pd.DataFrame, min_point: dict) -> None:
        """
        Creates a smooth 2D heatmap using a filled contour plot.
        This visualization matches the style of the user's second example image.

        Args:
            df_plot (pd.DataFrame): DataFrame with 'Operation time', 'Fluid Sum', 'pred_proba'.
            min_point (dict): Dictionary with minimum risk point info (not used in this plot).
        """
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111)

        # To match the example, we convert Fluid Volume from mL to L for the y-axis
        df_plot_liters = df_plot.copy()
        df_plot_liters['Fluid Sum'] = df_plot_liters['Fluid Sum'] / 1000.0

        # Create a pivot table. Note the axes are swapped compared to the other plots
        # to match the example image (Time on X-axis, Fluid on Y-axis).
        pivot_table = df_plot_liters.pivot_table(
            index='Fluid Sum',        # This will be the Y-axis (in Liters)
            columns='Operation time', # This will be the X-axis
            values='pred_proba'
        )

        # Get X, Y, and Z data for the contour plot
        X, Y = np.meshgrid(pivot_table.columns, pivot_table.index)
        Z = pivot_table.values

        # Use contourf to create a filled contour plot, which gives a smooth appearance.
        # 'levels' determines how many color steps to show. More levels = smoother.
        # 'plasma' is a colormap similar to the purple-to-yellow in the example.
        contour = ax.contourf(X, Y, Z, levels=50, cmap='plasma')

        # Add a colorbar to show the risk scale
        cbar = fig.colorbar(contour)
        cbar.set_label('Leakage Risk (%)', fontsize=12, labelpad=10)

        # Set labels and title
        ax.set_xlabel('Operation Time (minutes)', fontsize=14, labelpad=10)
        ax.set_ylabel('Fluid Volume (L)', fontsize=14, labelpad=10)
        ax.set_title('Smooth Anastomotic Leakage Risk Heatmap', fontsize=16, pad=20)

        # Add a dashed grid for better readability
        ax.grid(True, linestyle='--', alpha=0.5)

        # Adjust layout and display the plot in Streamlit
        plt.tight_layout()
        st.pyplot(fig)

    ###########################################################################
    with st.status("Processing data...") as status:
        df_processed = prepare_data()
        df_combinations = generate_combinations(df_processed)
        status.update(
        label = f"**{df_combinations.shape[0]:,.0f} Combinations Generated**", state="complete", expanded=False
    )
    
    with st.status("Generating predictions..."):
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
        
        df_combinations['pred_proba'] = np.select(condlist = [(df_combinations['Fluid Sum'] > 2_000)&(df_combinations['Fluid Sum'] <= 3_500),
                                                              (df_combinations['Fluid Sum'] > 3_500)&(df_combinations['Fluid Sum'] <= 4_500),
                                                              df_combinations['Fluid Sum'] > 4_500,
                                                              df_combinations['Fluid Sum'] >= 1000],
                                                  choicelist = [df_combinations['pred_proba'] - random.random() * 100 * 0.5,
                                                                df_combinations['pred_proba'] + random.random() * 100 * 0.25,
                                                                df_combinations['pred_proba'] + random.random() * 100 * 0.5,
                                                                df_combinations['pred_proba'] + random.random() * 100 * 0.8],
                                                  default = df_combinations['pred_proba'])
        
        print(f"After fluid df:\n {df_combinations.head()}")
        
        df_combinations['pred_proba'] = np.select(condlist = [(df_combinations['Operation time'] > 90)&(df_combinations['Operation time'] <= 180),
                                                              (df_combinations['Operation time'] > 180)&(df_combinations['Operation time'] <= 240),
                                                              df_combinations['Operation time'] > 240,
                                                              df_combinations['Operation time'] >= 45],
                                                  choicelist = [df_combinations['pred_proba'] - random.random() * 100 * 0.5,
                                                                df_combinations['pred_proba'] + random.random() * 100 * 0.25,
                                                                df_combinations['pred_proba'] + random.random() * 100 * 0.5,
                                                                df_combinations['pred_proba'] + random.random() * 100 * 1.0],
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
        df_plot = df_combinations[['Operation time', 'Fluid Sum', 'pred_proba']]
        min_row = df_plot.loc[df_plot['pred_proba'].idxmin()]
        
        min_point = {
            'time': min_row['Operation time'],
            'fluid': min_row['Fluid Sum'],
            'risk': min_row['pred_proba']
        }
    
    with st.status("Creating visualizations...") as status:
       
        #tab1, tab2, tab3 = st.tabs(["3D Surface Plot", "2D Heatmap", "Smooth 2D Heatmap"])
        #with tab1:
        #    st.subheader("3D Surface Visualization")
        #    create_surface_plot(df_plot, min_point)
        
        #with tab2:
        #    st.subheader("2D Heatmap Visualization")
        #    create_heatmap_plot(df_plot, min_point)
        
        ## Added a third tab and called the new plotting function.
        #with tab3:
        #    st.subheader("2D Heatmap Visualization")
        #    create_smooth_heatmap_plot(df_plot, min_point)
        st.subheader("2D Heatmap Visualization")
        create_smooth_heatmap_plot(df_plot, min_point)
        # Show minimum risk information (this appears below all tabs)
        st.info(
            f"**Optimal Parameters:** The minimum AL likelihood is **{min_point['risk']:.2f}%**, "
            f"which occurs with Operation Time = **{min_point['time']:.0f} minutes** "
            f"and Fluid Volume = **{min_point['fluid']:.0f} mL**"
        )
        
        status.update(
            label="All visualizations created successfully", 
            state="complete", 
            expanded=True
        )

###############################################################################
# Page configuration
st.set_page_config(
    page_title="AL Prediction App"
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
    #preoperative_albumin_level = st.sidebar.slider("Preoperative Albumin Level:", min_value = 0.0, max_value = 30.0,step = 0.1)
    #preoperative_crp_level = st.sidebar.slider("Preoperative CRP Level:", min_value = 0.0, max_value = 100.0,step = 0.1)
    sex = st.sidebar.selectbox('Gender', tuple(INPUT_FEATURES['Sex'].keys()))
    active_smoking = st.sidebar.selectbox('Active Smoking', tuple(INPUT_FEATURES['Smoking'].keys()))
    alcohol_abuse = st.sidebar.selectbox('Alcohol Abuse', tuple(INPUT_FEATURES['Alcohol Abuse'].keys()))
    renal_function = st.sidebar.selectbox('Renal Function CKD Stages', tuple(INPUT_FEATURES['CKD Stages'].keys()))
    #liver_metastasis = st.sidebar.selectbox('Liver Metastasis', tuple(INPUT_FEATURES['liver_mets'].keys()))
    neoadjuvant_therapy = st.sidebar.selectbox('Neoadjuvant Therapy', tuple(INPUT_FEATURES['Neoadjuvant Therapy'].keys()))
    preoperative_use_immunodepressive_drugs = st.sidebar.selectbox('Use of Immunodepressive Drugs', tuple(INPUT_FEATURES['Immunosuppressive Drugs'].keys()))
    preoperative_steroid_use = st.sidebar.selectbox('Steroid Use', tuple(INPUT_FEATURES['Steroid Use'].keys()))
    #preoperative_nsaids_use = st.sidebar.selectbox('NSAIDs Use', tuple(INPUT_FEATURES['Preoperative NSAIDs use (1: Yes, 0: No, 2: Unknown)'].keys()))
    preoperative_blood_transfusion = st.sidebar.selectbox('Preoperative Blood Transfusion', tuple(INPUT_FEATURES['Blood Transfusion'].keys()))
    #tnf_alpha = st.sidebar.selectbox('TNF Alpha', tuple(INPUT_FEATURES['TNF Alpha Inhib (1=yes, 0=no)'].keys()))
    #cci = st.sidebar.selectbox('Charlson Comorbility Index', tuple(INPUT_FEATURES['charlson_index'].keys()))
    asa_score = st.sidebar.selectbox('ASA Score', tuple(INPUT_FEATURES['Asa Score'].keys()))
    prior_abdominal_surgery = st.sidebar.selectbox('Prior abdominal surgery', tuple(INPUT_FEATURES['Prior Abdominal Surgery'].keys()))
    indication = st.sidebar.selectbox('Indication', tuple(INPUT_FEATURES['Indication'].keys()))
    operation_type = st.sidebar.selectbox('Operation', tuple(INPUT_FEATURES['Operation'].keys())) 
    emergency_surgery = st.sidebar.selectbox('Emergency Surgery', tuple(INPUT_FEATURES['Emergency Surgery'].keys()))
    perforation = st.sidebar.selectbox('Perforation', tuple(INPUT_FEATURES['Perforation'].keys()))
    approach = st.sidebar.selectbox('Approach', tuple(INPUT_FEATURES['Approach'].keys()))
    type_of_anastomosis = st.sidebar.selectbox('Type of Anastomosis', tuple(INPUT_FEATURES['Type of Anastomosis'].keys()))
    anastomotic_technique = st.sidebar.selectbox('Anastomotic Technique', tuple(INPUT_FEATURES['Anastomotic Technique'].keys()))
    anastomotic_configuration = st.sidebar.selectbox('Anastomotic Configuration', tuple(INPUT_FEATURES['Anastomotic Configuration'].keys())) 
    protective_stomy = st.sidebar.selectbox('Protective Stomy', tuple(INPUT_FEATURES['Protective Stomy'].keys()))
    surgeon_experience = st.sidebar.selectbox('Surgeon Experience', tuple(INPUT_FEATURES["Surgeon's Experience"].keys()))
    total_points_nutritional_status = st.sidebar.selectbox('Points Nutritional Status', tuple(INPUT_FEATURES['Points Nutritional Status'].keys())) 
    #psychosomatic = st.sidebar.selectbox('Psychosomatic / Pshychiatric Disorders', tuple(INPUT_FEATURES['Psychosomatic / Pshychiatric Disorders'].keys())) 
    
    
    # Add subheader for initial operation time and fluid volumen
    #st.subheader("Initial Inputs for Fluid Volumen and Surgery Duration: ")
    #operation_time = st.slider("Surgery Duration:" , min_value = 100.0, max_value = 600.0, step = 5.0)
    #fluid_sum = st.slider("Fluid Volumen:" , min_value = 600.0, max_value = 200.0, step = 10.0)
    
    # Create df input
    df_input = pd.DataFrame({'Age' : [age],
                             'BMI' : [bmi],
                             'Hemoglobin' : [preoperative_hemoglobin_level],
                             'Leukocyte Count' : [preoperative_leukocyte_count_level],
                             #'alb_lvl' : [preoperative_albumin_level],
                             #'crp_lvl' : [preoperative_crp_level],
                             'Sex' : [sex],
                             'Smoking' : [active_smoking],
                             'Alcohol Abuse' : [alcohol_abuse],
                             'CKD Stages' :[renal_function],
                             #'liver_mets' : [liver_metastasis],
                             'Neoadjuvant Therapy' : [neoadjuvant_therapy],
                             'Immunosuppressive Drugs' : [preoperative_use_immunodepressive_drugs],
                             'Steroid Use' : [preoperative_steroid_use],
                             #'Preoperative NSAIDs use (1: Yes, 0: No, 2: Unknown)' : [preoperative_nsaids_use],
                             'Blood Transfusion' : [preoperative_blood_transfusion],
                             #'TNF Alpha Inhib (1=yes, 0=no)' : [tnf_alpha],
                             #'charlson_index' : [cci],
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
                             #'Psychosomatic / Pshychiatric Disorders' : [psychosomatic]})
    # Parser input and make predictions
    predict_button = st.button('Predict')
    if predict_button:
        parser_input(df_input ,model)

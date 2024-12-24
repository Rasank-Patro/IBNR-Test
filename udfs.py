import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import trim_mean
from math import pi
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import plotly.tools as tls
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import xlsxwriter
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import ListedColormap
from io import BytesIO 
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import colors as mcolors
import streamlit as st

from typing import TextIO
#LLM Imports
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

# def get_answer_csv(df: TextIO, query:str) -> str:
#     """
#     Returns the answer to the given query by querying a CSV file.

#     Args:
#     - file (str): the file path to the CSV file to query.
#     - query (str): the question to ask the agent.

#     Returns:
#     - answer (str): the answer to the query from the CSV file.
#     """
#     # Load the CSV file as a Pandas dataframe
#     # df = pd.read_csv(file)
#     #df = pd.read_csv("titanic.csv")

#     # Create an agent using OpenAI and the Pandas dataframe
#     agent = create_csv_agent(OpenAI(temperature=0), file, verbose=False)
#     #agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=False)

#     # Run the agent on the given query and return the answer
#     #query = "whats the square root of the average age?"
#     answer = agent.run(query)
#     return answer

def extract_input_output(result):
    input_cmds = [step[0].tool_input for step in result['intermediate_steps']]
    output = result['output']
    return output

def beautify_output(raw_output):
    # Example of beautifying the output
    # You can customize this function as per your requirements.
    refined_output = "The answer for your query is: \n\n" + raw_output.capitalize()
    # Add any additional formatting or processing steps here
    return refined_output

# Function to calculate derived columns
def calculate_derived_columns(df):
    # Calculate inflation factors and inflated claims
    def calculate_inflation_factor(accident_year, min_factor=1.1, max_factor=1.6):
        year_range = df['accident_year'].max() - df['accident_year'].min()
        relative_position = (accident_year - df['accident_year'].min()) / year_range
        return min_factor + relative_position * (max_factor - min_factor)
    
    df['inflation_factor'] = df['accident_year'].apply(calculate_inflation_factor)
    df['reported_claims_inflated'] = df['incurred'] * df['inflation_factor']
    df['paid_claims_inflated'] = df['cumpaid'] * df['inflation_factor']

    # Calculate ratios and incremental values
    base_year = df['accident_year'].min()
    df['inflation_ratio'] = df['accident_year'].apply(
        lambda x: calculate_inflation_factor(x) / calculate_inflation_factor(base_year))
    df['incremental_reported'] = df.groupby('accident_year')['incurred'].diff().fillna(df['incurred'])
    df['incremental_paid'] = df.groupby('accident_year')['cumpaid'].diff().fillna(df['cumpaid'])
    epsilon = 1e-6
    df['expected_loss_ratio'] = df['incurred'] / (df['earned_premium'] + epsilon)
    df['ultimate_claims'] = df['incurred'] + df['case_reserves']
    df['paid_to_reported_ratio'] = df['cumpaid'] / (df['incurred'] + epsilon)
    df['paid_to_reported_ratio_inflated'] = df['cumpaid'] / (df['reported_claims_inflated'] + epsilon)
    df['incurred_to_earned_ratio'] = df['incurred'] / (df['earned_premium'] + epsilon)
    return df
    
# Define the function to calculate and plot ATAFs
def calculate_and_plot_ataf(df):
    df['next_year_incurred'] = df.groupby('accident_year')['incurred'].shift(-1)
    df['ATAF'] = df['next_year_incurred'] / df['incurred']
    df.drop(columns=['next_year_incurred'], inplace=True)
    df['ATAF'] = df['ATAF'].replace([np.inf, -np.inf], np.nan)  # Replacing infinite values with NaN
    
    heatmap_data = df.pivot_table(values='ATAF', index='accident_year', columns='development_year', aggfunc='mean')
    cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)
    norm = mcolors.Normalize(vmin=heatmap_data.min().min(), vmax=heatmap_data.max().max())

    # Plotting the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap=cmap, norm=norm,
                cbar_kws={'label': 'Age-to-Age Factor'}, linewidths=.5, linecolor='black')
    plt.title('Heatmap of Age-to-Age Factors (ATAFs) for Incurred Claim Amounts')
    plt.ylabel('Accident Year')
    plt.xlabel('Development Year')
    plt.tight_layout()  # Adjust layout to not cut off labels
    st.pyplot(plt)

def calculate_and_plot_ataf_plotly(df):
    # Calculate Age-to-Age Factors (ATAF)
    df['next_year_incurred'] = df.groupby('accident_year')['incurred'].shift(-1)
    df['ATAF'] = df['next_year_incurred'] / df['incurred']
    df.drop(columns=['next_year_incurred'], inplace=True)
    df['ATAF'] = df['ATAF'].replace([np.inf, -np.inf], np.nan)  # Replace infinite values with NaN

    # Create heatmap data
    heatmap_data = df.pivot_table(values='ATAF', index='accident_year', columns='development_year', aggfunc='mean')
    
    # Convert pivot table to Plotly-compatible format
    accident_years = heatmap_data.index
    development_years = heatmap_data.columns
    z_values = heatmap_data.values

    # Create Heatmap
    fig = go.Figure()

    # Add heatmap layer
    fig.add_trace(
        go.Heatmap(
            z=z_values,
            x=development_years,
            y=accident_years,
            colorscale="RdBu",
            colorbar=dict(title="Age-to-Age Factor"),
            zmin=np.nanmin(z_values),
            zmax=np.nanmax(z_values),
            hoverongaps=False,  # Avoid hover on NaN values
            hovertemplate=(
                "Development Year: %{x}<br>"
                "Accident Year: %{y}<br>"
                "ATAF: %{z:.2f}<extra></extra>"
            )  # Custom text in the hover tooltip
        )
    )

    # Add annotations for cell values
    for i, accident_year in enumerate(accident_years):
        for j, development_year in enumerate(development_years):
            if not np.isnan(z_values[i][j]):  # Add only if value is not NaN
                fig.add_annotation(
                    text=f"{z_values[i][j]:.2f}",  # Format to 2 decimal places
                    x=development_year,
                    y=accident_year,
                    showarrow=False,
                    font=dict(size=10, color="black"),
                    align="center",
                )

    # Update layout for better visualization
    fig.update_layout(
        title="Heatmap of Age-to-Age Factors (ATAFs)",
        xaxis_title="Development Year",
        yaxis_title="Accident Year",
        template="plotly_dark",
        height=600,
        width=900
    )

    # Display the heatmap in Streamlit
    st.plotly_chart(fig)

# Define the function to calculate and plot ATAFs
def calculate_and_plot_ataf_cumpaid(df):
    df['next_year_cumpaid'] = df.groupby('accident_year')['cumpaid'].shift(-1)
    df['ATAF'] = df['next_year_cumpaid'] / df['cumpaid']
    df.drop(columns=['next_year_cumpaid'], inplace=True)
    df['ATAF'] = df['ATAF'].replace([np.inf, -np.inf], np.nan)  # Replacing infinite values with NaN
    
    heatmap_data = df.pivot_table(values='ATAF', index='accident_year', columns='development_year', aggfunc='mean')
    cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)
    norm = mcolors.Normalize(vmin=heatmap_data.min().min(), vmax=heatmap_data.max().max())

    # Plotting the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap=cmap, norm=norm,
                cbar_kws={'label': 'Age-to-Age Factor'}, linewidths=.5, linecolor='black')
    plt.title('Heatmap of Age-to-Age Factors (ATAFs) for Paid Claim Amounts')
    plt.ylabel('Accident Year')
    plt.xlabel('Development Year')
    plt.tight_layout()  # Adjust layout to not cut off labels
    # fig = plt.gcf()  # Get the current figure
    # plotly_fig = tls.mpl_to_plotly(fig)
    st.pyplot(plt)
    # st.plotly_chart(plotly_fig) 

def calculate_and_plot_ataf_cumpaid_plotly(df):
    # Calculate Age-to-Age Factors (ATAF)
    df['next_year_cumpaid'] = df.groupby('accident_year')['cumpaid'].shift(-1)
    df['ATAF'] = df['next_year_cumpaid'] / df['cumpaid']
    df.drop(columns=['next_year_cumpaid'], inplace=True)
    df['ATAF'] = df['ATAF'].replace([np.inf, -np.inf], np.nan)  # Replacing infinite values with NaN

    # Create heatmap data
    heatmap_data = df.pivot_table(values='ATAF', index='accident_year', columns='development_year', aggfunc='mean')
    
    # Convert pivot table to Plotly-compatible format
    accident_years = heatmap_data.index
    development_years = heatmap_data.columns
    z_values = heatmap_data.values

    # Create Heatmap
    fig = go.Figure()

    # Add heatmap layer
    fig.add_trace(
        go.Heatmap(
            z=z_values,
            x=development_years,
            y=accident_years,
            colorscale="RdBu",
            colorbar=dict(title="Age-to-Age Factor"),
            zmin=np.nanmin(z_values),
            zmax=np.nanmax(z_values),
            hoverongaps=False,  # Avoid hover on NaN values
            hovertemplate=(
                "Development Year: %{x}<br>"
                "Accident Year: %{y}<br>"
                "ATAF: %{z:.2f}<extra></extra>"
            )  # Custom text in the hover tooltip
        )
    )

    # Add annotations for cell values
    for i, accident_year in enumerate(accident_years):
        for j, development_year in enumerate(development_years):
            if not np.isnan(z_values[i][j]):  # Add only if value is not NaN
                fig.add_annotation(
                    text=f"{z_values[i][j]:.2f}",  # Format to 2 decimal places
                    x=development_year,
                    y=accident_year,
                    showarrow=False,
                    font=dict(size=10, color="black"),
                    align="center",
                )

    # Update layout for better visualization
    fig.update_layout(
        title="Heatmap of Age-to-Age Factors (ATAFs)",
        xaxis_title="Development Year",
        yaxis_title="Accident Year",
        template="plotly_dark",
        height=600,
        width=900
    )

    # Display the heatmap in Streamlit
    st.plotly_chart(fig)

def calculate_ataf(df):
    df = df.set_index('accident_year')
    df['next_year_incurred'] = df.groupby('accident_year')['incurred'].shift(-1)
    df['ATAF'] = df['next_year_incurred'] / df['incurred']
    df.drop(columns=['next_year_incurred'], inplace=True)
    df['ATAF'] = df['ATAF'].replace([np.inf, -np.inf], np.nan)  # Replacing infinite values with NaN
    return df

def calculate_ataf_cumpaid(df):
    df = df.set_index('accident_year')
    df['next_year_cumpaid'] = df.groupby('accident_year')['cumpaid'].shift(-1)
    df['ATAF'] = df['next_year_cumpaid'] / df['cumpaid']
    df.drop(columns=['next_year_cumpaid'], inplace=True)
    df['ATAF'] = df['ATAF'].replace([np.inf, -np.inf], np.nan)  # Replacing infinite values with NaN
    return df
    
def prepare_pivot_claim_atafs(df):
    df = df.set_index('accident_year')
    df['next_year_cumpaid'] = df.groupby('accident_year')['cumpaid'].shift(-1)
    df['ATAF'] = df['next_year_cumpaid'] / df['cumpaid']
    df.drop(columns=['next_year_cumpaid'], inplace=True)
    df['ATAF'] = df['ATAF'].replace([np.inf, -np.inf], np.nan)  # Replacing infinite values with NaN
    return df.pivot_table(values='ATAF', index='accident_year', columns='development_year', aggfunc='mean')
    
def prepare_pivot_atafs(df):
    df = df.set_index('accident_year')
    df['next_year_incurred'] = df.groupby('accident_year')['incurred'].shift(-1)
    df['ATAF'] = df['next_year_incurred'] / df['incurred']
    df.drop(columns=['next_year_incurred'], inplace=True)
    df['ATAF'] = df['ATAF'].replace([np.inf, -np.inf], np.nan)
    return df.pivot_table(values='ATAF', index='accident_year', columns='development_year', aggfunc='mean')

def visualize_sma(pivot_atafs):
    spans = [10, 5, 3]
    for span in spans:
        pivot_atafs[f'SMA_{span}'] = pivot_atafs.apply(lambda row: row.dropna().tail(min(len(row), span)).mean(), axis=1)
    print("***********************************************")
    print(pivot_atafs)
    plt.figure(figsize=(10, 5))
    pivot_atafs['SMA_10'].plot(title='Simple Moving Average LDFs for the Latest 10 Years', marker='o', linestyle='-')
    plt.ylabel('LDF')
    plt.xlabel('Accident Year')
    plt.grid(True)
    st.pyplot(plt)

    # Comparative table
    sma_table = pd.DataFrame({
        'Latest 10 Years': pivot_atafs['SMA_10'],
        'Latest 5 Years': pivot_atafs['SMA_5'],
        'Latest 3 Years': pivot_atafs['SMA_3']
    }).T
    st.write("### Comparison between LDFs for Latest 10, 5, and 3 Years")
    st.dataframe(sma_table.style.background_gradient(cmap='YlGnBu', axis=1).format("{:.2f}"))
    return pivot_atafs

# def visualize_sma_plotly(pivot_atafs):
#     # Calculate Simple Moving Averages (SMAs)
#     spans = [10, 5, 3]
#     for span in spans:
#         pivot_atafs[f'SMA_{span}'] = pivot_atafs.apply(lambda row: row.dropna().tail(min(len(row), span)).mean(), axis=1)

#     # Plot interactive line chart
#     fig = go.Figure()

#     # Add traces for each SMA span
#     for span in spans:
#         fig.add_trace(
#             go.Scatter(
#                 x=pivot_atafs.index,
#                 y=pivot_atafs[f'SMA_{span}'],
#                 mode='lines+markers',
#                 name=f'SMA {span} Years'
#             )
#         )

#     # Update chart layout
#     fig.update_layout(
#         title="Simple Moving Average LDFs for the Latest Years",
#         xaxis_title="Accident Year",
#         yaxis_title="LDF",
#         template="plotly_white",
#         height=500,
#         width=800,
#         showlegend=True
#     )

#     # Display the chart in Streamlit
#     st.plotly_chart(fig)

#     # Comparative table
#     sma_table = pd.DataFrame({
#         'Latest 10 Years': pivot_atafs['SMA_10'],
#         'Latest 5 Years': pivot_atafs['SMA_5'],
#         'Latest 3 Years': pivot_atafs['SMA_3']
#     }).T

#     st.write("### Comparison between LDFs for Latest 10, 5, and 3 Years")
#     st.dataframe(sma_table.style.background_gradient(cmap='YlGnBu', axis=1).format("{:.2f}"))

#     return pivot_atafs

def visualize_sma_plotly(pivot_atafs):
    # Calculate Simple Moving Averages (SMAs)
    spans = [10, 5, 3]
    for span in spans:
        pivot_atafs[f'SMA_{span}'] = pivot_atafs.apply(lambda row: row.dropna().tail(min(len(row), span)).mean(), axis=1)

    # User input: Select accident year range
    min_year = pivot_atafs.index.min()
    max_year = pivot_atafs.index.max()
    selected_years = st.slider(
        "Select Accident Year Range:",
        min_value=int(min_year),
        max_value=int(max_year),
        value=(int(min_year), int(max_year))
    )

    # Filter data based on selected years
    filtered_data = pivot_atafs.loc[selected_years[0]:selected_years[1]]

    # Plot interactive line chart
    fig = go.Figure()

    # Add traces for each SMA span
    for span in spans:
        fig.add_trace(
            go.Scatter(
                x=filtered_data.index,
                y=filtered_data[f'SMA_{span}'],
                mode='lines+markers',
                name=f'SMA {span} Years'
            )
        )

    # Update chart layout
    fig.update_layout(
        title="Simple Moving Average LDFs for the Latest Years",
        xaxis_title="Accident Year",
        yaxis_title="LDF",
        template="plotly_white",
        height=500,
        width=800,
        showlegend=True
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig)

    # Comparative table
    sma_table = pd.DataFrame({
        'Latest 10 Years': filtered_data['SMA_10'],
        'Latest 5 Years': filtered_data['SMA_5'],
        'Latest 3 Years': filtered_data['SMA_3']
    }).T

    st.write("### Comparison between LDFs for Latest 10, 5, and 3 Years")
    st.dataframe(sma_table.style.background_gradient(cmap='YlGnBu', axis=1).format("{:.2f}"))

    return pivot_atafs
    
def visualize_vwa(pivot_atafs, df):
    # Create weights pivot table using incurred claims
    weights = df.pivot_table(values='incurred', index='accident_year', columns='development_year', aggfunc='sum')
    
    # Ensure only valid development years are considered
    valid_dev_years = [col for col in pivot_atafs.columns if col in weights.columns]
    pivot_atafs_valid = pivot_atafs[valid_dev_years]
    weights_valid = weights[valid_dev_years]

    # Define a function to calculate VWA for a given span
    def volume_weighted_average(series, weights, span):
        valid_data = series.dropna().tail(min(len(series), span))
        valid_weights = weights.loc[valid_data.index].tail(len(valid_data))
        if valid_data.empty or valid_weights.sum() == 0:
            return np.nan
        return (valid_data * valid_weights).sum() / valid_weights.sum()

    # Calculate VWA LDFs for various spans
    spans = [10, 5, 3]
    for span in spans:
        pivot_atafs[f'VWA_{span}'] = pivot_atafs_valid.apply(
            lambda row: volume_weighted_average(row, weights_valid.loc[row.name], span), axis=1)

    # Plot the VWA for the latest 10 years
    plt.figure(figsize=(10, 5))
    pivot_atafs['VWA_10'].plot(title='Volume Weighted Average LDFs for the Latest 10 Years', marker='o', linestyle='-')
    plt.ylabel('LDF')
    plt.xlabel('Accident Year')
    plt.grid(True)
    st.pyplot(plt)

    # Prepare and display a comparative table for VWAs
    vwa_table = pd.DataFrame({
        'Latest 10 Years': pivot_atafs['VWA_10'],
        'Latest 5 Years': pivot_atafs['VWA_5'],
        'Latest 3 Years': pivot_atafs['VWA_3']
    }).T
    st.write("### Comparison between LDFs for Latest 10, 5, and 3 Years")
    st.dataframe(vwa_table.style.background_gradient(cmap='YlGnBu', axis=1).format("{:.2f}"))
    return pivot_atafs

def visualize_vwa_plotly(pivot_atafs, df):
    # Create weights pivot table using incurred claims
    weights = df.pivot_table(values='incurred', index='accident_year', columns='development_year', aggfunc='sum')
    
    # Ensure only valid development years are considered
    valid_dev_years = [col for col in pivot_atafs.columns if col in weights.columns]
    pivot_atafs_valid = pivot_atafs[valid_dev_years]
    weights_valid = weights[valid_dev_years]

    # Define a function to calculate VWA for a given span
    def volume_weighted_average(series, weights, span):
        valid_data = series.dropna().tail(min(len(series), span))
        valid_weights = weights.loc[valid_data.index].tail(len(valid_data))
        if valid_data.empty or valid_weights.sum() == 0:
            return np.nan
        return (valid_data * valid_weights).sum() / valid_weights.sum()

    # Calculate VWA LDFs for various spans
    spans = [10, 5, 3]
    for span in spans:
        pivot_atafs[f'VWA_{span}'] = pivot_atafs_valid.apply(
            lambda row: volume_weighted_average(row, weights_valid.loc[row.name], span), axis=1)

    # User input: Select accident year range
    min_year = pivot_atafs.index.min()
    max_year = pivot_atafs.index.max()
    selected_years = st.slider(
        "Select Accident Year Range:",
        min_value=int(min_year),
        max_value=int(max_year),
        value=(int(min_year), int(max_year))
    )

    # Filter data based on selected years
    filtered_data = pivot_atafs.loc[selected_years[0]:selected_years[1]]

    # Interactive Plotly Line Chart
    fig = go.Figure()

    # Add traces for each VWA span
    for span in spans:
        fig.add_trace(
            go.Scatter(
                x=filtered_data.index,
                y=filtered_data[f'VWA_{span}'],
                mode='lines+markers',
                name=f'VWA {span} Years'
            )
        )

    # Update layout for better visualization
    fig.update_layout(
        title="Volume Weighted Average LDFs for the Selected Years",
        xaxis_title="Accident Year",
        yaxis_title="LDF",
        template="plotly_white",
        height=500,
        width=800,
        showlegend=True
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig)

    # Prepare and display a comparative table for VWAs
    vwa_table = pd.DataFrame({
        'Latest 10 Years': filtered_data['VWA_10'],
        'Latest 5 Years': filtered_data['VWA_5'],
        'Latest 3 Years': filtered_data['VWA_3']
    }).T

    st.write("### Comparison between LDFs for Latest 10, 5, and 3 Years")
    st.dataframe(vwa_table.style.background_gradient(cmap='YlGnBu', axis=1).format("{:.2f}"))

    return pivot_atafs
    
def geometric_average(series, span):
    valid_data = series.dropna().tail(min(len(series), span))
    if valid_data.empty:
        return np.nan
    # Calculate the geometric mean
    return np.prod(valid_data)**(1.0 / len(valid_data))

def visualize_ga(pivot_atafs):
    spans = [10, 5, 3]
    for span in spans:
        pivot_atafs[f'GA_{span}'] = pivot_atafs.apply(lambda row: geometric_average(row, span), axis=1)
    plt.figure(figsize=(10, 5))
    pivot_atafs['GA_10'].plot(title='Geometric Average LDFs for the Latest 10 Years', marker='o', linestyle='-')
    plt.ylabel('LDF')
    plt.xlabel('Accident Year')
    plt.grid(True)
    st.pyplot(plt)

    # Comparative table
    ga_table = pd.DataFrame({
        'Latest 10 Years': pivot_atafs['GA_10'],
        'Latest 5 Years': pivot_atafs['GA_5'],
        'Latest 3 Years': pivot_atafs['GA_3']
    }).T
    st.write("### Comparison between LDFs for Latest 10, 5, and 3 Years")
    st.dataframe(ga_table.style.background_gradient(cmap='YlGnBu', axis=1).format("{:.2f}"))
    return pivot_atafs

def visualize_ga_plotly(pivot_atafs):
    # Calculate Geometric Averages (GAs)
    spans = [10, 5, 3]
    for span in spans:
        pivot_atafs[f'GA_{span}'] = pivot_atafs.apply(lambda row: geometric_average(row, span), axis=1)

    # User input: Select accident year range
    min_year = pivot_atafs.index.min()
    max_year = pivot_atafs.index.max()
    selected_years = st.slider(
        "Select Accident Year Range:",
        min_value=int(min_year),
        max_value=int(max_year),
        value=(int(min_year), int(max_year))
    )

    # Filter data based on selected years
    filtered_data = pivot_atafs.loc[selected_years[0]:selected_years[1]]

    # Interactive Plotly Line Chart
    fig = go.Figure()

    # Add traces for each GA span
    for span in spans:
        fig.add_trace(
            go.Scatter(
                x=filtered_data.index,
                y=filtered_data[f'GA_{span}'],
                mode='lines+markers',
                name=f'GA {span} Years'
            )
        )

    # Update layout for better visualization
    fig.update_layout(
        title="Geometric Average LDFs for the Selected Years",
        xaxis_title="Accident Year",
        yaxis_title="LDF",
        template="plotly_white",
        height=500,
        width=800,
        showlegend=True
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig)

    # Prepare and display a comparative table for GAs
    ga_table = pd.DataFrame({
        'Latest 10 Years': filtered_data['GA_10'],
        'Latest 5 Years': filtered_data['GA_5'],
        'Latest 3 Years': filtered_data['GA_3']
    }).T

    st.write("### Comparison between LDFs for Latest 10, 5, and 3 Years")
    st.dataframe(ga_table.style.background_gradient(cmap='YlGnBu', axis=1).format("{:.2f}"))

    return pivot_atafs

def visualize_esa(pivot_atafs):
    alpha = 0.3
    spans = [10, 5, 3]
    for span in spans:
        pivot_atafs[f'ESA_{span}'] = pivot_atafs.apply(
            lambda row: row.dropna().tail(span).ewm(alpha=alpha).mean().iloc[-1] if len(row.dropna().tail(span)) > 0 else np.nan,
            axis=1)
    plt.figure(figsize=(10, 5))
    pivot_atafs['ESA_10'].plot(title='Exponential Smoothing Average LDFs for the Latest 10 Years', marker='o', linestyle='-')
    plt.ylabel('LDF')
    plt.xlabel('Accident Year')
    plt.grid(True)
    st.pyplot(plt)

    # Comparative table
    esa_table = pd.DataFrame({
        'Latest 10 Years': pivot_atafs['ESA_10'],
        'Latest 5 Years': pivot_atafs['ESA_5'],
        'Latest 3 Years': pivot_atafs['ESA_3']
    }).T
    st.write("### Comparison between LDFs for Latest 10, 5, and 3 Years")
    st.dataframe(esa_table.style.background_gradient(cmap='YlGnBu', axis=1).format("{:.2f}"))
    return pivot_atafs

def visualize_esa_plotly(pivot_atafs):
    alpha = 0.3
    spans = [10, 5, 3]

    # Calculate Exponential Smoothing Averages (ESAs)
    for span in spans:
        pivot_atafs[f'ESA_{span}'] = pivot_atafs.apply(
            lambda row: row.dropna().tail(span).ewm(alpha=alpha).mean().iloc[-1] 
            if len(row.dropna().tail(span)) > 0 else np.nan,
            axis=1
        )

    # User input: Select accident year range
    min_year = pivot_atafs.index.min()
    max_year = pivot_atafs.index.max()
    selected_years = st.slider(
        "Select Accident Year Range:",
        min_value=int(min_year),
        max_value=int(max_year),
        value=(int(min_year), int(max_year))
    )

    # Filter data based on selected years
    filtered_data = pivot_atafs.loc[selected_years[0]:selected_years[1]]

    # Interactive Plotly Line Chart
    fig = go.Figure()

    # Add traces for each ESA span
    for span in spans:
        fig.add_trace(
            go.Scatter(
                x=filtered_data.index,
                y=filtered_data[f'ESA_{span}'],
                mode='lines+markers',
                name=f'ESA {span} Years'
            )
        )

    # Update layout for better visualization
    fig.update_layout(
        title="Exponential Smoothing Average LDFs for the Selected Years",
        xaxis_title="Accident Year",
        yaxis_title="LDF",
        template="plotly_white",
        height=500,
        width=800,
        showlegend=True
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig)

    # Prepare and display a comparative table for ESAs
    esa_table = pd.DataFrame({
        'Latest 10 Years': filtered_data['ESA_10'],
        'Latest 5 Years': filtered_data['ESA_5'],
        'Latest 3 Years': filtered_data['ESA_3']
    }).T

    st.write("### Comparison between LDFs for Latest 10, 5, and 3 Years")
    st.dataframe(esa_table.style.background_gradient(cmap='YlGnBu', axis=1).format("{:.2f}"))

    return pivot_atafs
    
def visualize_median(pivot_atafs):
    spans = [10, 5, 3]
    for span in spans:
        pivot_atafs[f'Median_{span}'] = pivot_atafs.apply(lambda row: row.dropna().tail(span).median(), axis=1)
    plt.figure(figsize=(10, 5))
    pivot_atafs['Median_10'].plot(title='Median-Based LDFs for the Latest 10 Years', marker='o', linestyle='-')
    plt.ylabel('LDF')
    plt.xlabel('Accident Year')
    plt.grid(True)
    st.pyplot(plt)

    # Comparative table
    median_table = pd.DataFrame({
        'Latest 10 Years': pivot_atafs['Median_10'],
        'Latest 5 Years': pivot_atafs['Median_5'],
        'Latest 3 Years': pivot_atafs['Median_3']
    }).T
    st.write("### Comparison between LDFs for Latest 10, 5, and 3 Years")
    st.dataframe(median_table.style.background_gradient(cmap='YlGnBu', axis=1).format("{:.2f}"))
    return pivot_atafs

def visualize_median_plotly(pivot_atafs):
    # Calculate Median LDFs
    spans = [10, 5, 3]
    for span in spans:
        pivot_atafs[f'Median_{span}'] = pivot_atafs.apply(lambda row: row.dropna().tail(span).median(), axis=1)

    # User input: Select accident year range
    min_year = pivot_atafs.index.min()
    max_year = pivot_atafs.index.max()
    selected_years = st.slider(
        "Select Accident Year Range:",
        min_value=int(min_year),
        max_value=int(max_year),
        value=(int(min_year), int(max_year))
    )

    # Filter data based on selected years
    filtered_data = pivot_atafs.loc[selected_years[0]:selected_years[1]]

    # Interactive Plotly Line Chart
    fig = go.Figure()

    # Add traces for each Median span
    for span in spans:
        fig.add_trace(
            go.Scatter(
                x=filtered_data.index,
                y=filtered_data[f'Median_{span}'],
                mode='lines+markers',
                name=f'Median {span} Years'
            )
        )

    # Update layout for better visualization
    fig.update_layout(
        title="Median-Based LDFs for the Selected Years",
        xaxis_title="Accident Year",
        yaxis_title="LDF",
        template="plotly_white",
        height=500,
        width=800,
        showlegend=True
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig)

    # Prepare and display a comparative table for Medians
    median_table = pd.DataFrame({
        'Latest 10 Years': filtered_data['Median_10'],
        'Latest 5 Years': filtered_data['Median_5'],
        'Latest 3 Years': filtered_data['Median_3']
    }).T

    st.write("### Comparison between LDFs for Latest 10, 5, and 3 Years")
    st.dataframe(median_table.style.background_gradient(cmap='YlGnBu', axis=1).format("{:.2f}"))

    return pivot_atafs
    
def visualize_trimmed(pivot_atafs):
    def trimmed_mean_ldf(series, span):
        valid_data = series.dropna().tail(span)
        return trim_mean(valid_data, proportiontocut=0.1) if len(valid_data) > 0 else np.nan
    spans = [10, 5, 3]
    for span in spans:
        pivot_atafs[f'Trimmed_{span}'] = pivot_atafs.apply(lambda row: trimmed_mean_ldf(row, span), axis=1)
    plt.figure(figsize=(10, 5))
    pivot_atafs['Trimmed_10'].plot(title='Trimmed Mean LDFs for the Latest 10 Years', marker='o', linestyle='-')
    plt.ylabel('LDF')
    plt.xlabel('Accident Year')
    plt.grid(True)
    st.pyplot(plt)

    # Comparative table
    trimmed_table = pd.DataFrame({
        'Latest 10 Years': pivot_atafs['Trimmed_10'],
        'Latest 5 Years': pivot_atafs['Trimmed_5'],
        'Latest 3 Years': pivot_atafs['Trimmed_3']
    }).T
    st.write("### Comparison between LDFs for Latest 10, 5, and 3 Years")
    st.dataframe(trimmed_table.style.background_gradient(cmap='YlGnBu', axis=1).format("{:.2f}"))
    return pivot_atafs

def visualize_trimmed_plotly(pivot_atafs):
    def trimmed_mean_ldf(series, span):
        valid_data = series.dropna().tail(span)
        return trim_mean(valid_data, proportiontocut=0.1) if len(valid_data) > 0 else np.nan

    # Calculate Trimmed Mean LDFs
    spans = [10, 5, 3]
    for span in spans:
        pivot_atafs[f'Trimmed_{span}'] = pivot_atafs.apply(lambda row: trimmed_mean_ldf(row, span), axis=1)

    # User input: Select accident year range
    min_year = pivot_atafs.index.min()
    max_year = pivot_atafs.index.max()
    selected_years = st.slider(
        "Select Accident Year Range:",
        min_value=int(min_year),
        max_value=int(max_year),
        value=(int(min_year), int(max_year))
    )

    # Filter data based on selected years
    filtered_data = pivot_atafs.loc[selected_years[0]:selected_years[1]]

    # Interactive Plotly Line Chart
    fig = go.Figure()

    # Add traces for each Trimmed Mean span
    for span in spans:
        fig.add_trace(
            go.Scatter(
                x=filtered_data.index,
                y=filtered_data[f'Trimmed_{span}'],
                mode='lines+markers',
                name=f'Trimmed {span} Years'
            )
        )

    # Update layout for better visualization
    fig.update_layout(
        title="Trimmed Mean LDFs for the Selected Years",
        xaxis_title="Accident Year",
        yaxis_title="LDF",
        template="plotly_white",
        height=500,
        width=800,
        showlegend=True
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig)

    # Prepare and display a comparative table for Trimmed Means
    trimmed_table = pd.DataFrame({
        'Latest 10 Years': filtered_data['Trimmed_10'],
        'Latest 5 Years': filtered_data['Trimmed_5'],
        'Latest 3 Years': filtered_data['Trimmed_3']
    }).T

    st.write("### Comparison between LDFs for Latest 10, 5, and 3 Years")
    st.dataframe(trimmed_table.style.background_gradient(cmap='YlGnBu', axis=1).format("{:.2f}"))

    return pivot_atafs
    
def visualize_harmonic(pivot_atafs):
    def harmonic_mean_ldf(series, span):
        valid_data = series.dropna().tail(span)
        return len(valid_data) / np.sum(1.0 / valid_data) if len(valid_data) > 0 and all(valid_data > 0) else np.nan
    spans = [10, 5, 3]
    for span in spans:
        pivot_atafs[f'Harmonic_{span}'] = pivot_atafs.apply(lambda row: harmonic_mean_ldf(row, span), axis=1)
    plt.figure(figsize=(10, 5))
    pivot_atafs['Harmonic_10'].plot(title='Harmonic Mean LDFs for the Latest 10 Years', marker='o', linestyle='-')
    plt.ylabel('LDF')
    plt.xlabel('Accident Year')
    plt.grid(True)
    st.pyplot(plt)

    # Comparative table
    harmonic_table = pd.DataFrame({
        'Latest 10 Years': pivot_atafs['Harmonic_10'],
        'Latest 5 Years': pivot_atafs['Harmonic_5'],
        'Latest 3 Years': pivot_atafs['Harmonic_3']
    }).T
    st.write("### Comparison between LDFs for Latest 10, 5, and 3 Years")
    st.dataframe(harmonic_table.style.background_gradient(cmap='YlGnBu', axis=1).format("{:.2f}"))
    return pivot_atafs
    
# def calc_sma(pivot_atafs):
#     spans = [10, 5, 3]
#     for span in spans:
#         pivot_atafs[f'SMA_{span}'] = pivot_atafs.apply(lambda row: row.dropna().tail(min(len(row), span)).mean(), axis=1)
#     return pivot_atafs

# Calculate SMA
def calc_sma(pivot_atafs):
    spans = [10, 5, 3]
    for span in spans:
        column_name = f'SMA_{span}'
        print(f"Calculating {column_name}")
        pivot_atafs[column_name] = pivot_atafs.apply(
            lambda row: row.dropna().tail(min(len(row), span)).mean(), axis=1
        )
    return pivot_atafs
    
# def calc_vwa(pivot_atafs, df):
#     # Create weights pivot table using incurred claims
#     weights = df.pivot_table(values='incurred', index='accident_year', columns='development_year', aggfunc='sum')
    
#     # Ensure only valid development years are considered
#     valid_dev_years = [col for col in pivot_atafs.columns if col in weights.columns]
#     pivot_atafs_valid = pivot_atafs[valid_dev_years]
#     weights_valid = weights[valid_dev_years]

#     # Define a function to calculate VWA for a given span
#     def volume_weighted_average(series, weights, span):
#         valid_data = series.dropna().tail(min(len(series), span))
#         valid_weights = weights.loc[valid_data.index].tail(len(valid_data))
#         if valid_data.empty or valid_weights.sum() == 0:
#             return np.nan
#         return (valid_data * valid_weights).sum() / valid_weights.sum()

#     # Calculate VWA LDFs for various spans
#     spans = [10, 5, 3]
#     for span in spans:
#         pivot_atafs[f'VWA_{span}'] = pivot_atafs_valid.apply(
#             lambda row: volume_weighted_average(row, weights_valid.loc[row.name], span), axis=1)
#     return pivot_atafs

# Calculate VWA
def calc_vwa(pivot_atafs, df):
    weights = df.pivot_table(values='incurred', index='accident_year', columns='development_year', aggfunc='sum')
    valid_dev_years = [col for col in pivot_atafs.columns if col in weights.columns]
    pivot_atafs_valid = pivot_atafs[valid_dev_years]
    weights_valid = weights[valid_dev_years]

    def volume_weighted_average(series, weights, span):
        valid_data = series.dropna().tail(min(len(series), span))
        valid_weights = weights.loc[valid_data.index].tail(len(valid_data))
        if valid_data.empty or valid_weights.sum() == 0:
            return np.nan
        return (valid_data * valid_weights).sum() / valid_weights.sum()

    spans = [10, 5, 3]
    for span in spans:
        column_name = f'VWA_{span}'
        print(f"Calculating {column_name}")
        pivot_atafs[column_name] = pivot_atafs_valid.apply(
            lambda row: volume_weighted_average(row, weights_valid.loc[row.name], span), axis=1
        )
    return pivot_atafs

# def calc_esa(pivot_atafs):
#     alpha = 0.3
#     spans = [10, 5, 3]
#     for span in spans:
#         pivot_atafs[f'ESA_{span}'] = pivot_atafs.apply(
#             lambda row: row.dropna().tail(span).ewm(alpha=alpha).mean().iloc[-1] if len(row.dropna().tail(span)) > 0 else np.nan,
#             axis=1)
#     return pivot_atafs

# Calculate ESA
def calc_esa(pivot_atafs):
    alpha = 0.3
    spans = [10, 5, 3]
    for span in spans:
        column_name = f'ESA_{span}'
        print(f"Calculating {column_name}")
        pivot_atafs[column_name] = pivot_atafs.apply(
            lambda row: row.dropna().tail(span).ewm(alpha=alpha).mean().iloc[-1] 
            if len(row.dropna().tail(span)) > 0 else np.nan,
            axis=1
        )
    return pivot_atafs
    
# def calc_median(pivot_atafs):
#     spans = [10, 5, 3]
#     for span in spans:
#         pivot_atafs[f'Median_{span}'] = pivot_atafs.apply(lambda row: row.dropna().tail(span).median(), axis=1)
#     return pivot_atafs

# Calculate Median
def calc_median(pivot_atafs):
    spans = [10, 5, 3]
    for span in spans:
        column_name = f'Median_{span}'
        print(f"Calculating {column_name}")
        pivot_atafs[column_name] = pivot_atafs.apply(
            lambda row: row.dropna().tail(span).median(), axis=1
        )
    return pivot_atafs
    
# def calc_trimmed(pivot_atafs):
#     def trimmed_mean_ldf(series, span):
#         valid_data = series.dropna().tail(span)
#         return trim_mean(valid_data, proportiontocut=0.1) if len(valid_data) > 0 else np.nan
#     spans = [10, 5, 3]
#     for span in spans:
#         pivot_atafs[f'Trimmed_{span}'] = pivot_atafs.apply(lambda row: trimmed_mean_ldf(row, span), axis=1)
#     return pivot_atafs

# Calculate Trimmed Mean
def calc_trimmed(pivot_atafs):
    def trimmed_mean_ldf(series, span):
        valid_data = series.dropna().tail(span)
        return trim_mean(valid_data, proportiontocut=0.1) if len(valid_data) > 0 else np.nan

    spans = [10, 5, 3]
    for span in spans:
        column_name = f'Trimmed_{span}'
        print(f"Calculating {column_name}")
        pivot_atafs[column_name] = pivot_atafs.apply(
            lambda row: trimmed_mean_ldf(row, span), axis=1
        )
    return pivot_atafs
    
def calc_ga(pivot_atafs):
    spans = [10, 5, 3]
    for span in spans:
        pivot_atafs[f'GA_{span}'] = pivot_atafs.apply(lambda row: geometric_average(row, span), axis=1)
    return pivot_atafs
    
def calc_harmonic(pivot_atafs):
    def harmonic_mean_ldf(series, span):
        valid_data = series.dropna().tail(span)
        return len(valid_data) / np.sum(1.0 / valid_data) if len(valid_data) > 0 and all(valid_data > 0) else np.nan
    spans = [10, 5, 3]
    for span in spans:
        pivot_atafs[f'Harmonic_{span}'] = pivot_atafs.apply(lambda row: harmonic_mean_ldf(row, span), axis=1)
    return pivot_atafs

# def create_summary_table(pivot_atafs,df):
#     return pd.DataFrame({
#         'Simple Mean (10)': pivot_atafs['SMA_10'],
#         'Volume Weighted (10)': pivot_atafs['VWA_10'],
#         # 'Geometric Mean (10)': pivot_atafs['GA_10'],
#         'Exponential Smoothing (10)': pivot_atafs['ESA_10'],
#         'Median (10)': pivot_atafs['Median_10'],
#         'Trimmed Mean (10)': pivot_atafs['Trimmed_10']
#         # 'Harmonic Mean (10)': pivot_atafs['Harmonic_10']
#     })
def create_summary_table(pivot_atafs, selected_span):
    span_str = str(selected_span)
    return pd.DataFrame({
        f'Simple Mean ({span_str})': pivot_atafs[f'SMA_{span_str}'],
        f'Volume Weighted ({span_str})': pivot_atafs[f'VWA_{span_str}'],
        f'Exponential Smoothing ({span_str})': pivot_atafs[f'ESA_{span_str}'],
        f'Median ({span_str})': pivot_atafs[f'Median_{span_str}'],
        f'Trimmed Mean ({span_str})': pivot_atafs[f'Trimmed_{span_str}']
    })

# def visualize_heatmap(summary_table):
#     plt.figure(figsize=(12, 8))
#     sns.heatmap(summary_table, annot=True, fmt=".2f", cmap="coolwarm", linewidths=.5)
#     plt.title('Heatmap of LDF Averaging Methods (Latest 10 Years)')
#     plt.ylabel('Accident Year')
#     plt.xlabel('LDF Methods')
#     st.pyplot(plt)

def visualize_heatmap(summary_table, selected_span):
    # Add an accident year slider
    min_year = summary_table.index.min()
    max_year = summary_table.index.max()
    selected_years = st.slider(
        "Select Accident Year Range:",
        min_value=int(min_year),
        max_value=int(max_year),
        value=(int(min_year), int(max_year))
    )

    # Filter the summary table based on the selected range
    filtered_table = summary_table.loc[selected_years[0]:selected_years[1]]

    # Create the heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=filtered_table.values,
            x=filtered_table.columns,
            y=filtered_table.index,
            colorscale='RdBu',  # Supported colorscale
            colorbar=dict(title="LDF Values"),
            hoverongaps=False
        )
    )

    # Add annotations for each cell
    for i, row in enumerate(filtered_table.index):
        for j, col in enumerate(filtered_table.columns):
            value = filtered_table.iloc[i, j]
            if not pd.isna(value):  # Avoid annotating NaN values
                fig.add_annotation(
                    text=f"{value:.2f}",  # Format to 2 decimal places
                    x=col,
                    y=row,
                    showarrow=False,
                    font=dict(size=8, color="black", family="Arial Black"),
                    align="center"
                )

    # Update layout for better readability
    fig.update_layout(
        title=f"Heatmap of LDF Averaging Methods (Latest {selected_span} Years)",
        xaxis_title="LDF Methods",
        yaxis_title="Accident Year",
        template="plotly_white",
        height=600,
        width=900
    )

    # Display the heatmap in Streamlit
    st.plotly_chart(fig)
  
# def visualize_radar_chart(summary_table, year):
#     methods = summary_table.columns
#     angles = [n / float(len(methods)) * 2 * pi for n in range(len(methods))]
#     angles += angles[:1]  # Complete the circle

#     values = summary_table.loc[year].tolist()
#     values += values[:1]  # Complete the circle

#     fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
#     ax.fill(angles, values, color='red', alpha=0.25)
#     ax.plot(angles, values, color='blue', linewidth=2)
#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(methods, fontsize=10)
#     ax.set_title(f'LDF Comparison Across Methods for Accident Year {year}')
#     st.pyplot(fig)

def visualize_radar_chart(summary_table, year):
    methods = summary_table.columns
    values = summary_table.loc[year].tolist()

    # Close the radar chart by repeating the first value
    values += values[:1]
    methods = methods.tolist() + [methods[0]]

    # Create radar chart using Plotly
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=methods,
        fill='toself',  # Fill the area under the line
        name=f'Accident Year {year}',
        line=dict(color='blue', width=2),
        fillcolor='rgba(0, 0, 255, 0.2)',  # Transparent fill
        hovertemplate="<b>Method:</b> %{theta}<br><b>Value:</b> %{r:.2f}<extra></extra>"  # Custom hover text
    ))

    # Update layout for better visualization
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values) * 1.1]  # Add some padding for visibility
            )
        ),
        title=f"LDF Comparison Across Methods for Accident Year {year}",
        template="plotly_white",
        height=600,
        width=700
    )

    # Display the radar chart in Streamlit
    st.plotly_chart(fig)

# def visualize_boxplot(summary_table):
#     melted_summary = summary_table.melt(var_name='Method', value_name='LDF')

#     plt.figure(figsize=(12, 6))
#     sns.boxplot(data=melted_summary, x='Method', y='LDF')
#     plt.xticks(rotation=45)
#     plt.title('Distribution of LDFs Across Averaging Methods')
#     plt.ylabel('LDF')
#     st.pyplot(plt)

def visualize_boxplot(summary_table):
    # Melt the summary table to a long format
    melted_summary = summary_table.melt(var_name='Method', value_name='LDF')

    # Create an interactive boxplot using Plotly
    fig = px.box(
        melted_summary,
        x='Method',
        y='LDF',
        points="all",  # Show individual data points
        title="Distribution of LDFs Across Averaging Methods"
    )

    # Customize layout for better readability
    fig.update_layout(
        xaxis_title="Averaging Method",
        yaxis_title="LDF",
        template="presentation",
        height=500,
        width=900
    )

    # Rotate x-axis labels for readability
    fig.update_xaxes(tickangle=45)

    # Display the boxplot in Streamlit
    st.plotly_chart(fig)

def prepare_triangle(df):
    return df.pivot(index='accident_year', columns='development_year', values='incurred')

def prepare_claim_triangle(df):
    return df.pivot(index='accident_year', columns='development_year', values='cumpaid')

# # Chain-Ladder Method Dataframe
# def chain_ladder(triangle):
#     development_factors = []
#     for col in range(triangle.shape[1] - 1):
#         current_sum = triangle.iloc[:, col].sum()
#         next_sum = triangle.iloc[:, col + 1].sum()
#         factor = next_sum / current_sum if current_sum != 0 else 1
#         development_factors.append(factor)

#     projected_triangle = triangle.copy()
#     for col in range(1, projected_triangle.shape[1]):
#         for row in range(projected_triangle.shape[0]):
#             if pd.isna(projected_triangle.iloc[row, col]):
#                 previous_value = projected_triangle.iloc[row, col - 1]
#                 if not pd.isna(previous_value):
#                     projected_triangle.iloc[row, col] = previous_value * development_factors[col - 1]

#     return projected_triangle

def chain_ladder(triangle):
    development_factors = []
    for col in range(triangle.shape[1] - 1):
        current_sum = triangle.iloc[:, col].sum()
        next_sum = triangle.iloc[:, col + 1].sum()
        factor = next_sum / current_sum if current_sum != 0 else 1
        development_factors.append(factor)

    projected_triangle = triangle.copy()
    for col in range(1, projected_triangle.shape[1]):
        for row in range(projected_triangle.shape[0]):
            if pd.isna(projected_triangle.iloc[row, col]):
                previous_value = projected_triangle.iloc[row, col - 1]
                if not pd.isna(previous_value):
                    projected_triangle.iloc[row, col] = previous_value * development_factors[col - 1]

    # Fill in the values for all development years using the last available factor
    last_factor = development_factors[-1]
    for col in range(len(development_factors), projected_triangle.shape[1] - 1):
        for row in range(projected_triangle.shape[0]):
            previous_value = projected_triangle.iloc[row, col]
            if not pd.isna(previous_value):
                projected_triangle.iloc[row, col + 1] = previous_value * last_factor

    return projected_triangle

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=True, sheet_name='Sheet1')
        # Properly close the writer and save the file
        writer.close()  # This ensures the workbook is saved correctly
    
    processed_data = output.getvalue()
    return processed_data

# # Chain-Ladder Method
# def visualize_chain_ladder(triangle):
#     # Calculate development factors
#     factors = {}
#     for i in range(1, triangle.shape[1]):
#         valid_data = triangle.iloc[:, [i - 1, i]].dropna()
#         factors[i] = (valid_data.iloc[:, 1] / valid_data.iloc[:, 0]).mean()

#     # Fill in the triangle using the factors
#     completed_triangle = triangle.copy()
#     for i in range(1, completed_triangle.shape[1]):
#         for j in range(i, completed_triangle.shape[1]):
#             for k in range(len(completed_triangle)):
#                 if pd.isna(completed_triangle.iloc[k, j]):
#                     completed_triangle.iloc[k, j] = (
#                         completed_triangle.iloc[k, j - 1] * factors.get(j, 1)
#                     )

#     # Calculate IBNR reserve
#     ibnr_reserve = completed_triangle.sum(axis=1) - triangle.sum(axis=1, min_count=1)

#     # Define formatting function for better visualization
#     def format_number(x):
#         if x >= 1e6:
#             return f'{x/1e6:.1f}M'  # Millions
#         elif x >= 1e3:
#             return f'{x/1e3:.1f}K'  # Thousands
#         else:
#             return f'{x:.1f}'  # Less than 1000

#     # Define the Coolwarm color palette
#     cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)

#     # Normalize the color map to fit the range of data
#     norm = mcolors.Normalize(vmin=completed_triangle.min().min(), vmax=completed_triangle.max().max())

#     # Format the triangle values
#     formatted_triangle = completed_triangle.map(format_number)
    
#     # development_factors = []
#     # for col in range(triangle.shape[1] - 1):
#     #     current_sum = triangle.iloc[:, col].sum()
#     #     next_sum = triangle.iloc[:, col + 1].sum()
#     #     factor = next_sum / current_sum if current_sum != 0 else 1
#     #     development_factors.append(factor)

#     # projected_triangle = triangle.copy()
#     # for col in range(1, projected_triangle.shape[1]):
#     #     for row in range(projected_triangle.shape[0]):
#     #         if pd.isna(projected_triangle.iloc[row, col]):
#     #             previous_value = projected_triangle.iloc[row, col - 1]
#     #             if not pd.isna(previous_value):
#     #                 projected_triangle.iloc[row, col] = previous_value * development_factors[col - 1]

#     # ibnr_reserve = projected_triangle.sum(axis=1) - triangle.sum(axis=1, min_count=1)

#     # def format_number(x):
#     #     if x >= 1e6:
#     #         return f'{x/1e6:.1f}M'  # Millions
#     #     elif x >= 1e3:
#     #         return f'{x/1e3:.1f}K'  # Thousands
#     #     else:
#     #         return f'{x:.1f}'  # Keep as is if less than 1000

#     # # Define the Coolwarm color palette
#     # cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)
    
#     # # Normalize the color map to fit the range of data
#     # norm = mcolors.Normalize(vmin=projected_triangle.min().min(), vmax=projected_triangle.max().max())

#     # formatted_triangle = projected_triangle.map(format_number)

#     # Heatmap
#     plt.figure(figsize=(12, 8))
#     sns.heatmap(completed_triangle, cmap='coolwarm', annot=formatted_triangle, norm=norm, fmt="", linewidths=0.5, linecolor='black',annot_kws={'size': 6})
#     plt.title('Projected Incurred Claim Amount Triangle')
#     plt.ylabel('Accident Year')
#     plt.xlabel('Development Year')
#     st.pyplot(plt)

#     # Bar chart for IBNR reserves
#     plt.figure(figsize=(10, 6))
#     sns.barplot(x=ibnr_reserve.index, y=ibnr_reserve.values, palette="Blues_d",hue=ibnr_reserve.values)
#     # plt.grid(True, linestyle='--', linewidth=0.5)
#     plt.title("IBNR Reserves by Accident Year")
#     plt.xlabel("Accident Year")
#     plt.ylabel("IBNR Reserve Amount")
#     st.pyplot(plt)

#     # Line chart for ultimate claim amount
#     ultimate_claim_amount = completed_triangle.sum(axis=1)
#     plt.figure(figsize=(10, 6))
#     plt.plot(ultimate_claim_amount.index, ultimate_claim_amount.values, marker='o', linestyle='-', color='teal')
#     plt.title("Ultimate Claims Amount by Accident Year (Chain Ladder Method)")
#     plt.xlabel("Accident Year")
#     plt.ylabel("Ultimate Claim Amount")
#     st.pyplot(plt)

# Chain-Ladder Method
def visualize_chain_ladder(triangle):
    # Calculate development factors
    factors = {}
    for i in range(1, triangle.shape[1]):
        valid_data = triangle.iloc[:, [i - 1, i]].dropna()
        factors[i] = (valid_data.iloc[:, 1] / valid_data.iloc[:, 0]).mean()

    # Fill in the triangle using the factors
    completed_triangle = triangle.copy()
    for i in range(1, completed_triangle.shape[1]):
        for j in range(i, completed_triangle.shape[1]):
            for k in range(len(completed_triangle)):
                if pd.isna(completed_triangle.iloc[k, j]):
                    completed_triangle.iloc[k, j] = (
                        completed_triangle.iloc[k, j - 1] * factors.get(j, 1)
                    )
    
    # Accident Year Slider
    min_year = completed_triangle.index.min()
    max_year = completed_triangle.index.max()
    selected_years = st.slider(
        "Select Accident Year Range:",
        min_value=int(min_year),
        max_value=int(max_year),
        value=(int(min_year), int(max_year))
    )

    # Filter the triangle based on the selected accident years
    filtered_triangle = completed_triangle.loc[selected_years[0]:selected_years[1]]

     # Calculate IBNR reserve
    ibnr_reserve = filtered_triangle.sum(axis=1) - triangle.sum(axis=1, min_count=1)

    def format_number(value):
        if value >= 1e6:
            return f"{value / 1e6:.1f}M"  # Format as millions
        elif value >= 1e3:
            return f"{value / 1e3:.1f}K"  # Format as thousands
        else:
            return f"{value:.1f}"  # Leave as is for smaller numbers
        
    # Heatmap visualization using Plotly
    fig_heatmap = go.Figure(
        data=go.Heatmap(
            z=filtered_triangle.values,
            x=filtered_triangle.columns,
            y=filtered_triangle.index,
            colorscale='Viridis',
            colorbar=dict(title="Claim Amount"),
            hoverongaps=False,
        )
    )

    # Add the heatmap layer
    fig_heatmap.add_trace(
        go.Heatmap(
            z=filtered_triangle.values,
            x=filtered_triangle.columns,
            y=filtered_triangle.index,
            colorscale='Viridis',
            colorbar=dict(title="Claim Amount"),
            hoverongaps=False,
            hovertemplate=(
            "<b>Accident Year:</b> %{y}<br>"
            "<b>Development Year:</b> %{x}<br>"
            "<b>Claim Amount:</b> %{z:.0f}<extra></extra>"
            )
        )
    )
    
    # Add annotations for each cell
    for i, y in enumerate(filtered_triangle.index):
        for j, x in enumerate(filtered_triangle.columns):
            value = filtered_triangle.iloc[i, j]
            if not pd.isna(value):  # Skip NaN values
                formatted_value = format_number(value)
                fig_heatmap.add_annotation(
                    text=formatted_value,  # Format values as integers
                    x=x,
                    y=y,
                    showarrow=False,
                    font=dict(color="black", size=7.5),
                    align="center"
                )
                
    fig_heatmap.update_layout(
        title="Projected Claim Amount Triangle (Chain Ladder)",
        xaxis_title="Development Year",
        yaxis_title="Accident Year",
        template="plotly_white",
        height=600,
        width=900,
    )
    st.plotly_chart(fig_heatmap)


    # Format Ultimate Claim Amount for hover text
    ultimate_claim_amount = filtered_triangle.sum(axis=1)
    formatted_claim_amount = [format_number(val) for val in ultimate_claim_amount.values]
    
    fig_line = px.line(
        x=ultimate_claim_amount.index,
        y=ultimate_claim_amount.values,
        labels={"x": "Accident Year", "y": "Ultimate Claim Amount"},
        title="Ultimate Claims Amount by Accident Year (Chain Ladder Method)",
    )
    
    # Update traces for markers and hover text
    fig_line.update_traces(
        mode="lines+markers",
        line=dict(color="teal", width=2),
        marker=dict(size=8, color="white", line=dict(width=2, color="black")),  # Custom markers
        hovertemplate="<b>Accident Year:</b> %{x}<br><b>Ultimate Claim:</b> %{y:.2f}<extra></extra>",
    )
    
    # Enhance layout
    fig_line.update_layout(
        template="plotly_white",
        height=400,
        xaxis_title="Accident Year",
        yaxis_title="Ultimate Claim Amount",
        title_font_size=16,
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True, rangemode="tozero"),  # Adjust range dynamically
    )
    st.plotly_chart(fig_line)

    formatted_ibnr_reserve = [format_number(val) for val in ibnr_reserve.values]
    
    fig_bar = px.bar(
        x=ibnr_reserve.index,
        y=ibnr_reserve.values,
        labels={"x": "Accident Year", "y": "IBNR Reserve Amount"},
        title="IBNR Reserves by Accident Year",
        text=formatted_ibnr_reserve,  # Add formatted text (e.g., 1.2M, 2.3K)
    )
    
    # Update trace for colors and text
    fig_bar.update_traces(
        marker=dict(color=ibnr_reserve.values, colorscale="Blues", showscale=True),  # Dynamic coloring
        textposition="outside",
        texttemplate="%{text}",
    )
    
    # Enhance layout
    fig_bar.update_layout(
        template="plotly_white",
        height=500,
        xaxis_title="Accident Year",
        yaxis_title="IBNR Reserve Amount",
        title_font_size=16,
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
    )
    st.plotly_chart(fig_bar)

    # Create a summary DataFrame
    result_df = pd.DataFrame({
        "Ultimate Claims": ultimate_claim_amount,
        "IBNR Reserves (Chain Ladder Method)": ibnr_reserve
    })
    # result_df = result_df.set_index('accident_year')
    result_df.index.name = "Accident Year"
    st.divider()
    # Display the result DataFrame
    st.write("### Here's a preview of Ultimate Claims Amount & Loss Ratio Reserve Amounts using Chain Ladder Method")
    excel_file = to_excel(result_df)
    st.dataframe(result_df.tail())
    st.download_button(label="Download full summmary",
                       data=excel_file,
                       file_name="chain_ladder_method_summary.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")   

# # Bornhuetter-Ferguson (BF) Method
# def visualize_bf_method(triangle):
#     development_factors = []
#     for col in range(triangle.shape[1] - 1):
#         current_sum = triangle.iloc[:, col].sum()
#         next_sum = triangle.iloc[:, col + 1].sum()
#         factor = next_sum / current_sum if current_sum != 0 else 1
#         development_factors.append(factor)

#     projected_triangle = triangle.copy()
#     for col in range(1, projected_triangle.shape[1]):
#         for row in range(projected_triangle.shape[0]):
#             if pd.isna(projected_triangle.iloc[row, col]):
#                 previous_value = projected_triangle.iloc[row, col - 1]
#                 if not pd.isna(previous_value):
#                     projected_triangle.iloc[row, col] = previous_value * development_factors[col - 1]

#     bf_ultimate_claims = projected_triangle.sum(axis=1)
#     bf_reserves = (1 - (triangle.sum(axis=1) / bf_ultimate_claims)) * bf_ultimate_claims

#     # Bar chart for BF reserves
#     plt.figure(figsize=(10, 6))
#     sns.barplot(x=bf_reserves.index, y=bf_reserves.values, palette="Blues_d", hue=bf_reserves.values)
#     plt.title("BF Reserves by Accident Year")
#     plt.xlabel("Accident Year")
#     plt.ylabel("BF Reserve Amount")
#     st.pyplot(plt)

#     # Bar chart for ultimate claims
#     plt.figure(figsize=(10, 6))
#     sns.barplot(x=bf_ultimate_claims.index, y=bf_ultimate_claims.values, palette="Greens_d",hue=bf_ultimate_claims.values)
#     plt.title("Ultimate Claims Amount by Accident Year (BF Method)")
#     plt.xlabel("Accident Year")
#     plt.ylabel("Ultimate Claims Amount")
#     st.pyplot(plt)

    # incurred_claims = triangle.sum(axis=1)  # Sum of incurred claims for each accident year

    # # Prepare the data for stacked bar plot
    # data_for_comparison = pd.DataFrame({
    #     'Incurred Claims': incurred_claims,
    #     'BF Reserves': bf_reserves
    # })
    # # Create the plot
    # fig, ax = plt.subplots(figsize=(15, 6))
    # data_for_comparison.plot(kind='bar', stacked=True, ax=ax, color=['lightblue', 'salmon'])
    # ax.set_title("Incurred Claims vs BF Reserves by Accident Year", fontsize=16)
    # ax.set_xlabel("Accident Year", fontsize=12)
    # ax.set_ylabel("Amount", fontsize=12)
    # plt.tight_layout()
    
    # # Display the plot in Streamlit
    # st.pyplot(fig)

# Bornhuetter-Ferguson (BF) Method Visualization
def visualize_bf_method(triangle):
    # Calculate development factors
    development_factors = []
    for col in range(triangle.shape[1] - 1):
        current_sum = triangle.iloc[:, col].sum()
        next_sum = triangle.iloc[:, col + 1].sum()
        factor = next_sum / current_sum if current_sum != 0 else 1
        development_factors.append(factor)

    # Fill projected triangle using development factors
    projected_triangle = triangle.copy()
    for col in range(1, projected_triangle.shape[1]):
        for row in range(projected_triangle.shape[0]):
            if pd.isna(projected_triangle.iloc[row, col]):
                previous_value = projected_triangle.iloc[row, col - 1]
                if not pd.isna(previous_value):
                    projected_triangle.iloc[row, col] = previous_value * development_factors[col - 1]

    # Calculate BF ultimate claims and reserves
    bf_ultimate_claims = projected_triangle.sum(axis=1)
    bf_reserves = (1 - (triangle.sum(axis=1) / bf_ultimate_claims)) * bf_ultimate_claims

    # Accident Year Slider
    min_year = triangle.index.min()
    max_year = triangle.index.max()
    selected_years = st.slider(
        "Select Accident Year Range:",
        min_value=int(min_year),
        max_value=int(max_year),
        value=(int(min_year), int(max_year))
    )

    # Filter data based on the selected accident year range
    filtered_reserves = bf_reserves.loc[selected_years[0]:selected_years[1]]
    filtered_claims = bf_ultimate_claims.loc[selected_years[0]:selected_years[1]]

    # Format filtered reserves for better readability
    formatted_bf_reserves = [
        f"{val / 1e6:.1f}M" if val >= 1e6 else f"{val / 1e3:.1f}K" for val in filtered_reserves
    ]
    
    # Format filtered claims for better readability
    formatted_bf_claims = [
        f"{val / 1e6:.1f}M" if val >= 1e6 else f"{val / 1e3:.1f}K" for val in filtered_claims
    ]
    
    # Bar chart for ultimate claims with dynamic coloring
    fig_claims = px.bar(
        x=filtered_claims.index,
        y=filtered_claims.values,
        labels={"x": "Accident Year", "y": "Ultimate Claims Amount"},
        title="Ultimate Claims Amount by Accident Year (BF Method)",
        text=formatted_bf_claims,
    )
    fig_claims.update_traces(
        marker=dict(color=filtered_claims.values, colorscale="Greens", showscale=True),  # Dynamic coloring
        texttemplate="%{text}",
        textposition="outside",
    )
    fig_claims.update_layout(
        template="plotly_white",
        height=500,
        xaxis_title="Accident Year",
        yaxis_title="Ultimate Claims Amount",
    )
    st.plotly_chart(fig_claims)
    
    # Bar chart for BF reserves with dynamic coloring
    fig_reserves = px.bar(
        x=filtered_reserves.index,
        y=filtered_reserves.values,
        labels={"x": "Accident Year", "y": "BF Reserve Amount"},
        title="BF Reserves by Accident Year",
        text=formatted_bf_reserves,
    )
    fig_reserves.update_traces(
        marker=dict(color=filtered_reserves.values, colorscale="Blues", showscale=True),  # Dynamic coloring
        texttemplate="%{text}",
        textposition="outside",
    )
    fig_reserves.update_layout(
        template="plotly_white",
        height=500,
        xaxis_title="Accident Year",
        yaxis_title="BF Reserve Amount",
    )
    st.plotly_chart(fig_reserves)

    # Create a summary DataFrame
    result_df = pd.DataFrame({
        "Ultimate Claims": bf_ultimate_claims,
        "IBNR Reserves (BF Method)": bf_reserves
    })
    # result_df = result_df.set_index('accident_year')
    result_df.index.name = "Accident Year"
    st.divider()
    # Display the result DataFrame
    st.write("### Here's a preview of Ultimate Claims Amount & Loss Ratio Reserve Amounts using Bornhuetter-Ferguson (BF) Method")
    excel_file = to_excel(result_df)
    st.dataframe(result_df.tail())
    st.download_button(label="Download full summmary",
                       data=excel_file,
                       file_name="bf_method_summary.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")    

# # Loss Ratio Method Incurred
# def visualize_loss_ratio(df):
#     triangle = prepare_triangle(df)
#     earned_premiums = df.pivot(index='accident_year', columns='development_year', values='earned_premium')

#     incurred_claims = triangle.sum(axis=1)
#     total_earned_premiums = earned_premiums.sum(axis=1)
#     loss_ratios = incurred_claims / total_earned_premiums
#     expected_claims = total_earned_premiums * loss_ratios
#     loss_ratio_reserves = expected_claims - incurred_claims

#     # Bar chart for loss ratios 
#     plt.figure(figsize=(12, 7))
#     sns.barplot(x=loss_ratios.index, y=loss_ratios.values, palette="Reds_d",hue=loss_ratios.values)
#     plt.title("Loss Ratio by Accident Year")
#     plt.xlabel("Accident Year")
#     plt.ylabel("Loss Ratio")
#     st.pyplot(plt)

#     # Bar chart for ultimate claims
#     plt.figure(figsize=(10, 6))
#     sns.barplot(x=expected_claims.index, y=expected_claims.values, palette="Purples_d",hue=expected_claims.values)
#     plt.title("Ultimate Claims Amount by Accident Year (Loss Ratio Method)")
#     plt.xlabel("Accident Year")
#     plt.ylabel("Expected Claims Amount")
#     st.pyplot(plt)

#     # # Bar chart for loss ratio reserves
#     # plt.figure(figsize=(10, 6))
#     # sns.barplot(x=loss_ratio_reserves.index, y=loss_ratio_reserves.values, palette="Purples_d",hue=loss_ratio_reserves.values)
#     # plt.title("Reserve Amounts by Accident Year (Loss Ratio Method)")
#     # plt.xlabel("Accident Year")
#     # plt.ylabel("Loss Ratio Method Reserve Amount")
#     # st.pyplot(plt)

#     # Create a new DataFrame to display
#     result_df = pd.DataFrame({
#         # 'Accident Year': df.accident_year,
#         'Ultimate Claims': expected_claims
#     })
#     result_df.index.name = 'Accident Year'
#     # # Create a new column in the original DataFrame that maps the total earned premium to each accident year
#     # df['total_earned_premium'] = df['accident_year'].map(total_earned_premium)
#     st.write("Ultimate Claims Amount for each accident Year")
#     st.write(result_df)

def visualize_loss_ratio(df):
    # Prepare triangle and data
    triangle = prepare_triangle(df)
    earned_premiums = df.pivot(index='accident_year', columns='development_year', values='earned_premium')
    reported_losses = df.pivot(index='accident_year', columns='development_year', values='reported_loss')

    incurred_claims = triangle.sum(axis=1)
    total_earned_premiums = earned_premiums.sum(axis=1)
    total_reported_loss = reported_losses.sum(axis=1)
    loss_ratios = incurred_claims / total_earned_premiums
    expected_claims = total_earned_premiums * loss_ratios
    loss_ratio_reserves = expected_claims - total_reported_loss

    # Filter data using Accident Year Slider
    min_year = loss_ratios.index.min()
    max_year = loss_ratios.index.max()
    selected_years = st.slider(
        "Select Accident Year Range:",
        min_value=int(min_year),
        max_value=int(max_year),
        value=(int(min_year), int(max_year))
    )

    # Filter based on selected years
    filtered_loss_ratios = loss_ratios.loc[selected_years[0]:selected_years[1]]
    filtered_expected_claims = expected_claims.loc[selected_years[0]:selected_years[1]]
    filtered_reserves = loss_ratio_reserves.loc[selected_years[0]:selected_years[1]]

    # Bar chart for expected claims
    formatted_claims = [
        f"{val / 1e6:.1f}M" if val >= 1e6 else f"{val / 1e3:.1f}K" for val in filtered_expected_claims.values
    ]
    fig_expected_claims = px.bar(
        x=filtered_expected_claims.index,
        y=filtered_expected_claims.values,
        labels={"x": "Accident Year", "y": "Expected Claims Amount"},
        title="Ultimate Claims Amount by Accident Year (Loss Ratio Method)",
        text=formatted_claims,
    )
    fig_expected_claims.update_traces(
        marker=dict(color=filtered_expected_claims.values, colorscale="Purples", showscale=True),  # Dynamic coloring
        texttemplate="%{text}",
        textposition="outside",
    )
    fig_expected_claims.update_layout(
        template="plotly_white",
        height=500,
        xaxis_title="Accident Year",
        yaxis_title="Expected Claims Amount",
    )
    st.plotly_chart(fig_expected_claims)

    # Bar chart for reserves
    formatted_reserves = [
        f"{val / 1e6:.1f}M" if val >= 1e6 else f"{val / 1e3:.1f}K" for val in filtered_reserves.values
    ]
    fig_reserves = px.bar(
        x=filtered_reserves.index,
        y=filtered_reserves.values,
        labels={"x": "Accident Year", "y": "Reserve Amount"},
        title="Reserve Amounts by Accident Year (Loss Ratio Method)",
        text=formatted_reserves,
    )
    fig_reserves.update_traces(
        marker=dict(color=filtered_reserves.values, colorscale="Blues", showscale=True),  # Dynamic coloring
        texttemplate="%{text}",
        textposition="outside",
    )
    fig_reserves.update_layout(
        template="plotly_white",
        height=500,
        xaxis_title="Accident Year",
        yaxis_title="Reserve Amount",
    )
    st.plotly_chart(fig_reserves)

        # Bar chart for loss ratios
    fig_loss_ratios = px.bar(
        x=filtered_loss_ratios.index,
        y=filtered_loss_ratios.values,
        labels={"x": "Accident Year", "y": "Loss Ratio"},
        title="Loss Ratio by Accident Year",
        text=[f"{val:.2%}" for val in filtered_loss_ratios.values],  # Format as percentage
    )
    fig_loss_ratios.update_traces(
        marker=dict(color=filtered_loss_ratios.values, colorscale="Reds", showscale=True),  # Dynamic coloring
        texttemplate="%{text}",
        textposition="outside"
    )
    fig_loss_ratios.update_layout(
        template="plotly_white",
        height=500,
        xaxis_title="Accident Year",
        yaxis_title="Loss Ratio",
    )
    st.plotly_chart(fig_loss_ratios)

    # Create a summary DataFrame
    result_df = pd.DataFrame({
        "Ultimate Claims": filtered_expected_claims,
        "IBNR Reserves (Loss Ratio Method)": filtered_reserves
    })
    result_df.index.name = "Accident Year"
    st.divider()
    # Display the result DataFrame
    st.write("### Here's a preview of Ultimate Claims Amount & Loss Ratio Reserve Amounts using Loss Ratio Method")
    excel_file = to_excel(result_df)
    st.dataframe(result_df.tail())
    st.download_button(label="Download full summmary",
                       data=excel_file,
                       file_name="loss_ratio_method_summary.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")    

def visualize_claim_loss_ratio(df):
    triangle = prepare_claim_triangle(df)
    earned_premiums = df.pivot(index='accident_year', columns='development_year', values='earned_premium')
    reported_losses = df.pivot(index='accident_year', columns='development_year', values='reported_loss')

    incurred_claims = triangle.sum(axis=1)
    total_earned_premiums = earned_premiums.sum(axis=1)
    total_reported_loss = reported_losses.sum(axis=1)
    loss_ratios = incurred_claims / total_earned_premiums
    expected_claims = total_earned_premiums * loss_ratios
    loss_ratio_reserves = expected_claims - total_reported_loss

    # Filter data using Accident Year Slider
    min_year = loss_ratios.index.min()
    max_year = loss_ratios.index.max()
    selected_years = st.slider(
        "Select Accident Year Range:",
        min_value=int(min_year),
        max_value=int(max_year),
        value=(int(min_year), int(max_year))
    )

    # Filter based on selected years
    filtered_loss_ratios = loss_ratios.loc[selected_years[0]:selected_years[1]]
    filtered_expected_claims = expected_claims.loc[selected_years[0]:selected_years[1]]
    filtered_reserves = loss_ratio_reserves.loc[selected_years[0]:selected_years[1]]

    # Bar chart for expected claims
    formatted_claims = [
        f"{val / 1e6:.1f}M" if val >= 1e6 else f"{val / 1e3:.1f}K" for val in filtered_expected_claims.values
    ]
    fig_expected_claims = px.bar(
        x=filtered_expected_claims.index,
        y=filtered_expected_claims.values,
        labels={"x": "Accident Year", "y": "Expected Claims Amount"},
        title="Ultimate Claims Amount by Accident Year (Loss Ratio Method)",
        text=formatted_claims,
    )
    fig_expected_claims.update_traces(
        marker=dict(color=filtered_expected_claims.values, colorscale="Purples", showscale=True),  # Dynamic coloring
        texttemplate="%{text}",
        textposition="outside",
    )
    fig_expected_claims.update_layout(
        template="plotly_white",
        height=500,
        xaxis_title="Accident Year",
        yaxis_title="Expected Claims Amount",
    )
    st.plotly_chart(fig_expected_claims)

    # Bar chart for reserves
    formatted_reserves = [
        f"{val / 1e6:.1f}M" if val >= 1e6 else f"{val / 1e3:.1f}K" for val in filtered_reserves.values
    ]
    fig_reserves = px.bar(
        x=filtered_reserves.index,
        y=filtered_reserves.values,
        labels={"x": "Accident Year", "y": "Reserve Amount"},
        title="Reserve Amounts by Accident Year (Loss Ratio Method)",
        text=formatted_reserves,
    )
    fig_reserves.update_traces(
        marker=dict(color=filtered_reserves.values, colorscale="Blues", showscale=True),  # Dynamic coloring
        texttemplate="%{text}",
        textposition="outside",
    )
    fig_reserves.update_layout(
        template="plotly_white",
        height=500,
        xaxis_title="Accident Year",
        yaxis_title="Reserve Amount",
    )
    st.plotly_chart(fig_reserves)

        # Bar chart for loss ratios
    fig_loss_ratios = px.bar(
        x=filtered_loss_ratios.index,
        y=filtered_loss_ratios.values,
        labels={"x": "Accident Year", "y": "Loss Ratio"},
        title="Loss Ratio by Accident Year",
        text=[f"{val:.2%}" for val in filtered_loss_ratios.values],  # Format as percentage
    )
    fig_loss_ratios.update_traces(
        marker=dict(color=filtered_loss_ratios.values, colorscale="Reds", showscale=True),  # Dynamic coloring
        texttemplate="%{text}",
        textposition="outside"
    )
    fig_loss_ratios.update_layout(
        template="plotly_white",
        height=500,
        xaxis_title="Accident Year",
        yaxis_title="Loss Ratio",
    )
    st.plotly_chart(fig_loss_ratios)

    # Create a summary DataFrame
    result_df = pd.DataFrame({
        "Ultimate Claims": filtered_expected_claims,
        "IBNR Reserves (Loss Ratio Method)": filtered_reserves
    })
    result_df.index.name = "Accident Year"
    st.divider()
    # Display the result DataFrame
    st.write("### Here's a preview of Ultimate Claims Amount & Loss Ratio Reserve Amounts using Loss Ratio Method")
    excel_file = to_excel(result_df)
    st.dataframe(result_df.tail())
    st.download_button(label="Download full summmary",
                       data=excel_file,
                       file_name="loss_ratio_method_summary.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")    

def loss_ratio(df):
    triangle = prepare_claim_triangle(df)
    earned_premiums = df.pivot(index='accident_year', columns='development_year', values='earned_premium')
    reported_losses = df.pivot(index='accident_year', columns='development_year', values='reported_loss')
    
    incurred_claims = triangle.sum(axis=1)
    total_earned_premiums = earned_premiums.sum(axis=1)
    total_reported_loss = reported_losses.sum(axis=1)
    loss_ratios = incurred_claims / total_earned_premiums
    expected_claims = total_earned_premiums * loss_ratios
    loss_ratio_reserves = expected_claims - total_reported_loss
    return expected_claims

def loss_ratio_incurred(df):
    triangle = prepare_triangle(df)
    earned_premiums = df.pivot(index='accident_year', columns='development_year', values='earned_premium')
    reported_losses = df.pivot(index='accident_year', columns='development_year', values='reported_loss')
    
    incurred_claims = triangle.sum(axis=1)
    total_earned_premiums = earned_premiums.sum(axis=1)
    total_reported_loss = reported_losses.sum(axis=1)
    loss_ratios = incurred_claims / total_earned_premiums
    expected_claims = total_earned_premiums * loss_ratios
    loss_ratio_reserves = expected_claims - total_reported_loss
    return expected_claims

def loss_ratio_ibnr(df):
    triangle = prepare_claim_triangle(df)
    earned_premiums = df.pivot(index='accident_year', columns='development_year', values='earned_premium')
    reported_losses = df.pivot(index='accident_year', columns='development_year', values='reported_loss')
    
    incurred_claims = triangle.sum(axis=1)
    total_earned_premiums = earned_premiums.sum(axis=1)
    total_reported_loss = reported_losses.sum(axis=1)
    loss_ratios = incurred_claims / total_earned_premiums
    expected_claims = total_earned_premiums * loss_ratios
    loss_ratio_reserves = expected_claims - total_reported_loss
    return loss_ratio_reserves

def loss_ratio_ibnr_incurred(df):
    triangle = prepare_triangle(df)
    earned_premiums = df.pivot(index='accident_year', columns='development_year', values='earned_premium')
    reported_losses = df.pivot(index='accident_year', columns='development_year', values='reported_loss')
    
    incurred_claims = triangle.sum(axis=1)
    total_earned_premiums = earned_premiums.sum(axis=1)
    total_reported_loss = reported_losses.sum(axis=1)
    loss_ratios = incurred_claims / total_earned_premiums
    expected_claims = total_earned_premiums * loss_ratios
    loss_ratio_reserves = expected_claims - total_reported_loss
    return loss_ratio_reserves

def bf_method(triangle):
    development_factors = []
    for col in range(triangle.shape[1] - 1):
        current_sum = triangle.iloc[:, col].sum()
        next_sum = triangle.iloc[:, col + 1].sum()
        factor = next_sum / current_sum if current_sum != 0 else 1
        development_factors.append(factor)

    projected_triangle = triangle.copy()
    for col in range(1, projected_triangle.shape[1]):
        for row in range(projected_triangle.shape[0]):
            if pd.isna(projected_triangle.iloc[row, col]):
                previous_value = projected_triangle.iloc[row, col - 1]
                if not pd.isna(previous_value):
                    projected_triangle.iloc[row, col] = previous_value * development_factors[col - 1]

    bf_ultimate_claims = projected_triangle.sum(axis=1)
    bf_reserves = (1 - (triangle.sum(axis=1) / bf_ultimate_claims)) * bf_ultimate_claims
    return bf_ultimate_claims

def bf_method_ibnr(triangle):
    development_factors = []
    for col in range(triangle.shape[1] - 1):
        current_sum = triangle.iloc[:, col].sum()
        next_sum = triangle.iloc[:, col + 1].sum()
        factor = next_sum / current_sum if current_sum != 0 else 1
        development_factors.append(factor)

    projected_triangle = triangle.copy()
    for col in range(1, projected_triangle.shape[1]):
        for row in range(projected_triangle.shape[0]):
            if pd.isna(projected_triangle.iloc[row, col]):
                previous_value = projected_triangle.iloc[row, col - 1]
                if not pd.isna(previous_value):
                    projected_triangle.iloc[row, col] = previous_value * development_factors[col - 1]

    bf_ultimate_claims = projected_triangle.sum(axis=1)
    bf_reserves = (1 - (triangle.sum(axis=1) / bf_ultimate_claims)) * bf_ultimate_claims
    return bf_reserves

def chain_ladder(triangle):
    development_factors = []
    for col in range(triangle.shape[1] - 1):
        current_sum = triangle.iloc[:, col].sum()
        next_sum = triangle.iloc[:, col + 1].sum()
        factor = next_sum / current_sum if current_sum != 0 else 1
        development_factors.append(factor)

    projected_triangle = triangle.copy()
    for col in range(1, projected_triangle.shape[1]):
        for row in range(projected_triangle.shape[0]):
            if pd.isna(projected_triangle.iloc[row, col]):
                previous_value = projected_triangle.iloc[row, col - 1]
                if not pd.isna(previous_value):
                    projected_triangle.iloc[row, col] = previous_value * development_factors[col - 1]

    ibnr_reserve = projected_triangle.sum(axis=1) - triangle.sum(axis=1, min_count=1)
    ultimate_claim_amount = projected_triangle.sum(axis=1)
    return ultimate_claim_amount

def chain_ladder_ibnr(triangle):
    development_factors = []
    for col in range(triangle.shape[1] - 1):
        current_sum = triangle.iloc[:, col].sum()
        next_sum = triangle.iloc[:, col + 1].sum()
        factor = next_sum / current_sum if current_sum != 0 else 1
        development_factors.append(factor)

    projected_triangle = triangle.copy()
    for col in range(1, projected_triangle.shape[1]):
        for row in range(projected_triangle.shape[0]):
            if pd.isna(projected_triangle.iloc[row, col]):
                previous_value = projected_triangle.iloc[row, col - 1]
                if not pd.isna(previous_value):
                    projected_triangle.iloc[row, col] = previous_value * development_factors[col - 1]

    ibnr_reserve = projected_triangle.sum(axis=1) - triangle.sum(axis=1, min_count=1)
    ultimate_claim_amount = projected_triangle.sum(axis=1)
    return ibnr_reserve

def prepare_summary_dataframes(triangle, df):
    # Calculate required components
    development_factors = [triangle.iloc[:, col + 1].sum() / triangle.iloc[:, col].sum() 
                           for col in range(triangle.shape[1] - 1)]
    projected_triangle = triangle.copy()
    for col in range(1, projected_triangle.shape[1]):
        for row in range(projected_triangle.shape[0]):
            if pd.isna(projected_triangle.iloc[row, col]):
                projected_triangle.iloc[row, col] = projected_triangle.iloc[row, col - 1] * development_factors[col - 1]

    ibnr_reserve = projected_triangle.sum(axis=1) - triangle.sum(axis=1, min_count=1)
    bf_reserves = (1 - (triangle.sum(axis=1) / projected_triangle.sum(axis=1))) * projected_triangle.sum(axis=1)
    earned_premiums = df.pivot(index='accident_year', columns='development_year', values='earned_premium')
    total_earned_premiums = earned_premiums.sum(axis=1)
    loss_ratios = triangle.sum(axis=1) / total_earned_premiums
    expected_claims = total_earned_premiums * loss_ratios
    loss_ratio_reserves = expected_claims - triangle.sum(axis=1)

    # Prepare summary DataFrames
    summary_IBNR_df = pd.DataFrame({
        "Accident Year": projected_triangle.index,
        "IBNR Reserves (Chain Ladder)": ibnr_reserve.values,
        "IBNR Reserves (BF Method)": bf_reserves.values,
        "IBNR Reserves (Loss Ratio)": loss_ratio_reserves.values
    })

    summary_ultimate_claims_df = pd.DataFrame({
        "Accident Year": projected_triangle.index,
        "Ultimate Claims (Chain Ladder)": projected_triangle.sum(axis=1).values,
        "Ultimate Claims (BF Method)": projected_triangle.sum(axis=1).values,  # Replace if BF differs
        "Ultimate Claims (Loss Ratio)": expected_claims.values
    })

    return summary_IBNR_df, summary_ultimate_claims_df

# def visualize_ibnr_reserves_summary(summary_IBNR_df):
#     plt.figure(figsize=(14, 7))
#     bar_width = 0.25
#     x = np.arange(len(summary_IBNR_df["Accident Year"]))
#     plt.bar(x - bar_width, summary_IBNR_df["IBNR Reserves (Chain Ladder)"], bar_width, label="Chain Ladder", color="#1f77b4")
#     plt.bar(x, summary_IBNR_df["IBNR Reserves (BF Method)"], bar_width, label="BF Method", color="#ff7f0e")
#     plt.bar(x + bar_width, summary_IBNR_df["IBNR Reserves (Loss Ratio)"], bar_width, label="Loss Ratio", color="#2ca02c")
#     plt.title("Comparison of IBNR Reserves by Method", fontsize=16)
#     plt.xlabel("Accident Year", fontsize=12)
#     plt.ylabel("IBNR Reserves", fontsize=12)
#     plt.xticks(x, summary_IBNR_df["Accident Year"], rotation=45, fontsize=10)
#     plt.legend(title="Method", fontsize=10, title_fontsize=12)
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.tight_layout()
#     st.pyplot(plt)

#     st.write("### IBNR Reserves Summary Table")
#     st.dataframe(summary_IBNR_df.style.format("{:.2f}").set_table_styles(
#         [{'selector': 'th', 'props': [('font-size', '12pt')]}]))

def visualize_ibnr_reserves_summary(summary_IBNR_df):
    fig = px.bar(
        summary_IBNR_df,
        x="Accident Year",
        y=[
            "IBNR Reserves (Chain Ladder)",
            "IBNR Reserves (BF Method)",
            "IBNR Reserves (Loss Ratio)"
        ],
        labels={"value": "IBNR Reserves", "variable": "Method"},
        title="Comparison of IBNR Reserves by Method",
        barmode="group"
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Accident Year",
        yaxis_title="IBNR Reserves",
        height=600
    )
    st.plotly_chart(fig)

    # Display table
    st.write("### IBNR Reserves Summary Table")
    st.dataframe(summary_IBNR_df.style.format("{:.2f}"))

# def visualize_ultimate_claims_summary(summary_ultimate_claims_df):
#     plt.figure(figsize=(15, 7))
#     bar_width = 0.25
#     x = np.arange(len(summary_ultimate_claims_df["Accident Year"]))
#     plt.bar(x - bar_width, summary_ultimate_claims_df["Ultimate Claims (Chain Ladder)"], bar_width, label="Chain Ladder", color="#1f77b4")
#     plt.bar(x, summary_ultimate_claims_df["Ultimate Claims (BF Method)"], bar_width, label="BF Method", color="#ff7f0e")
#     plt.bar(x + bar_width, summary_ultimate_claims_df["Ultimate Claims (Loss Ratio)"], bar_width, label="Loss Ratio", color="#2ca02c")
#     plt.title("Comparison of Ultimate Claims by Method", fontsize=16)
#     plt.xlabel("Accident Year", fontsize=12)
#     plt.ylabel("Ultimate Claims", fontsize=12)
#     plt.xticks(x, summary_ultimate_claims_df["Accident Year"], rotation=45, fontsize=10)
#     plt.legend(title="Method", fontsize=10, title_fontsize=12)
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.tight_layout()
#     st.pyplot(plt)

#     st.write("### Ultimate Claims Summary Table")
#     st.dataframe(summary_ultimate_claims_df.style.format("{:.2f}").set_table_styles(
#         [{'selector': 'th', 'props': [('font-size', '12pt')]}]))

def visualize_ultimate_claims_summary(summary_ultimate_claims_df):
    fig = px.bar(
        summary_ultimate_claims_df,
        x="Accident Year",
        y=[
            "Ultimate Claims (Chain Ladder)",
            "Ultimate Claims (BF Method)",
            "Ultimate Claims (Loss Ratio)"
        ],
        labels={"value": "Ultimate Claims", "variable": "Method"},
        title="Comparison of Ultimate Claims by Method",
        barmode="group"
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Accident Year",
        yaxis_title="Ultimate Claims",
        height=600
    )
    st.plotly_chart(fig)

    # Display table
    st.write("### Ultimate Claims Summary Table")
    st.dataframe(summary_ultimate_claims_df.style.format("{:.2f}"))

def prepare_diagnostic_metrics(df):
    # Pivot data into triangles
    paid_to_reported_triangle = df.pivot(index='accident_year', columns='development_year', values='paid_to_reported_ratio')
    incurred_to_earned_triangle = df.pivot(index='accident_year', columns='development_year', values='incurred_to_earned_ratio')
    return paid_to_reported_triangle, incurred_to_earned_triangle

# def plot_heatmap_triangle(data, title, cmap='coolwarm'):
#     plt.figure(figsize=(14, 8))
#     sns.heatmap(
#         data,
#         annot=True,  # Display values on the heatmap
#         fmt=".2f",  # Limit decimal places
#         cmap=cmap,
#         linewidths=0.5,  # Add gridlines
#         cbar_kws={"label": "Ratio"}  # Label for the color bar
#     )
#     plt.title(title, fontsize=16)
#     plt.xlabel("Development Year", fontsize=12)
#     plt.ylabel("Accident Year", fontsize=12)
#     plt.xticks(rotation=45)
#     plt.yticks(rotation=0)
#     st.pyplot(plt)

def plot_heatmap_triangle(data, title):
    # Accident Year Slider
    min_year = data.index.min()
    max_year = data.index.max()
    selected_years = st.slider(
        "Select Accident Year Range:",
        min_value=int(min_year),
        max_value=int(max_year),
        value=(int(min_year), int(max_year))
    )

    # Filter data based on selected years
    filtered_data = data.loc[selected_years[0]:selected_years[1]]

    # Create interactive heatmap with Plotly
    fig = go.Figure(
        data=go.Heatmap(
            z=filtered_data.values,
            x=filtered_data.columns,
            y=filtered_data.index,
            colorscale="rdbu",
            colorbar=dict(title="Ratio"),
            hovertemplate="<b>Accident Year:</b> %{y}<br><b>Development Year:</b> %{x}<br><b>Ratio:</b> %{z:.2f}<extra></extra>"
        )
    )

    # Add annotations for each cell
    for i, y in enumerate(filtered_data.index):
        for j, x in enumerate(filtered_data.columns):
            value = filtered_data.iloc[i, j]
            if not pd.isna(value):  # Skip NaN values
                fig.add_annotation(
                    text=f"{value:.2f}",  # Format value with 2 decimal places
                    x=x,
                    y=y,
                    showarrow=False,
                    font=dict(color="black", size=10),
                    align="center"
                )

    # Update layout for better visualization
    fig.update_layout(
        title=title,
        xaxis_title="Development Year",
        yaxis_title="Accident Year",
        template="plotly_white",
        height=600,
        width=900,
    )

    # Display the heatmap
    st.plotly_chart(fig)

# Define a function to check weights and update session state
def check_weights():
    total = chain_value + bf_value + loss_value
    tolerance = 0.01  # Define a small tolerance, e.g., 0.01 or 0.001
    if abs(total - 1.0) <= tolerance:
        st.session_state['weights_correct'] = True
    else:
        st.session_state['weights_correct'] = False

## Line Break---------------------------------------------------------------------------------------------------------------------------------------

# # Function to forecast using Chain-Ladder Method
# def forecast_chain_ladder(triangle, future_years):
#     # max_dev_years = triangle.shape[1]
#     # development_factors = [triangle.iloc[:, col].sum() / triangle.iloc[:, col-1].sum() 
#     #                        for col in range(1, max_dev_years) if triangle.iloc[:, col-1].sum() != 0]
    
#     # # Average development factors for extrapolation
#     # average_factors = [sum(development_factors[:i+1]) / len(development_factors[:i+1]) for i in range(len(development_factors))]

#     # # Forecast new accident years
#     # last_accident_year = triangle.index[-1]
#     # for new_year in range(1, future_years + 1):
#     #     new_year_data = [triangle.iloc[-1, 0] * factor for factor in average_factors]
#     #     new_year_data = [new_year_data[i] if i < len(new_year_data) else new_year_data[-1] for i in range(max_dev_years)]
#     #     triangle.loc[last_accident_year + new_year] = new_year_data

#     max_dev_years = triangle.shape[1]
#     last_accident_year = triangle.index.max()

#     # Calculate initial development factors from existing data
#     development_factors = [triangle.iloc[:, i+1].dropna().sum() / triangle.iloc[:, i].dropna().sum() 
#                            for i in range(max_dev_years - 1) if not triangle.iloc[:, i].dropna().empty]

#     # Extend the triangle to include new accident years
#     new_index = list(triangle.index) + list(range(last_accident_year + 1, last_accident_year + 1 + future_years))
#     extended_triangle = pd.DataFrame(index=new_index, columns=range(1, max_dev_years + 1))
#     extended_triangle.update(triangle)

#     # Forecast each new accident year
#     for year in range(last_accident_year + 1, last_accident_year + future_years + 1):
#         for dev_year in range(1, max_dev_years + 1):
#             if dev_year == 1:
#                 # Assume the initial value can be an average or trend-based
#                 extended_triangle.at[year, dev_year] = extended_triangle.iloc[:, dev_year - 1].dropna().mean()
#             else:
#                 previous_value = extended_triangle.at[year, dev_year - 1]
#                 factor_index = dev_year - 2
#                 if factor_index < len(development_factors):
#                     extended_triangle.at[year, dev_year] = previous_value * development_factors[factor_index]
        
#         # Update development factors using the newly forecasted data
#         for i in range(max_dev_years - 1):
#             if not extended_triangle.iloc[:, i].dropna().empty and not extended_triangle.iloc[:, i+1].dropna().empty:
#                 development_factors[i] = extended_triangle.iloc[:, i+1].dropna().sum() / extended_triangle.iloc[:, i].dropna().sum()

#     st.write("Forecasted Triangle for Future Accident Years (Chain-Ladder Method):")
#     visualize_chain_ladder(extended_triangle)  # Reuse the existing function to visualize the extended triangle

# # # Function to forecast using Bornhuetter-Ferguson (BF) Method
# # def forecast_bf_method(triangle, future_years):
# #     development_factors = calculate_development_factors(triangle)
# #     bf_ultimate_claims = triangle.sum(axis=1) + (triangle.iloc[:, -1] * sum(development_factors[-future_years:]))  # Extend ultimate claims by future factors
# #     bf_reserves = bf_ultimate_claims - triangle.sum(axis=1, min_count=1)  # Update reserves
# #     visualize_bf_method(triangle.assign(Ultimate=bf_ultimate_claims, Reserves=bf_reserves))  # Modify existing function to include new columns

# # # Function to forecast using Loss Ratio Method
# # def forecast_loss_ratio(df, future_years):
# #     triangle = prepare_triangle(df)
# #     earned_premiums = df.pivot(index='accident_year', columns='development_year', values='earned_premium')
# #     loss_ratios = (triangle.sum(axis=1) / earned_premiums.sum(axis=1))
# #     expected_claims = earned_premiums.sum(axis=1) * loss_ratios
# #     for _ in range(future_years):
# #         # Project future values based on average loss ratio
# #         new_claims = expected_claims.iloc[-1] * np.mean(loss_ratios)
# #         expected_claims = pd.concat([expected_claims, pd.Series([new_claims])], ignore_index=True)
# #     visualize_loss_ratio(df.assign(Expected_Claims=expected_claims))  # Adjust function to handle new expected claims column

# # Updated prepare_triangle_with_forecast to initialize future years correctly
# def prepare_triangle_with_forecast(df, future_years):
#     # Prepare the initial triangle
#     triangle = df.pivot(index='accident_year', columns='development_year', values='incurred')
    
#     # Add new accident years for forecasting
#     max_accident_year = triangle.index.max()
#     max_development_year = triangle.columns.max()
    
#     # Create new rows for future accident years
#     for i in range(1, future_years + 1):
#         triangle.loc[max_accident_year + i, :] = [None] * len(triangle.columns)
    
#     return triangle, max_development_year

# def forecast_chain_ladder(triangle, future_years, max_development_year):
#     development_factors = []
#     for col in range(triangle.shape[1] - 1):
#         current_sum = triangle.iloc[:, col].sum()
#         next_sum = triangle.iloc[:, col + 1].sum()
#         factor = next_sum / current_sum if current_sum != 0 else 1
#         development_factors.append(factor)

#     projected_triangle = triangle.copy()
#     for col in range(1, projected_triangle.shape[1]):
#         for row in range(projected_triangle.shape[0]):
#             if pd.isna(projected_triangle.iloc[row, col]):
#                 previous_value = projected_triangle.iloc[row, col - 1]
#                 if not pd.isna(previous_value):
#                     projected_triangle.iloc[row, col] = previous_value * development_factors[col - 1]

#     # Forecast for future accident years
#     for i in range(1, future_years + 1):
#         for j in range(max_development_year + 1):  # Include all development years
#             if j == 0:  # Initialize the first development year for future accident years
#                 last_observed_year = projected_triangle.iloc[-i - 1, 0]
#                 projected_triangle.iloc[-i, j] = last_observed_year * development_factors[0]
#             elif j - 1 < len(development_factors):  # Ensure we don't exceed the range of development_factors
#                 previous_value = projected_triangle.iloc[-i, j - 1]
#                 projected_triangle.iloc[-i, j] = previous_value * development_factors[j - 1]
#             else:  # If no development factor is available, break the loop
#                 break

#     return projected_triangle

# # Corrected BF Method for cascading projections
# def forecast_bf_method(triangle, future_years, max_development_year):
#     development_factors = []
#     for col in range(triangle.shape[1] - 1):
#         current_sum = triangle.iloc[:, col].sum()
#         next_sum = triangle.iloc[:, col + 1].sum()
#         factor = next_sum / current_sum if current_sum != 0 else 1
#         development_factors.append(factor)

#     projected_triangle = triangle.copy()
#     for col in range(1, projected_triangle.shape[1]):
#         for row in range(projected_triangle.shape[0]):
#             if pd.isna(projected_triangle.iloc[row, col]):
#                 previous_value = projected_triangle.iloc[row, col - 1]
#                 if not pd.isna(previous_value):
#                     projected_triangle.iloc[row, col] = previous_value * development_factors[col - 1]

#     # Forecast for future accident years
#     for i in range(1, future_years + 1):
#         for j in range(max_development_year + 1):  # Include all development years
#             if j == 0:  # Initialize the first development year for future accident years
#                 last_observed_year = projected_triangle.iloc[-i - 1, 0]
#                 projected_triangle.iloc[-i, j] = last_observed_year * development_factors[0]
#             else:  # Cascading projection for subsequent development years
#                 previous_value = projected_triangle.iloc[-i, j - 1]
#                 projected_triangle.iloc[-i, j] = previous_value * development_factors[j - 1]

#     bf_ultimate_claims = projected_triangle.sum(axis=1)
#     bf_reserves = (1 - (triangle.sum(axis=1) / bf_ultimate_claims)) * bf_ultimate_claims

#     return projected_triangle, bf_reserves

# # Corrected Loss Ratio Method for cascading projections
# def forecast_loss_ratio_method(df, future_years):
#     triangle = prepare_triangle(df)
#     earned_premiums = df.pivot(index='accident_year', columns='development_year', values='earned_premium')

#     incurred_claims = triangle.sum(axis=1)
#     total_earned_premiums = earned_premiums.sum(axis=1)
#     loss_ratios = incurred_claims / total_earned_premiums
#     expected_claims = total_earned_premiums * loss_ratios

#     # Add new accident years for forecasting
#     max_accident_year = triangle.index.max()
#     max_development_year = triangle.columns.max()
#     for i in range(1, future_years + 1):
#         triangle.loc[max_accident_year + i, :] = [None] * triangle.shape[1]
#         for j in range(max_development_year + 1):
#             if j == 0:  # Initialize the first development year
#                 last_observed_year = incurred_claims.iloc[-1] * loss_ratios.mean()
#                 triangle.iloc[-i, j] = last_observed_year
#             else:  # Project subsequent development years
#                 triangle.iloc[-i, j] = triangle.iloc[-i, j - 1] * loss_ratios.mean()

#     return triangle, expected_claims
###################################################################
###################################################################
####################################################################
# Define UDF to transform data into a triangle format
def data_to_triangle(df, value_column):
    """
    Convert data into a triangle format where rows are accident years and
    columns are development years.

    Args:
    data (DataFrame): The original dataset.
    value_column (str): The column name for the values to be reshaped ('incurred' or 'cumpaid').

    Returns:
    DataFrame: Triangle format of the data.
    """
    max_dev_years = df['development_year'].max()
    triangle = pd.pivot_table(df, values=value_column, index='accident_year', columns='development_year', aggfunc=np.sum)
    # Ensure the triangle has all necessary columns (up to max development years)
    all_dev_years = range(1, max_dev_years + 1)
    for year in all_dev_years:
        if year not in triangle.columns:
            triangle[year] = np.nan
    return triangle.loc[:, sorted(triangle.columns)]

# Define UDF to apply the Chain Ladder method
def chain_ladder_method(triangle):
    """
    Apply the Chain Ladder method to estimate missing values in the claims triangle.

    Args:
    triangle (DataFrame): Claims data in triangle format.

    Returns:
    DataFrame: Completed triangle with estimated values.
    """
    # Calculate development factors
    factors = {}
    for i in range(1, triangle.shape[1]):
        valid_data = triangle[[i, i + 1]].dropna()
        factors[i] = (valid_data[i + 1] / valid_data[i]).mean()
    
    # Fill in the triangle using the factors
    for i in range(1, triangle.shape[1]):
        for j in range(i+1, triangle.shape[1] + 1):
            if pd.isna(triangle.at[triangle.index[-1] - j + i + 1, j]):
                triangle.at[triangle.index[-1] - j + i + 1, j] = triangle.at[triangle.index[-1] - j + i + 1, j-1] * factors.get(j-1, 1)
    return triangle

# # Define UDF to export DataFrame to Excel
# def dataframe_to_excel(dataframe, file_path):
#     """
#     Export a DataFrame to an Excel file.

#     Args:
#     dataframe (DataFrame): Data to export.
#     file_path (str): Path where the Excel file will be saved.
#     """
#     dataframe.to_excel(file_path, index=True)

# # Now process the dataset using these UDFs
# incurred_triangle = data_to_triangle(data, 'incurred')
# predicted_incurred = chain_ladder_method(incurred_triangle.copy())

# paid_triangle = data_to_triangle(data, 'cumpaid')
# predicted_paid = chain_ladder_method(paid_triangle.copy())

# # Define paths for Excel files
# incurred_file_path = '/mnt/data/Predicted_Incurred_Claims.xlsx'
# paid_file_path = '/mnt/data/Predicted_Paid_Claims.xlsx'

# # Export to Excel
# dataframe_to_excel(predicted_incurred, incurred_file_path)
# dataframe_to_excel(predicted_paid, paid_file_path)

# incurred_file_path, paid_file_path
def plot_forecast_line(predictions_df):
    # Melting the DataFrame to make it suitable for Plotly Express
    predictions_melted = predictions_df.reset_index().melt(id_vars=['Future Accident Year'], var_name='Development Year', value_name='Predicted Claims')
    
    fig = px.line(predictions_melted, x='Future Accident Year', y='Predicted Claims', color='Development Year',
                  title='Forecast of Future Accident Years by Development Year',
                  labels={'Predicted Claims': 'Predicted Claim Amount'})
    st.plotly_chart(fig, use_container_width=True)

def plot_forecast_heatmap(predictions_df):
    # Prepare data: flip the DataFrame so future accident years are rows and development years are columns
    data_for_heatmap = predictions_df

    # Generate the heatmap with annotations
    fig = px.imshow(data_for_heatmap,
                    labels=dict(x="Development Year", y="Future Accident Year", color="Predicted Claims"),
                    x=data_for_heatmap.columns,  # Development Years as columns
                    y=data_for_heatmap.index,    # Future Accident Years as rows
                    text_auto=True,  # Automatically add annotations with the cell values
                    title="Heatmap of Predicted Claims Over Future Accident Years")

    # Improve layout: rotate x-axis labels for better visibility if necessary
    fig.update_xaxes(side="bottom")
    fig.update_layout(xaxis=dict(tickangle=-45), height=600,width=900)
    st.plotly_chart(fig)


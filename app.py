import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import ListedColormap
from matplotlib import colors as mcolors
import seaborn as sns
from scipy.stats import trim_mean
import plotly.express as px
import plotly.graph_objects as go
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xlsxwriter
from io import BytesIO
import plotly.tools as tls
from sklearn.linear_model import LinearRegression
import os
from math import pi
from udfs import *

#LLM Imports
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

os.environ['API_KEY'] = st.secrets['api']
# gemini.api_key = st.secrets["api"]
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=API_KEY, temperature=0.2)

# Title of the app
st.title('IBNR Loss Reserves Usecase')

# Sidebar navigation
st.sidebar.title("Tool Capabilities")
option = st.sidebar.radio('Choose an option:', ['Input Dataset', 'Average LDFs', 
                                                'LDFs Summary Report', 'Actuary Methods for IBNR',
                                                'Diagnostic Metrics Evaluation',
                                                'Ultimate Claims & IBNR Reserves Summary Report','Future Accident Year Projections'])

# Ensure session state for dataset exists
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None

if option == 'Input Dataset':
    st.write("Welcome to the IBNR Loss Reserves application. Please upload your dataset to begin!")
    uploaded_file = st.file_uploader("Upload your CSV dataset file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.uploaded_df = df  # Store the dataset in session state
        st.write("Data Preview:")
        st.dataframe(df.head())
        prompt = st.text_area("Enter your prompt:")
        if st.button("Generate:"):
            if prompt: 
                st.write("LLM is generating your answer, please wait...")
                st.write(pandas_ai.run(df, prompt=prompt))
            else:
                st.warning("Please enter a prompt first.")
        st.divider()

        try:
            with st.spinner('Processing data...'):
                st.session_state.uploaded_df = calculate_derived_columns(df)
                st.success("Calculations for derived columns completed.")

            st.divider()

            heatmap_methods = ['Select an option','Show Age-to-Age Factors Heatmap for Incurred Claim Amount','Show Age-to-Age Factors Heatmap for Paid Claim Amount']
            selected_method = st.selectbox("Select an ATAF Heatmap chart", heatmap_methods)

            if selected_method == 'Show Age-to-Age Factors Heatmap for Incurred Claim Amount':
                with st.spinner('Generating heatmap...'):
                    calculate_and_plot_ataf_plotly(st.session_state.uploaded_df)     
                    user_query = st.text_input("Ask your question related to the heatmap:")
                    if user_query:
                        try:
                            with st.spinner("Processing your query..."):
                                # Assuming your DataFrame after modifications
                                df_to_process = calculate_ataf(st.session_state.uploaded_df)
                                agent_executor = create_pandas_dataframe_agent(
                                                    llm,
                                                    df_to_process,
                                                    agent_type="zero-shot-react-description",
                                                    verbose=True,
                                                    allow_dangerous_code=True,
                                                    return_intermediate_steps=True)
                                user_answer = agent_executor.invoke(user_query)
                                # st.write("Agent's Answer:", user_answer)
                                user_refined_ans= extract_input_output(user_answer)
                                refined_answer = beautify_output(user_answer['output'])
                                st.write(refined_answer)

                        except Exception as e:
                            st.error(f"Error processing your query: {e}")

            elif selected_method == 'Show Age-to-Age Factors Heatmap for Paid Claim Amount':
                with st.spinner('Generating heatmap...'):
                    calculate_and_plot_ataf_cumpaid_plotly(st.session_state.uploaded_df)
                    # user_query = st.text_input("Ask your question related to the heatmap:")
                    # if user_query:
                    #     try:
                    #         with st.spinner("Processing your query..."):
                    #             # Assuming your DataFrame after modifications
                    #             df_to_process = calculate_ataf_cumpaid(st.session_state.uploaded_df)
                    #             agent_executor = create_pandas_dataframe_agent(
                    #                                 llm,
                    #                                 df_to_process,
                    #                                 agent_type="zero-shot-react-description",
                    #                                 verbose=True,
                    #                                 allow_dangerous_code=True,
                    #                                 return_intermediate_steps=True)
                    #             user_answer = agent_executor.invoke(user_query)
                    #             refined_answer = beautify_output(user_answer['output'])
                    #             st.write(refined_answer)

                    #     except Exception as e:
                    #         st.error(f"Error processing your query: {e}")

        except Exception as e:
            st.error(f"Failed to process file: {e}")

else:
    if st.session_state.uploaded_df is not None:
        df = st.session_state.uploaded_df
        # calculate_ataf(st.session_state.uploaded_df)
        # calculate_ataf_cumpaid(st.session_state.uploaded_df)

        if option == 'Average LDFs':
            st.write("We have 7 averaging methods available to calculate LDFs from the ATAF. Please select one from the dropdown below:")

            claim_amount_method = ['Choose one method','Incurred Claim Amount','Paid Claim Amount']
            claim_method = st.selectbox("Select Incurred or Paid Claim Amount:", claim_amount_method)

            if claim_method == 'Incurred Claim Amount':
                averaging_methods = ['Select an averaging method','Simple Moving Average', 'Volume Weighted Average', 'Exponential Smoothing', 'Median-Based', 'Trimmed Mean']
                selected_method = st.selectbox("Select an averaging method:", averaging_methods)
                df = st.session_state.uploaded_df
                # calculate_ataf(st.session_state.uploaded_df)
                pivot_atafs = prepare_pivot_atafs(df)
                
                
                if selected_method == 'Simple Moving Average':
                    st.write(":green-background[:bulb: Hey! Simple Moving Average is best used when the ATAF data points are consistent and there are no significant outliers. It works best for scenarios where data variability is relatively low.]")
                    st.divider()
                    pivot_atafs=visualize_sma_plotly(pivot_atafs)
                    # st.divider()
                    # user_query = st.text_input("Ask your question related to the Simple Moving Average Values:")
                    # if user_query:
                    #     try:
                    #         with st.spinner("Processing your query..."):
                    #             agent_executor = create_pandas_dataframe_agent(
                    #                                 llm,
                    #                                 pivot_atafs,
                    #                                 agent_type="zero-shot-react-description",
                    #                                 verbose=True,
                    #                                 allow_dangerous_code=True,
                    #                                 return_intermediate_steps=True)
                    #             user_answer = agent_executor.invoke(user_query)
                    #             refined_answer = beautify_output(user_answer['output'])
                    #             st.write(refined_answer)
                    #     except Exception as e:
                    #         st.error(f"Error processing your query: {e}")     
                    
                elif selected_method == 'Volume Weighted Average':
                    st.write(":green-background[:bulb: Hey! We have incorporated the Incurred Claims Amount as weights while calculating the volume weighted average. This method is particularly very useful in scenarios where claim amounts are highly variable. By weighting more recent claims or those with larger impacts (such as higher claim amounts) more heavily, it can provide a more accurate reflection of recent trends and expected future developments.]")
                    st.divider()
                    pivot_atafs=visualize_vwa_plotly(pivot_atafs, df)
                    # st.divider()
                    # user_query = st.text_input("Ask your question related to the Volume Weighted Average Values:")
                    # if user_query:
                    #     try:
                    #         with st.spinner("Processing your query..."):
                    #             agent_executor = create_pandas_dataframe_agent(
                    #                                 llm,
                    #                                 pivot_atafs,
                    #                                 agent_type="zero-shot-react-description",
                    #                                 verbose=True,
                    #                                 allow_dangerous_code=True,
                    #                                 return_intermediate_steps=True)
                    #             user_answer = agent_executor.invoke(user_query)
                    #             refined_answer = beautify_output(user_answer['output'])
                    #             st.write(refined_answer)
                    #     except Exception as e:
                    #         st.error(f"Error processing your query: {e}")     
                    
                # elif selected_method == 'Geometric Average':
                #     pivot_atafs=visualize_ga_plotly(pivot_atafs)
                
                elif selected_method == 'Exponential Smoothing':
                    st.write(":green-background[:bulb: Hey! We have applied an ideal Exponential Smoothing factor for this calculation. This method is particularly useful when we want to smooth out short-term fluctuations and highlight long-term trends or cycles.]")
                    st.divider()
                    pivot_atafs=visualize_esa_plotly(pivot_atafs)
                    # st.divider()
                    # user_query = st.text_input("Ask your question related to the Exponential Smoothing Average Values:")
                    # if user_query:
                    #     try:
                    #         with st.spinner("Processing your query..."):
                    #             agent_executor = create_pandas_dataframe_agent(
                    #                                 llm,
                    #                                 pivot_atafs,
                    #                                 agent_type="zero-shot-react-description",
                    #                                 verbose=True,
                    #                                 allow_dangerous_code=True,
                    #                                 return_intermediate_steps=True)
                    #             user_answer = agent_executor.invoke(user_query)
                    #             refined_answer = beautify_output(user_answer['output'])
                    #             st.write(refined_answer)
                    #     except Exception as e:
                    #         st.error(f"Error processing your query: {e}")
                    
                elif selected_method == 'Median-Based':
                    st.write(":green-background[:bulb: Hey! Median Based Average is particularly robust against outliers, which is beneficial in handling claims data that often contains large, atypical payouts. It's best used for skewed distributions or when outliers are present and could distort the mean.]")
                    st.divider()
                    pivot_atafs=visualize_median_plotly(pivot_atafs)
                    # st.divider()
                    # user_query = st.text_input("Ask your question related to the Median Based Average Values:")
                    # if user_query:
                    #     try:
                    #         with st.spinner("Processing your query..."):
                    #             agent_executor = create_pandas_dataframe_agent(
                    #                                 llm,
                    #                                 pivot_atafs,
                    #                                 agent_type="zero-shot-react-description",
                    #                                 verbose=True,
                    #                                 allow_dangerous_code=True,
                    #                                 return_intermediate_steps=True)
                    #             user_answer = agent_executor.invoke(user_query)
                    #             refined_answer = beautify_output(user_answer['output'])
                    #             st.write(refined_answer)
                    #     except Exception as e:
                    #         st.error(f"Error processing your query: {e}")
                    
                elif selected_method == 'Trimmed Mean':
                    st.write(":green-background[:bulb: Hey! Trimmed Mean Average particularly strikes a balance between excluding outliers and retaining data. It's useful when the data set includes some extreme claims that should not overly influence the LDF but also contains valuable information in the outer ranges of data distribution.]")
                    st.divider()
                    pivot_atafs=visualize_trimmed_plotly(pivot_atafs)
                    # st.divider()
                    # user_query = st.text_input("Ask your question related to the Trimmed Mean Average Values:")
                    # if user_query:
                    #     try:
                    #         with st.spinner("Processing your query..."):
                    #             agent_executor = create_pandas_dataframe_agent(
                    #                                 llm,
                    #                                 pivot_atafs,
                    #                                 agent_type="zero-shot-react-description",
                    #                                 verbose=True,
                    #                                 allow_dangerous_code=True,
                    #                                 return_intermediate_steps=True)
                    #             user_answer = agent_executor.invoke(user_query)
                    #             refined_answer = beautify_output(user_answer['output'])
                    #             st.write(refined_answer)
                    #     except Exception as e:
                    #         st.error(f"Error processing your query: {e}")
                    
                # elif selected_method == 'Harmonic Mean':
                #     pivot_atafs=visualize_harmonic(pivot_atafs)
                    
            elif claim_method == 'Paid Claim Amount':
                averaging_methods = ['Select an averaging method','Simple Moving Average', 'Volume Weighted Average', 'Exponential Smoothing', 'Median-Based', 'Trimmed Mean']
                selected_method = st.selectbox("Select an averaging method:", averaging_methods)
                df = st.session_state.uploaded_df
                calculate_ataf_cumpaid(st.session_state.uploaded_df)
                pivot_claim_atafs = prepare_pivot_claim_atafs(df)
                
                if selected_method == 'Simple Moving Average':
                    st.write(":green-background[:bulb: Hey! Simple Moving Average is best used when the ATAF data points are consistent and there are no significant outliers. It works best for scenarios where data variability is relatively low.]")
                    st.divider()
                    pivot_atafs=visualize_sma_plotly(pivot_claim_atafs)
                    # st.divider()
                    # user_query = st.text_input("Ask your question related to the Simple Moving Average Values:")
                    # if user_query:
                    #     try:
                    #         with st.spinner("Processing your query..."):
                    #             agent_executor = create_pandas_dataframe_agent(
                    #                                 llm,
                    #                                 pivot_atafs,
                    #                                 agent_type="zero-shot-react-description",
                    #                                 verbose=True,
                    #                                 allow_dangerous_code=True,
                    #                                 return_intermediate_steps=True)
                    #             user_answer = agent_executor.invoke(user_query)
                    #             refined_answer = beautify_output(user_answer['output'])
                    #             st.write(refined_answer)
                    #     except Exception as e:
                    #         st.error(f"Error processing your query: {e}")  
                    
                elif selected_method == 'Volume Weighted Average':
                    st.write(":green-background[:bulb: Hey! We have incorporated the Paid Claims Amount as weights while calculating the volume weighted average. This method is particularly very useful in scenarios where claim amounts are highly variable. By weighting more recent claims or those with larger impacts (such as higher claim amounts) more heavily, it can provide a more accurate reflection of recent trends and expected future developments.]")
                    st.divider()
                    pivot_atafs=visualize_vwa_plotly(pivot_claim_atafs, df)
                    # st.divider()
                    # user_query = st.text_input("Ask your question related to the Volume Weighted Average Values:")
                    # if user_query:
                    #     try:
                    #         with st.spinner("Processing your query..."):
                    #             agent_executor = create_pandas_dataframe_agent(
                    #                                 llm,
                    #                                 pivot_atafs,
                    #                                 agent_type="zero-shot-react-description",
                    #                                 verbose=True,
                    #                                 allow_dangerous_code=True,
                    #                                 return_intermediate_steps=True)
                    #             user_answer = agent_executor.invoke(user_query)
                    #             refined_answer = beautify_output(user_answer['output'])
                    #             st.write(refined_answer)
                    #     except Exception as e:
                    #         st.error(f"Error processing your query: {e}")
                    
                # elif selected_method == 'Geometric Average':
                #     pivot_atafs=visualize_ga_plotly(pivot_claim_atafs)  
                
                elif selected_method == 'Exponential Smoothing':
                    st.write(":green-background[:bulb: Hey! We have applied an ideal Exponential Smoothing factor for this calculation. This method is particularly useful when we want to smooth out short-term fluctuations and highlight long-term trends or cycles.]")
                    st.divider()
                    pivot_atafs=visualize_esa_plotly(pivot_claim_atafs)
                    # st.divider()
                    # user_query = st.text_input("Ask your question related to the Exponential Smoothing Average Values:")
                    # if user_query:
                    #     try:
                    #         with st.spinner("Processing your query..."):
                    #             agent_executor = create_pandas_dataframe_agent(
                    #                                 llm,
                    #                                 pivot_atafs,
                    #                                 agent_type="zero-shot-react-description",
                    #                                 verbose=True,
                    #                                 allow_dangerous_code=True,
                    #                                 return_intermediate_steps=True)
                    #             user_answer = agent_executor.invoke(user_query)
                    #             refined_answer = beautify_output(user_answer['output'])
                    #             st.write(refined_answer)
                    #     except Exception as e:
                    #         st.error(f"Error processing your query: {e}")
                    
                elif selected_method == 'Median-Based':
                    st.write(":green-background[:bulb: Hey! Median Based Average is particularly robust against outliers, which is beneficial in handling claims data that often contains large, atypical payouts. It's best used for skewed distributions or when outliers are present and could distort the mean.]")
                    st.divider()
                    pivot_atafs=visualize_median_plotly(pivot_claim_atafs)
                    # st.divider()
                    # user_query = st.text_input("Ask your question related to the Median Based Average Values:")
                    # if user_query:
                    #     try:
                    #         with st.spinner("Processing your query..."):
                    #             agent_executor = create_pandas_dataframe_agent(
                    #                                 llm,
                    #                                 pivot_atafs,
                    #                                 agent_type="zero-shot-react-description",
                    #                                 verbose=True,
                    #                                 allow_dangerous_code=True,
                    #                                 return_intermediate_steps=True)
                    #             user_answer = agent_executor.invoke(user_query)
                    #             refined_answer = beautify_output(user_answer['output'])
                    #             st.write(refined_answer)
                    #     except Exception as e:
                    #         st.error(f"Error processing your query: {e}")
                    
                elif selected_method == 'Trimmed Mean':
                    st.write(":green-background[:bulb: Hey! Trimmed Mean Average particularly strikes a balance between excluding outliers and retaining data. It's useful when the data set includes some extreme claims that should not overly influence the LDF but also contains valuable information in the outer ranges of data distribution.]")
                    st.divider()
                    pivot_atafs=visualize_trimmed_plotly(pivot_claim_atafs)
                    # st.divider()
                    # user_query = st.text_input("Ask your question related to the Trimmed Mean Average Values:")
                    # if user_query:
                    #     try:
                    #         with st.spinner("Processing your query..."):
                    #             agent_executor = create_pandas_dataframe_agent(
                    #                                 llm,
                    #                                 pivot_atafs,
                    #                                 agent_type="zero-shot-react-description",
                    #                                 verbose=True,
                    #                                 allow_dangerous_code=True,
                    #                                 return_intermediate_steps=True)
                    #             user_answer = agent_executor.invoke(user_query)
                    #             refined_answer = beautify_output(user_answer['output'])
                    #             st.write(refined_answer)
                    #     except Exception as e:
                    #         st.error(f"Error processing your query: {e}")
                    
                # elif selected_method == 'Harmonic Mean':
                #     pivot_atafs=visualize_harmonic(pivot_claim_atafs)        
                    
        elif option == 'LDFs Summary Report':
            st.write("We have 3 different visualization graphs to help you analyze the Average LDFs better. Please select one from the dropdown below:")
            
            claim_amount_method = ['Choose one method','Incurred Claim Amount','Paid Claim Amount']
            claim_method = st.selectbox("Select Incurred or Paid Claim Amount:", claim_amount_method)

            if claim_method == 'Incurred Claim Amount':
                
                summary_options = ['Select a visualization method for the LDF Summary Report','Heatmap of LDF Averaging Methods', 'Radar Chart', 'Boxplot Charts']
                selected_summary_option = st.selectbox("Choose a visualization:", summary_options)
                df = st.session_state.uploaded_df
                calculate_ataf(st.session_state.uploaded_df)
                triangle = prepare_triangle(df)
                # summary_IBNR_df, summary_ultimate_claims_df = prepare_summary_dataframes(triangle, df)
                pivot_atafs = prepare_pivot_atafs(df)
                pivot_atafs=calc_sma(pivot_atafs)
                pivot_atafs=calc_vwa(pivot_atafs, df)
                # pivot_atafs=calc_ga(pivot_atafs)
                pivot_atafs=calc_esa(pivot_atafs)
                pivot_atafs=calc_median(pivot_atafs)
                pivot_atafs=calc_trimmed(pivot_atafs)
                # pivot_atafs=calc_harmonic(pivot_atafs)

                # Add a span selector
                spans = [10, 5, 3]
                selected_span = st.selectbox("Select Averaging Span (Years):", spans, index=0)

                summary_IBNR_df = create_summary_table(pivot_atafs, selected_span)
                st.divider()
    
                if selected_summary_option == 'Heatmap of LDF Averaging Methods':
                    visualize_heatmap(summary_IBNR_df,selected_span)
                    # st.divider()
                    # user_query = st.text_input("Ask your question related to the LDFs Summary Values:")
                    # if user_query:
                    #     try:
                    #         with st.spinner("Processing your query..."):
                    #             agent_executor = create_pandas_dataframe_agent(
                    #                                 llm,
                    #                                 summary_IBNR_df,
                    #                                 agent_type="zero-shot-react-description",
                    #                                 verbose=True,
                    #                                 allow_dangerous_code=True,
                    #                                 return_intermediate_steps=True)
                    #             user_answer = agent_executor.invoke(user_query)
                    #             refined_answer = beautify_output(user_answer['output'])
                    #             st.write(refined_answer)
                    #     except Exception as e:
                    #         st.error(f"Error processing your query: {e}")
                            
                elif selected_summary_option == 'Radar Chart':
                    min_year = df.accident_year.min()
                    max_year = df.accident_year.max()
                    year = st.selectbox("Select Accident Year:", range(min_year, max_year))
                    visualize_radar_chart(summary_IBNR_df, year)
                    # st.divider()
                    # user_query = st.text_input("Ask your question related to the LDFs Summary Values:")
                    # if user_query:
                    #     try:
                    #         with st.spinner("Processing your query..."):
                    #             agent_executor = create_pandas_dataframe_agent(
                    #                                 llm,
                    #                                 summary_IBNR_df,
                    #                                 agent_type="zero-shot-react-description",
                    #                                 verbose=True,
                    #                                 allow_dangerous_code=True,
                    #                                 return_intermediate_steps=True)
                    #             user_answer = agent_executor.invoke(user_query)
                    #             refined_answer = beautify_output(user_answer['output'])
                    #             st.write(refined_answer)
                    #     except Exception as e:
                    #         st.error(f"Error processing your query: {e}")
                    
                elif selected_summary_option == 'Boxplot Charts':
                    visualize_boxplot(summary_IBNR_df)
                    # st.divider()
                    # user_query = st.text_input("Ask your question related to the LDFs Summary Values:")
                    # if user_query:
                    #     try:
                    #         with st.spinner("Processing your query..."):
                    #             agent_executor = create_pandas_dataframe_agent(
                    #                                 llm,
                    #                                 summary_IBNR_df,
                    #                                 agent_type="zero-shot-react-description",
                    #                                 verbose=True,
                    #                                 allow_dangerous_code=True,
                    #                                 return_intermediate_steps=True)
                    #             user_answer = agent_executor.invoke(user_query)
                    #             refined_answer = beautify_output(user_answer['output'])
                    #             st.write(refined_answer)
                    #     except Exception as e:
                    #         st.error(f"Error processing your query: {e}")
                    
            elif claim_method == 'Paid Claim Amount':
                summary_options = ['Select a visualization method for the LDF Summary Report','Heatmap of LDF Averaging Methods', 'Radar Chart', 'Boxplot Charts']
                selected_summary_option = st.selectbox("Choose a visualization:", summary_options)
                df = st.session_state.uploaded_df
                calculate_ataf_cumpaid(st.session_state.uploaded_df)
                triangle = prepare_claim_triangle(df)
                # summary_IBNR_df, summary_ultimate_claims_df = prepare_summary_dataframes(triangle, df)
                pivot_claim_atafs = prepare_pivot_claim_atafs(df)
                pivot_atafs=calc_sma(pivot_claim_atafs)
                pivot_atafs=calc_vwa(pivot_claim_atafs, df)
                # pivot_atafs=calc_ga(pivot_atafs)
                pivot_atafs=calc_esa(pivot_claim_atafs)
                pivot_atafs=calc_median(pivot_claim_atafs)
                pivot_atafs=calc_trimmed(pivot_claim_atafs)
                # pivot_atafs=calc_harmonic(pivot_atafs)

                # Add a span selector
                spans = [10, 5, 3]
                selected_span = st.selectbox("Select Averaging Span (Years):", spans, index=0)
                
                summary_IBNR_df = create_summary_table(pivot_atafs, selected_span)
                st.divider()
    
                if selected_summary_option == 'Heatmap of LDF Averaging Methods':
                    visualize_heatmap(summary_IBNR_df,selected_span)
                    # st.divider()
                    # user_query = st.text_input("Ask your question related to the LDFs Summary Values:")
                    # if user_query:
                    #     try:
                    #         with st.spinner("Processing your query..."):
                    #             agent_executor = create_pandas_dataframe_agent(
                    #                                 llm,
                    #                                 summary_IBNR_df,
                    #                                 agent_type="zero-shot-react-description",
                    #                                 verbose=True,
                    #                                 allow_dangerous_code=True,
                    #                                 return_intermediate_steps=True)
                    #             user_answer = agent_executor.invoke(user_query)
                    #             refined_answer = beautify_output(user_answer['output'])
                    #             st.write(refined_answer)
                    #     except Exception as e:
                    #         st.error(f"Error processing your query: {e}")
                
                elif selected_summary_option == 'Radar Chart':
                    min_year = df.accident_year.min()
                    max_year = df.accident_year.max()
                    year = st.selectbox("Select Accident Year:", range(min_year, max_year))
                    visualize_radar_chart(summary_IBNR_df, year)
                    # st.divider()
                    # user_query = st.text_input("Ask your question related to the LDFs Summary Values:")
                    # if user_query:
                    #     try:
                    #         with st.spinner("Processing your query..."):
                    #             agent_executor = create_pandas_dataframe_agent(
                    #                                 llm,
                    #                                 summary_IBNR_df,
                    #                                 agent_type="zero-shot-react-description",
                    #                                 verbose=True,
                    #                                 allow_dangerous_code=True,
                    #                                 return_intermediate_steps=True)
                    #             user_answer = agent_executor.invoke(user_query)
                    #             refined_answer = beautify_output(user_answer['output'])
                    #             st.write(refined_answer)
                    #     except Exception as e:
                    #         st.error(f"Error processing your query: {e}")
                
                elif selected_summary_option == 'Boxplot Charts':
                    visualize_boxplot(summary_IBNR_df)
                    # st.divider()
                    # user_query = st.text_input("Ask your question related to the LDFs Summary Values:")
                    # if user_query:
                    #     try:
                    #         with st.spinner("Processing your query..."):
                    #             agent_executor = create_pandas_dataframe_agent(
                    #                                 llm,
                    #                                 summary_IBNR_df,
                    #                                 agent_type="zero-shot-react-description",
                    #                                 verbose=True,
                    #                                 allow_dangerous_code=True,
                    #                                 return_intermediate_steps=True)
                    #             user_answer = agent_executor.invoke(user_query)
                    #             refined_answer = beautify_output(user_answer['output'])
                    #             st.write(refined_answer)
                    #     except Exception as e:
                    #         st.error(f"Error processing your query: {e}")
                
        elif option == 'Actuary Methods for IBNR':
            st.write("Great! You have finally reached the Actuary Method implementation!")

            claim_amount_method = ['Choose one method','Incurred Claim Amount','Paid Claim Amount']
            claim_method = st.selectbox("Select Incurred or Paid Claim Amount:", claim_amount_method)

            if claim_method == 'Incurred Claim Amount':
                methods = ['Select an Actuary Method','Chain-Ladder Method', 'Bornhuetter-Ferguson (BF) Method', 'Loss Ratio Method']
                selected_method = st.selectbox("Choose an actuary method from the dropdown menu below:", methods)
                df = st.session_state.uploaded_df
                calculate_ataf(st.session_state.uploaded_df)
                value_column = 'incurred'
                triangle = data_to_triangle(df, value_column)
                # triangle = prepare_triangle(df)
    
                if selected_method == 'Chain-Ladder Method':
                    incurred_triangle = data_to_triangle(df, value_column)
                    predicted_incurred = chain_ladder_method(incurred_triangle.copy())
                    excel_file = to_excel(predicted_incurred)
                    st.divider()
                    st.write("Data Preview of Projected Claim Triangle for current accident years:")
                    st.dataframe(predicted_incurred.tail())
                    st.download_button(label="Download File",
                                       data=excel_file,
                                       file_name="chain_ladder_projected_incurred_claim_triangle.xlsx",
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                    st.divider()
                    visualize_chain_ladder(triangle)
                                                
                elif selected_method == 'Bornhuetter-Ferguson (BF) Method':
                    st.divider()
                    visualize_bf_method(triangle)
                    
                elif selected_method == 'Loss Ratio Method':
                    st.divider()
                    visualize_loss_ratio(df)

            elif claim_method == 'Paid Claim Amount':
                methods = ['Select an Actuary Method','Chain-Ladder Method', 'Bornhuetter-Ferguson (BF) Method', 'Loss Ratio Method']
                selected_method = st.selectbox("Choose an actuary method from the dropdown menu below:", methods)
                df = st.session_state.uploaded_df
                calculate_ataf_cumpaid(st.session_state.uploaded_df)
                triangle_claim = prepare_claim_triangle(df)
                if selected_method == 'Chain-Ladder Method':
                    value_column = 'cumpaid'
                    incurred_triangle = data_to_triangle(df, value_column)
                    predicted_incurred = chain_ladder_method(incurred_triangle.copy())
                    excel_file = to_excel(predicted_incurred)
                    st.divider()
                    st.write("Data Preview of Projected Claim Triangle for current accident years:")
                    st.dataframe(predicted_incurred.tail())
                    st.download_button(label="Download Chain Ladder Projected Claim Triangle for current accident years",
                                       data=excel_file,
                                       file_name="chain_ladder_projected_cumpaid_claim_triangle.xlsx",
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                    visualize_chain_ladder(triangle_claim)
                    
                elif selected_method == 'Bornhuetter-Ferguson (BF) Method':
                    st.divider()
                    visualize_bf_method(triangle_claim)
                    
                elif selected_method == 'Loss Ratio Method':
                    st.divider()
                    visualize_claim_loss_ratio(df)                  

        elif option == 'Diagnostic Metrics Evaluation':
            st.write("Let's analyze a few diagnostic metrics to analyze the dataset even better! Please select an option from the dropdown below:")
            diagnostic_options = ['Choose one option','Paid to Reported Incurred Claims Ratio', 'Incurred to Earned Premium Ratio Heatmap']
            selected_diagnostic_option = st.selectbox("Select Diagnostic Metric:", diagnostic_options)
            paid_to_reported_triangle, incurred_to_earned_triangle = prepare_diagnostic_metrics(df)

            if selected_diagnostic_option == 'Paid to Reported Incurred Claims Ratio':
                plot_heatmap_triangle(paid_to_reported_triangle, "Paid Claims to Reported Incurred Claims Ratio Heatmap")
                # st.divider()
                # user_query = st.text_input("Ask your question related to Paid To Reported Triangle:")
                # if user_query:
                #     try:
                #         with st.spinner("Processing your query..."):
                #             agent_executor = create_pandas_dataframe_agent(
                #                                 llm,
                #                                 paid_to_reported_triangle,
                #                                 agent_type="zero-shot-react-description",
                #                                 verbose=True,
                #                                 allow_dangerous_code=True,
                #                                 return_intermediate_steps=True)
                #             user_answer = agent_executor.invoke(user_query)
                #             refined_answer = beautify_output(user_answer['output'])
                #             st.write(refined_answer)
                #     except Exception as e:
                #         st.error(f"Error processing your query: {e}")
                            
            elif selected_diagnostic_option == 'Incurred to Earned Premium Ratio':
                plot_heatmap_triangle(incurred_to_earned_triangle, "Incurred Claims to Earned Premium Ratio Heatmap")
                # st.divider()
                # user_query = st.text_input("Ask your question related to Incurred to Earned Premium Ratio Triangle:")
                # if user_query:
                #     try:
                #         with st.spinner("Processing your query..."):
                #             agent_executor = create_pandas_dataframe_agent(
                #                                 llm,
                #                                 incurred_to_earned_triangle,
                #                                 agent_type="zero-shot-react-description",
                #                                 verbose=True,
                #                                 allow_dangerous_code=True,
                #                                 return_intermediate_steps=True)
                #             user_answer = agent_executor.invoke(user_query)
                #             refined_answer = beautify_output(user_answer['output'])
                #             st.write(refined_answer)
                #     except Exception as e:
                #         st.error(f"Error processing your query: {e}")

        elif option == 'Ultimate Claims & IBNR Reserves Summary Report':
            st.write("Let's analyze the comparisons between the outputs of each actuarial method! Please select an option from the dropdown below:")
            claim_amount_method = ['Choose one method', 'Incurred Claim Amount', 'Paid Claim Amount']
            claim_method = st.selectbox("Select Incurred or Paid Claim Amount:", claim_amount_method)
        
            if claim_method == 'Incurred Claim Amount':
                summary_options = ['Choose one option', 'Ultimate Claims Summary', 'IBNR Reserves Summary']
                selected_summary_option = st.selectbox("Select Summary Type:", summary_options)
                triangle = prepare_triangle(df)
                summary_IBNR_df, summary_ultimate_claims_df = prepare_summary_dataframes(triangle, df)
        
                if selected_summary_option == 'Ultimate Claims Summary':
                    visualize_ultimate_claims_summary(summary_ultimate_claims_df)
        
                    # Final Ultimate Claims Calculation
                    st.title("Calculate Final Ultimate Claims")
                    st.write(":green-background[:bulb: We will use a combination of all three actuary methods to get the final ultimate claims value. Please input weights for each actuary method. For example, Chain Ladder Method: 0.3, BF Method: 0.6, Loss Ratio Method: 0.1]")
        
                    # Input weights for the methods
                    chain_value = st.slider("Weight for Chain Ladder:", 0.0, 1.0, 0.3)
                    bf_value = st.slider("Weight for BF Method:", 0.0, 1.0, 0.6)
                    loss_value = st.slider("Weight for Loss Ratio Method:", 0.0, 1.0, 0.1)
        
                    total_weight = chain_value + bf_value + loss_value
        
                    if st.button('Show Final Ultimate Claims Amount'):
                        if abs(total_weight - 1.0) <= 0.01:
                            chain_amount = chain_ladder(triangle)
                            bf_amount = bf_method(triangle)
                            loss_ratio_amount = loss_ratio_incurred(df)
                    
                            # Calculate the final ultimate claims
                            final_ultimate_claims = (
                                chain_value * chain_amount +
                                bf_value * bf_amount +
                                loss_value * loss_ratio_amount
                            )
                    
                            # Convert Series to DataFrame for styling
                            final_ultimate_claims_df = final_ultimate_claims.reset_index()
                            final_ultimate_claims_df.columns = ["Accident Year", "Final Ultimate Claims Amount"]
                    
                            st.write("### Final Ultimate Incurred Claims by Accident Year")
                            st.dataframe(final_ultimate_claims_df.style.format("{:.2f}"))
                        else:
                            st.error(f"Weights must sum to 1.0. Current total: {total_weight:.2f}")
        
                elif selected_summary_option == 'IBNR Reserves Summary':
                    visualize_ibnr_reserves_summary(summary_IBNR_df)
        
                    # Final IBNR Reserves Calculation
                    st.title("Calculate Final IBNR Reserves")
                    st.write(":green-background[:bulb: We will use a combination of all three actuary methods to get the final IBNR reserves value. Please input weights for each actuary method. For example, Chain Ladder Method: 0.3, BF Method: 0.6, Loss Ratio Method: 0.1]")
        
                    # Input weights for the methods
                    chain_value = st.slider("Weight for Chain Ladder:", 0.0, 1.0, 0.3, key='Chain_IBNR')
                    bf_value = st.slider("Weight for BF Method:", 0.0, 1.0, 0.6, key='BF_IBNR')
                    loss_value = st.slider("Weight for Loss Ratio Method:", 0.0, 1.0, 0.1, key='Loss_IBNR')
        
                    total_weight = chain_value + bf_value + loss_value
        
                    if st.button('Show Final IBNR Reserves Amount'):
                        if abs(total_weight - 1.0) <= 0.01:
                            chain_amount = chain_ladder_ibnr(triangle)
                            bf_amount = bf_method_ibnr(triangle)
                            loss_ratio_amount = loss_ratio_ibnr_incurred(df)
                    
                            # Calculate the final IBNR reserves
                            final_ibnr_reserves = (
                                chain_value * chain_amount +
                                bf_value * bf_amount +
                                loss_value * loss_ratio_amount
                            )
                    
                            # Convert Series to DataFrame for styling
                            final_ibnr_reserves_df = final_ibnr_reserves.reset_index()
                            final_ibnr_reserves_df.columns = ["Accident Year", "Final IBNR Reserves Amount"]
                    
                            st.write("### Final IBNR Reserves by Accident Year")
                            st.dataframe(final_ibnr_reserves_df.style.format("{:.2f}"))
                        else:
                            st.error(f"Weights must sum to 1.0. Current total: {total_weight:.2f}")
               
            elif claim_method == 'Paid Claim Amount':
                summary_options = ['Choose one option','Ultimate Claims Summary', 'IBNR Reserves Summary']
                selected_summary_option = st.selectbox("Select Summary Type:", summary_options)
                triangle = prepare_claim_triangle(df)
                summary_IBNR_df, summary_ultimate_claims_df = prepare_summary_dataframes(triangle, df)
        
                if selected_summary_option == 'Ultimate Claims Summary':
                    visualize_ultimate_claims_summary(summary_ultimate_claims_df)
        
                    # Final Ultimate Claims Calculation
                    st.title("Calculate Final Ultimate Claims")
                    st.write(":green-background[:bulb: We will use a combination of all three actuary methods to get the final ultimate claims value. Please input weights for each actuary method. For example, Chain Ladder Method: 0.3, BF Method: 0.6, Loss Ratio Method: 0.1]")
        
                    # Input weights for the methods
                    chain_value = st.slider("Weight for Chain Ladder:", 0.0, 1.0, 0.3)
                    bf_value = st.slider("Weight for BF Method:", 0.0, 1.0, 0.6)
                    loss_value = st.slider("Weight for Loss Ratio Method:", 0.0, 1.0, 0.1)
        
                    total_weight = chain_value + bf_value + loss_value
        
                    if st.button('Show Final Ultimate Claims Amount'):
                        if abs(total_weight - 1.0) <= 0.01:
                            chain_amount = chain_ladder(triangle)
                            bf_amount = bf_method(triangle)
                            loss_ratio_amount = loss_ratio(df)
                    
                            # Calculate the final ultimate claims
                            final_ultimate_claims = (
                                chain_value * chain_amount +
                                bf_value * bf_amount +
                                loss_value * loss_ratio_amount
                            )
                    
                            # Convert Series to DataFrame for styling
                            final_ultimate_claims_df = final_ultimate_claims.reset_index()
                            final_ultimate_claims_df.columns = ["Accident Year", "Final Ultimate Claims Amount"]
                    
                            st.write("### Final Ultimate Incurred Claims by Accident Year")
                            st.dataframe(final_ultimate_claims_df.style.format("{:.2f}"))
                        else:
                            st.error(f"Weights must sum to 1.0. Current total: {total_weight:.2f}")
        
                elif selected_summary_option == 'IBNR Reserves Summary':
                    visualize_ibnr_reserves_summary(summary_IBNR_df)
        
                    # Final IBNR Reserves Calculation
                    st.title("Calculate Final IBNR Reserves")
                    st.write(":green-background[:bulb: We will use a combination of all three actuary methods to get the final IBNR reserves value. Please input weights for each actuary method. For example, Chain Ladder Method: 0.3, BF Method: 0.6, Loss Ratio Method: 0.1]")
        
                    # Input weights for the methods
                    chain_value = st.slider("Weight for Chain Ladder:", 0.0, 1.0, 0.3, key='Chain_IBNR')
                    bf_value = st.slider("Weight for BF Method:", 0.0, 1.0, 0.6, key='BF_IBNR')
                    loss_value = st.slider("Weight for Loss Ratio Method:", 0.0, 1.0, 0.1, key='Loss_IBNR')
        
                    total_weight = chain_value + bf_value + loss_value
        
                    if st.button('Show Final IBNR Reserves Amount'):
                        if abs(total_weight - 1.0) <= 0.01:
                            chain_amount = chain_ladder_ibnr(triangle)
                            bf_amount = bf_method_ibnr(triangle)
                            loss_ratio_amount = loss_ratio_ibnr(df)
                    
                            # Calculate the final IBNR reserves
                            final_ibnr_reserves = (
                                chain_value * chain_amount +
                                bf_value * bf_amount +
                                loss_value * loss_ratio_amount
                            )
                    
                            # Convert Series to DataFrame for styling
                            final_ibnr_reserves_df = final_ibnr_reserves.reset_index()
                            final_ibnr_reserves_df.columns = ["Accident Year", "Final IBNR Reserves Amount"]
                    
                            st.write("### Final IBNR Reserves by Accident Year")
                            st.dataframe(final_ibnr_reserves_df.style.format("{:.2f}"))
                        else:
                            st.error(f"Weights must sum to 1.0. Current total: {total_weight:.2f}")

        # elif option == 'Future Accident Year Projections':
        #     st.write("Let's analyze the comparisons between the outputs of each actuarial method! Please select an option from the dropdown below:")
        #     claim_amount_method = ['Choose one method','Incurred Claim Amount','Paid Claim Amount']
        #     claim_method = st.selectbox("Select Incurred or Paid Claim Amount:", claim_amount_method)

        #     st.write("Number of Forecast Years")
        #     future_years = st.slider("Enter number of future years to forecast:", min_value=1, max_value=10, value=5)
    
        #     if claim_method == 'Incurred Claim Amount':
        #         df = st.session_state.uploaded_df
        #         incurred_triangle = data_to_triangle(df, 'incurred')
        #         predicted_incurred = chain_ladder_method(incurred_triangle.copy())
        #         predicted_incurred.reset_index(inplace=True)
        #         predicted_incurred.columns = predicted_incurred.columns.astype(str)
        #         predicted_incurred.columns = predicted_incurred.columns.str.strip()
        #         st.dataframe(predicted_incurred)

        #         # Melt the dataset
        #         df_long = predicted_incurred.melt(
        #             id_vars=['accident_year'], 
        #             var_name='development_year', 
        #             value_name='incurred_claim_amount'
        #         )
                
        #         # Ensure the development_year is numeric
        #         df_long['accident_year'] = pd.to_numeric(df_long['accident_year'])
        #         df_long['development_year'] = pd.to_numeric(df_long['development_year'])  
        #         st.dataframe(df_long)
                  
        #         for lag in range(1, 4):  # Create 3 lag features
        #             df_long[f"lag_{lag}"] = df_long.groupby("accident_year")["incurred_claim_amount"].shift(lag)
                
        #         # Drop rows with NaN values (from lagging)
        #         df_long = df_long.dropna()
                
        #         # Encode categorical features
        #         df_long["accident_year"] = df_long["accident_year"].astype(int)
        #         st.dataframe(df_long)
                
        #         # Train-test split
        #         train_data = df_long[df_long["accident_year"] < 2015]
        #         test_data = df_long[df_long["accident_year"] == 2015]
                
        #         X_train = train_data[["accident_year", "development_year", "lag_1", "lag_2", "lag_3"]]
        #         y_train = train_data["incurred_claim_amount"]
        #         X_test = test_data[["accident_year", "development_year", "lag_1", "lag_2", "lag_3"]]
        #         y_test = test_data["incurred_claim_amount"]
                
        #         # LightGBM model
        #         train_set = lgb.Dataset(X_train, label=y_train)
        #         test_set = lgb.Dataset(X_test, label=y_test, reference=train_set)
                
        #         params = {
        #             "objective": "regression",
        #             "metric": "rmse",
        #             "boosting_type": "gbdt",
        #             "learning_rate": 0.1,
        #             "num_leaves": 31,
        #             "max_depth": -1,
        #             "seed": 42,
        #             "early_stopping": 10,  # Early stopping enabled
        #             "verbose": 1,         # Suppress printing during stopping
        #             "patience": 10  
        #         }

        #         model = lgb.train(params, train_set, valid_sets=[train_set, test_set])
                
        #         # Forecast for future accident years
        #         future_years = range(2016, 2026)
        #         future_predictions = []
            
        #         for year in future_years:
        #             year_predictions = []  # Store predictions for the current year
        #             for dev_year in range(1, 22):
        #                 lag_features = []
                        
        #                 for lag in range(1, 4):
        #                     # if dev_year > lag:
        #                     #     # Use predictions from the same year's earlier development years
        #                     #     lag_features.append(year_predictions[dev_year - 1 - lag])
        #                     if year == min(future_years) and dev_year <= 3:
        #                         # Use historical averages or some form of extrapolation for the first few lags
        #                         lag_features = [df_long[df_long['development_year'] == dev_year - lag]['incurred_claim_amount'].mean() for lag in range(1, 4)]
        #                     else:
        #                         # Retrieve lag values from the previous year's development years
        #                         previous_year = year - 1
        #                         last_dev_year = dev_year - lag + 21
        #                         lag_value = df_long[
        #                             (df_long["accident_year"] == previous_year) &
        #                             (df_long["development_year"] == last_dev_year)
        #                         ]["incurred_claim_amount"].values
                                
        #                         # Use fallback if historical data is missing
        #                         fallback_value = df_long["incurred_claim_amount"].mean()
        #                         lag_features.append(lag_value[0] if len(lag_value) > 0 else fallback_value)
                        
        #                 # Debugging: Inspect lag_features
        #                 print(f"Year: {year}, Dev Year: {dev_year}, Lag Features: {lag_features}")
                        
        #                 # Predict and store
        #                 row = [year, dev_year] + lag_features
        #                 pred = model.predict([row])[0]
        #                 year_predictions.append(pred)
                    
        #             future_predictions.extend(year_predictions)

        #         # Reshape predictions for presentation
        #         future_df = pd.DataFrame(future_predictions, columns=["incurred_amount"])
        #         future_df["accident_year"] = [year for year in future_years for _ in range(1, 22)]
        #         future_df["development_year"] = [i for _ in future_years for i in range(1, 22)]
                
        #         # Display results
        #         future_df_pivot = future_df.pivot(index="accident_year", columns="development_year", values="incurred_amount")
        #         # future_df_pivot.head()
        #         st.dataframe(future_df_pivot)
    
        elif option == 'Future Accident Year Projections':
            st.write("Let's analyze the comparisons between the outputs of each actuarial method! Please select an option from the dropdown below:")
            claim_amount_method = ['Choose one method', 'Incurred Claim Amount', 'Paid Claim Amount']
            claim_method = st.selectbox("Select Incurred or Paid Claim Amount:", claim_amount_method)
        
            st.write("Number of Forecast Years")
            future_years = st.slider("Enter number of future years to forecast:", min_value=1, max_value=10, value=5)
        
            if claim_method == 'Incurred Claim Amount':
                df = st.session_state.uploaded_df
                incurred_triangle = data_to_triangle(df, 'incurred')
                predicted_incurred = chain_ladder_method(incurred_triangle.copy())
                predicted_incurred.reset_index(inplace=True)
                # Ensure column names are converted to strings and stripped of any whitespace
                predicted_incurred.columns = [str(col).strip() for col in predicted_incurred.columns]
        
                # Extract the accident years and reshape for regression
                accident_years = predicted_incurred['accident_year'].values.reshape(-1, 1)
        
                # Specify the number of future years you want to forecast using the value from the slider
                future_accident_years = np.arange(accident_years[-1] + 1, accident_years[-1] + 1 + future_years).reshape(-1, 1)
        
                # Dictionary to hold models and predictions
                regression_models = {}
                predictions = {}
        
                # Loop through each development year to set up regression models
                for development_year in range(1, 22):
                    development_year_str = str(development_year)  # Convert integer to string to match column names
                    if development_year_str in predicted_incurred.columns:
                        # Get the claim amounts for the current development year
                        claim_amounts = predicted_incurred[development_year_str].values.reshape(-1, 1)
                
                        # Create the regression model and fit it
                        model = LinearRegression()
                        model.fit(accident_years, claim_amounts)
                        
                        # Store the model (optional, if you need to inspect or use the model later)
                        regression_models[development_year] = model
                        
                        # Predict the future claim amounts for the specified future years
                        predicted_claims = model.predict(future_accident_years).flatten()
                        predictions[development_year] = predicted_claims
                    
                    else:
                        st.error(f"Column {development_year_str} not found in DataFrame.")
        
                # Convert the predictions into a DataFrame for better visualization and use
                predictions_df = pd.DataFrame(predictions, index=np.arange(future_accident_years[0][0], future_accident_years[0][0] + future_years))
                predictions_df.index.name = 'Future Accident Year'
                st.dataframe(predictions_df)
                plot_forecast_line(predictions_df)
                plot_forecast_heatmap(predictions_df)

            elif claim_method == 'Paid Claim Amount':
                df = st.session_state.uploaded_df
                incurred_triangle = data_to_triangle(df, 'cumpaid')
                predicted_incurred = chain_ladder_method(incurred_triangle.copy())
                predicted_incurred.reset_index(inplace=True)
                # Ensure column names are converted to strings and stripped of any whitespace
                predicted_incurred.columns = [str(col).strip() for col in predicted_incurred.columns]
        
                # Extract the accident years and reshape for regression
                accident_years = predicted_incurred['accident_year'].values.reshape(-1, 1)
        
                # Specify the number of future years you want to forecast using the value from the slider
                future_accident_years = np.arange(accident_years[-1] + 1, accident_years[-1] + 1 + future_years).reshape(-1, 1)
        
                # Dictionary to hold models and predictions
                regression_models = {}
                predictions = {}
        
                # Loop through each development year to set up regression models
                for development_year in range(1, 22):
                    development_year_str = str(development_year)  # Convert integer to string to match column names
                    if development_year_str in predicted_incurred.columns:
                        # Get the claim amounts for the current development year
                        claim_amounts = predicted_incurred[development_year_str].values.reshape(-1, 1)
                
                        # Create the regression model and fit it
                        model = LinearRegression()
                        model.fit(accident_years, claim_amounts)
                        
                        # Store the model (optional, if you need to inspect or use the model later)
                        regression_models[development_year] = model
                        
                        # Predict the future claim amounts for the specified future years
                        predicted_claims = model.predict(future_accident_years).flatten()
                        predictions[development_year] = predicted_claims
                    
                    else:
                        st.error(f"Column {development_year_str} not found in DataFrame.")
        
                # Convert the predictions into a DataFrame for better visualization and use
                predictions_df = pd.DataFrame(predictions, index=np.arange(future_accident_years[0][0], future_accident_years[0][0] + future_years))
                predictions_df.index.name = 'Future Accident Year'
                st.dataframe(predictions_df)
                plot_forecast_line(predictions_df)
                plot_forecast_heatmap(predictions_df)

    else:
        st.error("Please upload and process the dataset first!")


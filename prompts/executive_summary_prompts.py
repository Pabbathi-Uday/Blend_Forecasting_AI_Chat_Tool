# from langchain.prompts import PromptTemplate
import streamlit as st
import config
key_category = config.key_category
target_variable = config.target_variable


gold_template = f"""
# Demand Forecasting Executive Summary Report

## 1. Overview

This report provides a concise summary of the demand forecasting model’s performance, highlighting key trends, forecast discrepancies, and comparative model insights. It outlines the modeling approach and data sources, with a focus on the Light Gradient Boosting Machine (LGBM) model. The analysis is organized around three pillars: trendline insights, residual patterns, and model forecast discrepancies. For each section, we spotlight only the top two stores showing the most impactful deviations, ensuring executive relevance and clarity. The report concludes with actionable recommendations to enhance forecasting accuracy and support data-driven decision-making.

## 2. Data Sources & Modeling Approach

### 2.1 Model Overview:

--> Clearly define which model or models were used for forecasting (e.g., time series forecasting, machine learning algorithms).

--> Rationale for Model Choice: Explain the reasons for selecting the chosen model. Consider aspects like prediction accuracy, ease of interpretability, scalability for future demand forecasting, and flexibility in handling different types of input data (e.g., categorical, continuous, or time-series data).

### 2.2 Performance Metrics & Accuracy:

| Evaluation Metric | [Metric 1] | [Metric 2] | [Metric 3] |
|--------|-------|-------|-------|
| Test | [Value] | [Value] | [Value] |
| Train | [Value] | [Value] | [Value] |
| Performance Takeaways | Explain key takeaways on model performance based on the given test and train metric values. | Explain key takeaways on model performance based on the given test and train metric values. | Explain key takeaways on model performance based on the given test and train metric values. |


## 3. Key Highlights

### 3.1 Trendline Analysis

This analysis highlights key shifts in demand by comparing actual values against rolling averages to surface peaks and dips. Each month’s anomalies are reviewed with a focus on promotions, seasonal patterns, and macroeconomic factors. For clarity, we limit the analysis to the two stores with the most significant changes, providing actionable insights into what’s driving demand shifts. These findings help decision-makers align promotional strategies, inventory, and staffing with anticipated demand.

#### 3.1.1 Monthly Trend Insights

<!-- IMPORTANT: Generate a detailed monthly analysis for EACH MONTH with significant trends or outliers. Continue until ALL months are covered. For each month, follow the format below. -->

##### [Month 1] Summary
[Summarize the major events or conditions for this month (e.g., holiday effects, promotional markdowns, weather influences).]

**Top 2 {key_category} Performers:**

1. **{key_category} [Identifier 1]:** [Include this store because it showed the highest change of X% in {target_variable}]
  - **Outlier Events Overview**:
    - **Anomaly Date [Event Number]**: [Exact date]
    - **{target_variable} Change**: [Percentage change and exact numerical values relative to the 5-week rolling average or baseline]
  - **Key Takeaways**: [For this outlier event, list all relevant factors such as store size, specific promotional activities, weather conditions, economic indicators (e.g., CPI, fuel prices), and additional data points that influenced sales. For every feature, explain % change value compared to baseline. Summarize into 2 sentences]

2. **{key_category} [Identifier 2]:** [Include this store because it showed the second highest change of Y% in {target_variable}]
  - **Outlier Events Overview**:
    - **Anomaly Date [Event Number]**: [Exact date]
    - **{target_variable} Change**: [Percentage change and exact numerical values relative to the 5-week rolling average or baseline]
  - **Key Takeaways**: [For this outlier event, list all relevant factors with % change values compared to baseline. Summarize into 2 sentences]

##### [Month 2] Summary
[Summarize the major events or conditions for this month (e.g., holiday effects, promotional markdowns, weather influences).]

**Top 2 {key_category} Performers:**

1. **{key_category} [Identifier 1]:** [Include this store because it showed the highest change of X% in {target_variable}]
  - **Outlier Events Overview**:
    - **Anomaly Date [Event Number]**: [Exact date]
    - **{target_variable} Change**: [Percentage change and exact numerical values relative to the 5-week rolling average or baseline]
  - **Key Takeaways**: [For this outlier event, list all relevant factors such as store size, specific promotional activities, weather conditions, economic indicators (e.g., CPI, fuel prices), and additional data points that influenced sales. For every feature, explain % change value compared to baseline. Summarize into 2 sentences]

2. **{key_category} [Identifier 2]:** [Include this store because it showed the second highest change of Y% in {target_variable}]
  - **Outlier Events Overview**:
    - **Anomaly Date [Event Number]**: [Exact date]
    - **{target_variable} Change**: [Percentage change and exact numerical values relative to the 5-week rolling average or baseline]
  - **Key Takeaways**: [For this outlier event, list all relevant factors with % change values compared to baseline. Summarize into 2 sentences]
...
...
...
...
##### [Month n] Summary
[Summarize the major events or conditions for this month (e.g., holiday effects, promotional markdowns, weather influences).]

**Top 2 {key_category} Performers:**

1. **{key_category} [Identifier 1]:** [Include this store because it showed the highest change of X% in {target_variable}]
  - **Outlier Events Overview**:
    - **Anomaly Date [Event Number]**: [Exact date]
    - **{target_variable} Change**: [Percentage change and exact numerical values relative to the 5-week rolling average or baseline]
  - **Key Takeaways**: [For this outlier event, list all relevant factors such as store size, specific promotional activities, weather conditions, economic indicators (e.g., CPI, fuel prices), and additional data points that influenced sales. For every feature, explain % change value compared to baseline. Summarize into 2 sentences]

2. **{key_category} [Identifier 2]:** [Include this store because it showed the second highest change of Y% in {target_variable}]
  - **Outlier Events Overview**:
    - **Anomaly Date [Event Number]**: [Exact date]
    - **{target_variable} Change**: [Percentage change and exact numerical values relative to the 5-week rolling average or baseline]
  - **Key Takeaways**: [For this outlier event, list all relevant factors with % change values compared to baseline. Summarize into 2 sentences]


### 3.2 Residual Analysis

"This analysis identifies outlier residuals that exceed a predefined threshold, with a focus on months where these anomalies occur. We use average Shapley values as the primary baseline to pinpoint influential features contributing to these discrepancies and compare them across the dataset. For months with significant anomalies, we provide a detailed monthly analysis highlighting the two top-performing stores, examining key factors like store size, weather conditions, and economic indicators."

#### 3.2.1 Monthly Residual Insights

<!-- IMPORTANT: Generate a detailed monthly analysis for EACH MONTH with significant residuals. Continue until ALL months are covered. For each month, follow the format below. -->

##### [Month 1] Summary
[Summarize the major events or conditions that influenced sales this month (e.g., holiday effects, promotional markdowns, weather influences) and explain how they contributed to forecast residuals.]

**Top 2 {key_category} with Highest Residuals:**

1. **{key_category} [Identifier 1]:** [Include this store because it showed the highest residual of X% between actual and forecast]
  - **Weekly Residual Change ([Specific Week Date]):** [Specify the difference between the actual and predicted values]
  - **Feature Contribution Analysis ([Specific Week Date]):**
    - **[Feature 1]:** Feature value of [value] with a Shapley value of [value], [compare to average Shapley value and explain significance]
    - **[Feature 2]:** Feature value of [value] with a Shapley value of [value], [compare to average Shapley value and explain significance]
    - **[Feature 3]:** Feature value of [value] with a Shapley value of [value], [compare to average Shapley value and explain significance]
  - **Key Takeaways ([Specific Week Date]):**
    - [Quantify differences between actual Shapley values and averages]
    - [Explain which features had outsized influence on forecast output]
    - [Provide key business takeaways from this analysis]

2. **{key_category} [Identifier 2]:** [Include this store because it showed the second highest residual of Y% between actual and forecast]
  - **Weekly Residual Change ([Specific Week Date]):** [Specify the difference between the actual and predicted values]
  - **Feature Contribution Analysis ([Specific Week Date]):**
    - **[Feature 1]:** Feature value of [value] with a Shapley value of [value], [compare to average Shapley value and explain significance]
    - **[Feature 2]:** Feature value of [value] with a Shapley value of [value], [compare to average Shapley value and explain significance]
    - **[Feature 3]:** Feature value of [value] with a Shapley value of [value], [compare to average Shapley value and explain significance]
  - **Key Takeaways([Specific Week Date]):**
    - [Quantify differences between actual Shapley values and averages]
    - [Explain which features had outsized influence on forecast output]
    - [Provide key business takeaways from this analysis]

##### [Month 2] Summary
[Summarize the major events or conditions that influenced sales this month (e.g., holiday effects, promotional markdowns, weather influences) and explain how they contributed to forecast residuals.]

**Top 2 {key_category} with Highest Residuals:**

1. **{key_category} [Identifier 1]:** [Include this store because it showed the highest residual of X% between actual and forecast]
  - **Weekly Residual Change ([Specific Week Date]):** [Specify the difference between the actual and predicted values]
  - **Feature Contribution Analysis ([Specific Week Date]):**
    - **[Feature 1]:** Feature value of [value] with a Shapley value of [value], [compare to average Shapley value and explain significance]
    - **[Feature 2]:** Feature value of [value] with a Shapley value of [value], [compare to average Shapley value and explain significance]
    - **[Feature 3]:** Feature value of [value] with a Shapley value of [value], [compare to average Shapley value and explain significance]
  - **Key Takeaways ([Specific Week Date]):**
    - [Quantify differences between actual Shapley values and averages]
    - [Explain which features had outsized influence on forecast output]
    - [Provide key business takeaways from this analysis]

2. **{key_category} [Identifier 2]:** [Include this store because it showed the second highest residual of Y% between actual and forecast]
  - **Weekly Residual Change ([Specific Week Date]):** [Specify the difference between the actual and predicted values]
  - **Feature Contribution Analysis ([Specific Week Date]):**
    - **[Feature 1]:** Feature value of [value] with a Shapley value of [value], [compare to average Shapley value and explain significance]
    - **[Feature 2]:** Feature value of [value] with a Shapley value of [value], [compare to average Shapley value and explain significance]
    - **[Feature 3]:** Feature value of [value] with a Shapley value of [value], [compare to average Shapley value and explain significance]
  - **Key Takeaways([Specific Week Date]):**
    - [Quantify differences between actual Shapley values and averages]
    - [Explain which features had outsized influence on forecast output]
    - [Provide key business takeaways from this analysis]
...
...
...
##### [Month n] Summary
[Summarize the major events or conditions that influenced sales this month (e.g., holiday effects, promotional markdowns, weather influences) and explain how they contributed to forecast residuals.]

**Top 2 {key_category} with Highest Residuals:**

1. **{key_category} [Identifier 1]:** [Include this store because it showed the highest residual of X% between actual and forecast]
  - **Weekly Residual Change ([Specific Week Date]):** [Specify the difference between the actual and predicted values]
  - **Feature Contribution Analysis ([Specific Week Date]):**
    - **[Feature 1]:** Feature value of [value] with a Shapley value of [value], [compare to average Shapley value and explain significance]
    - **[Feature 2]:** Feature value of [value] with a Shapley value of [value], [compare to average Shapley value and explain significance]
    - **[Feature 3]:** Feature value of [value] with a Shapley value of [value], [compare to average Shapley value and explain significance]
  - **Key Takeaways ([Specific Week Date]):**
    - [Quantify differences between actual Shapley values and averages]
    - [Explain which features had outsized influence on forecast output]
    - [Provide key business takeaways from this analysis]

2. **{key_category} [Identifier 2]:** [Include this store because it showed the second highest residual of Y% between actual and forecast]
  - **Weekly Residual Change ([Specific Week Date]):** [Specify the difference between the actual and predicted values]
  - **Feature Contribution Analysis ([Specific Week Date]):**
    - **[Feature 1]:** Feature value of [value] with a Shapley value of [value], [compare to average Shapley value and explain significance]
    - **[Feature 2]:** Feature value of [value] with a Shapley value of [value], [compare to average Shapley value and explain significance]
    - **[Feature 3]:** Feature value of [value] with a Shapley value of [value], [compare to average Shapley value and explain significance]
  - **Key Takeaways([Specific Week Date]):**
    - [Quantify differences between actual Shapley values and averages]
    - [Explain which features had outsized influence on forecast output]
    - [Provide key business takeaways from this analysis]

### 3.3 Forecast Discrepancy Analysis

"The purpose of this analysis is to identify weeks (time index granularity level) in which there are significant differences in the forecasts of each model for the same comparison time period. Based on these common weeks, we want to use the underlying shapley values to understand what might be causing this difference when this model is significantly different from the previous model. For months with significant differences, we provide a detailed monthly analysis highlighting the two top-performing stores, examining key factors like store size, weather conditions, and economic indicators."

#### 3.3.1 Monthly Forecast Discrepancy Insights

<!-- IMPORTANT: Generate a detailed monthly analysis for EACH MONTH with significant forecast discrepancies. Continue until ALL months are covered. For each month, follow the format below. -->

##### [Month 1] Summary
[Summarize the major events or conditions for this month (e.g., holiday effects, promotional markdowns, weather influences).]

**Top 2 {key_category} with Highest Forecast Discrepancies:**

1. **{key_category} [Identifier 1]:** [Include this store because it showed the highest forecast discrepancy of X% between models]
  - **Forecast Discrepancy ([Specific Week Date]):** [Specify the difference in forecast between the two models. Explain what this means in terms of forecast discrepancy]
  - **Feature Contributions:**
    - [Feature 1]: [Compare the SHAP values for this feature in both models. Explain why the models might prioritize this feature differently]
    - [Feature 2]: [Compare the SHAP values for this feature in both models. Explain why the models might prioritize this feature differently]
    - [Feature 3]: [Compare the SHAP values for this feature in both models. Explain why the models might prioritize this feature differently]
  - **Key Takeaways:**
    - [Explain WHY forecast difference appears in the models and WHY only these features show heavier differences between both models]
    - [Provide key business takeaways of model interpretation based on this analysis]

2. **{key_category} [Identifier 2]:** [Include this store because it showed the second highest forecast discrepancy of Y% between models]
  - **Forecast Discrepancy ([Specific Week Date]):** [Specify the difference in forecast between the two models. Explain what this means in terms of forecast discrepancy]
  - **Feature Contributions:**
    - [Feature 1]: [Compare the SHAP values for this feature in both models. Explain why the models might prioritize this feature differently]
    - [Feature 2]: [Compare the SHAP values for this feature in both models. Explain why the models might prioritize this feature differently]
    - [Feature 3]: [Compare the SHAP values for this feature in both models. Explain why the models might prioritize this feature differently]
  - **Key Takeaways:**
    - [Explain WHY forecast difference appears in the models and WHY only these features show heavier differences between both models]
    - [Provide key business takeaways of model interpretation based on this analysis]

##### [Month 2] Summary
[Summarize the major events or conditions for this month (e.g., holiday effects, promotional markdowns, weather influences).]

**Top 2 {key_category} with Highest Forecast Discrepancies:**

1. **{key_category} [Identifier 1]:** [Include this store because it showed the highest forecast discrepancy of X% between models]
  - **Forecast Discrepancy ([Specific Week Date]):** [Specify the difference in forecast between the two models. Explain what this means in terms of forecast discrepancy]
  - **Feature Contributions:**
    - [Feature 1]: [Compare the SHAP values for this feature in both models. Explain why the models might prioritize this feature differently]
    - [Feature 2]: [Compare the SHAP values for this feature in both models. Explain why the models might prioritize this feature differently]
    - [Feature 3]: [Compare the SHAP values for this feature in both models. Explain why the models might prioritize this feature differently]
  - **Key Takeaways:**
    - [Explain WHY forecast difference appears in the models and WHY only these features show heavier differences between both models]
    - [Provide key business takeaways of model interpretation based on this analysis]

2. **{key_category} [Identifier 2]:** [Include this store because it showed the second highest forecast discrepancy of Y% between models]
  - **Forecast Discrepancy ([Specific Week Date]):** [Specify the difference in forecast between the two models. Explain what this means in terms of forecast discrepancy]
  - **Feature Contributions:**
    - [Feature 1]: [Compare the SHAP values for this feature in both models. Explain why the models might prioritize this feature differently]
    - [Feature 2]: [Compare the SHAP values for this feature in both models. Explain why the models might prioritize this feature differently]
    - [Feature 3]: [Compare the SHAP values for this feature in both models. Explain why the models might prioritize this feature differently]
  - **Key Takeaways:**
    - [Explain WHY forecast difference appears in the models and WHY only these features show heavier differences between both models]
    - [Provide key business takeaways of model interpretation based on this analysis]
...
...
...
##### [Month n] Summary
[Summarize the major events or conditions for this month (e.g., holiday effects, promotional markdowns, weather influences).]

**Top 2 {key_category} with Highest Forecast Discrepancies:**

1. **{key_category} [Identifier 1]:** [Include this store because it showed the highest forecast discrepancy of X% between models]
  - **Forecast Discrepancy ([Specific Week Date]):** [Specify the difference in forecast between the two models. Explain what this means in terms of forecast discrepancy]
  - **Feature Contributions:**
    - [Feature 1]: [Compare the SHAP values for this feature in both models. Explain why the models might prioritize this feature differently]
    - [Feature 2]: [Compare the SHAP values for this feature in both models. Explain why the models might prioritize this feature differently]
    - [Feature 3]: [Compare the SHAP values for this feature in both models. Explain why the models might prioritize this feature differently]
  - **Key Takeaways:**
    - [Explain WHY forecast difference appears in the models and WHY only these features show heavier differences between both models]
    - [Provide key business takeaways of model interpretation based on this analysis]

2. **{key_category} [Identifier 2]:** [Include this store because it showed the second highest forecast discrepancy of Y% between models]
  - **Forecast Discrepancy ([Specific Week Date]):** [Specify the difference in forecast between the two models. Explain what this means in terms of forecast discrepancy]
  - **Feature Contributions:**
    - [Feature 1]: [Compare the SHAP values for this feature in both models. Explain why the models might prioritize this feature differently]
    - [Feature 2]: [Compare the SHAP values for this feature in both models. Explain why the models might prioritize this feature differently]
    - [Feature 3]: [Compare the SHAP values for this feature in both models. Explain why the models might prioritize this feature differently]
  - **Key Takeaways:**
    - [Explain WHY forecast difference appears in the models and WHY only these features show heavier differences between both models]
    - [Provide key business takeaways of model interpretation based on this analysis]

##### Model Preference Summary
<!-- At the end of this section, provide a concise summary comparing the tendencies of Model 1 vs. Model 2 across all months. Identify whether one model is more sensitive to short-term factors (e.g., promotions, recent sales) and the other to long-term macroeconomic variables (e.g., CPI, unemployment). -->

<!-- Recommend when each model should be used (e.g., tactical planning vs. strategic forecasting). Mention the need for periodic review of model feature sensitivity based on business priorities. -->



## 4. Summary of Findings

Based on our comprehensive analysis of the demand forecasting model, we have identified the following key insights for each month and store:

- **Trendline Analysis**:
  [For each [MONTH], provide detailed insights by identifying significant trends at both the store and overall business level. For each insight: ]
    - [Specify the **Store ID** and the **date(s)** where notable changes occurred.]
    - [Detail the **percentage change in sales** and identify the key drivers behind the fluctuation (e.g., weather patterns, store size, promotions).]
    - [Clearly explain how these factors influenced the observed trend and .  ]
    - [Where relevant, summarize overarching trends that apply across multiple stores. ]

- **Residual Analysis**:
 [For each [MONTH], provide store-specific insights by identifying the features that contributed most to residual errors. For each insight:]
    - [ Specify the **Store ID** and describe the primary contributors to the error by comparing Shapley values to their dataset-wide averages.]
    - [ Explain the **percentage deviation** and the driving factors — internal (e.g., promotions) or external (e.g., weather).  ]
    - [ Clearly connect these factors to their impact on the forecast for {target_variable}.]

- **Forecast Discrepancies**:
  [For each [MONTH], provide detailed insights by identifying discrepancies between Model 1 and Model 2. For each insight: ]
    - [ Specify the **Store ID** and describe the key features that influenced the discrepancy.  ]
    - [ Explain how these features behaved differently in the two models, highlighting the underlying causes (e.g., inventory strategies, demand shocks). ]
    - [ Quantify the business impact (e.g., lost sales, inventory shortages). ]

[ Ensure each insight is concrete, precise, and directly linked to observed data. Focus on delivering actionable insights that business stakeholders can use to drive strategic decisions.]


## 5. Recommendations & Next Steps

Based on our three analyses, we recommend the following specific actions:

### From Trendline Analysis:
1. **[Specific recommendation based on outlier events and provide actionable recommendations to improve decision-making based on detailed insights in trendline analysis]**: [Explanation of how this recommendation addresses specific findings from trendline analysis]
2. **[Specific recommendation based on influencing factors]**: [Explanation of how this recommendation addresses specific findings from trendline analysis]

### From Residual Analysis:
1. **[Specific recommendation based on Shapley value insights and provide actionable steps]**: [Explanation of how this recommendation addresses specific findings from residual analysis]
2. **[Specific recommendation based on feature contribution analysis]**: [Explanation of how this recommendation addresses specific findings from residual analysis]

### From Forecast Discrepancy Analysis:
1. **[Specific actionable recommendation for Business Stakeholders]**: [Provide explanation of how to use insights from forecast discrepancies to inform business planning in areas like marketing, inventory, and operations and provide actionable recommendations for improving future forecasts]
2. **[Specific recommendation for model improvement]**: [Explanation of how to recalibrate model sensitivity to specific features, aiming to bring results closer between models]
"""

def generate_prompt(gold_template, business_scenario, data_dict, model_description_summary,
                    forecast_analysis_summary, trendline_analysis_summary, residual_analysis_summary):
    
    prompt = f"""
            # Role
            As a Master Data Analyst with over 10 years of experience specializing in Forecasting Models and Feature
            Interpretability, you are an exceptional expert with unparalleled proficiency in following detailed
            instructions to perform precise analysis and complete reports. You excel at extracting meaningful insights from raw
            model data, using provided dictionaries and existing summaries to craft thorough, accurate executive business
            reports for stakeholders.
            With a deep business understanding of how individual features influence model performance, you excel in applying
            your expertise to fill structured templates, drawing actionable conclusions from the analysis
            of both predictive accuracy and feature impact.


            # Context
            You are provided with a business scenario - {business_scenario}. Based on this scenario, you have performed
            all three types of analysis - forecast discrepancy analysis, trendline analysis and residual analysis.
            Your team of business stakeholders have asked you for a detailed executive summary report. They have provided
            you with a gold template that you MUST strictly follow.
            In this gold template, your team has provided you with specific instructions that will guide you to map the right
            information in the right section of the gold tempalte. You must consider these instructions to be the top priority
            when you fill the template with executive summaries from each of the reports, model description summary and business
            scenario.

            # Feature Name Translation
            IMPORTANT: In all summaries and analyses, the original feature names are used. However, you MUST translate these
            to their more descriptive alternatives in your final report. Use this translation table to convert all feature
            references:

            | Original Feature Name | Use This Clearer Feature Name Instead |
            |----------------------|---------------------|
            | Store_ID | Store Identifier |
            | Week_Date | Week Start Date |
            | weekly_sales_lag_1w | Previous Week Sales |
            | weekly_sales_lag_52w | One Year Prior Sales |
            | weekly_sales_lag_104w | Two Years Prior Sales |
            | Temperature | Average Temperature |
            | Fuel_Price | Regional Fuel Price |
            | MarkDown1 | Promotional Discount 1 |
            | MarkDown2 | Promotional Discount 2 |
            | MarkDown3 | Promotional Discount 3 |
            | MarkDown4 | Promotional Discount 4 |
            | MarkDown5 | Promotional Discount 5 |
            | CPI | Consumer Price Index |
            | Unemployment | Regional Unemployment Rate |
            | IsHoliday | Holiday Week Indicator |
            | Type_A | Store Category A Flag |
            | Type_B | Store Category B Flag |
            | Type_C | Store Category C Flag |
            | Size | Store Square Footage |
            | year | Calendar Year |
            | week_of_month | Week Position in Month |
            | week_of_year | Week Number in Year |

            For example, if you see "The Store_ID feature showed strong correlation with sales" in the summaries,
            you MUST write "The Store Identifier feature showed strong correlation with sales" in your report.

            # Input Parameters
            {gold_template} - This is the exec summary report gold template that you need to fill using all three summaries
            provided - forecast analysis summary, trendline analysis summary, residual analysis summary, business scenario and
            model description.
            It has different sections such as Overview, Data & Models Used, Key Highlights, Summary of Findings, Recommendations
            & Next Steps and Conclusion. Each section has specific instructions to guide you on what information to fill in each
            part based on the provided summaries.

            {data_dict} - This data dictionary contains detailed definition, unit and interpretation for each column in the data
            used to build both demand forecasting models. It also contains the definition of
            of shapley value contributions of each feature to help understand how each feature impacts each model's predictions.
            Use this dictionary to better understand what each feature stands for. This dictionary will help you build meaningful
            context in your final response.

            {business_scenario} - This is the business scenario which outlines details about the business, challenge and solution
            that they are implementing. The solution provides a high level overview on what analysis has been done.

            {model_description_summary} - This is a detailed model description summary highlighting model performance metrics,
            feature importance, correlation information, etc.
            """
    if forecast_analysis_summary is not None:
        prompt += f"""          
            {forecast_analysis_summary} - This analysis summary covers the below points:
               - Analyze forecast discrepancies between two demand models.
               - Focus on economic indicators, promotions, and store-specific factors.
               - Use SHAP values to identify key feature impacts.
               - Compare feature sensitivity for better model calibration.
               - Provide insights for improving forecast accuracy and strategies.
        """
            
    prompt+= f"""
            {trendline_analysis_summary} - This analysis summary covers the below points:
               - Analyze significant sales fluctuations observed during key time periods across multiple locations.
               - Investigate the impact of various factors such as seasonal trends, external economic conditions, and store-specific characteristics on sales performance.
               - Provide insights to enhance business strategies and improve performance during peak retail periods.
               - Offer recommendations to optimize operational strategies, including inventory management and promotional activities.
               - Support future decision-making by identifying patterns and variables that influence sales trends.

            {residual_analysis_summary} - This analysis summary covers the below points:
               - Analyze sales anomalies during peak seasons.
               - Identify key factors affecting retail performance.
               - Provide insights for future strategic planning.
               - Guide inventory and promotional decision-making.

            # Step by Step Guidelines
            Let's think about this step-by-step:
            1) Start with going through the gold template in detail. Understand the instructions for each section clearly.
            2) Think about which summary is needed to fill in a particular section from the gold template. Start from the very
               first section - Overview, all the way to the final section - Conclusion. Build every section to be very detailed
               and covers all interesting insights. Summarize these insights well, considering all nuances. Each section
               MUST BE at-least 5-6 sentences long.
            3) Keep in mind that your final audience is a team of business stakeholders who will use the insights and recommendations
               you provide to make business decisions.
            4) Perform factual checks to ensure that the insights are strongly rooted in the content of the provided input
               summaries.
            5) For every month analysis, go as detailed as you can. Include AT-LEAST 8-9 sentences for each month within each
               analysis section in the gold template. Strongly emphasize the key factors influence the target variable for each
               analysis in each month and justify how these factors play a role in influencing the model's performance.
               If you do not follow this, you will receive a penalty.
            6) You must follow the gold template very closely. Once you have finished populating all sections in the gold template,
               return the completed exec summary gold template as your response.
            7) MOST IMPORTANT: As you write your report, translate ALL original feature names to their clearer alternatives using
               the feature name translation table provided above. NEVER use the original technical feature names in your final report.

            # Important Gold Template Instructions:
            1) Make sure you follow the exec summary gold template very closely and include all sections from the template.

            # Output
            Provide your response as the completed detailed exec summary template.

            # Key Conditions
            1) Articulate your response in a narrative manner, clear and easy to understand.
            2) Build the report to be very detailed in nature. Cover each and every nuance in the input summaries.
            3) Each section in the gold template must be atleast 7-8 sentences long.
            4) Use ONLY the clearer feature names (like "Store Identifier" instead of "Store_ID") throughout the entire report.

            Please provide your completed detailed executive summary report below:

    """
    return prompt


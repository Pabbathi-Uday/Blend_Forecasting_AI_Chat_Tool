#-----------------Trendline Prompt---------------------------------
gold_template = """
## Retail Sales Trendline Analysis Report

### Introduction
[Provide a concise overview of the analysis, including the time periods covered and key drivers of sales anomalies (e.g., holidays, economic conditions, {key_category} attributes). Set the stage for the detailed breakdown.]

### Monthly Trends Summary

#### [Month Year] Summary
[Summarize the major events or conditions that influenced trends this month (e.g., holiday effects, promotional markdowns, weather influences) and explain how they contributed to observed high and low points. Provide a high-level overview of the most significant feature-level changes by analyzing actual feature values and their impact on trend shifts.
Identify overarching patterns in how different features contributed to fluctuations over time. Provide integrated insights on the combined impact of all {key_category}-level data, highlighting key drivers behind trend deviations and summarizing how variations in feature-level impacts explain changes in the observed trend. Summarize the overall model performance and focus on the key drivers of deviations at the feature level
]

For every {key_category} — including cases where the same store has multiple outlier events on different dates — provide a detailed analysis as follows:

**{key_category} [Identifier] Analysis:**

- **Outlier Events Overview**: If multiple outlier events occurred for this store in a month, list each event by date. For each date, specify:
  - **Anomaly Date [GIVE THE EVENT NUMBER HERE INCASE OF MULTIPLE OUTLIERS]**: [Exact date] - GIVE ONLY DATE, DONT CALL IT WEEK OF ....
  - **{pred_var} Change**: [Percentage change and exact numerical values relative to the 5-week rolling average or baseline.]

If THERE IS A SINGLE OUTLIER EVENT FOR THIS STORE FOR THIS MONTH:
- **Primary Influencing Factors**: For each outlier event, list all relevant factors such as store size, specific promotional activities, weather conditions,
economic indicators (e.g., CPI, fuel prices), and any additional data points that influenced sales
AND MAKE SURE THAT, FOR EVERY FEATURE EXPLAIN % CHANGE VALUE COMPARED TO BASELINE, DONT GIVE ACTUAL VALUE OF BASELINE. Summarize into 2 sentences.

IF THERE ARE MULTIPLE OUTLIER EVENTS FOR THIS STORE FOR THIS MONTH:
- **Primary Influencing Factors (Common)**: For ALL the outlier events, list all COMMON relevant factors such as store size, specific promotional activities,
weather conditions, economic indicators (e.g., CPI, fuel prices), and any additional data points that influenced sales
AND MAKE SURE THAT, FOR EVERY FEATURE EXPLAIN % CHANGE VALUE COMPARED TO BASELINE, DONT GIVE ACTUAL VALUE OF BASELINE. Summarize into 2 sentences.
- **Primary Influencing Factors (Unique)**: For ALL the outlier events, list all (TOTAL-COMMON) relevant factors such as store size, specific promotional activities,
weather conditions, economic indicators (e.g., CPI, fuel prices), and any additional data points that influenced sales
AND MAKE SURE THAT, FOR EVERY FEATURE EXPLAIN % CHANGE VALUE COMPARED TO BASELINE, DONT GIVE ACTUAL VALUE OF BASELINE. Summarize into 2 sentences.

- **Insights**: Explain which features had an outsized influence on the forecast output for this {key_category} and what should be the key business takeaways of model interpretation based on this analysis, in a paragraph in detail.

[Repeat the above section for every {key_category} present in all week dates in this month.]
[**Ensure that every {key_category} from each month, including multiple different week dates for the same {key_category}, is included and compared.**]

### Business Implications and Strategic Recommendations
[Translate the aggregated insights into actionable business strategies. Discuss operational improvements such as inventory management, staffing adjustments, or promotional planning. Ensure recommendations are supported by specific data points from the analysis.]

### Conclusion
[Summarize the key takeaways from the report, emphasizing how the comprehensive monthly and {key_category}-specific insights contribute to a broader understanding of sales anomalies. Reinforce the strategic recommendations for future business operations.]
"""

map_prompt_template = """
        # Role:
        As an Expert Data Analyst specializing in forecasting models and feature interpretability, you possess deep expertise in time-series analysis,
        anomaly detection, and understanding the underlying drivers behind sales data deviations. Your proficiency enables you to quantitatively dissect how various features contribute to outliers, allowing you to pinpoint the causes of unexpected sales spikes or drops.

        # Input Parameters:
        {row_data} - This JSON object contains actual weekly sales count (the target variable), its corresponding rolling mean across the last 5 weeks (moving average) as a historical benchmark, and other dependent variables such as store size, fuel price, markdown values, economic indicators (CPI, Unemployment), holiday flags (IsHoliday), store type classifications, and lagged sales figures. Use this data to complete your task. You want to understand the reason why there is a high or low value in the target "y" variable based on the rolling mean values of the target and individual feature values.

        {data_dict} - This data dictionary provides comprehensive definitions, units, and interpretations for each variable in {row_data}. Leverage this information to accurately understand the real-world significance of each feature and how its deviation from historical trends may have contributed to the target sales count anomaly.

        # Task:
        Your task is to provide a highly detailed and quantitative analysis summary explaining the significantly high or low value in the target variable (weekly sales) compared to its rolling mean. You must do this by using the underlying features and their rolling means.
        You must use the rolling mean as a reference to identify the anomaly and then dissect if and how a specific set of features have deviated from their expected values (which should be close to the rolling mean average values for each feature), thereby contributing to the significantly different value in the target variable.

        # Step-by-Step Guidelines:
        1. **Introduction: Context & Objective** - Start with an introduction on what you observe in this row, for a specific Store ID and week date - with a particular focus on how the target variable differs from the rolling mean target value of the previous 5-week rolling mean (serving as a benchmark).

        2. **Identify Significant Deviations** - Perform a feature-to-rolling-mean feature comparison to understand which features have their value above or below the rolling mean value for the same feature and quantify the difference.
            - Compare each feature's value against its rolling mean counterpart to determine the extent of deviation.
            - Identify which features contribute the most to the significantly high or low target value by analyzing their variation from their rolling mean values.
            - Explain the expected normal behavior of each feature using its rolling mean as a reference point.
            - Analyze how the actual feature values deviate from this expected behavior and how this may contribute to the significantly high or low value of the actual target variable.
            - Quantify how far each feature deviated from its rolling mean (e.g., percentage difference or z-score).
            - If a feature deviated notably, explain how it could have influenced higher or lower sales.

        3. **Sensitivity Analysis** - Evaluate the impact of key feature deviations on the target variable.
            - Assess how changes in specific features (e.g., a higher price discount or a temperature drop) may have impacted total weekly sales count compared to the expected rolling mean.
            - Determine if the anomalies are a result of one or multiple factors and analyze their interactions.

        4. **Summary of Key Findings** - Conclude by synthesizing the analysis.
            - Highlight the primary feature deviations and their impact on the observed anomaly in weekly sales.
            - Avoid providing recommendations or actionable insights—focus on a clear and precise explanation of the deviation drivers.

        # Output:
        Provide your response in 2 paragraphs:
        - **Paragraph 1**: Introduction that sets the context and overview of the analysis.
        - **Paragraph 2**: Detailed feature-by-feature analysis with quantitative insights.

        # Important Tips:
        1) Always use the exact numbers provided, as mentioned in the input parameters. You may round these numbers if necessary, but do not truncate or make up any values.
        2) Ensure that your summary is concise, clear, and insightful, with each sentence contributing a distinct piece of analysis that collectively explains the observed anomaly.
        3) **Data Dictionary Usage**: Always use the data dictionary to interpret each feature. Understand how each feature is defined (e.g., whether it’s a lagged value or a seasonal effect).
        4) **Clear Presentation**: Keep the analysis structured and organized.
        5) **Use the right units** when discussing results.
        6) **Do not include section headers in the output. Only provide the two paragraphs based on the Output Format section above.**
"""


# Updated Reduce Prompt Template: Added explicit instructions to preserve all details.
reduce_prompt_template = """
# Role:
You are a highly skilled data analyst with deep expertise in synthesizing detailed sales analyses from multiple {key_category}. Your role is to merge refined individual summaries into one cohesive, aggregated report by populating the provided gold template. You excel at integrating quantitative insights and grouping similar trends to guide actionable business strategies.

# Task:
Your task is to synthesize the two refined summaries into a single unified report by populating the provided gold template.
In doing so, you must ensure that no monthly or store-level details are lost. If one summary contains details for a month, store, or specific date that the other does not, include all such unique details in your final report.

# Input:
- **{summary1}**: Refined Summary 1 - A refined summary that offers a comprehensive narrative of a specific outlier event in a {key_category} on a particular day, detailing the underlying causes, quantitative metrics (e.g., changes in {pred_var}), and relevant metadata.
- **{summary2}**: Refined Summary 2 - Another refined summary that provides an in-depth analysis of a different outlier event in a {key_category} on a particular day, detailing the underlying causes, quantitative metrics (e.g., changes in {pred_var}), and relevant metadata.
- **{gold_template}**: The detailed gold template outlining the structure for the final report, including sections such as Introduction, Monthly Trends Summary, Business Implications, and Conclusion. You must strictly adhere to this template.
- **{data_dict}**: A data dictionary containing detailed definitions, units, and interpretations for each feature used in the forecasting models. Utilize this to accurately interpret quantitative metrics and feature contributions.
- **{key_category}**: The primary identifier for analysis (e.g., Store ID, Product Category).
- **{pred_var}**: The performance metric under analysis (e.g., Weekly Sales, Monthly Revenue).

# Step-by-Step Guidelines
Let's go through this process step-by-step:
1- Review Existing Summary 1 – Go through the gold template and understand the questions and sections present. Identify the relevant context and answers needed for each section in Summary 1. Use Summary 1 to fill in the gold template, reaching an “intermediate stage” where it's partially filled with information from Summary 1, but not yet completed with Summary 2. Ensure you include key metadata, such as {key_category} and all associated dates, from Summary 1.
2- Review Existing Summary 2 – Go through the "intermediate stage" gold template filled with Summary 1. Familiarize yourself with the original, unfilled sections and questions in the gold template, and identify where answers and relevant context from Summary 2 are needed. Use Summary 2 to add answers and fill in these sections in the "intermediate stage" gold template. Once you have completed this, the template will reach the "complete stage," fully filled with answers and context from both Summary 1 and Summary 2.
3- Ensure the “Complete Stage” Format – Verify that the "complete stage" template matches the expected format of the original gold template.
4- Answer Accuracy – Make sure each section in the "complete stage" template thoroughly answers the questions and fills out the sections as defined in the original gold template.
5- Update Monthly Summary Section – For the Monthly Summary section, which focuses on summarizing overall sales performance and key drivers of deviations at the feature level, ensure it is updated based on both Summary 1 and Summary 2.

# Important Tips:
YOU MUST FOLLOW THE BELOW INSTRUCTIONS STRICTLY, OTHERWISE YOU WILL RECEIVE A PENALTY.
- **Preservation of Details:** Ensure that every monthly summary and every store-level detail from both summaries is retained in the final report. If one summary includes a date or a store not mentioned in the other, include that information in your final aggregation.
- **Strict Template Adherence:** Follow the structure provided by the gold template precisely. Do not omit or overwrite details; rather, merge them so that all insights are represented.
- **Comprehensive Merging:** When encountering overlapping or non-overlapping data across the summaries, synthesize the information to provide a unified narrative that covers all data points without duplication or loss.
- **Data Dictionary Usage:** Always use the data dictionary to interpret each feature. Understand how each feature is defined (e.g., whether it’s a lagged value or a seasonal effect).
- **Complete Date Capture:** For every {key_category}, YOU MUST CAPTURE ALL DATES CORRECTLY. This includes scenarios where a month and a {key_category} combination has more than two dates (for example, three or four or more); every date must be included in the final report.

# Output Structure:
The final report must integrate insights from all summaries without missing any data points (especially monthly {key_category} details) and provide clear, actionable strategic recommendations, with specific details (dates, store numbers, numerical values) supporting all insights.
"""
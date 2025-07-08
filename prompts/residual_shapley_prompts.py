gold_template =  """
## Retail Sales Residual Shapley Analysis Report

  ### Introduction
  [Provide a concise overview that introduces the analysis. Specify the time periods and {key_category} covered and highlight the key drivers of residual errors using Shapley values. Clearly explain how Shapley values help quantify the contribution of each feature (e.g., holidays, economic conditions, {key_category} attributes) to the forecast errors. Set the stage for a detailed breakdown by emphasizing the importance of analyzing Shapley values to understand why forecasts differ from actual sales.]

  ### Monthly Residual Analysis Summary

  <Follow the below format to outline details about Month Year summary for each month>

  #### [Month Year] Summary
  [Summarize the major events or conditions that influenced sales this month (e.g., holiday effects, promotional markdowns, weather influences) and explain how they contributed to forecast residuals.
  Provide a high-level overview of the most significant feature-level differences by comparing each feature’s Shapley contribution to its dataset-wide average for this month. Provide integrated insights explaining the combined impact of all the
  {key_category}-level data. Provide an integrated insight that explains the combined impact of the {key_category}-level findings, identifying overarching trends in how different features' Shapley contributions vary over time and how these variations explain residual errors. Summarize the key feature-level differences in Shapley values and how they explain variations in the forecast accuracy.
 ]


    ##### Shapley Value Contribution Analysis

    For every {key_category}, analyze the residual changes and feature contributions using Shapley values. Analyze the Shapley contributions of key features across all weeks and provide a detailed explanation of how these feature-level variations influenced forecast residuals.
    <Follow the below format to outline the residual changes and feature contributions using Shapley values for all week dates within the specific affected {key_category}>

    - **{key_category}:** [Provide the unique identifier for the {key_category}, e.g., store, region, product category, etc.]

      - **Weekly Residual Change ([Specific Week Date 1]):** [Specify the difference between the actual and predicted values for the {key_category}.],
      - **Feature Contribution Analysis ([Specific Week Date 1]):**
        - For each feature, include both the **feature value** (e.g., for feature `X`, the value might be 98.10, or for feature `Y`, it might be 1,000,000, etc) and its corresponding **Shapley value**.
        - Analyze the feature-level Shapley values for this {key_category} and compare them against the dataset-wide average Shapley values.
        - Identify features where the Shapley contribution in this instance significantly differs from its average Shapley value.
        - **Order features by their Shapley value contributions**, from most to least influential.
      - **Insights** ([Specific Week Date 1]):
          - Quantify these differences and explain how they influenced the forecast residual (e.g., a +0.25 increase in Shapley value compared to the average).
          - Explain which features had an outsized influence on the forecast output for this period and what should be the key business takeaways of model interpretation based on this analysis.
         Example:
          #### Store_ID 2
          - **Weekly Residual Change (August 12, 2012):** The model predicted sales of $2,076,648.75, while actual sales were $1,919,917.03, resulting in a residual error of $156,731.72.
          - **Feature Contribution Analysis (August 12, 2012):**
            - **weekly_sales_lag_1w:** Feature value of $1,805,999.79 with a Shapley value of $729,555.99, significantly higher than the average of $2,268.07, indicating a strong positive influence on the prediction.
            - **Size:** Feature value of 126,512 square feet with a Shapley value of $50,151.81, compared to the average of $3,612.18, reflecting the store's large size contribution to the overprediction.
            - **CPI:** Feature value of $3.884 with Shapley value of $27,079.45, contrasting with the average negative contribution of -$2,790.49, indicating an unexpected positive influence of inflation.
            - **MarkDown4:** Feature value of $43,000 with Shapley value of $43,901.32, much higher than the average of $8,536.57, indicating overestimated promotional discounts.
         - ** Insights (August 12, 2012):** These deviations suggest that the model overestimated the effects of recent sales trends, store size, inflation, and markdowns, contributing to the forecast error.

      - **Weekly Residual Change ([Specific Week Date 2]):** [Specify the difference between the actual and predicted values for the {key_category}.]
      - **Feature Contribution Analysis ([Specific Week Date 2]):**
        - For each feature, include both the **feature value** (e.g., for feature `X`, the value might be 98.10, or for feature `Y`, it might be 1,000,000, etc) and its corresponding **Shapley value**.
        - Analyze the feature-level Shapley values for this {key_category} and compare them against the dataset-wide average Shapley values.
        - Identify features where the Shapley contribution in this instance significantly differs from its average Shapley value.
        - **Order features by their Shapley value contributions**, from most to least influential.
      - **Insights** ([Specific Week Date 1]):
          - Quantify these differences and explain how they influenced the forecast residual (e.g., a +0.25 increase in Shapley value compared to the average).
          - Explain which features had an outsized influence on the forecast output for this period and what should be the key business takeaways of model interpretation based on this analysis.

      ..
      ..

  [Repeat the above section for every {key_category} present in this month.]
  [**Ensure that every {key_category} from each month, including multiple different week dates for the same {key_category}, is included and compared.**]

  ### Business Implications and Strategic Recommendations
  [Translate the aggregated insights into actionable business strategies. Discuss operational improvements such as inventory management, staffing adjustments, or promotional planning. Ensure recommendations are supported by specific data points and quantitative insights from Shapley analysis.]

  ### Conclusion
  [Summarize the key takeaways from the report, emphasizing how comprehensive monthly and {key_category}-specific insights contribute to a broader understanding of Shapley contributions. Reinforce the strategic recommendations for future business operations.]
"""


map_prompt_template = """

        # Role:
        As an Expert Data Analyst specializing in interpretable machine learning and model diagnostics, you possess deep expertise in residual analysis,
        Shapley value interpretation, and understanding the underlying drivers behind forecast errors. Your proficiency enables you to quantitatively dissect
        how various features contribute to high residuals, allowing you to pinpoint the causes of significant forecast errors in this model.

        # Input Parameters:
        {row_data} - This JSON object contains the residual error for a given forecasted instance and its corresponding feature-level Shapley values,
        which quantify each feature's contribution to the prediction. Use this data to analyze why the model's forecast differs significantly from the actual value.

        {avg_shapley_values} - This JSON object contains the dataset-wide average Shapley values for each feature, serving as a baseline for comparison.
        Leverage this data to determine whether the feature contributions in {row_data} are unusual or expected.

        {data_dict} - This data dictionary provides comprehensive definitions, units, and interpretations for each variable in {row_data}.
        Use this to accurately understand the real-world significance of each feature and its influence on residual errors.

        # Task:
        Your task is to provide a highly detailed and quantitative analysis explaining the high residual error observed in the forecast using Shapley values.
        You must compare the instance-specific Shapley values in  {row_data} with the average Shapley values from {avg_shapley_values} to identify key features
        where the Shapley value differs significantly from its dataset-wide average. Specifically, quantify how much each feature’s Shapley value in {row_data}
        differs from its corresponding average Shapley value in {avg_shapley_values}, and use this difference to determine the primary factors contributing to
        the forecast error.

        # Introduction:
        Begin by setting the context—identify the model being analyzed and specify the instance under review. Explain how residual analysis helps in detecting
        forecasting inaccuracies and why Shapley values are instrumental in interpreting these differences.

        # Feature-by-Feature Analysis:
        For each key feature, analyze how much its Shapley value in {row_data} differs from its dataset-wide average in  {avg_shapley_values}. Quantify this difference (e.g., percentage difference or absolute difference) and interpret its significance using the definitions in {data_dict}. Additionally, explicitly include the corresponding feature values (e.g., the actual feature `X` value of 98.10, or feature `Y` value of 1,000,000, etc) directly from {row_data} to explain how the actual feature value and the Shapley value both contribute to the residual error.
        For example: If feature `X` has a Shapley value of $121,152.49 and the actual feature value is 98.10, compare this feature value with the Shapley value and explain how the interaction between these values leads to the discrepancy between forecasted and actual values. The same approach applies for all features. Discuss how these feature values and Shapley values influence the forecasted residual error, considering model-specific assumptions and data characteristics.

        # Quantitative Insights:
        Synthesize your findings by referencing specific numeric differences (such as a +0.25 increase in Shapley contribution compared to the average) to
        articulate the primary drivers behind the high residual. Avoid including any final conclusions or recommendations for model improvement.

        # Step-by-Step Guidelines:
        1. Start with an introduction that contextualizes the analysis, specifying the model and instance under review, and outline the importance of residual
        diagnostics in forecasting.
        2. For each key feature, state its definition (using {data_dict}), compare its Shapley value to the dataset-wide average, quantify the difference, and discuss
        the model interpretability insights derived from this comparison.
        3. Conclude the analysis with a summary of key feature differences without providing actionable recommendations.

        # Output:
        Provide your response in 2 paragraphs:
        <Paragraph 1> - Introduction that sets the context and overview of the analysis.
        <Paragraph 2> - Detailed feature-by-feature analysis with quantitative insights.
        # Important Tips
          1) Always use the exact numbers provided, as mentioned in the input parameters. You may round these numbers if necessary, but do not truncate or make up any values
          2) Ensure that your summary is concise, clear, and insightful, with each sentence contributing a distinct piece of analysis that collectively explains the observed residual error
          3) Data Dictionary Usage: Always use the data dictionary to interpret each feature. Understand how each feature is defined (e.g., whether it’s a lagged value or a seasonal effect).
          4) Clear Presentation: Keep the analysis clear and organized by following the provided structure.
          5) Do not output any section headers, ONLY UPTO 2 PARAGRAPHS BASED ON THE # Output Format section above.
          6) Use the right units, when you discuss results.
"""

reduce_prompt_template = """

# Role:
As an Expert Data Analyst specializing in Shapley value analysis and interpretability of machine learning models, you possess advanced expertise in understanding feature-level contributions and analyzing the underlying drivers behind model prediction errors. Your skills enable you to dissect how Shapley values highlight each feature's contribution to model behavior, offering deep insights into what causes significant deviations in model predictions.

# Context:
Your organization uses a forecasting model to predict future values over a specified period. You've generated Shapley values to quantify how individual features contribute to model errors for each prediction. Your task is to leverage these Shapley values to provide a comprehensive analysis of the causes behind model inaccuracies, breaking down contributions at both the feature and time period levels. You'll be completing a detailed analysis report based on Shapley value results, ensuring that the report is structured according to predefined sections such as introduction, feature-level contributions, and business insights.

# Input Parameters:
{gold_template}: This is the golden template you will fill based on Shapley value analysis results. The template includes sections like the introduction, monthly analysis, feature-level contributions, and business implications. Follow the specific structure and ensure that all required sections are addressed based on the analysis from both summaries.

{summary1}: This summary provides insights into the Shapley value contributions for a specific time period or multiple time periods. It breaks down how each feature contributed to model behavior and deviations, providing quantitative data for feature-level insights. Note that Summary 1 may include multiple stores or time periods.

{summary2}: This summary provides insights into the Shapley value contributions for a different time period or multiple time periods. It may overlap with Summary 1 in terms of stores, time periods, or feature insights. Carefully assess both summaries to correctly merge and consolidate information.

{data_dict}: This dictionary contains detailed definitions, units, and descriptions of the features used in the Shapley analysis. Refer to this to correctly interpret each feature's role in the analysis.

{key_category}: This refers to the key categories used for structuring the analysis (e.g., time period, product category, region). Ensure that the report aligns with the correct categories.

# Task:
Your task is to complete the golden template using the insights from both summaries. Address the following key areas:

1. **Quantifying Feature Contributions**: For each feature, explain its Shapley value and the actual feature value contribution to the model errors, highlighting any significant deviations compared to the baseline average feature Shapley values (as provided in the summaries).
2. **Residual Error Breakdown**: Use the Shapley values to identify key drivers of model errors, explaining how the most influential features impacted the predictions during different time periods.
3. **Feature-Level Insights**: Focus on explaining anomalies or significant shifts in feature contributions, particularly where individual features caused notable deviations in predictions.
4. **Business Implications**: Based on the feature contributions, provide insights on how these model behaviors impact the business outcomes. Offer suggestions on how business operations or strategies can be adjusted based on the insights derived from the Shapley analysis.
5. **Monthly Analysis**: For each time period, summarize the overall model performance and focus on the key drivers of deviations at the feature level. Ensure all overlapping stores or features are merged appropriately.

# Critically Important: Store Preservation and Consolidation Instructions
1. **YOU MUST INCLUDE ALL STORES FROM BOTH SUMMARIES WITHOUT EXCEPTION**. Your final output must contain every single store that appears in either summary1 or summary2.
2. **First, identify all unique stores from both summaries.** Make a comprehensive list of all stores mentioned in both summaries.
3. **For each store, check if it appears in both summaries or only one summary:**
   - If a store appears in both summaries, merge the insights carefully.
   - If a store appears in only one summary, include it as-is in the final output.
4. **For stores appearing multiple times within the same month, consolidate all entries into one merged entry per store per month:**
   - Keep each date-specific analysis intact (e.g., "Weekly Residual Change (July 15, 2012)" and "Weekly Residual Change (July 29, 2012)").
   - Maintain the chronological order of dates within each store's monthly entry.
   - Ensure that each date's specific feature contributions remain clearly labeled with their respective dates.
   - Consolidate insights that apply to the entire month while preserving date-specific insights.
5. **Do not skip any store or date-specific analysis for any reason.** Every store and every date analysis must be included.
6. **Before finalizing, verify that all stores from both summaries are represented in your output with all their respective date analyses.**

# Step-by-Step Guidelines:
1. **Identify All Unique Stores**
   - Create a complete list of all stores mentioned in both Summary 1 and Summary 2.
   - For each store, identify all the dates/periods it appears in both summaries.
   - This list will be your checklist to ensure no store or date analysis is missed.

2. **Review Summary 1**
   - Carefully examine Summary 1 for relevant information.
   - Extract all stores and their time periods from Summary 1.
   - Fill in the golden template with Summary 1 insights.

3. **Review Summary 2**
   - Examine Summary 2 for additional content.
   - Extract all stores and their time periods from Summary 2.
   - For each store in Summary 2:
     - If the store is already in your template from Summary 1:
       - Check if there are new dates for this store.
       - For the same month, add new date-specific analyses while preserving existing ones.
       - Consolidate insights for the same store within the same month, keeping date-specific analyses clearly labeled.
     - If the store is not in your template yet, add it as a new entry with all its date-specific analyses.

4. **Consolidate Multiple Entries for the Same Store in the Same Month**
   - For each store that appears multiple times within the same month:
     - Maintain the format: "**Store_ID: X**" once at the top.
     - For each date-specific analysis, keep the format: "* **Weekly Residual Change (Specific Date):**"
     - Ensure all date-specific feature contributions are clearly labeled with their respective dates.
     - Present the dates in chronological order within each store's monthly entry.
     - Example structure for a store with multiple dates:
       ```
       **Store_ID: 27**
       * **Weekly Residual Change (July 15, 2012):** [analysis]
       * **Feature Contribution Analysis (July 15, 2012):** [detailed features]
       * **Insights (July 15, 2012):** [insights]
       * **Weekly Residual Change (July 29, 2012):** [analysis]
       * **Feature Contribution Analysis (July 29, 2012):** [detailed features]
       * **Insights (July 29, 2012):** [insights]
       ```

5. **Ensure Comprehensive Coverage**
   - Double-check that every store from your initial list is included in the final output.
   - Verify that all date-specific analyses for each store are preserved.
   - If any store or date analysis is missing, go back to the relevant summary and add it.

6. **Update Monthly Summary Section**
   - For the Monthly Summary section, ensure you merge analyses from both summaries.
   - Provide a consolidated overview of the month while highlighting key trends across different dates.

# Important Golden Template Instructions:
1. **Ensure that each {key_category} covers all dates associated with it.**
2. **Feature Contribution Quantification:** Quantify how each feature's Shapley value and the actual feature value contributes to the model's errors. Include numerical details to highlight deviations.
3. **Monthly Analysis Structure:** For each month, summarize the model's performance and explain the shifts in feature-level contributions.
4. **Actionable Business Insights:** Focus on delivering recommendations that can help improve forecasting accuracy.
5. **Accuracy and Clarity:** Ensure that all details related to Shapley values, residuals, and model performance are accurately described.
6. **Metadata Inclusion:** Ensure that all relevant metadata is included to make the analysis actionable.
7. **Template Format:** Follow the exact structure of the golden template.

# Final Validation:
Before submitting your final output, perform the following checks:
1. **Store Completeness Check:** Verify that EVERY store from BOTH summaries is included in your final output.
2. **Date Completeness Check:** Ensure that ALL date-specific analyses for each store are preserved.
3. **Consolidation Quality Check:** Confirm that stores with multiple entries within the same month are properly consolidated into a single entry while preserving all date-specific analyses.
4. **Format Consistency Check:** Confirm that the final output follows the correct format of the golden template.
5. **Chronological Order Check:** Verify that date-specific analyses within each store's entry are presented in chronological order.

# Output:
Provide the completed Shapley analysis report by filling out the gold template. Focus on explaining the Shapley contributions of each feature, how these contributions affect model predictions, and the business implications based on the observed deviations.

"""

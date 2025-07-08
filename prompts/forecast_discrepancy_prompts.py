gold_template = """

## Retail Sales Forecast Discrepancy Analysis Report

  ### Introduction
  [Provide a concise overview that introduces this analysis. Include the time periods covered, the key drivers of the forecast discrepancies (e.g., holidays, economic conditions, store attributes), and set the 
  stage for the detailed breakdown.]

  ### Monthly Forecast Discrepancy Summary

    <Follow the below format to outline details about Month Year summary for each month>

    #### [Month Year] Summary
    [Summarize the major events or conditions for this month (e.g., holiday effects, promotional markdowns, weather influences).Provide integrated insights explaining the combined impact of all the 
    detailed {key_category}-level data, identifying overarching trends and underlying business reasons, DONT LIST THE {key_category}]

    For every {key_category} this month — including cases where the same store has forecast discrepancies in different week dates (MAKE SURE YOU CAPTURE
    EACH DATE IN EACH {key_category}— provide a detailed analysis as follows:

    •	[{key_category} Forecast Discrepancy Analysis]:
      <Follow the below format to outline forecast discrepancy and contributions for each week date within the specific affected {key_category}>
      o	Forecast Discrepancy ([Specific Week Date 1]): [Specify the difference in forecast between the two models for the {key_category}. Explain what this means in terms of forecast discrepancy.]
      o	Feature Contributions: [Compare the SHAP values for each key feature in both models. Identify which features drive the biggest differences in predictions between the models.
                                Consider why the models might prioritize features differently. Differences in SHAP values highlight how each model processes information, affecting
                                how they respond to changes in key variables. Fill this part as bullet points for each key feature. Use these SHAPley values to explain WHY
                                forecast difference appears in the models. Also explain WHY only these features show heavier differences between both models and what should be the key
                                business takeaways of model interpretation based on this analysis.]
      o	Forecast Discrepancy ([Specific Week Date 2]): [Specify the difference in forecast between the two models for the {key_category}. Explain what this means in terms of forecast discrepancy.]
      o	Feature Contributions: [Compare the SHAP values for each key feature in both models. Identify which features drive the biggest differences in predictions between the models.
                                Consider why the models might prioritize features differently. Differences in SHAP values highlight how each model processes information, affecting
                                how they respond to changes in key variables. Fill this part as bullet points for each key feature. Use these SHAPley values to explain WHY
                                forecast difference appears in the models. Also explain WHY only these features show heavier differences between both models and what should be the key
                                business takeaways of model interpretation based on this analysis.]
      ..
      ..

    [Repeat the above section for every {key_category} present in this month.]

    [Repeat the above [Month Year] Summary structure for every month in the data. ONLY keep this above section at monthly level.
     **Ensure that every store from each month, including multiple different week dates for the same store, is included and compared.**]

  ### Business Implications and Strategic Recommendations
  [Translate the aggregated insights into actionable business strategies. Ensure recommendations are supported by specific data points from the analysis. For the Tech Team,
  investigate how to recalibrate the sensitivity of Model 2 to specific features, aiming to bring its results closer to those of Model 1. For Business Stakeholders, use the
  insights from forecast discrepancies to inform business planning in areas like marketing, inventory, and operations. Ensure forecasts are adjusted to reflect real-time economic
  factors and {key_category}-specific data, enhancing decision-making. Translate the aggregated insights into actionable business strategies. Discuss operational improvements such
  as inventory management, staffing adjustments, or promotional planning. Ensure recommendations are supported by specific data points from the analysis.]

  ### Conclusion
  [Summarize the key takeaways from the report, emphasizing how the detailed monthly and {key_category}-specific insights contribute to a broader understanding of the forecast
  differences between both models.
  Emphasize which factors each model is sensitive to and how these insights can guide strategic decisions. Reinforce the final strategic recommendations for future business operations,
  ensuring the business is prepared for varying forecast scenarios.]


"""

# ------------------------- MAP Prompt Template -------------------------
map_prompt_template = """
        # Role
        As a Master Data Analyst with over 10 years of experience specializing in Forecasting Models and Feature Interpretability, you are an exceptional expert with unparalleled proficiency in following detailed
        instructions to perform precise analysis and complete reports. You excel at extracting meaningful insights from raw model data, using provided dictionaries and templates to craft thorough, accurate reports.

        # Input Parameters
        {row_data} - This row contains contains actual sales, predictions, residuals, and individual feature SHAP value contributions for two forecasting models - Model 1 and Model 2. This data
        is structured as a json where each key is the feature name and value is the numeric value for that feature. Use this data to complete your task.
        You can identify the forecast difference by the forecast_diff variable. The unit of the forecast difference is weekly sales amount.
        The variable forecast_diff is defined as the difference between predicted weekly sales amounts provided by both models.

        {data_dict} - This data dictionary contains detailed definition, unit and interpretation for each column in the above input parameter - {row_data}.
        Use this dictionary to better understand what each feature stands for. This dictionary will help you build meaningful context in your final response.

        {pred_var} - This is the name of the predicted variable in the underlying data.

        # Context
        Your business has run two demand forecasting models - Model 1 and Model 2. You calculated the difference in predictions of both models {pred_var} as the forecast discrepancy value which is saved for the variable {forecast_disc}.
        Your technical team and business stakeholders have asked you for a detailed summary to help explain this forecast discrepancy between both models. This summary should be very detailed, narrative and connect all the dots
        between the feature shapley values of both models and their predictions to explain their forecast difference discrepancy.

        # Task
        Your task is to provide a highly detailed and quantitative analysis related to the forecast difference (indicated by the "forecast_diff" field)
        between Model 1 and Model 2. In your summary, you need to focus on finding the answers to the below questions:
        1) Quantify how much is the forecast difference between both models?
        2) Analyze the feature shapley contributions towards each model prediction of weekly sales amount. Are there features that contribute differently to each model?
        3) Once you identify these high-impactful features, discuss the magnitude by which these features contribute differently to each model predictions.
        4) Conclude the reasons of this forecast discrepancy by outlining which features and by how much do they impact each individual model prediction.


        # Step-by-Step Guidelines:
        Let's think about this step-by-step:
        1. **Introduction:**
          - State the context of the analysis. Outline what the weekly sales predictions are for each model and by how much (forecast_diff) do they differ in USD.
          - Briefly explain the role that different feature contributions play in explaining the underlying forecast difference.
          - Use metadata such as store id and week date to explain which store and what time did this forecast difference occur.

        2. **Quantitative Analysis & Insights:**
           - For each feature, assess how both Model 1 and Model 2 are impacted by or favor certain features, focusing on the magnitude (how much influence the
             feature has) and direction (whether the feature increases or decreases the forecast).
           - Compare the contributions across models, identifying which features have a stronger impact in one model over the other, and explain the reasons behind these differences.
           - Relate each feature’s SHAP value to its description and unit in the data dictionary, and discuss what these insights mean for the tech team (e.g., model tuning or
             adjustments) and business stakeholders (e.g., actionable decisions or focus areas).
           - Offer clear action items for the tech team (such as model recalibration) and practical business implications (such as the need to align forecasts with
             specific business priorities, like focusing on economic indicators for strategic planning)

        # Output
          Give your response in 2 paragraphs summarizing the key reasons why the forecast differences appear between both models. Quantify your results
          by using the feature numeric values. Tie the quantified insights with business takeaways. This output summary must be easily understood by
          both the business stakeholders as well as the technical leads. Make sure your summary is concise, and does not include repeatitive sentences.
          Each sentence must be very insightful and try to connect the dots in a narrative manner to paint the overall picture.

        # Output Format
          Follow the below provided output format carefully:

          <Paragraph 1> - Introduction

          <Paragraph 2> - Quantitative Analysis & Insights


          Your output MUST have at-least 2 paragraphs

        # Important Tips
          1) Accuracy: Make sure the analysis is quantitative—refer to specific SHAP values, residuals, and forecast errors when possible.
          2) Data Dictionary Usage: Always use the data dictionary to interpret each feature. Understand how each feature is defined (e.g., whether it’s a lagged value or a seasonal effect).
          3) Clear Presentation: Keep the analysis clear and organized by following the provided structure.
          4) Do not output any section headers, ONLY UPTO 2 PARAGRAPHS BASED ON THE # Output Format section above.
          5) Use the right units, when you discuss forecasts, make sure to word your insights considering total weekly sales predicted by each model.

        Please provide your comprehensive and quantitative summary below:

"""

# ------------------------- REDUCE Prompt Template -------------------------
reduce_prompt_template = """
      # Role
        As a Master Data Analyst with over 10 years of experience specializing in Forecasting Models and Feature Interpretability, you are an exceptional expert with unparalleled proficiency in following detailed
        instructions to perform precise analysis and complete reports. You excel at extracting meaningful insights from raw model data, using provided dictionaries and templates to craft thorough, accurate reports.
        With a deep technical understanding of how individual features influence model performance, you excel in applying your expertise to fill structured templates, drawing actionable conclusions from the analysis
        of both predictive accuracy and feature impact.

      # Context
        Your business has run two demand forecasting models - Model 1 and Model 2. You calculated the difference in predictions of both models {pred_var} as the forecast discrepancy value
        which is saved for the variable {forecast_disc}.
        Your technical team and business stakeholders have asked you for a detailed report to help explain this forecast discrepancy between both models. They have provided you with a gold template that you MUST strictly follow.
        In this gold template, your team has provided you with key questions that will guide you in your analysis. You must consider these questions to be the top priority to answer for each section.

      # Input Parameters
        {gold_template} - This is the summary report gold template that you need to fill using both the summaries provided above. It has different sections such as introduction, monthly forecast discrepancy summary, business
        implications and conclusion. Each section has specific instructions to guide you on what information to fill in each part based on the provided summaries.

        {summary1} - This is Summary 1 which gives you a comprehensive, knowledgeable and insightful story on a specific forecast discrepancy event occurring in a store on a particular week date.
        It outlines exactly what are the underlying causes of forecast discrepancies between both models.
        It does a good job to quantify this difference in the form of feature shapley contributions, thus, explaining HOW individual features play a role in creating these forecast discrepancies. This report also ties these technical insights to business key takeaways to cater
        to audiences that are technical leads as well as business stakeholders. You need to use this existing summary as your starting point and refine it further based on the new document provided below. This report contains crucial metadata such as store id and week date which help tie an insight back to the
        exact data point when it occurred.

        {summary2} - This is Summary 2 which gives you a comprehensive, knowledgeable and insightful story on a specific forecast discrepancy event occurring in a store on a particular week date. It outlines exactly what
        are the underlying causes of forecast discrepancies between both models.
        It does a good job to quantify this difference in the form of feature shapley contributions, thus, explaining HOW individual features play a role in creating these forecast discrepancies. This report also ties these technical insights to business key takeaways to cater
        to audiences that are technical leads as well as business stakeholders. You need to use this existing summary as your starting point and refine it further based on the new document provided below. This report contains crucial metadata such as store id and week date which help tie an insight back to the
        exact data point when it occurred.

        {data_dict} - This data dictionary contains detailed definition, unit and interpretation for each column in the data used to build both demand forecasting models. It also contains the definition of
        shapley value contributions of each feature to help understand how each feature impacts each model's predictions.
        Use this dictionary to better understand what each feature stands for. This dictionary will help you build meaningful context in your final response.

        {key_category} - This is the key_category which you will use to fill the gold template provided above.

      # Task
        Your task is to fill in the gold template provided above based on key information from both summaries and data dictionary information.
        The gold template has multiple sections, with each section containing a set of instructions.
        Read through each section carefully, understand the instructions provided and use the summaries to fill in the information in the relevant sections.
        Your final output response must be the COMPLETELY FILLED template with all detailed information from both Summary 1 and Summary 2.
        Ensure that you capture key metadata such as the date, store id, and {key_category} details properly when you fill the gold template.

      # Step-by-Step Guidelines

        Let's go through this process step-by-step:
          1- Review Existing Summary 1 – Go through the gold template and understand the questions and sections present. Identify the relevant context and answers needed for each section in Summary 1. Use Summary 1 to fill in the gold template, reaching an “intermediate stage” where it's partially filled with information from Summary 1, but not yet completed with Summary 2. Ensure you include key metadata, such as {key_category}, store id, and all the [Week Dates] for each {key_category} from Summary 1.
          2- Review Existing Summary 2 – Go through the "intermediate stage" gold template filled with Summary 1. Familiarize yourself with the original, unfilled sections and questions in the gold template, and identify where answers and relevant context from Summary 2 are needed. Use Summary 2 to add answers and fill in these sections in the "intermediate stage" gold template. Once you have completed this, the template will reach the "complete stage," fully filled with answers and context from both Summary 1 and Summary 2.
          3- The Complete stage should capture all the key information from both Summary 1 and Summary 2, including all the {key_category}, store ids, and all the [Week Dates] from both summaries. No {key_category}, store id, or associated [week dates] should go missing.
          4- In cases where for a given month and {key_category} there are more than two dates available (for example, three or four dates), ensure that **ALL** such dates are captured and clearly listed.
          5- **Explicit Merging Instruction:** When merging Summary 1 and Summary 2, if a store (i.e., a {key_category}) appears in both summaries, you must:
              - Include the store ID only once.
              - Concatenate the lists of all associated week dates from both summaries.
              - Ensure no week date is lost, and avoid duplications if the same date appears in both.
          6- Ensure the “Complete Stage” Format – Verify that the "complete stage" template matches the expected format of the original gold template.
          7- Answer Accuracy – Make sure each section in the "complete stage" template thoroughly answers the questions and fills out the sections as defined in the original gold template.
          8- Update Monthly Summary Section which focuses on summarizing overall sales performance and key drivers of deviations at the feature level, ensuring it is updated based on both Summary 1 and Summary 2.

      # Important Gold Template Instructions:
      0) Make sure that every {key_category} from both the summaries is included in the final summary.
      1) Make sure that each {key_category} covers every single date associated with it, and ensure that for every store id, every associated week date is captured.
      2) Within feature contributions, for each bullet point feature, follow the instructions in the gold template strictly to explain the impact of features on the model forecast differences and consider why the model favors these features to make such predictions.
      3) Instead of sharing obvious insights such as "Showed a notable difference, with contributions of $86,178.50 in Model 1 and $63,440.89 in Model 2, suggesting Model 1 places more emphasis on store size." Focus more on how to interpret this result and what that means on how each model uses the key variables in its predictions. Use the data dictionary here to connect the dots and really interpret what it means for these features to have a specific shapley value number and how that translates to what the business takeaway should be.
         Examples of insightful sentences for forecast discrepancy include:
         Example 1: SHAP values were $123,498.39 for Model 1 and $88,645.85 for Model 2, indicating that Model 1 assigns greater importance to its key features. Notably, Model 1’s strong emphasis on store size suggests it perceives physical capacity as a crucial determinant of sales performance, potentially capturing the impact of foot traffic, inventory space, and operational scalability more effectively than Model 2.
         Example 2: The store size feature contributes $90,059.18 in Model 1 compared to $68,939.65 in Model 2, highlighting that Model 1 places greater importance on store size. Model 1 likely considers store size an influential factor because a larger physical space enables better inventory management, improved product availability, and the capacity to serve more customers simultaneously, all of which can boost sales. For businesses, this emphasizes the value of optimizing store layouts and investing in larger spaces to maximize sales potential, especially in high-traffic areas.
         Example 3: The SHAP value for the unemployment rate is $15,897.06 in Model 1 compared to $25,294.77 in Model 2, suggesting that Model 2 places significantly more emphasis on the unemployment rate. This indicates that Model 2 recognizes the strong influence of unemployment levels on consumer spending power, with higher unemployment likely leading to reduced sales. For businesses, this highlights the importance of monitoring labor market trends, as shifts in unemployment could directly affect demand and should be factored into strategic planning, particularly for pricing and marketing decisions.
      4) Provide the overall conclusion ONLY AT THE MONTH LEVEL, NOT AT THE {key_category} LEVEL, and IN DETAIL FOLLOWING ALL THE RULES FROM THE GOLDEN TEMPLATE.

      # Output
        Provide your output as the "complete stage" gold template directly which contains all key information from both Summary 1 and Summary 2, with merged lists for store IDs that appear in both summaries.

      # Important Tips
        1) Accuracy: Make sure the analysis is quantitative—refer to specific SHAP values, residuals, and forecast errors when possible.
        2) Data Dictionary Usage: Always use the data dictionary to interpret each feature. Understand how each feature is defined (e.g., whether it’s a lagged value or a seasonal effect).
        3) Clear Presentation: Keep the analysis clear and organized by following the provided structure.
        4) Triple check the numbers you present in the summary.
        5) Make sure the "complete stage" template follows the correct format present in the original gold template.
        6) ***If a specific {key_category} contains more than one date, please make sure all the dates are captured within that {key_category} section.***
        7) **For every store id, ensure that every associated week date is captured.**
        8) **For cases where a month and a {key_category} have more than two dates, capture every single date (even if there are three, four, or more).**
        9) Follow the above gold template instructions thoroughly otherwise you will receive a penalty.
       10) Make sure you provide the accurate numbers. For eg: "Model 2 shows a greater impact ($20,564.40) than Model 1 ($44,384.42)" - This statement does not make sense since Model 1 impact is a larger number than Model 2. Double-check all numbers and make sure that such mistakes are avoided. You will be penalized for this mistake.
       11) For every {key_category}, EVERY ASSOCIATED DATE MUST BE CAPTURED. (A {key_category} can have more than one [Week Date] for every [Month Year])
       12) No {key_category} should go missing, aka all the {key_category} from Summary 1 and Summary 2 should be included in the "Complete Stage"
       13) MAKE SURE TO GIVE THE "Overview" of the "[Month Year] Summary" IN DETAIL FOLLOWING ALL THE RULES FROM THE GOLDEN TEMPLATE OR YOU WILL BE PENALIZED

      Please provide your comprehensive and quantitative "complete stage" template below:
"""

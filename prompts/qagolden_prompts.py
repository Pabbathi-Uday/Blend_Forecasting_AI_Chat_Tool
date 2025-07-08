shapley_prompt_template = """
# Store Performance Analysis Question Generator with Answers

## Role:

You are a Retail Analytics Expert specializing in interpreting sales forecasting models and Shapley value explanations. Your expertise lies in identifying patterns across time periods and extracting actionable insights from complex retail performance data.
Generate as many question-answer pairs as you can from the following summary. If some stores or features are missing details, skip them. Focus on generating high-quality insights based on the data provided.

## Input Parameters:
{summary}-  A Shapley summary of store performance analyses which its Data covers few months (e.g., July 2012, August 2012, September 2012),
Data includes n stores (e.g., IDs: 10, 14, 20, 28, 31, 38, 45). This summary contains:
        - Monthly overview sections (e.g., "Summary")
        - Multiple store-specific analyses within each month
        - Feature contribution details with Shapley values for each store
        - Prediction accuracy metrics including residual errors
        - Detailed feature Shapley values compared to their average impact across the dataset

## Task:
Generate all possible insightful, specific questions that address both the monthly patterns and individual store performance details across every month and store listed in the summary, along with well-reasoned answers to these questions. Ensure no stores are skipped. These question-answer pairs should help retail executives and data scientists better understand forecast errors and feature contributions, with particular emphasis on Shapley value analysis.

Return a complete set of question-answer pairs, covering:
  - Monthly insights
  - Store-specific performance
  - Shapley value explanations
  - Actionable recommendations
  -Temporal patterns

## Key Requirements:

  - Iterate through **every month** in the summary without omission.
  - Each month must have its own dedicated set of questions and answers.
  - **List every month by name** (e.g., July, August, September) to ensure no month is missed.
  - For each month, list all stores mentioned and generate questions for every store without omission.
  - For each store, generate questions across all five categories with at least the minimum specified number of questions per category.
  - If a store appears in multiple months, generate a new set of questions for each appearance with time-specific insights.
  - Questions must never skip stores or months due to summarization or brevity.

## Step-by-Step Guidelines:

1. **Monthly Section Identification:**
   - Identify every monthly summary block.
   - List all stores discussed within the month.

2. **Store Iteration:**
   - For each store mentioned in the month, extract:
     - Store ID
     - Dates of analysis
     - Predicted vs. actual sales amounts
     - Residual errors
     - Top contributing features with Shapley values
     - Comparison of Shapley values to dataset-wide averages

4. **Generate five categories of questions with corresponding answers for each store in each month:**

   ### A. Monthly Summary Questions (3-5 per month):
   - What factors might explain why [feature] had a [positive/negative] impact across stores in [month]?
   - How do the key drivers identified in the [month] summary reflect broader market conditions?
   - What might explain the atypical behavior of [feature] noted in the [month] summary?
   - How might the forecasting model be adjusted to better account for unexpected feature influences?

   ### B. Store-Specific Performance Questions (3-5 Per Store):
   - What explains the consistent over/underprediction pattern for Store ID [X] on [DATE]?
   - Why did Store ID [X] on [DATE] experience a substantial over/underprediction of $[AMOUNT] on [SPECIFIC DATE]?
   - Why does Store ID [X] on [DATE] show different sensitivity to [feature] compared to others?
   - What combination of factors led to the accurate/inaccurate forecast for Store [X] in the [WEEK] week of [MONTH]?

   ### C. Shapley Value Analysis Questions (4-5 Per Store):
   - For Store [X] on [DATE], which features had the most significant impact on the forecast error based on their Shapley values?
   - How did the features with significant Shapley values contribute to the discrepancy between actual and predicted sales for Store [X]?
   - Why did [FEATURE] for Store [X] have a Shapley value of $[AMOUNT] on [DATE], which significantly [exceeds/falls below] its dataset-wide average of $[AMOUNT]?
   - What explains the unusual pattern where [FEATURE] showed a Shapley value of $[AMOUNT] for Store [X] on [DATE] but had minimal impact in other stores?
   - How did the interaction between [FEATURE1] (Shapley: $[AMOUNT]) and [FEATURE2] (Shapley: $[AMOUNT]) affect the forecast accuracy for Store [X] on [DATE]?

   ### D. Actionable Insight Questions (3-5 Per Store):
   - Given the high Shapley value of [FEATURE] ($[AMOUNT]) for Store [X], what targeted interventions might improve forecast accuracy?
   - How should inventory management strategies be adjusted for Store [X] given the significant impact of [FEATURE] (Shapley: $[AMOUNT])?
   - What forecasting model adjustments might reduce the sensitivity to [FEATURE] for locations like Store [X]?
   - How could we better account for the extreme Shapley values observed for [FEATURE] ($[AMOUNT] vs average $[AMOUNT]) in Store [X]'s predictions on [DATE]?

   ### E. Temporal Pattern Questions (3-5 Per Store):
   - How did the influence of [FEATURE] (as measured by Shapley values) on Store [X]'s sales forecasts change throughout [MONTH]?
   - What explains the shift in Shapley value importance from [FEATURE1] ($[AMOUNT]) to [FEATURE2] ($[AMOUNT]) for Store [X] between [DATE1] and [DATE2]?
   - Why might the Shapley value of [FEATURE] have increased/decreased from $[AMOUNT] to $[AMOUNT] for Store [X] during [MONTH]?
   - What seasonal factors might explain the changing Shapley values observed for [FEATURE] in Store [X]'s forecasts on [DATE]?

5. **For each question, provide an answer that:**
   - Is directly extracted from the corresponding lines in the summary.
   - Uses only the relevant section where the information is found.
   - Does not require the model to infer or generate new interpretations.
   - Maintains factual accuracy based on the provided data.
   - References specific values and patterns from the summary.

Return a list of generated question-answer pairs in a **plain text list** format.

## Output Structure:
[
Question: [Generated Question]
Answer: [Generated Answer]
,
Question: [Generated Question]
Answer: [Generated Answer]
,
...
,
...
,
...
,
Question: [Generated Question]
Answer: [Generated Answer]
]
Make sure to strictly follow this format. The output must start with '[' and end with ']'. Each entry must follow the format 'Question: [Generated Question]' and 'Answer: [Generated Answer]'. No additional text, explanations, or formatting should be included.

**IMPORTANT Notes:**
  - **All stores must be processed**—none can be skipped.
  - Ensure equal attention is given to **every store** for each month.
  - The final output must cover **all possible questions** without leaving any store or feature unexplored.
  - Each question must have a corresponding answer directly extracted from the provided summary.
  - The answers should strictly use the relevant lines from the summary without inference.
  - Ensure every answer references specific data points (stores, dates, features, values).
  - Make sure to generate questions for all stores listed for each month.
"""
tredline_prompt_template = """
# Store Performance Trendline Analysis Question Generator with Answers

## Role:

You are a Retail Analytics Expert specializing in interpreting sales trends, anomalies, and outlier events. Your expertise lies in identifying patterns across time periods and extracting actionable insights from complex retail performance data, particularly focusing on sales deviations from expected trendlines.

## Input Parameters:
{summary} - A Trendline Analysis summary of store performance which covers few months (e.g., October 2010, November 2010, December 2010). Data includes multiple stores (e.g., IDs: 2, 4, 6, 10, 13, 14, 20, 23, 27). This summary contains:
        - Monthly overview sections (e.g., "Summary")
        - Multiple store-specific analyses within each month
        - Detailed outlier event overviews including specific dates and percentage changes
        - Weekly sales change metrics comparing actual sales to rolling averages
        - Primary influencing factors (both common and unique) affecting sales anomalies

## Task:

Generate all possible insightful, specific questions that address both the monthly patterns and individual store performance details across every month and store listed in the summary, along with well-reasoned answers to these questions. Ensure no stores are skipped. These question-answer pairs should help retail executives and data scientists better understand sales anomalies and influencing factors, with particular emphasis on trendline deviation analysis.

Return a complete set of question-answer pairs, covering:
  - Monthly insights
  - Store-specific performance
  - Trendline deviation explanations
  - Actionable recommendations
  - Temporal patterns

## Key Requirements:

  - Iterate through every month in the summary.
  - For each month, list all stores mentioned and generate questions for every store without omission.
  - For each store, generate questions across all five categories with at least the minimum specified number of questions per category.
  - If a store appears in multiple months, generate a new set of questions for each appearance with time-specific insights.
  - Questions must never skip stores or months due to summarization or brevity.

## Step-by-Step Guidelines:

1. **Monthly Section Identification:**
   - Identify every monthly summary block.
   - List all stores discussed within the month.

2. **Store Iteration:**
   - For each store mentioned in the month, extract:
     - Store ID
     - Dates of anomaly events
     - Overview of primary influencing factors:
      - Common factors impacting anomalies (e.g., store size, foot traffic)
      - Unique factors per event (e.g., economic indicators, fuel prices, weather conditions, prior sales trends)

3. **Generate five categories of questions with corresponding answers for each store in each month:**

   ### A. Monthly Anomaly Pattern Questions (3-5 per month):
   - What common factors drove sales anomalies across multiple stores in [month]?
   - How did [economic indicator] influence overall sales patterns in [month]?
   - Why did [specific factor] have a widespread impact on store performance in [month]?
   - What seasonal elements contributed to the sales patterns observed in [month]?

   ### B. Store-Specific Performance Questions (3-5 Per Store):
   - What explains the significant [increase/decrease] of [PERCENTAGE]% in Store ID [X]'s sales on [SPECIFIC DATE]?
   - Why did Store ID [X] experience $[AMOUNT] in sales compared to its rolling average of $[AMOUNT] [SPECIFIC DATE]?
   - How did Store ID [X]'s [attribute] contribute to its anomalous performance [SPECIFIC DATE]?
   - What combination of factors led to Store [X]'s exceptional performance in the [WEEK] week of [MONTH]?

   ### C. Trendline Deviation Analysis Questions (4-5 Per Store):
   - For Store [X] on [DATE], which factors had the most significant impact on the [PERCENTAGE]% deviation from expected sales?
   - How did the combination of [FACTOR1] and [FACTOR2] contribute to Store [X]'s sales exceeding its rolling average by $[AMOUNT] [SPECIFIC DATE]?
   - Why did Store [X] show a [PERCENTAGE]% [increase/decrease] while similar stores experienced different outcomes [SPECIFIC DATE]?
   - What explains the unusual pattern where Store [X] had [PERCENTAGE]% [higher/lower] sales than expected despite [CONTRADICTING FACTOR] [SPECIFIC DATE]?
   - How did the interplay between [FACTOR1] and [FACTOR2] create the observed sales anomaly for Store [X] on [DATE]?

   ### D. Actionable Insight Questions (3-5 Per Store):
   - Given the influence of [FACTOR] on Store [X]'s performance [SPECIFIC DATE], what targeted strategies might capitalize on similar conditions in the future?
   - How should inventory planning be adjusted for Store [X] [SPECIFIC DATE] given the significant impact of [FACTOR] on its sales?
   - What operational adjustments might help stores like [X] better respond to [FACTOR] in future scenarios?
   - How could forecasting models be improved to better account for the impact of [FACTOR] observed in Store [X]'s performance?

   ### E. Temporal Comparison Questions (3-5 Per Store):
   - How did Store [X]'s response to [FACTOR] change between [DATE1] and [DATE2]?
   - What explains the shift in primary drivers from [FACTOR1] to [FACTOR2] for Store [X] between [DATE1] and [DATE2]?
   - Why might the impact of [FACTOR] have increased/decreased from [PERCENTAGE1]% to [PERCENTAGE2]% for Store [X] during [MONTH]?
   - What sequential patterns can be observed in Store [X]'s performance throughout [MONTH]?

5. **For each question, provide an answer that:**
   - Is directly extracted from the corresponding lines in the summary.
   - Uses only the relevant section where the information is found.
   - Does not require the model to infer or generate new interpretations.
   - Maintains factual accuracy based on the provided data.
   - References specific values and patterns from the summary.

Return a list of generated question-answer pairs in a **plain text list** format.

## Output Structure:
[
Question: [Generated Question]
Answer: [Generated Answer]
,
Question: [Generated Question]
Answer: [Generated Answer]
,
...
,
...
,
...
,
Question: [Generated Question]
Answer: [Generated Answer]
]
Make sure to strictly follow this format. The output must start with '[' and end with ']'. Each entry must follow the format 'Question: [Generated Question]' and 'Answer: [Generated Answer]'. No additional text, explanations, or formatting should be included.

**IMPORTANT Notes:**
  - **All stores must be processed**—none can be skipped.
  - Ensure equal attention is given to **every store** for each month.
  - The final output must cover **all possible questions** without leaving any store or feature unexplored.
  - Each question must have a corresponding answer directly extracted from the provided summary.
  - The answers should strictly use the relevant lines from the summary without inference.
  - Ensure every answer references specific data points (stores, dates, sales figures, percentages, factors).
  - Make sure to generate questions for all stores listed for each month.

"""
forecast_prompt_template = """
# Forecast Discrepancy Analysis Question Generator

## Role:

You are a Forecast Discrepancy Analysis Expert specializing in interpreting model predictions and understanding the sources of forecast variations between different predictive models.
## Input Parameters:
{summary} - A Forecast Discrepancy summary of store performance which covers few months (e.g., October 2010, November 2010, December 2010). Data includes multiple stores (e.g., IDs: 2, 4, 6, 10, 13, 14, 20, 23, 27). This summary contains:
        - Monthly overview sections (e.g., "Summary")
        - Multiple store-specific analyses within each month
        - Forecast Discrepancy Insights, including specific dates and variance amounts
        - Comparison of Model Predictions, highlighting differences in forecasted sales
        - Key Feature Contributions, explaining why forecast differences occurred

## Task:

Generate insightful, specific questions that address model forecast discrepancies along with well-reasoned answers, feature contributions, and underlying prediction variations. The goal is to help data scientists and business analysts understand nuanced differences between predictive models. Ensure no stores are skipped.

Return a complete set of question-answer pairs, covering:
  - Monthly insights
  - Store-specific performance
  - Forecast discrepancies explanations
  - Actionable recommendations
  - Temporal patterns


## Step-by-Step Guidelines:

1. **Monthly Section Identification:**
   - Identify every monthly summary block.
   - List all stores discussed within the month.

2. **Store Iteration:**
   - For each store mentioned in the month, extract:
     - Store ID
     - Dates of analysis
     - Forecast discrepancies, including predicted sales from different models and their variance
     - Analysis of differences in model predictions based on feature sensitivity
     - Top contributing features with Shapley values
     - Key contributing factors affecting forecast variance

3. **Generate five categories of questions with corresponding answers for each store in each month:**

   ### A. Monthly Model Comparison Questions (3-5 per month):
   - How do the underlying assumptions of Model 1 and Model 2 differ in interpreting sales data in the [month]?
   - What systematic differences exist in how the models prioritize various features in the [month]?
   - in the [month], Why do the models show consistent/inconsistent predictions across different stores?

   ### B. Forecast Discrepancy Analysis Questions (3-5 Per Store):
   - What explains the $[X] forecast discrepancy between models for Store ID [Y] on [DATE]?
   - For Store ID [Y] on [DATE], How do the feature contributions reveal different prediction strategies between the models?
   - For Store ID [Y], Why does one model appear more sensitive to specific features than the other on [DATE]?

   ### C. Feature Contribution Insights (4-5 Per Store):
   - How do the Shapley values for [FEATURE] differ between Model 1 and Model 2 for Store ID [X] on [DATE]?
   - What can the variation in Shapley values tell us about each model's predictive logic on [DATE]?
   - Why does [FEATURE] have a significantly different impact across the two models on [DATE]?

   ### D. Predictive Model Optimization Questions (3-5 Per Store):
   - What adjustments could reduce the forecast discrepancy for Store ID [X] on [DATE]?
   - For Store ID [Y] on [DATE],How might we reconcile the different feature interpretations between the models?
   - For Store ID [Y], What additional features could help improve prediction consistency on [DATE]?

   ### E. Temporal and Contextual Pattern Questions (3-5 Per Store):
   - How do the model discrepancies change across different dates for For Store ID [Y], on [DATE]?
   - What contextual factors might explain the varying model performances on [DATE] For Store ID [Y],?
   - For Store ID [Y] on [DATE], How do seasonal or economic conditions influence the models' prediction differences?

5. **For each question, provide an answer that:**
   - Is directly extracted from the corresponding lines in the summary.
   - Uses only the relevant section where the information is found.
   - Does not require the model to infer or generate new interpretations.
   - Maintains factual accuracy based on the provided data.
   - References specific values and patterns from the summary.

Return a list of generated question-answer pairs in a **plain text list** format.

## Output Structure:
[
Question: [Generated Question]
Answer: [Generated Answer]
,
Question: [Generated Question]
Answer: [Generated Answer]
,
...
,
...
,
...
,
Question: [Generated Question]
Answer: [Generated Answer]
]
Make sure to strictly follow this format. The output must start with '[' and end with ']'. Each entry must follow the format 'Question: [Generated Question]' and 'Answer: [Generated Answer]'. No additional text, explanations, or formatting should be included.

**IMPORTANT Notes:**
  - **All stores must be processed**—none can be skipped.
  - Ensure equal attention is given to **every store** for each month.
  - The final output must cover **all possible questions** without leaving any store or feature unexplored.
  - Each question must have a corresponding answer directly extracted from the provided summary.
  - The answers should strictly use the relevant lines from the summary without inference.
  - Ensure every answer references specific data points (stores, dates, sales figures, percentages, factors).
  - Make sure to generate questions for all stores listed for each month.

"""
score_prompt_template = """
## Role:
You are an expert evaluator of question-answer (QA) quality, focusing on technical document analysis.

## Input Parameters:
- {document_summary}: A summary of the original document's key analytical insights
- {generated_qa}: A question-answer pair containing:
  - Question: The generated investigative query
  - Answer: The corresponding response

## Task:
Evaluate the quality of the question-answer pair against the document summary.

## Step-by-Step Guidelines:
1. Carefully read the document summary
2. Assess the answer:
   - Answer Correctness
     - Factual accuracy
     - Traceability to source document
     - Technical nuance interpretation
     - Comprehensiveness of answer
     - Conciseness and clarity
3. Apply the scoring criteria:
   - 10: Exceptional - Perfect alignment
   - 9: Excellent - Nearly flawless
   - 8: Very Good - Minor improvable aspects
   - 7: Good - Some notable limitations
   - 6: Satisfactory - Significant improvement needed
   - 5: Mediocre - Substantial understanding gaps
   - 4: Poor - Major misinterpretations
   - 3: Very Poor - Fundamentally misaligned
   - 2: Minimal Relevance - Barely connected
   - 1: Incorrect - Complete misunderstanding
   - 0: Irrelevant - No connection to source

## Output Structure:
Return ONLY the integer score from 0-10 representing the QA pair's quality
"""
  
def generate_prompt(business_scenario, model_info, model_hyperparams,
                    metrics_train, metrics_test, feature_importance,
                    p_value, dw_stat, column_description, high_corr_df, target_data_desc, corr_thresh=0.8):
    """
    Generates a detailed prompt for summarizing the demand forecasting model.
    """
    prompt = f"""# **Role:**
    You are an expert data scientist with strong business communication skills. Your task is to present a **clear, detailed, and business-focused** summary of a **demand forecasting model** to key stakeholders. The goal is to bridge the gap between **technical insights** and **real-world business impact**, ensuring decision-makers understand the **why** and **how** behind the model’s predictions.

    # **Task:**
    Generate a structured, in-depth **business-oriented summary** covering the following:

    ## **1. Model Overview** (One Paragraph)
    - Explain the **model’s purpose and function** in simple business terms.
    - Describe how it **improves sales forecasting** and addresses challenges like **inventory mismanagement, staffing inefficiencies, and revenue loss**.
    - Provide **business context**, including challenges like **seasonal demand variability** and **supply chain disruptions**.
    - Highlight **LGBM model’s advantages** over traditional methods.
    - Summarize the **types of analysis conducted**, but **only if relevant to the business scenario** (e.g., statistical diagnostics, feature importance analysis, correlation assessments).

    ## **2. Methodology & Hyperparameters** (Bullet Points)
    - Explain how the model **processes historical sales data** and incorporates factors like **seasonality, promotions, and store characteristics**.
    - List **all hyperparameters** with:
      - Their **values**
      - Allowed **ranges**
      - **Business significance**
      -> For every hyperparameter in a ROW
    - Discuss **trade-offs** in hyperparameter settings regarding **forecast accuracy, overfitting risk, and training time**.

    ## **3. Performance Evaluation & Business Interpretation** (Bullet Points with Detail)
    - Assess model accuracy using **RMSE, MAE, and SMAPE** for **training and test datasets**.
    - Determine **relevant metrics** based on the **statistical properties** of the target variable (min, max, std, mean, median).
    - Evaluate model performance:
      - **RMSE vs. Standard Deviation (std)**
        - **Good:** RMSE is close to or lower than std → Errors are within expected demand variability.
        - **Poor:** RMSE is significantly higher → Model errors exceed natural demand fluctuations.
      - **MAE vs. Mean & Median**
        - **Good:** MAE is much lower than mean/median → Small absolute errors relative to demand.
        - **Poor:** MAE is close to or exceeds mean/median → Large absolute errors impact reliability.
      - **SMAPE vs. Demand Ranges**
        - **Good:** SMAPE < 20% → Reasonable percentage errors.
        - **Poor:** SMAPE > 50% → Model struggles with relative accuracy.
        - **Critical Case:** If **target variable is close to 0**, SMAPE may be unreliable due to division errors.
    - Communicate **trade-offs** based on the above analysis (if any)(e.g., a model with **low RMSE but high SMAPE** may have strong absolute accuracy but weak relative error handling).
    - DELIVER an **overall performance review** considering **business context and model results**. EXPLAIN IN DETAIL WITHOUT USING ANY GENERIC STATEMENTS

    ## **4. Business Impact & Use Cases** (One Paragraph)
    - Link **forecast accuracy** to **concrete business outcomes**.
    - Explain how decision-makers can **apply model insights** to daily operations.

    ## **5. Feature Importance Analysis (SHAP Insights)** (Bullet Points)
    - List the **top 5 influential features** and their impact on sales forecasts.
    - Provide **business-friendly takeaways**: Why do these features matter? How can businesses leverage these insights?
    - Discuss the **role of lag features** (e.g., previous week’s sales) in capturing sales trends.

    ## **6. Correlation Analysis & Business Insights** (Bullet Points)
    - From all the given **highly correlated features** identify their **business relationships**.
    - Discuss whether these correlations introduce **biases** in forecasting.

    ## **7. Statistical Diagnostics & Model Reliability** (Bullet Points)
    - **Breusch-Pagan Test p-value**
      - Explain **heteroscedasticity** and its implications.
      - Interpret the current value in business terms.
      - Suggest mitigation strategies if needed.
    - **Durbin-Watson Statistic**
      - Explain implications for **forecast stability**.
      - Interpret the value in a business context.
      - Recommend solutions for detected issues.

    ## **8. Cost-Benefit & Risk Analysis** (Bullet Points)
    - **Quantify financial risks** of forecast inaccuracies.
    - Conduct an **expanded sensitivity analysis**.
    - Outline **risk mitigation strategies**.

    ## **9. Recommendations & Next Steps** (Bullet Points)
    - Provide **actionable recommendations** for improving model performance.
      - Suggest updates/new Hyperparameters to be considered based on test and train metrics and the previous hyper parameters, and the given model

    ---

    # **Input Parameters:**

    ## **1. Business Scenario**
    - {business_scenario}

    ## **2. Model Information**
    - {model_info}
    - **Hyperparameters:** {model_hyperparams}

    ## **3. Model Performance Metrics**

    ### **Train Set:**
    - RMSE: {metrics_train["rmse"]:.2f}
    - MAE: {metrics_train['mae']:.2f}
    - SMAPE: {metrics_train['smape']:.2f}%

    ### **Test Set:**
    - RMSE: {metrics_test['rmse']:.2f}
    - MAE: {metrics_test['mae']:.2f}
    - SMAPE: {metrics_test['smape']:.2f}%

    ## **4. Feature Importance (SHAP Values):**
    - {feature_importance}

    ## **5. Statistical Diagnostics:**
    - Breusch-Pagan Test p-value: {p_value:.4f}
    - Durbin-Watson Statistic: {dw_stat:.2f}

    ## **6. Column Descriptions & Data Types:**
    - {column_description}

    ## **7. High-Correlation Features (> {corr_thresh})**
    (Each pair is formatted as ['Feature_1', 'Feature_2', 'Correlation']):
    - {high_corr_df}

    ## **8. Target Variable Data Description**
    - {target_data_desc}

    ---

    # **Execution Guidelines:**
    - Use **clear section headings and bullet points** for readability.
    - Avoid **excessive technical jargon**—focus on **business impact**.
    - Provide **concrete examples** linking model performance to business challenges.
    - Frame recommendations in terms of **ROI, efficiency, and strategic advantage**.
    - **Do not hallucinate** if data is insufficient.
    - Avoid generic statements and go into detail

    ---

    # **Output Format:**
    ✔ Clearly explains **model function and accuracy**  
    ✔ Connects **model insights to business operations & financial impact**  
    ✔ Highlights **forecast risks and mitigation strategies**  
    ✔ Provides **data-driven recommendations** to improve sales forecasting effectiveness  
    """
    return prompt


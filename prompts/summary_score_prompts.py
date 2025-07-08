# ---------------------------
# Prompt Templates
# ---------------------------
EXTRACTION_PROMPT = """
Role:
You are a highly skilled text analyst specializing in extracting targeted sections from aggregated texts.
Task:
Given:
- Aggregated Summary: {docx_text}
- Store ID: {store_id}
- Week Date: {week_date}
Task:
Extract the section of the text that corresponds to the specified given Store ID and Week Date. 
Output:
Return only the section corresponding to the specified Store ID and Week Date with no additional commentary.
If no relevant section is found, return an empty string: "". 
"""

SECTION_EXTRACTION_PROMPT = """
Role:
You are a skilled document analyst who can accurately extract sections of text from a larger document.
Task:
Given:
- Aggregated Executive Summary: {executive_text}
- Section Heading: {section_heading}
Task:
Extract the section of the text that corresponds to the specified given section heading.
Output:
Return only the section that begins with the given heading, up to the next heading or end of document. No extra commentary.
If no relevant section is found, return an empty string: "". 
"""

COHERENCE_PROMPT = """
Role:
You are an experienced evaluator of text coherence.
Task:
Analyze the following Individual Summary:
{individual_summary}
Task:
Given the Individual Summary, evaluate its coherence and Output 1 if it is coherent (logical progression, clear transitions), or 0 if not.
Output: 
1 or 0 only with no explanation or additional text.
"""

# HALLUCINATION_PROMPT = """
# Role:
# You are an evaluator specialized in detecting hallucinations in AI summaries.
# Task:
# Given:
# - Ground Truth: {ground_truth}
# - Individual Summary: {individual_summary}
# task:
# Evaluate the accuracy of the references in the Individual Summary against the Ground Truth and Output 0 if all references are accurate, or 1 if any misinformation is present.
# Output:
# 1 or 0 only with no explanation or additional text.
# """

HALLUCINATION_PROMPT = """
Role:
You are an evaluator specialized in detecting hallucinations in AI summaries.
Task:
Given:
- Ground Truth: {ground_truth}
- Individual Summary: {individual_summary}

Only focus on information in the Ground Truth that is directly related to what is mentioned in the Individual Summary. Ignore any Ground Truth sections not referenced in the summary.

If all facts and references in the summary are supported by corresponding content in the Ground Truth, output 0.
If there is any factual inconsistency or unsupported claim (hallucination), output 1.

Output only the number: 0 or 1.
End.
"""


CONCISENESS_PROMPT = """
Role:
You are an evaluator of summary conciseness.
Task:
Given:
- Ground Truth: {ground_truth}
- Individual Summary: {individual_summary}
Task:
Evaluate the conciseness of the Individual Summary in relation to the Ground Truth and Output 1 if Individual Summary efficiently conveys the key facts from Ground Truth without redundancy, or 0 if verbose.
Output:
1 or 0 only with no explanation or additional text.
"""
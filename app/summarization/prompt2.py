"""
Research Paper Summarization: Key Points Extraction for Researchers
Note: Always return the summary in the same language as the original paper.
"""


# # Mini-Prompt 1: Basic Information
# basic_info_prompt = """
# Extract the essential identifying information from this research paper.
#
# | Information | Details |
# |-------------|---------|
# | Title | [Paper title] |
# | Authors | [Author names] |
# | Publication | [Journal/Conference, Year] |
# | Field | [Research domain] |
# | Keywords | [Key terms] |
# | DOI/URL | [Link if available] |
# """
#
# # Mini-Prompt 2: Research Objectives & Abstract
# objectives_prompt = """
# Extract the following:
#
# 1. Primary research question/objective (2-3 bullet points)
# 2. Condensed abstract summary (2-3 sentences)
# 3. Main contributions (3-5 bullet points)
# 4. Theoretical foundation/frameworks
#
# Keep each point short and focused on a single idea.
# """
#
# # Mini-Prompt 3: Methodology Details
# methodology_prompt = """
# Extract key methodology details:
#
# 1. Base model(s) used
# 2. Architecture summary
# 3. Dataset(s) with name, size, and characteristics
# 4. Experimental setup (conditions, controls, parameters)
# 5. Implementation details (hardware/software used)
#
# Format as a structured table with clear categories.
# """
#
# # Mini-Prompt 4: Key Equations & Technical Approach
# equations_prompt = """
# Extract and explain key equations:
#
# | Equation | Purpose | Explanation |
# |----------|---------|-------------|
# | [Equation 1] | [What it calculates] | [Explanation] |
# | [Equation 2] | [What it calculates] | [Explanation] |
#
# Include 2-3 equations central to the methodology or findings.
# """
#
# # Mini-Prompt 5: Results & Performance
# results_prompt = """
# Summarize the performance:
#
# 1. 5-7 bullet points for primary findings
# 2. Performance metrics (accuracy, F1, etc.)
# 3. Comparison with previous work or baselines
# 4. Highlights from ablation studies
#
# Present performance in a table if applicable:
#
# | Metric | Value | Compared to Baseline |
# |--------|-------|-----------------------|
# """
#
# # Mini-Prompt 6: Critical Analysis
# analysis_prompt = """
# Identify strengths and weaknesses:
#
# | Strengths | Limitations |
# |-----------|-------------|
# | [Point 1] | [Limitation 1] |
# | [Point 2] | [Limitation 2] |
# | [Point 3] | [Limitation 3] |
#
# Also note:
# - Key assumptions made
# - Generalizability of findings
# - Ethical concerns, if mentioned
# """
#
# # Mini-Prompt 7: Implications & Future Work
# implications_prompt = """
# Extract future-oriented content:
#
# 1. Research implications (field-level impact)
# 2. Practical/industry applications
# 3. Future research directions
# 4. Unanswered or open questions
#
# Use 3-4 concise bullet points.
# """
#
# # Mini-Prompt 8: Executive Summary
# executive_summary_prompt = """
# Create a concise executive summary (max 250 words):
#
# 1. One-paragraph overview
# 2. Problem being addressed
# 3. Approach and methods
# 4. Key results
# 5. Significance of the findings
# """


# basic_info_prompt = """You are extracting basic information from a research paper. Please provide:
#
# 1. Paper title
# 2. Authors and affiliations
# 3. Publication date and venue/journal
# 4. DOI/URL if available
# 5. Citation information
# 6. Research domain/field
#
# Format as a simple table with categories and details."""
#
# # Mini-Prompt 2: Research Objectives and Abstract
# objectives_prompt = """Based on the research paper, please extract:
#
# 1. The primary research question/objective (2-3 bullet points)
# 2. A condensed abstract summary (2-3 sentences capturing the essence)
# 3. The main contributions (3-5 key bullet points)
# 4. The theoretical foundation/frameworks underlying the research
#
# Keep each bullet point to 1-2 sentences, focused on a single idea."""
#
# # Mini-Prompt 3: Methodology Details
# methodology_prompt = """Extract the key methodological details from the paper:
#
# 1. Base model(s) used (if applicable)
# 2. System/model architecture (concise description)
# 3. Datasets used (names, sizes, characteristics)
# 4. Experimental setup (conditions, controls, parameters)
# 5. Implementation details (hardware, software, computational resources)
#
# Present this information in a structured table format."""
#
# # Mini-Prompt 4: Key Equations and Technical Approach
# equations_prompt = """Identify and explain the most important equations and technical approaches:
#
# 1. Extract 2-3 key equations/formulations
# 2. For each equation, explain:
#    - What it calculates
#    - Its purpose in the paper
#    - How it relates to the overall methodology
#
# Format as a table with columns for Equation, Purpose, and Explanation."""
#
# # Mini-Prompt 5: Results and Performance
# results_prompt = """Summarize the main results and performance metrics:
#
# 1. Primary findings (5-7 bullet points)
# 2. Performance metrics (accuracy, F1, BLEU, etc.)
# 3. Comparison to prior or competing approaches
# 4. Key insights from ablation studies
#
# Present performance metrics in a table with columns for Metric, Value, and Comparison to Previous Work."""
#
# # Mini-Prompt 6: Critical Analysis
# analysis_prompt = """Analyze the strengths and limitations of the paper:
#
# 1. Clearly stated limitations (3-4 bullet points)
# 2. Key assumptions made by the authors
# 3. Assessment of generalizability of findings
# 4. Ethical considerations mentioned
#
# Present as a comparison table with Strengths and Limitations columns."""
#
# # Mini-Prompt 7: Implications and Future Work
# implications_prompt = """Extract information about implications and future directions:
#
# 1. Research implications (how this advances the field)
# 2. Practical/industry applications of the findings
# 3. Future research directions identified by the authors
# 4. Unresolved questions that emerge from this work
#
# Provide as 3-4 concise bullet points focusing on significance and future work."""
#
# # Mini-Prompt 8: Executive Summary
# executive_summary_prompt = """Create a concise executive summary of the paper with these components:
#
# 1. One-paragraph overview (3-5 sentences)
# 2. The problem being solved
# 3. The approach taken
# 4. The key results
# 5. Why this matters
#
# Keep the entire summary under 250 words for quick reference."""


# ================= ================= ================= ================= ================= =================


basic_info_prompt = """# Basic Paper Information

Generate a concise summary of the paper's essential metadata using the table below. Ensure all details are accurately extracted and easy for researchers to scan. Provide the output in the same language as this prompt. If a specific detail cannot be found, provide an empty string (`""`) for that item or cell, do not use placeholder text.

| Information       | Details                          |
|-------------------|----------------------------------|
| **Title** | [Full title of the paper]        |
| **Authors** | [Complete list of authors]       |
| **Publication Venue** | [Journal/Conference, Year]   |
| **Research Field**| [Primary domain or discipline]   |
| **Keywords** | [Relevant terms and topics - use bullet points if multiple] |
"""

research_focus_prompt = """# Core Research Focus

Summarize the central aim, problem, contribution, and significance of the paper. Present the information clearly and concisely using bullet points.  Provide the output in the same language as this prompt. If a specific detail cannot be found, provide an empty string (`""`) for that item or cell, do not use placeholder text.

* **Research Question:** [What is being investigated? State this clearly.]
* **Problem Statement:** [What specific gap or issue does the paper address? Be direct.]
* **Main Contribution:** [What is the core offering, innovation, or finding? Highlight the novelty.]
* **Significance:** [Why is this research important for the field or practice? Briefly explain the impact.]
"""

abstract_prompt = """# Abstract Summary

Break down the paper's abstract into its fundamental components for quick comprehension. Present the information concisely using bullet points. Provide the output in the same language as this prompt. If a specific detail cannot be found, provide an empty string (`""`) for that item or cell, do not use placeholder text.

* **Background:** [Brief context leading to the study]
* **Problem:** [The specific issue the paper tackles]
* **Methodology:** [Approach, methods, or techniques used]
* **Key Findings:** [Main results or discoveries - use sub-bullets if needed]
* **Conclusion:** [Primary takeaway or implication]
"""

methods_prompt = """# Methodology Summary

Describe how the research was conducted, focusing on key aspects like study design, data, techniques, and evaluation. Present the information concisely using bullet points, with sub-bullets for details. Provide the output in the same language as this prompt. If a specific detail cannot be found, provide an empty string (`""`) for that item or cell, do not use placeholder text.

* **Study Design:** [e.g., Experimental, Simulation, Case Study, etc.]
* **Dataset(s):**
    * Source: [Where the data came from]
    * Size: [Amount of data]
    * Key Characteristics: [Important features or properties]
    * Preprocessing: [Main steps taken to prepare data]
* **Techniques/Models:** [Specific models, algorithms, or frameworks used - list key ones]
* **Evaluation:**
    * Metrics: [How performance/success was measured - list key metrics]
    * Setup: [Briefly describe evaluation setup if notable]
* **Tools & Software:** [Libraries, platforms, hardware specifics if critical]
"""

results_prompt = """# Key Results

List and explain the paper's main outcomes and their importance. Use the table for primary findings and bullet points for comparisons to prior work. Keep descriptions and insights brief and impactful. Provide the output in the same language as this prompt. If a specific detail cannot be found, provide an empty string (`""`) for that item or cell, do not use placeholder text.

| Finding # | Description of Result         | Significance / Insight           |
|-----------|-------------------------------|----------------------------------|
| 1         | [What was observed/found?]    | [Why this result is important or novel?] |
| 2         | [Another key result]          | [Its implication or contribution]  |
| 3         | [Third main finding]          | [What we learn from this]        |
| ...       | [Add more rows as needed]     | [Corresponding insight]          |

**Comparison to Prior Work:**
* [Highlight how these results differ from or improve upon previous research.]
* [Mention specific previous work if comparison is direct.]
* [Explain why the improvement or difference matters.]
"""

visuals_prompt = """# Important Figures & Tables

Highlight the most critical visualizations and tabular data from the paper. Explain their content and why they are important for understanding the research. Use the table below.

| Visual Element  | Brief Description             | Key Insight or Interpretation            |
|-----------------|-------------------------------|--------------------------------------|
| **Figure [Number]**| [What the figure depicts or shows] | [What key point or data trend does it illustrate?] |
| **Table [Number]** | [Summary of data/content in the table] | [What conclusion or comparison can be drawn from this table?] |
| **Figure [Number]**| [Another key visualization]   | [Why is this figure crucial for the results or argument?] |
| ...             | [Add more rows as needed]     | [Corresponding insight]              |
"""

limitations_prompt = """# Limitations & Future Work

Detail the limitations encountered during the research and outline suggested future directions. Use bullet points for both limitations and future work. Be concise. Provide the output in the same language as this prompt. If a specific detail cannot be found, provide an empty string (`""`) for that item or cell, do not use placeholder text.

**Limitations:**
* **Theoretical:** [Conceptual limits of the approach, with brief impact]
* **Methodological:** [Issues with design or procedure, with brief impact]
* **Data-Related:** [Constraints due to data quality/availability, with brief impact]
* [Add other relevant limitations]

**Future Work Suggestions:**
* [Proposed next steps or improvements to the current work.]
* [New areas or questions for future research based on these findings.]
* [Potential experiments or applications to explore.]
"""

contributions_prompt = """# Main Contributions

List all major contributions of the paper, categorized by type. Explain how each contribution adds value or novelty to the field. Use bullet points, with sub-bullets for novelty/advancement. Provide the output in the same language as this prompt. If a specific detail cannot be found, provide an empty string (`""`) for that item or cell, do not use placeholder text.

* **Theoretical:** [New framework, concept, or insight introduced]
    * Novelty/Advancement: [How it extends or changes existing theory]
* **Methodological:** [New method, algorithm, or model developed]
    * Novelty/Advancement: [What makes it different, better, or more efficient?]
* **Empirical:** [Significant findings or results from experiments/data]
    * Novelty/Advancement: [Why these results matter or what they demonstrate?]
* **Practical:** [Applications, systems, or tools developed]
    * Novelty/Advancement: [Real-world relevance or utility]
* [Add other relevant contributions]

**Most Noteworthy Contribution:** [Briefly summarize the single biggest impact or most innovative aspect of the paper.]
"""

related_work_prompt = """# Related Work

Show how this research fits into the existing landscape of studies and what specific gaps it addresses. Use the table to compare this work to previous approaches and list the addressed gap using bullet points. Provide the output in the same language as this prompt. If a specific detail cannot be found, provide an empty string (`""`) for that item or cell, do not use placeholder text.

| Topic/Area      | Previous Approaches             | This Paper's Innovation / Difference   |
|-----------------|----------------------------------|----------------------------------------|
| **[Relevant Area 1]**| [Summary of how prior work handled this] | [What new approach, technique, or finding is introduced here?] |
| **[Relevant Area 2]**| [Other related methods or studies] | [How does this paper build upon or deviate from them?] |
| **[Relevant Area 3]**| [Existing theories or models]    | [Enhancements, alternatives, or validations provided by this work] |
| ...             | [Add more rows as needed]        | [Corresponding innovation]             |

**Gap Addressed:**
* [What specific problem, limitation, or missing piece in the existing literature does this paper tackle?]
"""

applications_prompt = """# Practical Applications

Explore potential real-world applications of the research findings or methods. Use the table to detail potential use cases, required conditions, and feasibility. Provide the output in the same language as this prompt. If a specific detail cannot be found, provide an empty string (`""`) for that item or cell, do not use placeholder text.

| Domain/Industry | Potential Use Case or Application   | Key Requirements or Dependencies        | Feasibility/Timeline (e.g., Short/Med/Long term) |
|-----------------|--------------------------------------|-------------------------------------|-------------------|
| **[Domain 1]** | [How can the results/methods be used here?] | [What data, technology, or infrastructure is needed?] | [Estimated time to potential deployment]  |
| **[Domain 2]** | [Another potential application area] | [Factors affecting feasibility or adoption] | [Estimated time to potential deployment]  |
| **[Domain 3]** | [Innovative potential application] | [Challenges or conditions for implementation] | [Estimated time to potential deployment]  |
| ...             | [Add more rows as needed]            | [Corresponding requirements]          | [Corresponding timeline]  |

**Most Promising Use Case:** [Briefly highlight the application with the highest potential impact or feasibility.]
"""

technical_prompt = """# Technical Details

Provide a concise summary of the paper's specific technical aspects. Use the table for algorithms, architecture, implementation, and performance. Provide the output in the same language as this prompt. If a specific detail cannot be found, provide an empty string (`""`) for that item or cell, do not use placeholder text.

| Component         | Description                        | Key Configuration or Parameters         |
|-------------------|-------------------------------------|--------------------------------------|
| **Algorithm(s)** | [What specific algorithm(s) are central?] | [Key hyperparameters, variations used, etc.] |
| **Model/Architecture**| [Type or design of the model/system] | [Number of layers, components, specific structure details] |
| **Implementation**| [Languages, key libraries, environment specifics] | [Frameworks used (TensorFlow, PyTorch, etc.), notable dependencies] |
| **Performance** | [Key performance metrics reported]    | [Results achieved (e.g., Accuracy %, F1 score, latency ms)] |
| ...               | [Add more rows as needed]           | [Corresponding details]              |

"""

quick_summary_prompt = """# Quick Summary

Provide a highly concise summary of the entire paper, suitable for a quick grasp of its core message. Include both a brief paragraph and a single-sentence version. Provide the output in the same language as this prompt. If a specific detail cannot be found, provide an empty string (`""`) for that item or cell, do not use placeholder text.

**Brief Summary (3–5 Sentences):**
[Write a concise summary covering the paper's motivation, core method, main findings, and overall significance.]

**One-Sentence Summary:**
[Write a single, impactful sentence that captures the paper’s most important contribution or finding.]
"""

reading_guide_prompt = """# Reading Guide

Help researchers quickly navigate the paper by highlighting the most important sections and the key information found within them. Suggest an efficient reading path. Use the table for key sections and bullet points for the reading path. Provide the output in the same language as this prompt. If a specific detail cannot be found, provide an empty string (`""`) for that item or cell, do not use placeholder text.

| Section Name      | Key Information or Reason to Focus Here |
|-------------------|------------------------------------|
| **[Section Name 1]**| [What is the main idea or critical takeaway from this section?] |
| **[Section Name 2]**| [Why is this section particularly insightful or important for understanding the work?] |
| **[Section Name 3]**| [What key details or results are presented here?] |
| ...               | [Add more rows as needed]          |
| **[Conclusion Section]**| [Main takeaways and future implications.] |

**Recommended Reading Path:**
* [Suggest an efficient order to read the key sections for maximum understanding (e.g., Abstract -> Introduction -> Methods (key parts) -> Results (key figures/tables) -> Conclusion).]
"""

equations_prompt = """# Key Equations

Highlight and explain the major equations presented in the paper. For each equation, describe its purpose, define its variables, and explain its significance to the research. Use the table below. Use LaTeX format ($$...$$ for block, $...$ for inline) for equations. Provide the output in the same language as this prompt. If a specific detail cannot be found, provide an empty string (`""`) for that item or cell, do not use placeholder text.

| Equation          | Purpose or Role in the Paper      | Why It Matters to the Research     |
|-------------------|-----------------------------------|------------------------------------|
| $$ [Equation 1] $$ | [What the equation calculates, models, or represents] | [Its role in the method, results, or theory] |
| $$ [Equation 2] $$ | [Purpose of this equation]       | [Its impact on the conclusions or findings] |
| $$ [Equation 3] $$ | [Purpose of this equation]       | [How it supports the overall argument] |
| ...               | [Add more rows as needed]         | [Corresponding significance]       |
"""

executive_summary_prompt = """# Executive Summary

Provide a high-level summary of the paper, tailored for research leads, grant reviewers, or collaborators. Focus on the problem, solution, key results, and implications using concise bullet points. Provide the output in the same language as this prompt. If a specific detail cannot be found, provide an empty string (`""`) for that item or cell, do not use placeholder text.

* **Research Problem:** [Clear articulation of the challenge the paper addresses]
* **Proposed Solution:** [Brief overview of the method, model, or approach introduced]
* **Major Results:** [Highlights of the most significant findings or achievements - use sub-bullets if needed]
* **Implications:** [Practical, theoretical, or future impact of the work]
* **Relevance:** [Why this paper is important and should be paid attention to]
"""

# ================= ================= ================= ================= ================= =================




# 1. Basic Paper Information
# basic_info_prompt = """# Basic Paper Information
#
# Extract all essential metadata to clearly identify and classify the research paper. Focus on accurately capturing the publication details.
#
# | Information       | Details                          |
# |-------------------|----------------------------------|
# | Title             | [Full title of the paper]        |
# | Authors           | [Complete list of authors]       |
# | Publication Venue | [Journal/Conference, Year]       |
# | Research Field    | [Primary domain or discipline]   |
# | Keywords          | [Relevant terms and topics]      |
# """
#
# # 2. Core Research Focus
# research_focus_prompt = """# Core Research Focus
#
# Summarize the central aim of the paper. Clearly articulate the main question, the addressed problem, and the novelty of the contribution.
#
# | Element              | Details                                         |
# |----------------------|-------------------------------------------------|
# | Research Question    | [What is being investigated?]                  |
# | Problem Statement    | [What gap or issue does the paper address?]    |
# | Main Contribution    | [What is the core offering or innovation?]     |
# | Significance         | [Why is this research important?]              |
# """
#
# # 3. Abstract Summary
# abstract_prompt = """# Abstract Summary
#
# Break down the abstract into its fundamental components for easier comprehension.
#
# | Component     | Details                               |
# |---------------|----------------------------------------|
# | Background     | [Brief context of the study]          |
# | Research Problem | [Specific issue the paper solves]    |
# | Methodology    | [Approach or technique used]         |
# | Key Findings   | [Main results or discoveries]         |
# | Conclusion     | [Primary takeaway from the study]     |
# """
#
#
# # 4. Methodology Summary
# methods_prompt = """# Methodology Summary
#
# Describe how the research was conducted, including data, tools, and procedures.
#
# | Component        | Details                                   |
# |------------------|-------------------------------------------|
# | Study Design     | [Experimental, simulation, case study etc.] |
# | Dataset          | [Source, size, preprocessing, etc.]       |
# | Techniques Used  | [Models, algorithms, or frameworks used]  |
# | Evaluation Metrics| [How success or performance was measured]|
# | Tools & Software | [Libraries, platforms, hardware specifics]|
# """
#
#
# # 5. Key Results
# results_prompt = """# Key Results
#
# List and explain the main outcomes, their impact, and how they compare to past work.
#
# | Finding # | Description of Result         | Significance / Insight           |
# |-----------|-------------------------------|----------------------------------|
# | 1         | [What was observed]           | [Why it matters]                 |
# | 2         | [What was observed]           | [Why it matters]                 |
# | 3         | [What was observed]           | [Why it matters]                 |
#
# **Comparison to Prior Work:** [Highlight how these results differ or improve upon previous research.]
# """
#
#
# # 6. Important Figures & Tables
# visuals_prompt = """# Important Figures & Tables
#
# Highlight the most critical visualizations and tabular data, explaining their importance.
#
# | Figure/Table | Description                     | Insight or Interpretation            |
# |--------------|----------------------------------|--------------------------------------|
# | Figure 1     | [What it shows]                  | [Why it's important]                |
# | Table 2      | [Data/content summary]           | [What we learn from it]             |
# | Figure 3     | [Trend or structure depicted]    | [Significance to conclusions]       |
# """
#
#
# # 7. Limitations & Future Work
# limitations_prompt = """# Limitations & Future Work
#
# Detail the limitations encountered in the research and outline proposed future directions.
#
# | Type           | Limitation Description          | Potential Impact                     |
# |----------------|----------------------------------|--------------------------------------|
# | Theoretical     | [Conceptual limits]              | [Effect on validity/generalizability]|
# | Methodological  | [Design or procedure issues]     | [Effect on robustness]               |
# | Data-Related    | [Data quality, availability]     | [Effect on conclusions]              |
#
# **Future Work Suggestions:**
# - [Proposed improvement or next step]
# - [New areas to explore]
# - [Potential experiments or applications]
# """
#
#
# # 8. Main Contributions
# contributions_prompt = """# Main Contributions
#
# List all major contributions by type, and explain how each adds value.
#
# | Category       | Contribution Summary             | Novelty or Advancement                |
# |----------------|----------------------------------|----------------------------------------|
# | Theoretical     | [New framework or insight]       | [How it extends theory]               |
# | Methodological  | [New method/model]               | [What makes it different or better]   |
# | Empirical       | [Results from data/experiments]  | [Why they matter]                     |
# | Practical       | [Applications or systems]        | [Real-world relevance]                |
#
# **Most Noteworthy Contribution:** [Summarize the biggest impact of the paper]
# """
#
# # 9. Related Work
# related_work_prompt = """# Related Work
#
# Show how this research fits into the existing landscape and what gaps it fills.
#
# | Topic/Area      | Previous Approaches             | This Paper's Innovation               |
# |------------------|----------------------------------|----------------------------------------|
# | Area 1           | [Summary of prior methods]       | [What’s new in this work]             |
# | Area 2           | [Prior attempts or models]       | [Improvements or alternatives]        |
# | Area 3           | [Old techniques or theories]     | [Enhancements introduced here]        |
#
# **Gap Addressed:** [What missing element or inefficiency this paper tackles]
# """
#
# # 10. Practical Applications
# applications_prompt = """# Practical Applications
#
# Explore how the research can be applied in real-world domains.
#
# | Domain/Industry | Use Case or Application         | Requirements or Dependencies        | Expected Timeline |
# |------------------|----------------------------------|-------------------------------------|-------------------|
# | Domain 1         | [What can be done]               | [Tech, data, adoption needs]        | [Short/Med/Long]  |
# | Domain 2         | [Another use case]               | [Feasibility factors]               | [Short/Med/Long]  |
# | Domain 3         | [Innovative potential]           | [Deployment conditions]             | [Short/Med/Long]  |
#
# **Most Promising Use Case:** [Brief highlight of top application potential]
# """
#
#
# # 11. Technical Details
# technical_prompt = """# Technical Details
#
# Dive into the specific technical aspects, including algorithms, architecture, and implementation details.
#
# | Component     | Description                        | Configuration or Parameters         |
# |---------------|-------------------------------------|--------------------------------------|
# | Algorithm     | [What algorithm is used]            | [Hyperparameters, version etc.]     |
# | Model/Architecture | [Type or design used]          | [Layers, connections, components]   |
# | Implementation| [Languages, packages, environment] | [Frameworks, hardware specifics]    |
# | Performance   | [Observed performance]              | [Accuracy, latency, etc.]           |
#
# **Code Repository:** [Link if available or mention if not provided]
# """
#
#
# # 12. Quick Summary
# quick_summary_prompt = """# Quick Summary
#
# Provide an overview of the entire paper in both concise and single-sentence formats.
#
# **Brief Summary (3–5 Sentences):**
# [Include motivation, methodology, findings, and significance.]
#
# **One-Sentence Summary:**
# [A compact summary capturing the paper’s core message.]
# """
#
#
# # 13. Reading Guide
# reading_guide_prompt = """# Reading Guide
#
# Help readers focus on the most insightful sections.
#
# | Section Name      | Key Information or Reason to Read |
# |-------------------|------------------------------------|
# | [Section A]       | [Main idea or takeaway]            |
# | [Section B]       | [Core implementation detail]       |
# | [Section C]       | [Critical results or discussion]   |
#
# **Recommended Reading Path:** [Suggestion for efficient reading – e.g., skip intro, read methods, then results]
# """
#
#
# # 14. Key Equations
# equations_prompt = """# Key Equations
#
# Highlight and explain major equations in the paper.
#
# | Equation         | Purpose                          | Variable Explanation                 | Why It Matters                     |
# |------------------|-----------------------------------|---------------------------------------|------------------------------------|
# | [Equation 1]     | [What it calculates or models]    | [Define each term]                    | [Its role in the paper]            |
# | [Equation 2]     | [Purpose]                         | [Define each term]                    | [Impact on method/results]         |
# | [Equation 3]     | [Purpose]                         | [Define each term]                    | [How it supports the conclusions]  |
# """
#
# # 15. Executive Summary
# executive_summary_prompt = """# Executive Summary
#
# Offer a high-level summary tailored for research leads, grant reviewers, or collaborators.
#
# | Section       | Description                         |
# |----------------|--------------------------------------|
# | Research Problem | [Clear articulation of the challenge] |
# | Proposed Solution| [Brief on method/model introduced]   |
# | Major Results   | [Highlights of key findings]         |
# | Implications    | [Practical, theoretical impact]      |
# | Relevance       | [Why this paper should be read]      |
# """

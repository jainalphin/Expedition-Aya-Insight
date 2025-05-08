"""
Research Paper Summarization: Key Points Extraction for Researchers
Note: Always return the summary in the same language as the original paper.
"""
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

resource_link_prompt = """Find the original link of this document from the website for this text \n\n {text} \n\n  Respond only with the Link"""
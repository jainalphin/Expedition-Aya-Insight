"""
Research Paper Summarization: Key Points Extraction for Researchers
Note: Always return the summary in the same language as the original paper.
"""
comprehensive_research_paper_prompt = """
# Research Paper Summarization: Comprehensive Analysis Guide

**Note: Always return the summary in the same language as the original paper.**

Please analyze the provided research paper and generate a comprehensive summary covering all the following sections. If a specific detail cannot be found, provide an empty string (`""`) for that item or cell, do not use placeholder text.

---

### 1. Basic Paper Information

Generate a concise summary of the paper's essential metadata using the table below:

| Information       | Details                          |
|-------------------|----------------------------------|
| **Title** | [Full title of the paper]        |
| **Authors** | [Complete list of authors]       |
| **Publication Venue** | [Journal/Conference, Year]   |
| **Research Field**| [Primary domain or discipline]   |
| **Keywords** | [Relevant terms and topics - use bullet points if multiple] |

---

### 2. Abstract Summary

Break down the paper's abstract into its fundamental components:

* **Background:** [Brief context leading to the study]
* **Problem:** [The specific issue the paper tackles]
* **Methodology:** [Approach, methods, or techniques used]
* **Key Findings:** [Main results or discoveries - use sub-bullets if needed]
* **Conclusion:** [Primary takeaway or implication]

---

### 3. Methodology Summary

Describe how the research was conducted:

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

---

### 4. Key Results

List and explain the paper's main outcomes and their importance:

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

---

## 5. Technical Details

Provide a concise summary of the paper's specific technical aspects:

| Component         | Description                        | Key Configuration or Parameters         |
|-------------------|-------------------------------------|--------------------------------------|
| **Algorithm(s)** | [What specific algorithm(s) are central?] | [Key hyperparameters, variations used, etc.] |
| **Model/Architecture**| [Type or design of the model/system] | [Number of layers, components, specific structure details] |
| **Implementation**| [Languages, key libraries, environment specifics] | [Frameworks used (TensorFlow, PyTorch, etc.), notable dependencies] |
| **Performance** | [Key performance metrics reported]    | [Results achieved (e.g., Accuracy %, F1 score, latency ms)] |
| ...               | [Add more rows as needed]           | [Corresponding details]              |

---

### 6. Key Equations

Highlight and explain the major equations presented in the paper (use LaTeX format):

| Equation          | Purpose or Role in the Paper      | Why It Matters to the Research     |
|-------------------|-----------------------------------|------------------------------------|
| $$ [Equation 1] $$ | [What the equation calculates, models, or represents] | [Its role in the method, results, or theory] |
| $$ [Equation 2] $$ | [Purpose of this equation]       | [Its impact on the conclusions or findings] |
| $$ [Equation 3] $$ | [Purpose of this equation]       | [How it supports the overall argument] |
| ...               | [Add more rows as needed]         | [Corresponding significance]       |

---

### 7. Related Work

Show how this research fits into the existing landscape:

| Topic/Area      | Previous Approaches             | This Paper's Innovation / Difference   |
|-----------------|----------------------------------|----------------------------------------|
| **[Relevant Area 1]**| [Summary of how prior work handled this] | [What new approach, technique, or finding is introduced here?] |
| **[Relevant Area 2]**| [Other related methods or studies] | [How does this paper build upon or deviate from them?] |
| **[Relevant Area 3]**| [Existing theories or models]    | [Enhancements, alternatives, or validations provided by this work] |
| ...             | [Add more rows as needed]        | [Corresponding innovation]             |

**Gap Addressed:**
* [What specific problem, limitation, or missing piece in the existing literature does this paper tackle?]

---

### 8. Limitations & Future Work

Detail the limitations encountered and suggested future directions:

**Limitations:**
* **Theoretical:** [Conceptual limits of the approach, with brief impact]
* **Methodological:** [Issues with design or procedure, with brief impact]
* **Data-Related:** [Constraints due to data quality/availability, with brief impact]
* [Add other relevant limitations]

**Future Work Suggestions:**
* [Proposed next steps or improvements to the current work.]
* [New areas or questions for future research based on these findings.]
* [Potential experiments or applications to explore.]

---

### 9. Practical Applications

Explore potential real-world applications of the research findings:

| Domain/Industry | Potential Use Case or Application   | Key Requirements or Dependencies        | Feasibility/Timeline |
|-----------------|--------------------------------------|-------------------------------------|-------------------|
| **[Domain 1]** | [How can the results/methods be used here?] | [What data, technology, or infrastructure is needed?] | [Short/Med/Long term]  |
| **[Domain 2]** | [Another potential application area] | [Factors affecting feasibility or adoption] | [Short/Med/Long term]  |
| **[Domain 3]** | [Innovative potential application] | [Challenges or conditions for implementation] | [Short/Med/Long term]  |
| ...             | [Add more rows as needed]            | [Corresponding requirements]          | [Corresponding timeline]  |

**Most Promising Use Case:** [Briefly highlight the application with the highest potential impact or feasibility.]

---

*Please provide a thorough analysis covering all these sections based on the research paper content.*

{research_paper}
"""
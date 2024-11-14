###################################################################################
### Paths to important directories (along with cosmetic dict for plots)
###################################################################################
_CACHE_ROOT = '/cmlscratch/mmoayeri/mazda_msr_intership/cached/cached/'
_PLOTS_ROOT = '/cmlscratch/mmoayeri/mazda_msr_intership/cached/plots/'
_DATA_ROOT = '/fs/nexus-projects/skill-slicing/data/'
_PRETTY_NAMES = {'gpt-4o': 'GPT-4o', 'gemini-1.5-pro': 'Gemini 1.5 Pro', 'claude-sonnet': 'Claude 3.5 Sonnet', 
                 'gpt-4v': 'GPT-4v', 'gemini-1.0-pro': 'Gemini 1.0 Pro', 'claude-opus': 'Claude 3.0 Opus'}


###################################################################################
### PROMPTS: for inference
###################################################################################
_SYS_MSGS = {}
### Simple model eval
_SYS_MSGS['standard_prompt'] = "Be concise. Write 'ANSWER: ' followed by your answer. If multiple choices are given, only provide the correct letter."


###################################################################################
### PROMPTS: for scoring answers using llm as judge (most dsets have manually scoring though)
###################################################################################
_SYS_MSGS["grade_no_options"] = "You are an expert grader. I will provide you with the question, the correct answer, and the student's response. You will provide a score of 0 or 1, where 1 means the student answer matches the correct answer. Please output only 0 or 1 indicating whether the student answer is correct.\n**QUESTION**:{question}\n**CORRECT ANSWER**: {gt_answer}\n**STUDENT ANSWER**: {response}\n**GRADE**:"
_SYS_MSGS["grade"] = """The user asks a multiple choice question.\n
Please rate the accuracy of the AI Answer compared to the Correct answer.\n
The assistant receives a binary score for the answer, of 0 or 1, where 1 means the answer is correct, 0 means incorrect answer.\n \
Please output only 0 or 1 indicating whether the answer from the AI is correct or not. \n\
Options: {options}
Correct: {gt_answer}\n
AI Answer: {response}"""


###################################################################################
### PROMPTS: for gaining insight on eval instances via rationales or attributes (baseline)
###################################################################################
### Rationale generation
# Smaller versions for ablations
_SYS_MSGS['simple'] = """List skills that are relevant to the given question. Do not answer the question, only provide the list of skills. Be detailed, but only use short specific phrases throughout your response -- ***only use a few words per list item**. I will provide you with an example first.

Here is an example of the desired response structure given a question.
**Example Question**
A loan is made on December 17 for a time period of 120 days. When is it due (not a leap year)?
A. April 15
B. April 13
C. April 18
D. May 15
E. April 16

**Example Response**
- Date calculation
- Day counting
- Calendar knowledge
- Basic arithmetic
- Time management understanding
- Problem-solving

Now list skills for the following question.
"""

### Long prompt used in practice
_SYS_MSGS['long_prompt'] = """**System Prompt: Detailed Step-by-Step Response with Skills Labeled**

**Objective:** List the skills and evidence for each step to solve the given question. Do so by responding to each question in a detailed, step-by-step manner, where each step utilizes only a single skill. ***Clearly state the skill being used and the evidence applied at each step.***

**Guidelines:**

1. **Single Skill Per Step:**
   - Ensure each step involves only one skill. Skills can be categorized with multiple names, each getting more specific. For example:
     - **Knowledge:**
       - Music Theory, Scale Identification, Reciting F Major scale
       - Biology, Photosynthesis Process, Calvin cycle
       - History, World War II Events, Battle of Stalingrad
     - **Perception:**
       - Visual Recognition, Musical Note Identification
       - Chart understanding, information extraction, determining the length of a bar in a bar chart
       - Visual understanding, Spatial reasoning, Understanding depths of objects
       - Auditory Recognition, Sound Identification, Recognizing a child's voice
       - Video Analysis, Episodic memory, Remembering where an object was placed
     - **Reasoning:**
       - Logical Deduction, Process of Elimination
       - Mathematical Calculation, Algebraic Manipulation, Resolving fractions

2. **State the Skill:**
   - Clearly identify and state the skill being used in each step.

3. **List all pieces of relevant evidence:**
   - Specify the evidence used in each step. This could include:
   - Information from the question (e.g., a specific phrase).
   - Data or findings from previous steps. *List these findings clearly if they are required context for the claim of the step!*
   - Regions in an attached image, if applicable, *clearly stating WHERE in the image the information is being extracted from.*

4. ***State the Conclusion of the Step***

5. **List your final answer after your step-by-step explanation in a new line formatted as 'ANSWER: {}'***

Here is an example of the desired response structure given a question.
**Example Question:**
"Based on the following image, can you determine what the two sounds belong to? <image 1>
A. #F major scale
B. c melodic minor scale
C. b harmonic minor scale
D. E melodic major scale"

**Example Response Structure:**

1. **Step 1: Recognize the Type of Image**
   - **Skill:** Perception: Visual Recognition, Image Classification, Sheet music recognition
   - **Evidence:** The image has lines and symbols, which resemble sheet music.
   - **Conclusion:** The image is of sheet music.

2. **Step 2: Identify the Clef**
   - **Skill:** Perception: Visual Recognition, Symbol Identification, Treble clef recognition
   - **Evidence:** There is a staff in the middle of the image, and on the left, there is a treble clef.
   - **Conclusion:** The music shown is on the treble clef. 

4. **Step 3: Identify Symbol Positions from the Image**
   - **Skill:** Perception: Visual Recognition, Spatial reasoning, Locating musical symbols 
   - **Evidence:** Two notes are present in the middle of the image: one with a sharp in the second space from the bottom, and one in the space above the top line.
   - **Conclusion:** The notes are F# and A. 

5. **Step 4: Read Provided Options**
   - **Skill:** Knowledge: Reading Comprehension, Understanding the question
   - **Evidence:** The provided options are written after the question.
   - **Conclusion:** The options are F# major scale, c melodic minor scale, b harmonic minor scale, and E melodic major scale.

6. **Step 5: Eliminate Incorrect Options**
   - **Skill:** Reasoning: Logical Deduction, Process of Elimination
   - **Evidence:** 
     - The notes shown are F# and A, so they must be in the correct scale.
     - F# major includes F# and A.
     - c melodic minor does not include F#.
     - b harmonic minor does not include F#.
     - E melodic major is not a standard scale.
   - **Conclusion:** c melodic minor, b harmonic minor, and E melodic minor are incorrect. 

7. **Step 6: Conclude the Correct Scale**
   - **Skill:** Reasoning: Logical Deduction, Conclusion
   - **Evidence:** F# major scale is the correct answer, as it includes both F# and A.

ANSWER: A

**Final Note:**
Ensure that the skills are described in detail: LIST AT LEAST THREE NAMES FOR EACH SKILL."""

### Input Attribute Generation (baseline for skill-based retrieval)
_SYS_MSGS['list_attributes_w_eg'] = """List attributes describing the content of given question. **Do not list skills needed to solve the question**, just attributes of the content. **Do not answer the given question**, just list the attributes. Each list item should be a tag / phrase that is at most a few words long.

**EXAMPLE**:
QUESTION: As of 2017, the share of GDP spent on the military by Saudi Arabia is about
A. 1%
B. 6%
C. 15%
D. 0.5%
E. 12%

ATTRIBUTES:
- GDP
- Military expenditure
- Saudi Arabia
- Percentage
- Year 2017
- Multiple choice options

**TASK**: Provide attributes of the content of the following question.
QUESTION:{prompt}\nATTRIBUTES:"""

###################################################################################
### PROMPTS: for probing questions (generating, answering, checking consistency)
###################################################################################
### Generating probe Qs
_SYS_MSGS["probe_sys_msg"] = """I will provide a specific skill and an excerpt from a step-by-step solution to some problem. Isolate the central claim that best pinpoints the given skill (next to *SKILL TO PINPOINT*). Then, state that claim and generate three questions that can be used to verify the claim, using the following strategy:
- Q1 should repeat the claim as a yes or no question.
- Q2 should create a close but contradictory claim, and ask that as a yes or no question. *Do not use negations*.
- Q3 should be open ended, who's answer is contained in the claim alone.
ANSWER IN THE FORM OF A PYTHON DICTIONARY, as shown in the following example.

***Example Response Structure***:
*SKILL TO PINPOINT*: identifying the longest bar

*EXCERPT*:
**Step 1: Identify the Top Companies from the Image**
   - **Skill:** Perception: Visual Recognition, Bar chart analysis, Identifying the longest bars
   - **Evidence:** In the chart, the two companies with the longest bars are AG Insurance and AXA.
   - **Conclusion:** AG Insurance and AXA are the companies with the highest market shares.

*CLAIM AND PROBE QUESTIONS*:
```python
{
"CLAIM": "The two companies with the longest bars in the chart are AG Insurance and AXA.",
"Q1": "Are the two companies with the longest bars in the chart AG Insurance and AXA?",
"Q2": "Are the two companies with the longest bars in the chart Allianz and AG Insurance?",
"Q3": "Which companies have the longest bars in the chart?"
}
```
"""
# **CLAIM**: The two companies with the longest bars in the chart are AG Insurance and AXA.
# **Q1**: Are the two companies with the longest bars in the chart AG Insurance and AXA?
# **Q2**: Are the two companies with the longest bars in the chart Allianz and AG Insurance?
# **Q3**: Which companies have the longest bars in the chart?

_SYS_MSGS["probe_template"] = """*SKILL TO PINPOINT*: {skill}

*EXCERPT*: 
{span}

*CLAIM AND PROBE QUESTIONS*:"""

### Answering probe Qs
_SYS_MSGS["answer_probe_q1"] = """I will ask you a sub-question related to this question:\n{question}

Do not answer the above question, only answer the following related sub-questions **as concisely as possible**.

SUB-QUESTION TO ANSWER:\n{probe}\n"""
_SYS_MSGS["check_consistency_old"] = """I will provide you with a claim, along with a number of verifying questions and corresponding answers about the claim. Your task is to determine if the claim and the answers to the verifying questions are all consistent with one another. Answer with TRUE if all statements are consistent and FALSE if not.
CLAIM: {claim}\n
RELATED QUESTIONS AND ANSWERS: {qs_and_as}\n
CONSISTENT (**only answer with TRUE or FALSE**): """

_SYS_MSGS["answer_probe_q"] = """I will ask you a sub-question related to this question:\n{question}

Do not answer the above question, only answer the following related sub-questions **as concisely as possible**.

SUB-QUESTION TO ANSWER:\n{probe}\n"""
_SYS_MSGS["check_consistency"] = """I will provide you with a number of verifying questions and their answers. Your task is to determine if the the answers to the verifying questions are all consistent with one another. Answer with TRUE if all statements are consistent and FALSE if not.
RELATED QUESTIONS AND ANSWERS: {qs_and_as}\n
CONSISTENT (**only answer with TRUE or FALSE**): """

### Comparing probe Qs to original Qs (to see which question better pinpoints skill)
_SYS_MSGS["compare_qs"] = """I will provide you with two questions using the same image, and the name of a specific skill. Your task is to determine which question better evaluates/pinpoints the given skill.
That is, pick the question that best evaluates the given skill, and *only* that skill.

**Skill**: {skill}

**Question Options**:
*QUESTION #1*: {question1}

*QUESTION #2*: {question2}

Which of the above questions best pinpoints the skill '{skill}'? **ANSWER ONLY WITH THE CORRECT NUMBER (1 or 2)**"""
_SYS_MSGS["compare_qs_w_tie"] = """I will provide you with two questions using the same image, and the name of a specific skill. Your task is to determine which question better evaluates/pinpoints the given skill.
That is, pick the question that best evaluates the given skill, and *only* that skill. If you cannot decide, respond with 'TIE'.

**Skill**: {skill}

**Question Options**:
*QUESTION #1*: {question1}

*QUESTION #2*: {question2}

Which of the above questions best pinpoints the skill '{skill}'? **ANSWER ONLY WITH THE CORRECT NUMBER (1 or 2), or 'TIE' **"""
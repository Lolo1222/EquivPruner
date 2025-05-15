COT_EXAMPLES = None
# COT_TASK_DESC = """<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}. Additionally, follow these rules for generating responses:\n\n1. If the problem can be solved using fast thinking (quick reasoning or straightforward solutions), output `<FAST>` followed by the fast-thinking solution process, formatted as: `<FAST> + fast-thinking solution`.\n2. If the problem requires slow thinking (detailed analysis, complex reasoning, or step-by-step problem-solving), output `<SLOW>` followed by the slow-thinking solution process, formatted as: `<SLOW> + slow-thinking solution`.\n\nExamples:\n- Fast-thinking solution: `<FAST> This problem can be solved quickly by applying [method/principle]. The answer is \\boxed{{answer}}.`\n- Slow-thinking solution: `<SLOW> This problem requires detailed analysis. First, [step 1]. Then, [step 2]. Finally, [step 3]. The answer is \\boxed{{answer}}.`\n\nAlways choose the appropriate format based on the complexity and requirements of the question.<|im_end|>"""
# PROBLEM_FORMAT_STR = """<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"""
COT_TASK_DESC = "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>"
PROBLEM_FORMAT_STR = """<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"""

SEP = "\n\n"
# For math-sheperd
# SEP = "ки\n"

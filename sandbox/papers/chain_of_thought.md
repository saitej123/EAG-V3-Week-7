# Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Wei et al., 2022)

**Authors:** Jason Wei, Xuezhi Wang, Dale Schutmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc V. Le, Denny Zhou  
**Venue:** NeurIPS 2022  
**Core idea:** Ask language models to produce **intermediate natural-language reasoning steps** before the final answer, dramatically improving multi-step reasoning without fine-tuning on many tasks.

## Problem setting

Large language models (LLMs) scale well on many NLP benchmarks but often fail on tasks requiring **multi-hop reasoning**: grade-school math word problems (GSM8K), commonsense QA with implicit steps, symbolic manipulation, and date understanding. Standard prompting asks for an answer directly (“Question → Answer”), which encourages shortcut guessing when the model must combine several facts.

Chain-of-thought (CoT) prompting instead provides a few **demonstrations** where the model sees human-written rationale lines ending in the correct answer. At test time the model continues the pattern: it emits its own rationale, then the final answer.

## Linear stepwise reasoning (defining property)

CoT traces are **linear narratives**: Step 1, Step 2, Step 3, …, therefore the answer. There is **no interleaving of external actions**—no search API calls, no calculator tools, no environment observations mid-stream. The entire reasoning process unfolds as **continuous text in the context window**.

This contrasts with agent frameworks that alternate Thought / Action / Observation. CoT stays inside the language channel from prompt to completion.

**Format example (conceptual):**  
*Q: Roger has 5 tennis balls. He buys 2 cans of 3 balls each. How many now?*  
*A: Roger started with 5. Each can has 3, so 2 cans is 6. 5 + 6 = 11. The answer is 11.*

The paper shows that **order matters**: steps should mirror human-readable decomposition rather than jumping to the result.

## Mechanism: why it helps

Several hypotheses appear in the paper and follow-ups:

1. **Decomposition pressure:** forcing intermediate sentences allocates capacity to subproblems instead of one-shot pattern completion.
2. **Increased compute in depth:** longer generations give the transformer more layers of self-attention over its own partial reasoning (though the paper focuses on prompting, not architectural change).
3. **In-context algorithm induction:** demonstrations teach a procedural template the model imitates on new inputs.

CoT gains are largest on models above ~100B parameters in the original study; smaller models may produce fluent but incorrect chains.

## Gradients and learning (fine-tuning context)

While the headline result is **prompting**, the paper and successors discuss **supervised fine-tuning on rationale-augmented datasets**. When labels include reasoning steps, **error signals from the final answer propagate backward through intermediate tokens** in the standard language-modeling loss. Each reasoning token participates in the computation graph; the model learns which intermediate hops correlate with correct outcomes.

This is distinct from reinforcement learning with sparse terminal rewards only: teacher-forced rationales provide **dense token-level supervision** along the chain. Follow-up work (STaR, rationalization pipelines) generates synthetic chains and filters by answer correctness.

**Implication for training:** intermediate reasoning tokens are not passive text—they receive **gradient updates** linking phrasing of steps to task success when fine-tuned with chain labels.

## Experimental highlights

- **GSM8K:** PaLM 540B with CoT prompting reaches high accuracy versus direct answering; gains shrink if demonstrations omit reasoning steps.
- **StrategyQA, SVAMP, AQuA, Date Understanding, Sports Understanding, SayCan (robot planning language):** consistent improvements when chains are enabled.
- **Ablation:** using wrong rationale text that still ends with the right answer hurts performance, suggesting the model uses step content—not just length.

## Variants discussed

- **Zero-shot CoT:** append “Let’s think step by step” without full demonstrations; surprisingly effective on some tasks.
- **Self-consistency:** sample multiple chains, majority-vote final numeric answers; reduces variance from a single greedy decode.
- **Complexity-based prompting:** select longer demonstration chains for harder problems.

## Limitations

- **Faithfulness:** models may produce plausible but unfaithful rationales (post-hoc justification). CoT text is not guaranteed to reflect internal computation.
- **Cost:** longer outputs increase latency and inference price.
- **Brittleness:** formatting changes, irrelevant text in demonstrations, or adversarial injections in context can derail the chain.
- **No grounding:** without tools, arithmetic errors persist unless the model’s weights encode correct operations.

## Comparison axis for multi-paper evals

When comparing to **ReAct-style agents**, CoT emphasizes **uninterrupted internal monologue**. ReAct pauses the monologue to act in an environment, then resumes with observations. CoT is the baseline for “pure language reasoning traces.”

## Retrieval-friendly phrases

- linear stepwise reasoning  
- intermediate reasoning tokens before the final answer  
- backpropagation through reasoning steps when fine-tuned with chain labels  
- in-context demonstrations of multi-hop rationale  
- no tool or environment actions during the chain  

## Summary

Chain-of-thought prompting elicits **step-by-step natural language reasoning** before answers, improving multi-hop tasks via in-context learning. Its defining structural property is a **single forward narrative** without external actions; training extensions treat each reasoning token as part of the supervised path from question to label.

## Extended discussion for RAG evaluators

**Prompt sensitivity:** swapping demonstration order or using semantically similar but numerically wrong chains reduces GSM8K accuracy, indicating models imitate procedure—not merely output length.

**Arithmetic without tools:** CoT does not invoke calculators; errors like 17×13 mistakes persist unless the base model already encodes arithmetic. Tool-augmented variants (Program-of-Thought, PAL) export steps to Python—outside vanilla CoT.

**Multilingual behavior:** later work shows CoT transfers unevenly; reasoning templates in English sometimes help non-English questions when demonstrations are English-only.

**Evaluation metrics:** accuracy on final answer only; partial credit for correct intermediate steps is rarely scored in automatic benchmarks, hiding cases where rationale is wrong but guess is right.

**Dataset contamination concerns:** GSM8K-style problems may appear in pretraining corpora; CoT gains must be interpreted alongside contamination audits in frontier models.

For cross-paper questions about **learning influence across intermediate steps**, cite CoT fine-tuning: token-level cross-entropy assigns **gradient credit** along the rationale span toward the final label, unlike sparse terminal-only rewards in naive RL setups.

## Additional benchmark context

**BIG-bench and beyond:** CoT-style instructions appear in many BIG-bench tasks; gains correlate with model scale more than prompt length alone.

**Educational use:** instructors use CoT demonstrations to teach exam-style worked solutions; models mimic layout (Given …, Step 1 …) even when arithmetic is wrong.

**Token budget planning:** production systems cap max rationale length to control cost; truncation mid-chain hurts final accuracy on hard math sets.

**Relation to scratchpad methods:** scratchpad training explicitly labels intermediate variables; CoT is the zero-shot / few-shot prompting analogue without parameter updates.

## Paper reproduction checklist

Researchers reproducing Wei et al. (2022) should fix model scale (PaLM 540B in original), match demonstration count (8-shot typical), and use identical decoding (greedy vs sampling). Self-consistency requires drawing ≥5 samples at non-zero temperature. Report both exact-match and flexible numeric parse for math sets because formatting differences ("11" vs "11.0") affect scoring scripts independently of reasoning quality.

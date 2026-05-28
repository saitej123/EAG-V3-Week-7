# ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022)

**Authors:** Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao  
**Venue:** ICLR 2023  
**Core idea:** Interleave **natural-language reasoning traces** with **task-specific actions** (search, lookup, interact) so models ground plans in external evidence instead of hallucinating facts.

## Motivation

Pure chain-of-thought reasoning keeps all computation inside parametric memory. For knowledge-intensive tasks (HotpotQA, FEVER fact verification, AlfWorld household simulation), models hallucinate entities or relations not stored reliably in weights.

Classic **action-only** agents (say, a reinforcement learning policy mapping observations to tool calls) lack explicit deliberation—they struggle on tasks needing multi-step plans with interpretable intermediate intent.

**ReAct** combines both: the model writes **Thought** lines explaining what it intends, chooses an **Action** from a allowed set, receives an **Observation** from the environment, then continues.

## Interleaving reasoning and tool actions

A typical trace looks like:

```
Thought 1: I need the birthplace of the director of film X.
Action 1: Search[director of film X]
Observation 1: The director is Jane Doe, born in Oslo.
Thought 2: I should confirm Oslo details for the question.
Action 2: Lookup[Oslo population]
Observation 2: ...
Thought 3: I can answer now ...
Action 3: Finish[Oslo, 693494]
```

**Key structural difference from chain-of-thought:** the narrative **stops** at each Action boundary. The model cannot proceed until the environment returns Observation text appended to context. Reasoning and acting **alternate** rather than forming one uninterrupted paragraph.

This design reduces unsupported factual leaps because Thoughts must reconcile with Observations before the next Thought.

## Action spaces by benchmark

- **HotpotQA / FEVER:** Wikipedia API with `Search[entity]` and `Lookup[keyword]` returning article sentences.
- **AlfWorld:** textual household commands (`go to`, `take`, `put`) with simulator feedback.

Prompts include few-shot ReAct trajectories demonstrating valid Thought/Action/Observation formatting.

## Intermediate signals and feedback quality

**Observations** inject **external feedback** into the context. The model must update beliefs when observations contradict earlier Thoughts. Paper ablations show removing Thoughts hurts success rates—actions alone wander; removing observations collapses grounding.

**Signal path:** action correctness and observation formatting strongly affect downstream reasoning. Noisy search snippets or ambiguous simulator messages propagate through later Thoughts.

For training analysis, each Thought conditions on prior Observations; gradients during fine-tuning (where available) flow through action tokens and subsequent reasoning, but the paper’s main experiments are **in-context prompting** with frozen models.

## Empirical results (selected)

- **HotpotQA:** ReAct outperforms imitation-learning action-only baselines and matches or beats CoT on questions needing retrieval, with lower hallucination rate in human evals.
- **FEVER:** ReAct improves fact verification when claims require lookup.
- **AlfWorld:** interleaved reasoning helps multi-step plans in interactive environments.

**Failure modes:** repetitive search loops, invalid action syntax, ignoring contradictory observations, over-long contexts when many steps accumulate.

## Comparison to chain-of-thought

| Aspect | Chain-of-thought | ReAct |
|--------|------------------|-------|
| External tools | No | Yes (search, lookup, env) |
| Trace shape | Linear text | Alternating Thought/Action/Observation |
| Grounding | Parametric memory only | Retrieved / simulated evidence |
| Best for | Math, symbolic puzzles with in-weight skills | Knowledge-intensive QA, interactive tasks |

Eval query **H** contrasts **intermediate reasoning treatment**: CoT keeps reasoning inside a continuous language chain; ReAct **segments** reasoning around environment-facing actions.

## Design lessons for agents

1. **Explicit Thoughts improve action selection** even without gradient updates—interpretability bonus for debugging trajectories.
2. **Tool interfaces should return concise Observations**; dumping entire pages can dilute attention.
3. **Stop conditions** (`Finish[answer]`) must be part of the action grammar to terminate loops.

## Limitations

- Prompt engineering overhead: few-shot trajectories are long.
- Latency stacks with each environment call.
- Security: unconstrained action spaces risk unsafe tool use in open deployments.
- Does not replace parametric reasoning for tasks where tools add no value.

## Retrieval-friendly phrases

- interleaving reasoning and tool actions  
- Thought Action Observation loop  
- external observations as intermediate signals  
- grounding language plans in retrieved evidence  
- pauses internal narrative to act in the environment  

## Summary

ReAct synergizes **verbal reasoning** with **executable actions**, using environment observations to steer multi-step tasks. Its hallmark is **alternation** between deliberation and action—not a single uninterrupted reasoning monologue.

## Extended discussion for RAG evaluators

**Context window growth:** each Action/Observation pair appends tokens; long HotpotQA episodes may hit context limits unless observations are truncated or summarized.

**Prompt brittleness:** invalid Action syntax (missing brackets, wrong API name) yields empty observations; models must recover in subsequent Thoughts—failure modes include infinite Search loops.

**Human evaluation:** paper reports reduced hallucinated entities versus CoT on knowledge tasks because Observations anchor entities to retrieved text.

**Fine-tuning variants:** follow-ups (FireAct, agent tuning) supervise Action formats; gradients then flow through action tokens conditioned on prior Thoughts, mixing language planning with discrete tool choice.

**Safety:** open-web Search actions require filtering; ReAct assumes benign Wikipedia snapshots in academic benchmarks.

When contrasting with chain-of-thought for exam-style synthesis, emphasize: CoT = **continuous internal trace**; ReAct = **segmented trace punctuated by environment feedback** and **tool actions** that supply intermediate evidence beyond parametric memory.

## Additional benchmark context

**WebShop and e-commerce simulators:** later agent papers adopt ReAct formatting for product search trajectories; observation strings mimic catalog snippets.

**API cost modeling:** each Action may trigger paid search; budgets limit max steps per episode in production agents derived from ReAct prompts.

**Structured parsing:** deployments wrap Action lines with regex parsers; malformed outputs trigger retry prompts or fallback to CoT-only mode.

**Memory modules:** long-horizon variants summarize prior Observations into a scratchpad while preserving ReAct alternation at the outer loop.

## Paper reproduction checklist

ReAct reproductions should log full trajectories (Thought/Action/Observation) for error analysis, cap max steps to prevent runaway loops, and use the same Wikipedia snapshot or API wrapper as the benchmark harness. Human evaluators often score **supporting evidence**—whether final answers cite observation text—separate from exact match. When comparing to CoT on HotpotQA, ensure both conditions receive identical retrieval corpora; CoT-only baselines must not silently access Wikipedia unless that is the intended ablation.

## Industrial agent stacks influenced by ReAct

Modern tool-using assistants (search-augmented chat, code interpreters, browser agents) inherit the **Thought → tool call → tool result → resume** pattern even when internal APIs differ from the paper's bracket syntax. Observability products parse trajectories into spans labeled "planning" vs "tool" for latency tracing. Evaluators grading cross-paper synthesis should note ReAct's **intermediate observations** as first-class inputs to later thoughts, whereas chain-of-thought treats all intermediate material as self-generated language without external provenance tags.

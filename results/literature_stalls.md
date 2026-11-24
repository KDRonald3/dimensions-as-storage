# Training tiny language models faster: what the research says about stand-stills

Literature review made on 22 November 2026 by a Claude Opus 5.5 subagent, at the owner's request.
The subagent confirmed 37 papers through the arXiv listing service and read the relevant sections
of eight of them; most papers came from its own knowledge and were then checked against arXiv.
Claude (the main session) checked that six of the key papers exist under the titles given
(sources 6, 7, 9, 16, 17, 19) and that the code settings named below exist. Nothing else here was
checked independently, and none of the remedies has been tried in this project yet.

## 1. The answer in short

- Researchers call this a **loss plateau** followed by **abrupt learning**. The plain-English name
  is a stand-still; some also call it a **phase transition** or **stagewise learning**. It is well
  documented in small transformers on synthetic tasks. It is *not* grokking: grokking is about
  held-out data, and this stall is on the training data.
- The usual explanation: the model sits near a flat "saddle" (a point where the slope is almost
  zero in every direction). An attention pattern (which words each word looks at) has to form
  first, and it forms slowly and invisibly. Whether a run gets out quickly is partly luck.
- Best-supported remedies, roughly in order: **reshape the data** (skewed frequencies, a warm-up
  on a subset, or mixing in easier examples), **sharpen or normalize attention**, **start from
  larger weights**, **use a smaller Adam epsilon or bigger batches**, and **run several seeds**.
- Weight decay and grokking tricks such as Grokfast have little evidence for stalls on training
  data.

## 2. What the stand-still is, and why it happens

A **saddle point** is a spot in parameter space (the space of all possible weight values) where
the loss is flat in most directions. Gradient descent (the rule that nudges the weights downhill)
slows to a crawl there. Dauphin et al. (2014) argue that in big parameter spaces, saddles rather
than true dead ends cause most plateaus. Saxe et al. (2013) solved simple deep networks exactly.
They showed long plateaus, then rapid drops, then a new plateau, one "mode" at a time. Later work
calls this **saddle-to-saddle dynamics** (Jacot et al. 2021; Abbe et al. 2023). That matches the
facts arriving one kind at a time. Michaud et al. (2023) describe skills learned as separate
"quanta".

**Simplicity bias** means networks learn simple statistics first, such as how often things occur,
before combinations. Belrose et al. (2024) measured this. The models here learn what each animal
is "usually" like before they learn how the season changes it, which looks like the same thing.
Chen et al. (2023) found a sudden loss drop when attention heads begin tracking grammar.

In transformers the bottleneck is usually **attention**:

- Hoffmann et al. (2023) traced their "Eureka moments" to the softmax, the step that turns
  attention scores into weights. Early on it spreads attention almost evenly, so the gradients
  that would teach it to focus are tiny.
- Gopalani & Hu (2025) found that during the plateau the model learns an easy partial solution.
  Meanwhile attention progresses quietly underneath.
- Barak et al. (2022) proved for a related problem that progress continues during the flat
  stretch, but the score does not show it.

**Luck:** Edelman et al. (2023) show that success depends on how "rich" (wide), patient or lucky a
run is. Width acts as a parallel search for lucky neurons, which fits wider models getting through
sooner. Hu et al. (2023) found that some seeds take "detours" costing up to three times longer.
The thread-count result of this project shows the same sensitivity.

## 3. What to try, most promising first

**1. Reshape the data (curriculum).**
- **What to change:** use `usage="zipf"` (already in RunConfig), or train first on a few animals
  (`animals`) and then switch to all of them. For the season plateau, oversample sentences where
  the season actually changes the answer. That last idea is the reviewer's own and untested.
- **What was found:** Zucchet et al. (2025) used a synthetic fact-recall task close to this one.
  They found a plateau before facts are learned that grows almost linearly with the number of
  individuals. Skewed frequencies (best exponent 0.6 to 0.8) or a warm-up on a subset shortened
  it. They say its length depends more on the data than on model size. Kirsch et al. (2022) found
  that biasing part of each batch toward an easier fixed task "completely eliminates the loss
  plateau". Fu et al. (2023) got similar results by mixing.
- **Strength of evidence:** moderate to strong, with consistent results across three groups.
- **Carry-over to models this small:** Zucchet's models had about 44M+ parameters, but the
  mechanism is about the data, so it should carry over fairly well.
- **Work:** small.

**2. Sharpen or normalize attention.**
- **What to change:** in model.py, scale the attention scores up, for example by passing `scale=`
  to `scaled_dot_product_attention`. Hoffmann's "tau = 1/3" means scores 3 times larger.
  Alternatively, use NormSoftmax, which divides the scores by their measured spread.
- **What was found:** Hoffmann et al. (2023) on multi-step tasks; answering a yes/no question
  needs recall plus comparison, which is two steps. Their baseline learned the hidden step in 7 of
  10 seeds after about 207 epochs. NormSoftmax learned it in 10 of 10 after about 28, and in 36 of
  36 runs across different learning rates. Gopalani & Hu (2025) removed the plateau by nudging
  attention toward the right pattern.
- **Strength of evidence:** moderate. It includes a 1-layer, 4-head model.
- **Carry-over:** fairly good.
- **Work:** small to medium.

**3. Larger starting weights.**
- **What to change:** `--init-std` in model.py (now 0.02). Try 0.05 or 0.1.
- **What was found:** Gopalani & Hu (2025) found that a smaller starting scale gives a longer
  plateau. Theory agrees: small starting weights sit near a saddle (Jacot et al. 2021). But Kumar
  et al. (2023) found that too large a scale delays feature learning ("lazy" training).
- **Strength of evidence:** mixed. There may be a sweet spot.
- **Work:** trivial.

**4. Run several seeds and treat escape time as a distribution.**
- **What to change:** run 3 to 5 seeds per setting, report the median and the spread, or keep the
  first run that gets through.
- **What was found:** Hu et al. (2023) and Edelman et al. (2023) show that luck matters. Hu et al.
  also found that normalization layers or smaller batches removed the detours.
- **Strength of evidence:** strong that luck matters.
- **Work:** compute only.

**5. Smaller Adam epsilon.**
- **What to change:** pass `eps=1e-12` to `AdamW` in train.py (the current default is 1e-8).
  Epsilon is a small number Adam adds to stop its steps blowing up. If gradients are about that
  small, it quietly shrinks the steps.
- **What was found:** Kirsch et al. (2022) found that a smaller epsilon "more than halv[ed] the
  plateau length". Wortsman et al. (2023) saw this effect only in very large models.
- **Strength of evidence:** one paper on plateaus.
- **Tip:** first log how big the gradients are. If they are far above 1e-8, this setting will not
  matter.
- **Work:** trivial.

**6. Bigger batch.**
- **What to change:** `batch`.
- **What was found:** Kirsch et al. (2022) found a power law: bigger batches gave shorter
  plateaus.
- **Strength of evidence:** conflicting. Hu et al. (2023) found *smaller* batches removed detours.
- **Work:** trivial.

**7. Give more weight to the hard part, or add a helper loss.**
- **What to change:** up-weight the loss on the yes/no answer word, which is one word in five of a
  question, or on words right after a season. This is the reviewer's suggestion.
- **What was found:** Fu et al. (2023) shortened plateaus with an auxiliary loss, an extra target
  that predicts the hidden facts (weight 0.1). Pezeshki et al. (2020) show that features which are
  easy to learn starve harder ones of gradient.
- **Strength of evidence:** indirect.
- **Work:** medium.

**8. Learning rate.**
- **What to change:** `learning_rate`.
- **What was found:** Gopalani & Hu (2025) found that smaller rates give longer plateaus. The
  calibration of this project found 0.03 erratic, so there is little room. Wortsman et al. (2023)
  show that QK-layernorm (normalizing the attention inputs) lets small models train stably at
  higher rates. That might open room for a higher rate.
- **Strength of evidence:** moderate.
- **Work:** medium.

**9. Width.** Wider models get through faster, as already found here. This is the most reliable
lever (Edelman et al. 2023).

## 4. What is unlikely to help, or is believed without support

- **"A flat score means nothing is happening."** False. Progress is hidden (Barak et al. 2022;
  Gopalani & Hu 2025). So a stopping rule may call a run flat shortly before it jumps. A
  **progress measure** (an internal quantity that keeps changing while the score stands still),
  such as how much attention goes to the season word, would show this (Nanda et al. 2023).
- **"It's a local minimum, a true dead end."** Usually it is a saddle (Dauphin et al. 2014).
- **Weight decay** (a steady pull of the weights toward zero). It drives grokking, which is about
  held-out data (Power et al. 2022; Liu et al. 2022). In Hoffmann et al. (2023) it *lowered* the
  success rate.
- **Grokfast** (Lee et al. 2024). It closes the gap between training and held-out scores. The
  paper reports no speed-up on the training curve. It is cheap but unproven for this case.
- **StableMax / softmax collapse** (Prieto et al. 2025). This applies near zero loss, which is not
  the situation here.
- **Cyclical learning rates and warm restarts** (Smith 2015; Loshchilov & Hutter 2016). These are
  general speed-ups. No study was found showing they end plateaus (unconfirmed).
- **Longer warm-up.** It mainly buys stability (Kalra & Barkeshli 2024; Wortsman et al. 2023).
  Hoffmann et al. (2023) found a longer warm-up helped "somewhat".

## 5. Open questions and what could not be confirmed

- Almost no evidence comes from models under about 400K parameters. Gopalani & Hu's single-layer
  model had embedding size 256, while the models here have 4 to 32. Results may not hold at width
  8.
- No paper found tested this exact case: a plateau where the model ignores a conditioning word
  such as the season.
- The batch-size evidence points both ways.
- No study of learning-rate schedules or restarts as plateau remedies could be confirmed.
- Zucchet et al. give the speed-up only in figures, so there is no number for it. Several of
  Gopalani & Hu's hyperparameter results are likewise given only in figures, without stated
  directions.

## 6. Sources

1. Saxe, McClelland, Ganguli (2013). Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. https://arxiv.org/abs/1312.6120 — Plateaus and sudden drops, derived exactly.
2. Dauphin et al. (2014). Identifying and attacking the saddle point problem in high-dimensional non-convex optimization. https://arxiv.org/abs/1406.2572 — Plateaus come from saddles, not local minima.
3. Jacot et al. (2021). Saddle-to-Saddle Dynamics in Deep Linear Networks. https://arxiv.org/abs/2106.15933 — Small starting weights mean visiting one saddle after another.
4. Abbe, Boix-Adsera, Misiakiewicz (2023). SGD learning on neural networks: leap complexity and saddle-to-saddle dynamics. https://arxiv.org/abs/2302.11055 — Staged learning in theory.
5. Barak et al. (2022). Hidden Progress in Deep Learning: SGD Learns Parities Near the Computational Limit. https://arxiv.org/abs/2207.08799 — Progress continues invisibly during a plateau.
6. Edelman et al. (2023). Pareto Frontiers in Neural Feature Learning: Data, Compute, Width, and Luck. https://arxiv.org/abs/2309.03800 — Width and luck trade off against each other.
7. Hoffmann et al. (2023). Eureka-Moments in Transformers. https://arxiv.org/abs/2310.12956 — Softmax causes the plateau; NormSoftmax and a lower temperature fix it.
8. Gopalani, Lubana, Hu (2024). Abrupt Learning in Transformers: A Case Study on Matrix Completion. https://arxiv.org/abs/2410.22244 — Plateau then a sharp drop, as attention forms.
9. Gopalani, Hu (2025). What Happens During the Loss Plateau? https://arxiv.org/abs/2506.13688 — Partial solutions, effects of attention bias, starting scale and learning rate.
10. Chen et al. (2023). Sudden Drops in the Loss. https://arxiv.org/abs/2309.07311 — Simplicity bias and sudden drops in language models.
11. Belrose et al. (2024). Neural Networks Learn Statistics of Increasing Complexity. https://arxiv.org/abs/2402.04362 — Simple statistics are learned first.
12. Michaud et al. (2023). The Quantization Model of Neural Scaling. https://arxiv.org/abs/2303.13506 — Skills are learned as separate chunks.
13. Olsson et al. (2022). In-context Learning and Induction Heads. https://arxiv.org/abs/2209.11895 — An attention circuit forms at a phase change.
14. Singh et al. (2024). What needs to go right for an induction head? https://arxiv.org/abs/2404.07129 — The sub-circuits behind a sudden jump.
15. Reddy (2023). The mechanistic basis of data dependence and abrupt learning in an in-context classification task. https://arxiv.org/abs/2312.03002 — Nested steps cause sharp transitions.
16. Zucchet et al. (2025). How do language models learn facts? https://arxiv.org/abs/2503.21676 — Plateau before facts; skewed data or a subset warm-up shortens it.
17. Kirsch et al. (2022). General-Purpose In-Context Learning by Meta-Learning Transformers. https://arxiv.org/abs/2212.04458 — Batch size, Adam epsilon and biased data each shorten plateaus.
18. Fu et al. (2023). Breaking through the learning plateaus of in-context learning in Transformer. https://arxiv.org/abs/2309.06054 — Warm-up, mixing and an auxiliary loss.
19. Hu et al. (2023). Latent State Models of Training Dynamics. https://arxiv.org/abs/2308.09543 — Seed-dependent detours.
20. Pezeshki et al. (2020). Gradient Starvation. https://arxiv.org/abs/2011.09468 — Easy features starve hard ones of gradient.
21. Kumar et al. (2023). Grokking as the Transition from Lazy to Rich Training Dynamics. https://arxiv.org/abs/2310.06110 — Starting scale controls when feature learning begins.
22. Wortsman et al. (2023). Small-scale proxies for large-scale Transformer training instabilities. https://arxiv.org/abs/2309.14322 — QK-layernorm, epsilon and learning-rate sensitivity.
23. Power et al. (2022). Grokking. https://arxiv.org/abs/2201.02177 — Defines grokking (delayed generalization).
24. Liu, Michaud, Tegmark (2022). Omnigrok. https://arxiv.org/abs/2210.01117 — Weight norm explains grokking.
25. Nanda et al. (2023). Progress measures for grokking via mechanistic interpretability. https://arxiv.org/abs/2301.05217 — Measuring hidden progress.
26. Lee et al. (2024). Grokfast. https://arxiv.org/abs/2405.20233 — Amplifies slow gradients; tested on generalization only.
27. Prieto et al. (2025). Grokking at the Edge of Numerical Stability. https://arxiv.org/abs/2501.04697 — Softmax collapse near zero loss.
28. Thilak et al. (2022). The Slingshot Mechanism. https://arxiv.org/abs/2206.04817 — Adam's late-training cycles and grokking.
29. Smith (2015). Cyclical Learning Rates for Training Neural Networks. https://arxiv.org/abs/1506.01186 — Cyclical learning rates; general speed-up.
30. Loshchilov, Hutter (2016). SGDR: Stochastic Gradient Descent with Warm Restarts. https://arxiv.org/abs/1608.03983 — Warm restarts; general speed-up.
31. Kalra, Barkeshli (2024). Why Warmup the Learning Rate? https://arxiv.org/abs/2406.09405 — What warm-up does: mainly stability.

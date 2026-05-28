# Direct Preference Optimisation

DPO aligns language models to human preferences without a separate reinforcement-learning loop. Given a prompt and two completions—one preferred, one dispreferred—the loss increases the log-probability margin for the chosen response while regularising against a frozen reference policy.

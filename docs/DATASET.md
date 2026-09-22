# Dataset and Causal Framing

Created by School of AI and School of QC.

The repository generates the Mackey–Glass delay system:

$$x_{t+1}=x_t+\frac{\beta x_{t-\tau}}{1+x_{t-\tau}^{n}}-\gamma x_t.$$

The defaults use $\beta=0.2$, $\gamma=0.1$, $\tau=17$, $n=10$, initial value 1.2, and a
400-step burn-in. No random noise or external data is used, so the raw series is deterministic.

Each supervised row contains current value, 19 preceding values, and the next value as target.
Rows are ordered and split 60% train, 20% validation, and 20% test. The first 80 training
reservoir states are discarded as washout.

The test target never enters scaling, readout fitting, alpha selection, or classical training.
Current and past test inputs causally drive the reservoir during streaming evaluation, which is
permitted because those values would be observed at forecast time.

Mackey–Glass is useful for studying nonlinear temporal dynamics but is not representative of
finance, weather, medicine, energy, or operational demand. External validity requires new data,
new splits, and domain-specific evaluation.

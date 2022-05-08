We want to solve the task using REINFOCE. We will be experimenting with

- Vanilla v.s. temporal structures
- No baseline v.s. a value function baseline

using a neural network with fully connected layers, which has input tensor size corresponding to the observation_space.shape, two intermediate layers each with size 64, and output size of action_space.n. As a reminder, the REINFORCE objective for the vanilla version is
$$V(\pi_\theta) = \mathbb{E}[\Big(\sum\limits_{t=0}^{T}\log\pi_\theta(a_t | s_t)\Big)\Big(\sum\limits_{t'=0}^{T} \gamma^{t'} r_{t'}\Big)].$$
The REINFORCE objective with temporal structure is
$$V(\pi_\theta) = \mathbb{E}[\sum\limits_{t=0}^{T}\log\pi_\theta(a_t | s_t)\Big(\sum\limits_{t'=t}^{T} \gamma^{t'} r_{t'}\Big)].$$

The REINFORCE objective with temporal structure and baseline is
$$V(\pi_\theta) = \mathbb{E}[\sum\limits_{t=0}^{T}\log\pi_\theta(a_t | s_t)\Big(\sum\limits_{t'=t}^{T} \gamma^{t'} r_{t'}-\gamma^t b(s_t)\Big)],$$
where $b(s_t)$ is the baseline.
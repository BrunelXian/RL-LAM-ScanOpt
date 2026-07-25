# PPO Reward Model Claim Boundary

- This surrogate is a terminal reward emulator trained on FEA teacher-labelled scan-order data.
- It is not the physical teacher.
- It is not a PPO policy.
- PPO candidates must still be independently validated by Abaqus.
- Strong PPO claims require PPO-only candidate generation plus Abaqus validation.
- The current artifact supports only a supervised surrogate reward-model claim.
- No online Abaqus PPO, PPO training, or PPO candidate generation was performed in Stage B.

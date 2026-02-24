
- dqn-1/DQN_15: 73c90232ae3c528cd8dca4062795d8e6b3082d09
    - 500左右的highest block


log2 reward 对比

- 对比
    - QRDQN_15: 不log2
    - QRDQN_16: 与11相比，添加了log2 reward
- 结论
    - 添加log2reward，会显著改善训练
    - rollout/highest_block，log2reward始终更好
    - eval/highest:max: log2reward始终更好，可以达到2048，而不加log2则较少情况达到2048，以1024为主
    - 二者都有问题：eval/highest:min、eval/highest:avg 太菜，上下限差距太大。可能是Q-Leaning系列算法的问题.



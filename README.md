Meta-Q-Learning
=============================================
This paper introduces Meta-Q-Learning (MQL), a new off-policy algorithm for meta-Reinforcement Learning (meta-RL). MQL builds upon three simple ideas. First, we show that Q-learning is competitive with state-of-the-art meta-RL algorithms if given access to a context variable that is a representation of the past trajectory. Second, a multi-task objective to maximize the average reward across the training tasks is an effective method to meta-train RL policies. Third, past data from the meta-training replay buffer can be recycled to adapt the policy on a new task using off-policy updates. MQL draws upon ideas in propensity estimation to do so and thereby amplifies the amount of available data for adaptation. Experiments on standard continuous-control benchmarks suggest that MQL compares favorably with the state of the art in meta-RL.

This repository provides the implementation of [Meta-Q-learning](https://arxiv.org/abs/1910.00125). If you use this code please cite the paper using the bibtex reference below.

```
@misc{fakoor2019metaqlearning,
    title={Meta-Q-Learning},
    author={Rasool Fakoor and Pratik Chaudhari and Stefano Soatto and Alexander J. Smola},
    year={2019},
    eprint={1910.00125},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
## Getting Started
```
python run_script.py --env cheetah-dir --gpu_id 0 --seed 0
```

'env' can be humanoid-dir, ant-dir, cheetah-vel, cheetah-dir, ant-goal, and walker-rand-params. The code works on GPU and CPU machine. For the experiments in this paper, we used [p3.2xlarge](https://aws.amazon.com/ec2/instance-types/p3/). For complete list of hyperparameters, please refer to the paper appendix.


In order to run this code, you will need to install Pytorch and MuJoCo. If you face any problem, please follow [PEARL](https://github.com/katerakelly/oyster/) steps to install.  

## New Environments
In order to run code with a new environment, you will need to first define an entry in ./configs/pearl_envs.json. Look at ./configs/abl_envs.json as a reference. In addation, you will need to add an env's code to rlkit/env/.

## Acknowledgement
- **rand_param_envs** and **rlkit** are completely based/copied on/from following repositories:
[rand_param_envs](https://github.com/dennisl88/rand_param_envs/tree/4d1529d61ca0d65ed4bd9207b108d4a4662a4da0) and
[PEARL](https://github.com/katerakelly/oyster/). Thanks to their authors to make them available.
We include them here to make it easier to run and work with this repository.


## License

This code is licensed under the CC-BY-NC-4.0 License.

# Contact

Please open an issue on [issues tracker](https://github.com/amazon-research/meta-q-learning/issues) to report problems or to ask questions or send an email to me, [Rasool Fakoor](https://github.com/rasoolfa).

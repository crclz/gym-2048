from play2048.application.dqn_train import make_model, make_sub_process_env


if __name__ == "__main__":
    env = make_sub_process_env(1, eval=False)

    device = "cpu"

    dqn_model = make_model(env, "dqn", device)

    dqn_conv_model = make_model(env, "dqn-conv", device)

    print("dqn_model is", dqn_model.policy)

    print("dqn_conv_model is", dqn_conv_model.policy)
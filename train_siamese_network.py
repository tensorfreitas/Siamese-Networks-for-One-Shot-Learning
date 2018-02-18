from siamese_network import SiameseNetwork


def main():
    dataset_path = 'Omniglot Dataset'
    use_augmentation = True
    learning_rate = 10e-4
    batch_size = 32
    # Learning Rate multipliers for each layer
    learning_rate_multipliers = {}
    learning_rate_multipliers['Conv1'] = 1
    learning_rate_multipliers['Conv2'] = 1
    learning_rate_multipliers['Conv3'] = 1
    learning_rate_multipliers['Conv4'] = 1
    learning_rate_multipliers['Dense1'] = 1
    # l2-regularization penalization for each layer
    l2_penalization = {}
    l2_penalization['Conv1'] = 1e-2
    l2_penalization['Conv2'] = 1e-2
    l2_penalization['Conv3'] = 1e-2
    l2_penalization['Conv4'] = 1e-2
    l2_penalization['Dense1'] = 1e-2
    # Path where the logs will be saved
    #tensorboard_log_path = './logs/siamese_net_sgd_lr10e-4_augmentation_regularization1e-2_momentum_0_9'
    tensorboard_log_path = './logs2/test'
    siamese_network = SiameseNetwork(
        dataset_path=dataset_path,
        learning_rate=learning_rate,
        batch_size=batch_size, use_augmentation=use_augmentation,
        learning_rate_multipliers=learning_rate_multipliers,
        l2_regularization_penalization=l2_penalization,
        tensorboard_log_path=tensorboard_log_path
    )
    # Final layer-wise momentum (mu_j in the paper)
    momentum = 0.9
    # linear epoch slope evolution
    momentum_slope = 0.01
    support_set_size = 20
    evaluate_each = 100
    number_of_train_iterations = 1000000
    siamese_network.train_siamese_network(number_of_iterations=number_of_train_iterations,
                                          support_set_size=support_set_size, 
                                          final_momentum=momentum, 
                                          momentum_slope=momentum_slope,
                                          evaluate_each=evaluate_each)


if __name__ == "__main__":
    main()

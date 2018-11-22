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
    l2_penalization['Dense1'] = 1e-4
    # Path where the logs will be saved
    tensorboard_log_path = './logs/siamese_net_lr10e-4'
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
    evaluate_each = 1000
    number_of_train_iterations = 1000000

    validation_accuracy = siamese_network.train_siamese_network(number_of_iterations=number_of_train_iterations,
                                                                support_set_size=support_set_size,
                                                                final_momentum=momentum,
                                                                momentum_slope=momentum_slope,
                                                                evaluate_each=evaluate_each, 
                                                                model_name='siamese_net_lr10e-4')
    if validation_accuracy == 0:
        evaluation_accuracy = 0
    else:
        # Load the weights with best validation accuracy
        siamese_network.model.load_weights('./models/siamese_net_lr10e-4.h5')
        evaluation_accuracy = siamese_network.omniglot_loader.one_shot_test(siamese_network.model,
                                                                        20, 40, False)
    
    print('Final Evaluation Accuracy = ' + str(evaluation_accuracy))


if __name__ == "__main__":
    main()

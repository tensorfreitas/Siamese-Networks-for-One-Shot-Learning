import GPy
import GPyOpt
import keras.backend as K

from siamese_network import SiameseNetwork

current_model_number = 0


def main():

    hyperparameters = [{'name': 'learning_rate', 'type': 'continuous',
                        'domain': (10e-6, 10e-4)},
                       {'name': 'momentum', 'type': 'continuous',
                        'domain': (0.0, 1.0)},
                       {'name': 'momentum_slope', 'type': 'continuous',
                        'domain': (0.001, 0.1)},
                       {'name': 'Conv1_multiplier', 'type': 'discrete',
                        'domain': (0.01, 0.1, 1, 10)},
                       {'name': 'Conv2_multiplier', 'type': 'discrete',
                        'domain': (0.01, 0.1, 1, 10)},
                       {'name': 'Conv3_multiplier', 'type': 'discrete',
                        'domain': (0.01, 0.1, 1, 10)},
                       {'name': 'Conv4_multiplier', 'type': 'discrete',
                        'domain': (0.01, 0.1, 1, 10)},
                       {'name': 'Dense1_multiplier', 'type': 'discrete',
                        'domain': (0.01, 0.1, 1, 10)},
                       {'name': 'l2_penalization_Conv1', 'type': 'discrete',
                        'domain': (0, 0.0001, 0.001, 0.01, 0.1)},
                       {'name': 'l2_penalization_Conv2', 'type': 'discrete',
                        'domain': (0, 0.0001, 0.001, 0.01, 0.1)},
                       {'name': 'l2_penalization_Conv3', 'type': 'discrete',
                        'domain': (0, 0.0001, 0.001, 0.01, 0.1)},
                       {'name': 'l2_penalization_Conv4', 'type': 'discrete',
                        'domain': (0, 0.0001, 0.001, 0.01, 0.1)},
                       {'name': 'l2_penalization_Dense1', 'type': 'discrete',
                        'domain': (0, 0.0001, 0.001, 0.01, 0.1)}]

    def bayesian_optimization_function(x):
        dataset_path = 'Omniglot Dataset'

        current_learning_rate = float(x[:, 0])
        current_momentum = float(x[:, 1])
        current_momentum_slope = float(x[:, 2])
        current_conv1_multiplier = float(x[:, 3])
        current_conv2_multiplier = float(x[:, 4])
        current_conv3_multiplier = float(x[:, 5])
        current_conv4_multiplier = float(x[:, 6])
        current_dense1_multiplier = float(x[:, 7])
        current_conv1_penalization = float(x[:, 8])
        current_conv2_penalization = float(x[:, 9])
        current_conv3_penalization = float(x[:, 10])
        current_conv4_penalization = float(x[:, 11])
        current_dense1_penalization = float(x[:, 12])

        model_name = 'siamese_net_lr_' + str(current_learning_rate) + \
            'momentum_' + str(current_momentum) + '_slope_' + \
            str(current_momentum_slope)

        global current_model_number
        current_model_number += 1
        tensorboard_log_path = './logs/' + str(current_model_number)

        # Learning Rate multipliers for each layer
        learning_rate_multipliers = {}
        learning_rate_multipliers['Conv1'] = current_conv1_multiplier
        learning_rate_multipliers['Conv2'] = current_conv2_multiplier
        learning_rate_multipliers['Conv3'] = current_conv3_multiplier
        learning_rate_multipliers['Conv4'] = current_conv4_multiplier
        learning_rate_multipliers['Dense1'] = current_dense1_multiplier
        # l2-regularization penalization for each layer
        l2_penalization = {}
        l2_penalization['Conv1'] = current_conv1_penalization
        l2_penalization['Conv2'] = current_conv2_penalization
        l2_penalization['Conv3'] = current_conv3_penalization
        l2_penalization['Conv4'] = current_conv4_penalization
        l2_penalization['Dense1'] = current_dense1_penalization
        K.clear_session()
        siamese_network = SiameseNetwork(
            dataset_path=dataset_path,
            learning_rate=current_learning_rate,
            batch_size=32, use_augmentation=True,
            learning_rate_multipliers=learning_rate_multipliers,
            l2_regularization_penalization=l2_penalization,
            tensorboard_log_path=tensorboard_log_path
        )

        current_model_number += 1

        support_set_size = 20
        evaluate_each = 500
        number_of_train_iterations = 100000
        
        validation_accuracy = siamese_network.train_siamese_network(number_of_iterations=number_of_train_iterations,
                                                                    support_set_size=support_set_size,
                                                                    final_momentum=current_momentum,
                                                                    momentum_slope=current_momentum_slope,
                                                                    evaluate_each=evaluate_each,
                                                                    model_name=model_name)

        if validation_accuracy == 0:
            evaluation_accuracy = 0
        else:        
            # Load the weights with best validation accuracy
            siamese_network.model.load_weights('models/' + model_name + '.h5')
            evaluation_accuracy = siamese_network.omniglot_loader.one_shot_test(siamese_network.model,
                                                                                20, 40, False)
        print("Model: " + model_name +
              ' | Accuracy: ' + str(evaluation_accuracy))
        K.clear_session()
        return 1 - evaluation_accuracy

    optimizer = GPyOpt.methods.BayesianOptimization(
        f=bayesian_optimization_function, domain=hyperparameters)

    optimizer.run_optimization(max_iter=100)

    print("optimized parameters: {0}".format(optimizer.x_opt))
    print("optimized eval_accuracy: {0}".format(1 - optimizer.fx_opt))


if __name__ == "__main__":
    main()

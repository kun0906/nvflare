
def plot_robust_aggregation_all():
    """
       single point: 'adaptive_krum' 'krum' 'adaptive_krum+rp' 'krum+rp' 'median' 'mean'

       average points: 'adaptive_krum_avg' 'krum_avg' 'adaptive_krum+rp_avg' 'krum+rp_avg'
                            'median_avg' 'trimmed_mean' 'geometric_median'


    Args:
        start:

    Returns:

    """
    plt.close()

    fig, axes = plt.subplots(nrows=3, ncols=5, sharey=None, figsize=(15, 10))  # width, height
    # axes = axes.reshape((1, -1))

    j_col = 0
    for start in range(0, 110, 4):
        i_row = 0
        for METRIC in ['accuracy', 'l2_error', 'time_taken']:
            try:
                print(f'\nstart: {start}, {METRIC}')
                global_accs = {}
                start2 = start + 7
                method_txt_files = [
                    # # # # # Aggregated results: single point
                    ('adaptive_krum', f'log/output_{JOBID}_{start}.out'),
                    ('krum', f'log/output_{JOBID}_{start + 1}.out'),
                    ('median', f'log/output_{JOBID}_{start + 2}.out'),
                    ('mean', f'log/output_{JOBID}_{start + 3}.out'),
                    # ('exp_weighted_mean', f'log/output_{JOBID}_{start + 7}.out'),

                    # # # Aggregated results: average point
                    # ('adaptive_krum_avg', f'log/output_{JOBID}_{start2}.out'),
                    # ('krum_avg', f'log/output_{JOBID}_{start2 + 1}.out'),
                    # ('adaptive_krum+rp_avg', f'log/output_{JOBID}_{start2 + 2}.out'),
                    # ('krum+rp_avg', f'log/output_{JOBID}_{start2 + 3}.out'),
                    # ('medoid_avg', f'log/output_{JOBID}_{start2 + 4}.out'),
                    # ('trimmed_mean', f'log/output_{JOBID}_{start2 + 5}.out'),
                    # ('geometric_median', f'log/output_{JOBID}_{start2 + 6}.out'),

                ]
                case_name = extract_case_info(f'log/output_{JOBID}_{start}.out')
                print(case_name, flush=True)
                # Example usage
                namespace_params = extract_namespace(f'log/output_{JOBID}_{start}.out')
                # if (namespace_params['server_epochs'] in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.8, 9.0, 10.0]
                #         or namespace_params['labeling_rate'] in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.8, 9.0, 10.0]
                #         or namespace_params['num_clients'] in []):
                #     return
                # print(namespace_params)
                if (namespace_params['server_epochs'] == SERVER_EPOCHS and namespace_params['labeling_rate'] != 0.0
                        and namespace_params['num_clients'] == NUM_CLIENTS):
                    pass

                    print(namespace_params)
                else:
                    return

                title = ', '.join(['num_clients:' + str(namespace_params['num_clients']),
                                   'classes_cnt:' + str(namespace_params['server_epochs']),
                                   'large_value:' + str(namespace_params['labeling_rate'])])
                for method, txt_file in method_txt_files:
                    namespace_params = extract_namespace(txt_file)
                    method0 = namespace_params['aggregation_method']
                    if method != method0:
                        print(f'{method} != {method0}, ', txt_file, flush=True)
                        continue
                    try:
                        results = parse_file(txt_file, metric=METRIC)
                        global_accs[(method, txt_file)] = results
                    except Exception as e:
                        print(e, method0, txt_file, flush=True)
                        # traceback.print_exc()

                aggregation_methods = list(global_accs.keys())
                makers = ['o', '+', 's', '*', 'v', '.', 'p', 'h', 'x', '8', '1', '^', 'D', 'd']
                for i in range(len(aggregation_methods)):
                    agg_method, txt_file = aggregation_methods[i]
                    label = agg_method
                    if METRIC == 'accuracy':
                        vs = [vs['shared_acc'] for vs in global_accs[(agg_method, txt_file)]]
                    elif METRIC == 'l2_error':
                        vs = [vs['l2_error'] for vs in global_accs[(agg_method, txt_file)]]
                    elif METRIC == 'time_taken':
                        vs = [vs['time_taken'] for vs in global_accs[(agg_method, txt_file)]]
                        if agg_method == 'median_avg': continue
                    # elif METRIC == 'misclassification_error':
                    #     vs = [vs['shared_acc'] for vs in global_accs[(agg_method, txt_file)]]
                    #     vs = [1 - v for v in vs]
                    else:
                        raise NotImplementedError(METRIC)
                    ys, ys_errs = zip(*vs)
                    xs = range(len(ys))
                    print(agg_method, txt_file, ys, flush=True)
                    # print(agg_method, txt_file, [f"{v[0]:.2f}/{v[1]:.2f}" for v in vs], flush=True)

                    # axes[i_row, j_col].plot(xs, ys, label=label, marker=makers[i])
                    # axes[i_row, j_col].errorbar(xs, ys, yerr=ys_errs, label=label, marker=makers[i], capsize=3)
                    axes[i_row, j_col].plot(xs, ys, label=label, marker=makers[i])  # Plot the line
                    # axes[i_row, j_col].fill_between(xs,
                    #                                 [y - e for y, e in zip(ys, ys_errs)],
                    #                                 [y + e for y, e in zip(ys, ys_errs)],
                    #                                 color='blue', alpha=0.3)  # label='Error Area'
                plt.xlabel('Epochs', fontsize=10)
                if METRIC == 'loss':
                    ylabel = 'Loss'
                elif METRIC == 'l2_error':
                    ylabel = 'L_2 Error'
                elif METRIC == 'time_taken':
                    ylabel = 'Time Taken'
                # elif METRIC == 'misclassification_error':
                #     ylabel = 'Misclassification Error'
                else:
                    ylabel = 'Accuracy'

                FONTSIZE = 20
                axes[i_row, j_col].set_ylabel(ylabel, fontsize=FONTSIZE)

                variable = str(namespace_params['labeling_rate'])
                axes[i_row, j_col].set_title(f'start:{start}, {variable}', fontsize=FONTSIZE)
                axes[i_row, j_col].legend(fontsize=6.5, loc='lower right', fontsie=FONTSIZE)

            except Exception as e:
                print(e)
            i_row += 1
        j_col += 1

    # attacker_ratio = NUM_BYZANTINE_CLIENTS / (NUM_HONEST_CLIENTS + NUM_BYZANTINE_CLIENTS)
    # title = (f'{model_type}_cnn' + '$_{' + f'{num_server_epoches}+1' + '}$' +
    #          f':{attacker_ratio:.2f}-{LABELING_RATE:.2f}')
    # plt.suptitle(f'Global Model ({JOBID}), start:{start}, {title}, {METRIC}', fontsize=10)
    plt.suptitle(f'Global Model ({JOBID}), {title}\n{case_name}', fontsize=FONTSIZE)
    # Adjust layout to prevent overlap
    plt.tight_layout()
    # fig_file = (f'{IN_DIR}/{model_type}_{LABELING_RATE}_{AGGREGATION_METHOD}_'
    #             f'{SERVER_EPOCHS}_{NUM_HONEST_CLIENTS}_{NUM_BYZANTINE_CLIENTS}_accuracy.png')
    fig_file = f'global_{JOBID}.png'
    print(fig_file)
    # os.makedirs(os.path.dirname(fig_file), exist_ok=True)
    plt.savefig(fig_file, dpi=300)
    plt.show()
    plt.close()


if __name__ == '__main__':

    # SPAMBASE:

    # MNIST :


    # SENTIMENT 140 :







    # plot_robust_aggregation()

    ######################### MNIST Results 20250313 ###############################################
    # JOBID = 256611  # it works, log_large_values_20250214 with fixed large values

    # # Random noise injection with alpha=10, client_epochs=20, batch_size=512, epoch=10, num_clients=20, f=8
    # JOBID = 273327  # for Model Poisoning Attacks, random noise injection to model updates

    # # Random noise injection with alpha=10, client_epochs=20, batch_size=512, epoch=100, num_clients=50, f=23
    # JOBID = 274003  # for Model Poisoning Attacks, random noise injection to model updates

    # # # Large Value with alpha=10, client_epochs=20, batch_size=512, epoch=10, num_clients=20, f=8
    # JOBID = 273345  # for Model Poisoning Attacks, Large values to model updates

    # # # Large Value with alpha=10, client_epochs=20, batch_size=512, epoch=100, num_clients=50, f=23
    # JOBID = 274024  # for Model Poisoning Attacks, Large values to model updates

    ### Label flipping

    # # Flip labels with alpha=10, client_epochs=1, batch_size=512, epoch=10, num_clients=20, f=8
    # JOBID = 273678  # for Data Poisoning Attacks, flip labels for Byzantine clients

    # # Flip labels with alpha=10, client_epochs=1, batch_size=512, epoch=100, num_clients=50, f=23
    # JOBID = 273720  # for Data Poisoning Attacks, flip labels for Byzantine clients

    ###########################Sentiment140###########################################################

    # # JOBID = 273742
    # # JOBID = 273327
    # JOBID = 274219      # Sentiment140 with 30% data
    # JOBID = 274249  # Sentiment140 with 10% data with different batch sizes
    # JOBID = 274271  # Sentiment140 with 10% data with different alpha
    # JOBID = 274320  # Sentiment140 with 10% data with different alpha

    # # Random noise injection with alpha=10, client_epochs=5, batch_size=512, epoch=10, num_clients=20, f=8,
    # JOBID = 274502  # Sentiment140 with 10% data with different alpha

    # # Random noise injection with alpha=10, client_epochs=5, batch_size=512, epoch=100, num_clients=50, f=823,
    # JOBID = 274433  # Sentiment140 with 10% data with different alpha

    # # Large Value with alpha=10, client_epochs=5, batch_size=512, epoch=10, num_clients=20, f=8,
    # JOBID = 274527  # Sentiment140 with 10% data with different alpha

    # Large Value with alpha=10, client_epochs=5, batch_size=512, epoch=100, num_clients=50, f=23,
    # JOBID = 274794 # Sentiment140 with 10% data with different alpha

    ### Label flipping

    # # Flip labels with alpha=10, client_epochs=5, batch_size=512, epoch=10, num_clients=20, f=8
    # JOBID = 274839  # for Data Poisoning Attacks, flip labels for Byzantine clients
    # # Flip labels with alpha=10, client_epochs=5, batch_size=512, epoch=100, num_clients=50, f=23
    # JOBID = 274863 # Sentiment140 with 10% data with different alpha


    # JOBID = 274594
    # JOBID = 274462  # Shakespeare with 50 features and  different alpha

    JOBID = 345592      # MNIST: random Noise Injection Attack

    # JOBID = 345089  # MNIST: random Noise Injection Attack

    SERVER_EPOCHS = 100
    NUM_CLIENTS = 50

    # #  # NewsCategory
    # JOBID = 274668  # NewsCategory with different alpha, num_clients=20, epochs=10
    # JOBID = 274713  # NewsCategory with different alpha, num_clients=50, epochs=100

    # # Fakenews
    # JOBID = 274735  # Fakenews with different alpha,  num_clients=20, epochs=10
    # JOBID = 274765  # Fakenews with different alpha,  num_clients=50, epochs=100


    # # METRIC = 'loss'
    # METRIC = 'misclassification_error'  # or misclassification Rate
    # METRIC = "l2_error"  # 'accuracy'  # l2_error, time_taken
    METRIC = 'accuracy'
    for start in range(0, 100, 4):
        try:
            print(f'\nstart: {start}')
            plot_robust_aggregation(start, METRIC)
        except Exception as e:
            print(e)

    plot_robust_aggregation_all()

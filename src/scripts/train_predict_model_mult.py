import argparse
from src.common.load_data import load_data_mul, load_data
import os
from src.common.score import cal_score
from src.model.model_mul import SequenceModel


def main(parser):
    args = parser.parse_args()
    training_set = args.training_set
    test_set_per_lg = args.test_set_per_lg
    eval_set = args.eval_set
    use_cuda = args.use_cuda
    is_evaluate = args.is_evaluate
    model_dir = args.model_dir
    model_type = args.model_type
    model_name = args.model_name
    wandb_project = args.wandb_project
    is_sweeping = args.is_sweeping
    best_result_config = args.best_result_config
    exec_model(model_type, model_name, model_dir, training_set, test_set_per_lg, eval_set, is_evaluate, use_cuda,
               wandb_project, is_sweeping, best_result_config)


def exec_model(model_type, model_name, model_dir, training_set, test_set_per_lg, eval_set, is_evaluate, use_cuda,
               wandb_project, is_sweeping, best_result_config):
    if model_dir == '':
        df_train, _ = load_data_mul(os.getcwd() + training_set, False, ['label', 'label_lg'])
        df_train = df_train[0:10]
        model = SequenceModel(model_type, model_name, use_cuda, None, [2,2], wandb_project,
                            is_sweeping, is_evaluate, best_result_config, True, len(df_train['features'][0]))

        if is_evaluate:
            df_eval, _ = load_data_mul(os.getcwd() + eval_set, False, ['label', 'label_lg'])
            df_eval = df_eval[0:10]
        # df_train = pd.concat([df_train, df_eval])
        model.train(df_train, df_eval, ['label', 'label_lg'])

    else:
        model = SequenceModel(model_type, os.getcwd() + model_dir, use_cuda)

    for test_dir in test_set_per_lg:
        ###### Predict test set ########
        df_test, _ = load_data(os.getcwd() + test_dir, False)
        df_test = df_test[0:10]
        y_predict, _ = model.predict(df_test, task_id=0)
        cal_score(df_test, y_predict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--training_set",
                        default="/data/train_mu_lg.tsv",
                        type=str,
                        help="This parameter is the relative dir of training set.")

    parser.add_argument("--eval_set",
                        default="/data/eval_mu_lg.tsv",
                        type=str,
                        help="This parameter is the relative dir of eval set.")

    parser.add_argument("--test_set_per_lg",
                        default=["/data/test_es.tsv", "/data/test_en.tsv"],
                        nargs='+',
                        help="This parameter is the relative dir of eval set.")

    parser.add_argument("--use_cuda",
                        default=False,
                        action='store_true',
                        help="This parameter should be True if cuda is present.")

    parser.add_argument("--is_evaluate",
                        default=True,
                        action='store_true',
                        help="This parameter should be True if you want to split train in train and dev.")

    parser.add_argument("--model_dir",
                        default='',
                        type=str,
                        help="This parameter is the relative dir of model for predict.")

    parser.add_argument("--model_type",
                        default="rembert",
                        type=str,
                        help="This parameter is the relative type of model to trian and predict.")

    parser.add_argument("--model_name",
                        default="google/rembert",
                        type=str,
                        help="This parameter is the relative name of model to trian and predict.")

    parser.add_argument("--wandb_project",
                        default="AuTexTification",
                        type=str,
                        help="This parameter is the name of wandb project.")

    parser.add_argument("--is_sweeping",
                        default=False,
                        action='store_true',
                        help="This parameter should be True if you use sweep search.")

    parser.add_argument("--best_result_config",
                        default="",
                        type=str,
                        help="This parameter is the file with best hyperparameters configuration.")

    main(parser)

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn
from src.model.ClassificationModel_mult import ClassificationModelMult
import torch.nn.functional as F
import torch


class SequenceModel():
    def __init__(self, model_type, model_name, use_cuda, labels_len=None, labels_list=None, wandb_project=None,
                 is_sweeping=False, is_evaluate=False, best_result_config=None, is_trainig=False, value_head=0):
        self.is_evaluate = is_evaluate
        ############ Hyperparameters #####################
        model_args = {
            'learning_rate': 2e-5,
            'num_train_epochs': 3,
            'reprocess_input_data': True,
            'overwrite_output_dir': True,
            'process_count': 10,
            'train_batch_size': 8,
            'eval_batch_size': 8,
            'max_seq_length': 400,
            'manual_seed': 4260,
            "config": {
                 "classifier_dropout_prob": 0.2,
            },
            # "scheduler": "cosine_with_hard_restarts_schedule_with_warmup",
            'use_multiprocessing': False,
            'use_multiprocessing_for_evaluation': False,
            'evaluate_during_training': self.is_evaluate,
            'evaluate_during_training_steps': 100000,
            'multiprocessing_chunksize': 500,
            'fp16': True,
            'fp16_opt_level': '01',
            'wandb_project': wandb_project,
            'tensorboard_dir': 'tensorboard',
            'value_head': value_head
        }
        if is_trainig:
            wandb_config = {}
            sweep_config = {}
            if wandb_project:
                parse_wandb_param(wandb_config, model_args)
                if is_sweeping:
                    sweep_config = wandb_config
                    parse_wandb_param(sweep_config, model_args)
            if best_result_config:
                sweep_result = pd.read_csv(os.getcwd() + best_result_config)
                best_params = sweep_result.to_dict()
                print(best_params)
                parse_wandb_param(best_params, model_args)

            self.model = ClassificationModelMult(model_type, model_name, num_labels=labels_len, labels_list=labels_list ,use_cuda=use_cuda,
                                             args=model_args, sweep_config=sweep_config)
        else:
            self.model = ClassificationModelMult(model_type, model_name, use_cuda=use_cuda, labels_list=labels_list, args=model_args)

    def train(self, df_train, df_eval=None, labels_name=None):
        if self.is_evaluate and not isinstance(df_eval, pd.DataFrame):
            df_train, df_eval = train_test_split(df_train, test_size=0.2, train_size=0.8, random_state=1)

        return self.model.train_model(df_train, eval_df=df_eval, shuffle_each_epoch=True, f1=sklearn.metrics.f1_score, acc=sklearn.metrics.accuracy_score, labels_name=labels_name)

    def predict(self, df_test, is_softmax=False, task_id=None):
        labels_fic = len(df_test) * [0]
        labels_fic = pd.Series(labels_fic)
        if 'text_a' in df_test.columns:
            text_a = df_test['text_a']
            text_b = df_test['text_b']
            df_result = pd.concat([text_a, text_b], axis=1)
        else:
            text = df_test['text']
            features = df_test['features']
            text.reset_index(drop=True, inplace=True)
            features.reset_index(drop=True, inplace=True)
            df_result = pd.concat([text, labels_fic, features], axis=1)

        value_in = df_result.values.tolist()
        y_predict, model_outputs_test = self.model.predict(value_in, task_id=task_id)
        if is_softmax:
            model_outputs_test = F.softmax(torch.from_numpy(model_outputs_test), dim=-1)
            model_outputs_test = model_outputs_test.detach().cpu().numpy()
        y_predict = np.argmax(model_outputs_test, axis=1)
        return y_predict, model_outputs_test


def parse_wandb_param(sweep_config, model_args):
    # Extracting the hyperparameter values
    cleaned_args = {}
    layer_params = []
    param_groups = []
    for key, value in sweep_config.items():
        if isinstance(value, dict):
            value = value[0]
        if key.startswith("layer_"):
            # These are layer parameters
            layer_keys = key.split("_")[-1]

            # Get the start and end layers
            start_layer = int(layer_keys.split("-")[0])
            end_layer = int(layer_keys.split("-")[-1])

            # Add each layer and its value to the list of layer parameters
            for layer_key in range(start_layer, end_layer):
                layer_params.append(
                    {"layer": layer_key, "lr": value,}
                )
        elif key.startswith("params_"):
            # These are parameter groups (classifier)
            params_key = key.split("_")[-1]
            param_groups.append(
                {
                    "params": [params_key],
                    "lr": value,
                    "weight_decay": model_args.weight_decay
                    if "bias" not in params_key
                    else 0.0,
                }
            )
        else:
            # Other hyperparameters (single value)
            cleaned_args[key] = value
    cleaned_args["custom_layer_parameters"] = layer_params
    cleaned_args["custom_parameter_groups"] = param_groups

    # Update the model_args with the extracted hyperparameter values
    model_args.update(cleaned_args)

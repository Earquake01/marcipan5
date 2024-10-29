# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 14:07:29 2021

@author: chumache
"""
import json
import os
import time

import mlflow
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
from torch import nn, optim
from torch.optim import lr_scheduler

import transforms
from dataset import get_test_set, get_training_set, get_validation_set
from model import generate_model
from opts import parse_opts
from train import train_epoch
from utils import Logger, adjust_learning_rate, save_checkpoint
from validation import val_epoch


def chdir_to_projects_root():
    project_dir = os.path.split(find_dotenv())[0]
    os.chdir(project_dir)


def read_env_variables():
    load_dotenv(find_dotenv())


if __name__ == "__main__":
    chdir_to_projects_root()

    # считаем переменные в рабочее окружение
    read_env_variables()

    mlflow_server_uri = os.getenv("MLFLOW_SERVER_URI")

    # задаем настройки MlFlow
    mlflow.set_tracking_uri(mlflow_server_uri)
    mlflow.set_experiment("emotion_classifier")

    opt = parse_opts()
    n_folds = 1
    test_accuracies = []

    mlflow.pytorch.autolog(log_models=False)
    with mlflow.start_run():
        model_name = "_".join((opt.model, opt.fusion, str(opt.sample_duration)))
        mlflow.log_param("model_name", model_name)

        if opt.device != "cpu":
            opt.device = "cuda" if torch.cuda.is_available() else "cpu"

        pretrained = opt.pretrain_path != "None"

        # opt.result_path = 'res_'+str(time.time())
        if not os.path.exists(opt.result_path):
            os.makedirs(opt.result_path)

        opt.arch = "{}".format(opt.model)
        opt.store_name = "_".join([opt.dataset, opt.model, str(opt.sample_duration)])

        for fold in range(n_folds):
            # if opt.dataset == 'RAVDESS':
            #    opt.annotation_path = '/lustre/scratch/chumache/ravdess-develop/annotations_croppad_fold'+str(fold+1)+'.txt'

            print(opt)
            with open(
                os.path.join(
                    opt.result_path, "opts" + str(time.time()) + str(fold) + ".json"
                ),
                "w",
            ) as opt_file:
                json.dump(vars(opt), opt_file)

            torch.manual_seed(opt.manual_seed)
            model, parameters = generate_model(opt)

            criterion = nn.CrossEntropyLoss()
            criterion = criterion.to(opt.device)

            if not opt.no_train:
                video_transform = transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotate(),
                        transforms.ToTensor(opt.video_norm_value),
                    ]
                )

                training_data = get_training_set(opt, spatial_transform=video_transform)

                train_loader = torch.utils.data.DataLoader(
                    training_data,
                    batch_size=opt.batch_size,
                    shuffle=True,
                    num_workers=opt.n_threads,
                    pin_memory=True,
                )

                train_logger = Logger(
                    os.path.join(opt.result_path, "train" + str(fold) + ".log"),
                    ["epoch", "loss", "prec1", "prec5", "lr"],
                )
                train_batch_logger = Logger(
                    os.path.join(opt.result_path, "train_batch" + str(fold) + ".log"),
                    ["epoch", "batch", "iter", "loss", "prec1", "prec5", "lr"],
                )

                optimizer = optim.SGD(
                    parameters,
                    lr=opt.learning_rate,
                    momentum=opt.momentum,
                    dampening=opt.dampening,
                    weight_decay=opt.weight_decay,
                    nesterov=False,
                )
                scheduler = lr_scheduler.ReduceLROnPlateau(
                    optimizer, "min", patience=opt.lr_patience
                )

            if not opt.no_val:
                video_transform = transforms.Compose(
                    [transforms.ToTensor(opt.video_norm_value)]
                )

                validation_data = get_validation_set(
                    opt, spatial_transform=video_transform
                )

                val_loader = torch.utils.data.DataLoader(
                    validation_data,
                    batch_size=opt.batch_size,
                    shuffle=False,
                    num_workers=opt.n_threads,
                    pin_memory=True,
                )

                val_logger = Logger(
                    os.path.join(opt.result_path, "val" + str(fold) + ".log"),
                    ["epoch", "loss", "prec1", "prec5"],
                )
                test_logger = Logger(
                    os.path.join(opt.result_path, "test" + str(fold) + ".log"),
                    ["epoch", "loss", "prec1", "prec5"],
                )

            best_prec1 = 0
            best_loss = 1e10
            if opt.resume_path:
                print("loading checkpoint {}".format(opt.resume_path))
                checkpoint = torch.load(opt.resume_path)
                assert opt.arch == checkpoint["arch"]
                best_prec1 = checkpoint["best_prec1"]
                opt.begin_epoch = checkpoint["epoch"]
                model.load_state_dict(checkpoint["state_dict"])

            for i in range(opt.begin_epoch, opt.n_epochs + 1):
                if not opt.no_train:
                    adjust_learning_rate(optimizer, i, opt)
                    train_epoch(
                        i,
                        train_loader,
                        model,
                        criterion,
                        optimizer,
                        opt,
                        train_logger,
                        train_batch_logger,
                    )
                    state = {
                        "epoch": i,
                        "arch": opt.arch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_prec1": best_prec1,
                    }
                    save_checkpoint(state, False, opt, fold)

                if not opt.no_val:
                    validation_loss, prec1, prec5 = val_epoch(
                        i, val_loader, model, criterion, opt, val_logger
                    )
                    is_best = prec1 > best_prec1
                    best_prec1 = max(prec1, best_prec1)
                    state = {
                        "epoch": i,
                        "arch": opt.arch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_prec1": best_prec1,
                    }

                    save_checkpoint(state, is_best, opt, fold)

                    mlflow.log_metric("val_loss", validation_loss, step=i)
                    mlflow.log_metric("val_prec1", prec1, step=i)
                    mlflow.log_metric("val_prec5", prec5, step=i)

            if opt.test:
                test_logger = Logger(
                    os.path.join(opt.result_path, "test" + str(fold) + ".log"),
                    ["epoch", "loss", "prec1", "prec5"],
                )

                video_transform = transforms.Compose(
                    [transforms.ToTensor(opt.video_norm_value)]
                )

                test_data = get_test_set(opt, spatial_transform=video_transform)

                # load best model
                best_state = torch.load(
                    "%s/%s_best" % (opt.result_path, opt.store_name)
                    + str(fold)
                    + ".pth"
                )
                model.load_state_dict(best_state["state_dict"])

                test_loader = torch.utils.data.DataLoader(
                    test_data,
                    batch_size=opt.batch_size,
                    shuffle=False,
                    num_workers=opt.n_threads,
                    pin_memory=True,
                )

                test_loss, test_prec1, test_prec5 = val_epoch(
                    10000, test_loader, model, criterion, opt, test_logger
                )

                mlflow.log_metric("test_loss", test_loss)
                mlflow.log_metric("test_prec1", test_prec1)
                mlflow.log_metric("test_prec5", test_prec5)

                with open(
                    os.path.join(
                        opt.result_path, "test_set_bestval" + str(fold) + ".txt"
                    ),
                    "a",
                ) as f:
                    f.write("Prec1: " + str(test_prec1) + "; Loss: " + str(test_loss))
                test_accuracies.append(test_prec1)

        if not opt.no_train and not opt.no_val:
            mlflow.pytorch.log_model(model, artifact_path="model")
            mlflow.log_params(vars(opt))

        with open(os.path.join(opt.result_path, "test_set_bestval.txt"), "a") as f:
            f.write(
                "Prec1: "
                + str(np.mean(np.array(test_accuracies)))
                + "+"
                + str(np.std(np.array(test_accuracies)))
                + "\n"
            )

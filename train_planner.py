"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import  load_model, save_model
from .datasets.road_dataset import load_data
from .metrics import PlannerMetric

from torchview import draw_graph

def train(
    exp_dir: str = "logs",
    model_name: str = "mlp_planner",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    print(model)
    #model_graph = draw_graph(model=model, device="meta")
    #model_graph.visual_graph
    model = model.to(device)
    model.train()

    train_data = load_data("drive_data/train", shuffle=True, transform_pipeline="default", batch_size=batch_size, num_workers=2)
    val_data = load_data("drive_data/val", shuffle=False)

    # create loss function and optimizer
    if model_name == "transformer_planner":
        loss_func = torch.nn.HuberLoss()
    else:
        loss_func = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    #optimizer = torch.optim.Adam()

    global_step = 0
    metrics = {"train_err": PlannerMetric(), "val_err": PlannerMetric()}
    min_err = 1000.0

    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        for key in metrics:
            metrics[key].reset()

        model.train()
        #TODO: see what returns from the data loader
        for batch in train_data:
            track_left, track_right, waypoints, waypoints_mask = batch["track_left"].to(device), batch["track_right"].to(device), batch["waypoints"].to(device), batch["waypoints_mask"].to(device)

            # TODO: implement training step
            with torch.autograd.detect_anomaly():
                preds = model(track_left, track_right)
            error = loss_func(preds, waypoints)
            if torch.isnan(error):
                print("Nan in error")
                return
            error_masked = error * waypoints_mask[..., None]

            longitudinal_error = error_masked[:,0].sum() / ( waypoints_mask.sum().float() + 1e-8)
            lateral_error = error_masked[:,1].sum() / (waypoints_mask.sum().float() + 1e-8)
            l1_error = longitudinal_error + lateral_error


            metrics["train_err"].add(preds, waypoints, waypoints_mask)
            
            optimizer.zero_grad()
            l1_error.backward()
            optimizer.step()

            global_step += 1

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()

            for batch in val_data:
                track_left, track_right, waypoints, waypoints_mask = batch["track_left"].to(device), batch["track_right"].to(device), batch["waypoints"].to(device), batch["waypoints_mask"].to(device)

                # TODO: compute validation accuracy
                test_pred = model(track_left, track_right)

                metrics["val_err"].add(test_pred, waypoints, waypoints_mask)


        # log average train and val accuracy to tensorboard
        epoch_train_err = torch.as_tensor(metrics["train_err"].compute()["l1_error"])
        epoch_val_err = torch.as_tensor(metrics["val_err"].compute()["l1_error"])

        epoch_train_long_err = torch.as_tensor(metrics["train_err"].compute()["longitudinal_error"])
        epoch_val_long_err = torch.as_tensor(metrics["val_err"].compute()["longitudinal_error"])

        epoch_train_lat_err = torch.as_tensor(metrics["train_err"].compute()["lateral_error"])
        epoch_val_lat_err = torch.as_tensor(metrics["val_err"].compute()["lateral_error"])

        logger.add_scalar('train_l1_error', (torch.tensor(metrics["train_err"].compute()["l1_error"])).item(), global_step)
        logger.add_scalar('val_l1_error', (torch.tensor(metrics["val_err"].compute()["l1_error"])).item(), global_step)

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_l1_error={epoch_train_err:.4f} "
                f"val_l1_error={epoch_val_err:.4f}\n"
                f"train_longitudinal_error={epoch_train_long_err:.4f} "
                f"val_longitudinal_error={epoch_val_long_err:.4f}\n"
                f"train_lateral_error={epoch_train_lat_err:.4f} "
                f"val_lateral_error={epoch_val_lat_err:.4f}\n"
            )
            if(epoch_val_err < min_err):
                min_err = epoch_val_err
                torch.save(model.state_dict(), log_dir / f"{model_name}-earlystop.th")
                print(f"Model saved to {log_dir / f'{model_name}-earlystop.th'}")

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")

    # save and overwrite the model in the root directory for grading
    model.load_state_dict(torch.load(log_dir / f"{model_name}-earlystop.th", weights_only=True))
    model.eval()
    save_model(model)

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2024)
    # optional: additional model hyperparamters
    #parser.add_argument("--num_blocks", type=int, default=1)
    # pass all arguments to train
    train(**vars(parser.parse_args()))

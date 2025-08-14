'''
Training script for LeNet-5.
'''

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import random
import time

import data
from models.torch_lenet import TorchLeNet

BATCH_SIZE = 64
PATIENCE = 2  # epochs without val improvement
NUM_WORKERS = 4

def set_seed(seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_optimizer(optimizer: str, model_params, lr: float):
    if optimizer == "sgd":
        return torch.optim.SGD(model_params, lr=lr)
    elif optimizer == "adam":
        return torch.optim.Adam(model_params, lr=lr)
    elif optimizer == "adamw":
        return torch.optim.AdamW(model_params, lr=lr)
    else:
        return torch.optim.SGD(model_params, lr=lr)  # Default to SGD

def parse_args():
    parser = argparse.ArgumentParser(description="LeNet-5 CLI")

    parser.add_argument("--framework", type=str, default="torch")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--rotation_degrees", type=int, default=0)
    parser.add_argument("--crop_padding", type=int, default=0)
    parser.add_argument("--duplicate_with_augment", action="store_true", default=False)
    parser.add_argument(
        "--init",
        type=str,
        default="kaiming",
        choices=["orthogonal", "kaiming", "xavier"],
    )
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam", "adamw"])
    parser.add_argument(
            "--activation",
            type=str,
            default="relu",
            choices=["relu", "tanh", "sigmoid"],
        )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")    
    parser.add_argument("--seed", type=int, default=12)
    
    args = parser.parse_args()

    print("---------------------------------------------")
    print(f"Framework:                  {args.framework}")
    print(f"Validation Split:           {args.val_split}")
    print(f"Rotation Degrees:           {args.rotation_degrees}")
    print(f"Crop Padding:               {args.crop_padding}")
    print(f"Duplicate with Augment:     {args.duplicate_with_augment}")
    print(f"Init method:                {args.init}")
    print(f"Optimizer:                  {args.optimizer}")
    print(f"Activation:                 {args.activation}")
    print(f"Learning Rate:              {args.lr}")
    print(f"Random Seed:                {args.seed}")
    print("---------------------------------------------")

    return parser.parse_args()

def main():
    args = parse_args()
    train_dataset, test_dataset, val_dataset = data.get_MNIST(args.val_split, args.rotation_degrees, args.crop_padding, args.duplicate_with_augment)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)

    # framework, val_split, rotation_degrees, crop_padding, duplicate_with_augment, init, optimizer, activation, lr, seed
    save_path = f'models/ckpts/lenet_{args.framework}_{args.val_split}_{args.rotation_degrees}_{args.crop_padding}_{args.duplicate_with_augment}_{args.init}_{args.optimizer}_{args.activation}_{args.lr}_{args.seed}.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lenet5 = TorchLeNet(act_fn=args.activation, init=args.init).to(device)

    print("---------------------------------------------")
    print(lenet5)
    lenet5.param_count()
    print("---------------------------------------------")

    optimizer = get_optimizer(args.optimizer, lenet5.parameters(), args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Train with early stopping if val loss does not improve
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    train_time = 0

    for epoch in range(1, 100):
        lenet5.train()
        start_time = time.time()
        epoch_loss = 0
        val_loss = 0

        for x,y in train_loader:
            x,y = x.to(device), y.to(device)

            optimizer.zero_grad()
            y_pred = lenet5(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss

        end_time = time.time()
        epoch_duration = end_time - start_time
        train_time += epoch_duration

        lenet5.eval()
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device), y.to(device)
                y_pred = lenet5(x)
                loss = criterion(y_pred, y)
                val_loss += loss

        print(f'Epoch {epoch} | {epoch_duration:.2f}s | Train Loss: {epoch_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping at epoch {epoch} after {PATIENCE} epochs without val improvement.")
            break

    print(f'Total training time: {train_time:.2f}s')

    torch.save(lenet5.state_dict(), save_path)
    print(f'Saved model at {save_path}.')

    # Evaluate on test set

if __name__ == '__main__':
    main()
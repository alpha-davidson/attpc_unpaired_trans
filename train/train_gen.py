import os
import math
import argparse
import torch
import torch.utils.tensorboard
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from utils.dataset import *
from utils.misc import *
from utils.data import *
from models.vae_gaussian import *
from models.vae_flow import *
from models.flow import add_spectral_norm, spectral_norm_power_iteration
from evaluation import *
import matplotlib.pyplot as plt
import numpy as np

def get_unique_filename(directory, base_filename):
    base, ext = os.path.splitext(base_filename)
    candidate = os.path.join(directory, base_filename)
    i = 1
    while os.path.exists(candidate):
        candidate = os.path.join(directory, f"{base}{i}{ext}")
        i += 1
    return candidate

epoch_losses = []
epoch_val_losses = []
val_epochs = [] 
epoch_numbers = []
lr_change_log = []

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gaussian', choices=['flow', 'gaussian'])
parser.add_argument('--latent_dim', type=int, default=256)
parser.add_argument('--num_steps', type=int, default=256)
parser.add_argument('--beta_1', type=float, default=1e-4)
parser.add_argument('--beta_T', type=float, default=0.02)
parser.add_argument('--sched_mode', type=str, default='linear')
parser.add_argument('--flexibility', type=float, default=1.0)
parser.add_argument('--truncate_std', type=float, default=2.0)
parser.add_argument('--latent_flow_depth', type=int, default=14)
parser.add_argument('--latent_flow_hidden_dim', type=int, default=256)
parser.add_argument('--num_samples', type=int, default=4)
parser.add_argument('--sample_num_points', type=int, default=256)
parser.add_argument('--kl_weight', type=float, default=0.001)
parser.add_argument('--residual', type=eval, default=True, choices=[True, False])
parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])
parser.add_argument('--dataset_path', type=str, default='data/rectangle/rectangle_noisy.npy')
parser.add_argument('--categories', type=str_list, default=['Mg22'])
parser.add_argument('--scale_mode', type=str, default=None)
parser.add_argument('--train_batch_size', type=int, default=128)
parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--train_ratio', type=float, default=0.8)
parser.add_argument('--lr', type=float, default=2e-3)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--max_grad_norm', type=float, default=10)
parser.add_argument('--end_lr', type=float, default=1e-4)
parser.add_argument('--sched_start_epoch', type=int, default=200*THOUSAND)  
parser.add_argument('--sched_end_epoch', type=int, default=400*THOUSAND)  
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
parser.add_argument('--log_root', type=str, default='./logs_gen')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_epochs', type=int, default=1000*THOUSAND) 
parser.add_argument('--val_freq', type=int, default=100)  
parser.add_argument('--test_freq', type=int, default=300)
parser.add_argument('--test_size', type=int, default=400)
parser.add_argument('--tag', type=str, default=None)
args = parser.parse_args()
seed_all(args.seed)

def setup_logging(args):
    # Initialize logger, TensorBoard writer, and checkpoint manager based on args
    if args.logging:
        assert args.tag is not None, "You must provide a --tag to name the log directory."
        log_dir = os.path.join(args.log_root, f"GEN_{args.tag}")
        os.makedirs(log_dir, exist_ok=True)
        logger = get_logger('train', log_dir)
        writer = torch.utils.tensorboard.SummaryWriter(log_dir)
        ckpt_mgr = CheckpointManager(log_dir)
        log_hyperparams(writer, args)
    else:
        logger = get_logger('train', None)
        writer = BlackHole()
        ckpt_mgr = BlackHole()
    return logger, writer, ckpt_mgr

def load_data(args, logger):
    # Load dataset, split into training and validation, create dataloaders
    logger.info('Loading datasets...')
    data = np.load(args.dataset_path)
    data = torch.from_numpy(data).float()
    n_train = int(len(data) * args.train_ratio)
    train_data = data[:n_train]
    val_data = data[n_train:]
    train_dset = TensorDataset(train_data)
    val_dset = TensorDataset(val_data)
    train_loader = DataLoader(train_dset, batch_size=args.train_batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dset, batch_size=args.val_batch_size)
    return train_loader, val_loader

def build_model(args, logger):
    # Construct model and apply spectral norm if requested
    logger.info('Building model...')
    if args.model == 'gaussian':
        model = GaussianVAE(args).to(args.device)
    elif args.model == 'flow':
        model = FlowVAE(args).to(args.device)
    else:
        raise ValueError(f"Unsupported model type: {args.model}")
    logger.info(repr(model))
    if args.spectral_norm:
        add_spectral_norm(model, logger=logger)
    return model

def load_checkpoint_if_exists(checkpoint_path, model, optimizer, scheduler, args, logger):
    # Load checkpoint if exists, return start epoch and previous logs or defaults
    start_epoch = 1
    epoch_losses = []
    epoch_val_losses = []
    epoch_numbers = []
    val_epochs = []
    lr_change_log = []
    resuming = False

    if os.path.exists(checkpoint_path):
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=args.device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch = ckpt['epoch'] + 1
        epoch_losses = ckpt.get('losses', [])
        epoch_val_losses = ckpt.get('val_losses', [])
        epoch_numbers = ckpt.get('epoch_numbers', [])
        val_epochs = ckpt.get('val_epochs', [])
        lr_change_log = ckpt.get('lr_change_log', [])
        resuming = True
        logger.info(f"Resumed at epoch {start_epoch}")

    return start_epoch, epoch_losses, epoch_val_losses, epoch_numbers, val_epochs, lr_change_log, resuming

def validate_inspect(model, val_loader, args, epoch, logger, writer):
    # Evaluate model on validation set and log average loss
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0].to(args.device)
            loss = model.get_loss(x, kl_weight=args.kl_weight)
            total_loss += loss.item()
            count += 1
    avg_val_loss = total_loss / count
    logger.info(f"[Validation] Epoch {epoch} | Avg Val Loss: {avg_val_loss:.6f}")
    writer.add_scalar('val/loss', avg_val_loss, epoch)
    return avg_val_loss

def train_loop(args):
    # Main training loop including training, validation, checkpoint saving, and logging
    logger, writer, ckpt_mgr = setup_logging(args)
    train_loader, val_loader = load_data(args, logger)
    model = build_model(args, logger)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_scheduler(
        optimizer,
        start_epoch=args.sched_start_epoch,
        end_epoch=args.sched_end_epoch,
        start_lr=args.lr,
        end_lr=args.end_lr
    )

    checkpoint_filename = f"resume_ckpt_{args.tag}.pt" if args.tag else "resume_ckpt.pt"
    checkpoint_path = os.path.join(args.log_root if args.logging else '.', checkpoint_filename)

    start_epoch, epoch_losses, epoch_val_losses, epoch_numbers, val_epochs, lr_change_log, resuming = \
        load_checkpoint_if_exists(checkpoint_path, model, optimizer, scheduler, args, logger)

    prev_lr = optimizer.param_groups[0]['lr']
    plots_dir = os.path.join(args.log_root, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    try:
        for epoch in range(start_epoch, args.max_epochs + 1):
            running_loss = 0.0
            running_count = 0

            model.train()
            for batch in train_loader:
                x = batch[0].to(args.device)
                optimizer.zero_grad()
                if args.spectral_norm:
                    spectral_norm_power_iteration(model, n_power_iterations=1)

                loss = model.get_loss(x, kl_weight=args.kl_weight)
                loss.backward()
                grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

                running_loss += loss.item()
                running_count += 1

                current_lr = optimizer.param_groups[0]['lr']
                if current_lr != prev_lr:
                    lr_change_log.append((epoch, prev_lr, current_lr))
                    prev_lr = current_lr

            scheduler.step()
            avg_epoch_loss = running_loss / running_count

            epoch_losses.append(avg_epoch_loss)
            epoch_numbers.append(epoch)

            logger.info(f'[Train] Epoch {epoch} | Avg Loss {avg_epoch_loss:.6f} | Grad {grad_norm:.4f} | LR {optimizer.param_groups[0]["lr"]:.6e}')
            writer.add_scalar('train/loss', avg_epoch_loss, epoch)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('train/grad_norm', grad_norm, epoch)

            if epoch % 20 == 0 or epoch == args.max_epochs:
                val_loss = validate_inspect(model, val_loader, args, epoch, logger, writer)
                epoch_val_losses.append(val_loss)
                val_epochs.append(epoch)

                opt_states = {
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }
                ckpt_mgr.save(model, args, 0, others=opt_states, step=epoch)
                torch.save({
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'epoch': epoch,
                    'losses': epoch_losses,
                    'val_losses': epoch_val_losses,
                    'epoch_numbers': epoch_numbers,
                    'val_epochs': val_epochs,
                    'lr_change_log': lr_change_log,
                }, checkpoint_path)

    except KeyboardInterrupt:
        logger.info('Training interrupted. Saving checkpoint...')
        torch.save({
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'epoch': epoch,
            'losses': epoch_losses,
            'val_losses': epoch_val_losses,
            'epoch_numbers': epoch_numbers,
            'lr_change_log': lr_change_log,
        }, checkpoint_path)
        logger.info('Checkpoint saved.')

    plot_loss(args, epoch_losses, epoch_numbers, epoch_val_losses, val_epochs, lr_change_log, plots_dir, resuming, start_epoch)

def plot_loss(args, epoch_losses, epoch_numbers, epoch_val_losses, val_epochs, lr_change_log, plots_dir, resuming, start_epoch):
    # Plot training and validation losses with smoothing and save the figure
    def moving_average(data, window_size):
        if len(data) < window_size:
            return data
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    window_size = 5
    loss_array = np.array(epoch_losses)
    epoch_array = np.array(epoch_numbers)
    val_loss_array = np.array(epoch_val_losses)
    val_epoch_array = np.array(val_epochs)

    smoothed_losses = moving_average(loss_array, window_size)
    smoothed_epochs = epoch_array[len(epoch_array) - len(smoothed_losses):]

    smoothed_val_losses = moving_average(val_loss_array, window_size)
    smoothed_val_epochs = val_epoch_array[len(val_epoch_array) - len(smoothed_val_losses):]

    plt.figure()
    plt.plot(epoch_array, loss_array, label="raw train loss", color='gray', alpha=0.3)
    plt.plot(val_epoch_array, val_loss_array, label="raw val loss", color='gray', alpha=0.3, linestyle='--')
    plt.plot(smoothed_epochs, smoothed_losses, label="smoothed train loss", color='blue', linewidth=2)
    plt.plot(smoothed_val_epochs, smoothed_val_losses, label="smoothed val loss", color='green', linewidth=2)
    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (log scale)")
    plt.title(f"Training and Validation Loss vs. Epochs (Smoothed) - {args.tag}")

    combined_min_loss = min(min(smoothed_losses), min(smoothed_val_losses))
    combined_max_loss = max(max(smoothed_losses), max(smoothed_val_losses))

    if combined_max_loss > 70:
        plt.ylim(combined_min_loss, 70)
    else:
        plt.ylim(combined_min_loss, combined_max_loss)

    plt.axvline(x=start_epoch, color='red', linestyle='--', label='Resumed')
    plt.legend()
    plt.savefig(get_unique_filename(plots_dir, "plot_loss_epochs.png"))
    plt.show()

    # Log LR and Val Loss info to file
    tag_suffix = f"_{args.tag}" if args.tag else ""
    lr_val_log_path = os.path.join(args.log_root, f"lr_val_log{tag_suffix}.txt")
    mode = "a" if resuming else "w"
    with open(lr_val_log_path, mode) as f:
        if not resuming:
            f.write("Learning Rate Changes Log\n")
            f.write("=" * 30 + "\n\n")
            f.write("Validation Loss Log\n")
            f.write("=" * 30 + "\n")

        for (epoch_num, old_lr, new_lr) in lr_change_log:
            f.write(f"Epoch {epoch_num}: LR {old_lr:.6e} → {new_lr:.6e}\n")

        f.write("\n")

        for epoch_num, val_loss in zip(val_epochs, epoch_val_losses):
            f.write(f"Epoch {epoch_num}: Val Loss {val_loss:.6f}\n")

if __name__ == "__main__":
    # Entry point for running training script
    args = parse_args()
    seed_all(args.seed)
    train_loop(args)
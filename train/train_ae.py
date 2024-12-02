import os
import argparse
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
from collections import deque
from torch.utils.data import DataLoader, TensorDataset

from utils.dataset import *
from utils.misc import *
from utils.data import *
from utils.transform import *
from models.autoencoder import *
from evaluation import EMD_CD
import matplotlib.pyplot as plt

train_losses = []
val_losses = []
train_itters = []
val_itters = []
# Arguments
parser = argparse.ArgumentParser()
# Model arguments
parser.add_argument('--latent_dim', type=int, default=256)
parser.add_argument('--num_steps', type=int, default=200)
parser.add_argument('--beta_1', type=float, default=1e-4)
parser.add_argument('--beta_T', type=float, default=0.05)
parser.add_argument('--sched_mode', type=str, default='linear')
parser.add_argument('--flexibility', type=float, default=0.0)
parser.add_argument('--residual', type=eval, default=True, choices=[True, False])
parser.add_argument('--resume', type=str, default=None)

# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default='data/Mg22_Unpair/Mg22_experimental_1280.npy')
parser.add_argument('--categories', type=str_list, default=['Ar46'])
parser.add_argument('--scale_mode', type=str, default=None)
parser.add_argument('--train_batch_size', type=int, default=128)
parser.add_argument('--val_batch_size', type=int, default=32)
parser.add_argument('--train_ratio', type=float, default=0.8)
parser.add_argument('--rotate', type=eval, default=False, choices=[True, False])

# Optimizer and scheduler
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--max_grad_norm', type=float, default=10)
parser.add_argument('--end_lr', type=float, default=1e-4)
parser.add_argument('--sched_start_epoch', type=int, default=150*THOUSAND)
parser.add_argument('--sched_end_epoch', type=int, default=300*THOUSAND)

# Training
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
parser.add_argument('--log_root', type=str, default='./logs_ae')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_iters', type=int, default= 100 * THOUSAND)
parser.add_argument('--val_freq', type=float, default= 1 * THOUSAND)
parser.add_argument('--window_size', type=int, default= 1000)
parser.add_argument('--tag', type=str, default=None)
parser.add_argument('--num_val_batches', type=int, default=-1)
parser.add_argument('--num_inspect_batches', type=int, default=1)
parser.add_argument('--num_inspect_pointclouds', type=int, default=4)
args = parser.parse_args()
seed_all(args.seed)

train_loss_window = deque(maxlen=args.window_size)
# Logging
if args.logging:
    log_dir = get_new_log_dir(args.log_root, prefix='AE_4D_', postfix='_' + args.tag if args.tag is not None else '')
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    ckpt_mgr = CheckpointManager(log_dir)
else:
    logger = get_logger('train', None)
    writer = BlackHole()
    ckpt_mgr = BlackHole()
logger.info(args)

# Datasets and loaders
transform = None
if args.rotate:
    transform = RandomRotate(180, ['pointcloud'], axis=1)
logger.info('Transform: %s' % repr(transform))
logger.info('Loading datasets...')


# Load the data
data = np.load(args.dataset_path)

# Convert to PyTorch Tensors
data = torch.from_numpy(data).float()

n_train = int(len(data) * args.train_ratio)

# Split the data
train_data = data[:n_train]
val_data = data[n_train:]

# Create TensorDatasets
train_dset = TensorDataset(train_data)
val_dset = TensorDataset(val_data)

train_iter = get_data_iterator(DataLoader(
    train_dset,
    batch_size=args.train_batch_size,
    num_workers=0,
))
val_loader = DataLoader(val_dset, batch_size=args.val_batch_size)


# Model
logger.info('Building model...')
if args.resume is not None:
    logger.info('Resuming from checkpoint...')
    ckpt = torch.load(args.resume)
    model = AutoEncoder(ckpt['args']).to(args.device)
    model.load_state_dict(ckpt['state_dict'])
else:
    model = AutoEncoder(args).to(args.device)
logger.info(repr(model))


# Optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), 
    lr=args.lr, 
    weight_decay=args.weight_decay
)
scheduler = get_linear_scheduler(
    optimizer,
    start_epoch=args.sched_start_epoch,
    end_epoch=args.sched_end_epoch,
    start_lr=args.lr,
    end_lr=args.end_lr
)

# Train, validate 
def train(it):
    # Load data
    batch = next(train_iter)
    x = batch[0].to(args.device)

    # Reset grad and model state
    optimizer.zero_grad()
    model.train()

    # Forward
    loss = model.get_loss(x)

    # Backward and optimize
    loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    scheduler.step()
    
    #taking window average of the training loss
    train_loss_window.append(loss.item())
    if len(train_loss_window) == args.window_size:
        if it % args.window_size == 0:
            train_loss_avg = np.mean(train_loss_window)
            train_losses.append(train_loss_avg)
            train_itters.append(it)
    
    logger.info('[Train] Iter %04d | Loss %.6f | Grad %.4f ' % (it, loss.item(), orig_grad_norm))
    
    
        
    writer.add_scalar('train/loss', loss, it)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
    writer.add_scalar('train/grad_norm', orig_grad_norm, it)
    writer.flush()


def validate_loss(it):
    model.eval()

    total_loss = 0.0
    num_batch = 0
    with torch.no_grad():  # Deactivate gradients for the following code
        for i, batch in enumerate(tqdm(val_loader, desc='Validate')):
            num_batch += 1
            x = batch[0].to(args.device)
            # Compute loss
            loss = model.get_loss(x)
            total_loss += loss.item()

    # Compute and return average loss
    avg_loss = total_loss / num_batch
    val_losses.append(avg_loss)
    val_itters.append(it)
    return avg_loss

# Main loop
logger.info('Start training...')
try:
    it = 1
    while it <= args.max_iters:
        train(it)
        if it % args.val_freq == 0 or it == args.max_iters:
            with torch.no_grad():
                cd_loss = validate_loss(it)
            opt_states = {
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            ckpt_mgr.save(model, args, cd_loss, opt_states, step=it)
        it += 1

except KeyboardInterrupt:
    logger.info('Terminating...')

plt.plot(train_itters, train_losses, label="Training loss")
plt.plot(val_itters, val_losses, label="Validation loss")
plt.legend()
plt.title("Loss vs. Iterations for general Argon dataset")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.savefig('plot_loss_exp.png')
plt.show()

print("END")
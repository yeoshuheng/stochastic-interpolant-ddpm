import torch, os
from src.DDPM import DDPM, get_loss
from torch.optim import Adam
from src.dataset import data_loader
from src.config import CONFIG
import logging

LOG_FILE = os.getcwd() + "/logs"
if not os.path.exists(LOG_FILE):
    os.makedirs(LOG_FILE)
LOG_FILE = LOG_FILE + "/ddpm_default_run.log"

def main():
    print("Starting job.")
    logger = logging.getLogger(__name__)
    handler = logging.FileHandler(LOG_FILE, mode="w")
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    logger.info("job started")

    # Donwnload CIFAR10 data
    train_loader, test_loader = data_loader()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Selected device: {}".format(device))

    # Initialise model & optimizer
    model = DDPM()
    model.to(device)
    optimizer = Adam(model.parameters(), lr=0.01)
    epoch = CONFIG["epoch"]
    T = CONFIG["total_timesteps"]
    batch_size = CONFIG["batch_size"]

    logger.info("configuration done")

    for e in range(epoch): # Default training process (Non stochastic-interpolant)
        logger.info("Finished Epoch: " + str(e))
        for i, img_ in enumerate(train_loader):
            print("Epoch: {} | Iteration: {}".format(e, i))
            img = img_[0]
            optimizer.zero_grad()
            t = torch.randint(0, T, (batch_size, ), device=device).type(torch.int64)
            loss = get_loss(img, t, model)
            loss.backward()
            optimizer.step()
        
    path = "./../checkpoints/"
    if not os.path.exists(path):
        os.makedirs(path)

    # Save final model
    torch.save(model.state_dict(), os.path.join(path, "ddpm.pt"))
    logger.log("model saved")

if __name__ == "__main__":
    main()
    print("========== Ending script ==========")
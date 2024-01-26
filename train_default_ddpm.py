import torch, os
from src.DDPM import DDPM, get_loss
from torch.optim import Adam
from src.dataset import data_loader
from src.config import CONFIG

def main():
    print("Starting job.")
    
    # Donwnload CIFAR10 data
    train_loader, test_loader = data_loader()

    # Initialise model & optimizer
    
    model = DDPM()
    optimizer = Adam(model.parameters(), lr=0.01)
    epoch = CONFIG["epoch"]
    T = CONFIG["total_timesteps"]
    batch_size = CONFIG["batch_size"]

    for e in range(epoch): # Default training process (Non stochastic-interpolant)
        for i, img_ in enumerate(train_loader):
            print("Epoch: {} | Iteration: {}".format(e, i))
            img = img_[0]
            optimizer.zero_grad()
            t = torch.randint(0, T, (batch_size, )).type(torch.int64)
            loss = get_loss(img, t, model)
            loss.backward()
            optimizer.step()
        
    path = "./../checkpoints/"
    if not os.path.exists(path):
        os.makedirs(path)
    # Save final model
    torch.save(model.state_dict(), os.path.join(path, "ddpm.pt"))

if __name__ == "__main__":
    main()
    print("========== Ending script ==========")
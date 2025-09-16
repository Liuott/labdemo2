# train_cifar_fast.py
import time, argparse
import torch, torchvision as tv
import torchvision.transforms as T
from torch import amp
from torchvision.datasets import CIFAR10

def get_data(bs):
    mean = (0.4914, 0.4822, 0.4465); std = (0.2470, 0.2435, 0.2616)
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    #after download,change to false
    train = tv.datasets.CIFAR10(root="./data", train=True,  download=False, transform=train_tf)
    test  = tv.datasets.CIFAR10(root="./data", train=False, download=False, transform=test_tf)
    train_ld = torch.utils.data.DataLoader(train, batch_size=bs, shuffle=True,
                                           num_workers=0, pin_memory=True, persistent_workers=False)
    test_ld  = torch.utils.data.DataLoader(test, batch_size=bs*2, shuffle=False,
                                           num_workers=0, pin_memory=True, persistent_workers=False)
    return train_ld, test_ld

def accuracy(model, loader, device):
    model.eval(); correct=total=0
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total   += y.size(0)
    return correct/total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.4)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--arch", type=str, default="resnet18") 
    ap.add_argument("--amp", action="store_true")
    args, _ = ap.parse_known_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    train_ld, test_ld = get_data(args.batch)
    print("Data ready:", len(train_ld), "train iters;", len(test_ld), "test iters")

   
    if args.arch == "resnet18":
        model = tv.models.resnet18(weights=None, num_classes=10)
    else:
        raise NotImplementedError("add your ResNet9 here")
    model.to(device).to(memory_format=torch.channels_last)

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.wd, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_ld)
    )
    enabled_amp = (device == "cuda" and args.amp)           # 只有 CUDA 才启用 AMP
    scaler = amp.GradScaler("cuda", enabled=enabled_amp) 
    

    t0 = time.time()
    for ep in range(1, args.epochs+1):
        model.train()
        for x,y in train_ld:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with amp.autocast("cuda", enabled=enabled_amp):         # ✅ 新写法
                out = model(x)
                loss = loss_fn(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()

        acc = accuracy(model, test_ld, device)
        print(f"epoch {ep:02d}/{args.epochs} | test acc={acc:.4f}")
    elapsed = time.time() - t0
    print(f"Final test acc={acc:.4f} | wall time={elapsed:.1f}s")


    torch.save(model.state_dict(), "cifar_fast.pth")

if __name__ == "__main__":
    main()

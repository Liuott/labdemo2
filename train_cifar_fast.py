
import time, argparse, json, os
import torch, torchvision as tv
import torchvision.transforms as T
from torch import amp
from torchvision.datasets import CIFAR10


def resnet18_cifar10(num_classes=10):
    """
    Replace the ImageNet stem (7×7 s=2 + maxpool) with a 3×3 s=1 conv and remove maxpool.
    On 32×32 images, this preserves spatial detail and usually boosts accuracy.
    """
    m = tv.models.resnet18(weights=None, num_classes=num_classes)
    m.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = torch.nn.Identity()
    return m

def get_data(bs):
    """
    Create train/test DataLoaders.
    - Augmentation: RandomCrop(32, padding=4) + RandomHorizontalFlip
      If your torchvision has TrivialAugmentWide/RandomErasing, we enable them (safe defaults).
    - Normalization uses CIFAR-10 channel stats.
    """
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    aug = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]

    # Optional operators are added only if present in your torchvision version.
    # They are safe, light regularizers that often help generalization.
    if getattr(T, "TrivialAugmentWide", None) is not None:
        aug += [T.TrivialAugmentWide()]

    aug += [T.ToTensor(), T.Normalize(mean, std)]


    if getattr(T, "RandomErasing", None) is not None:
        aug += [T.RandomErasing(p=0.25, scale=(0.02, 0.2), value='random')]

    train_tf = T.Compose(aug)
    test_tf  = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    train = CIFAR10(root="./data", train=True,  download=True, transform=train_tf)
    test  = CIFAR10(root="./data", train=False, download=True, transform=test_tf)

    # Use ~half of logical CPUs; adjust if your input pipeline is the bottleneck.
    num_workers = max(2, (os.cpu_count() or 2) // 2)
    train_ld = torch.utils.data.DataLoader(
        train, batch_size=bs, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=True
    )
    test_ld = torch.utils.data.DataLoader(
        test, batch_size=bs*2, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=True
    )
    return train_ld, test_ld


def evaluate(model, loader, device, topk=(1,)):
    """
    Compute top-k accuracy.
    Returns a dict: {1: acc@1, 5: acc@5, ...} where values are floats in [0, 1].
    """
    model.eval()
    total = 0
    correct = {k: 0 for k in topk}
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            out = model(x)
            total += y.size(0)
            maxk = max(topk)
            _, pred = out.topk(maxk, dim=1, largest=True, sorted=True)
            pred = pred.t()
            eq = pred.eq(y.view(1, -1).expand_as(pred))
            for k in topk:
                correct[k] += eq[:k].reshape(-1).float().sum().item()
    return {k: correct[k] / total for k in topk}

# Build SGD optimizer with separate weight decay for parameters that should not be decayed.
def build_opt(model, lr, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: 
            continue
        if p.ndim == 1 or n.endswith(".bias"):   
            no_decay.append(p)
        else:
            decay.append(p)                     
    param_groups = [
        {'params': decay, 'weight_decay': wd},
        {'params': no_decay, 'weight_decay': 0.0},
    ]
    return torch.optim.SGD(param_groups, lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)

def main():
    # Parse command line arguments.
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--arch", type=str, default="resnet18")
    ap.add_argument("--label_smooth", type=float, default=0.05)
    ap.add_argument("--amp", action="store_true")
    args, _ = ap.parse_known_args()

    # Setup device and cudnn benchmark.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    # Load data.
    train_ld, test_ld = get_data(args.batch)
    print("Data ready:", len(train_ld), "train iters;", len(test_ld), "test iters")

    # Create model.
    if args.arch == "resnet18":
        model = resnet18_cifar10(num_classes=10)
    else:
        raise NotImplementedError("Only resnet18 is provided in this minimal script.")
    model.to(device).to(memory_format=torch.channels_last)

    # Loss, optimizer, LR scheduler, AMP scaler.
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
    opt = build_opt(model, args.lr, args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_ld),
        pct_start=0.2, div_factor=10.0, final_div_factor=100.0
    )

    # Use AMP if requested and on CUDA.
    enabled_amp = (device == "cuda" and args.amp)
    scaler = amp.GradScaler("cuda", enabled=enabled_amp)

    # For tracking best accuracy.
    best_acc1, best_ep = 0.0, 0
    t0 = time.time()

    # Train and evaluate.
    for ep in range(1, args.epochs + 1):
        model.train()
        for x, y in train_ld:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with amp.autocast("cuda", enabled=enabled_amp):
                out = model(x)
                loss = loss_fn(out, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()

        metrics = evaluate(model, test_ld, device, topk=(1, 5))
        acc1, acc5 = metrics[1], metrics[5]
        if acc1 > best_acc1:
            best_acc1, best_ep = acc1, ep
            torch.save(model.state_dict(), "cifar_fast_best.pth")
        print(f"epoch {ep:02d}/{args.epochs} | acc@1={acc1:.4f} acc@5={acc5:.4f} "
              f"| best@1={best_acc1:.4f} (ep {best_ep}) | loss={loss.item():.4f}")

    # Final evaluation.
    elapsed = time.time() - t0
    final = evaluate(model, test_ld, device, topk=(1, 5))
    print(f"Final acc@1={final[1]:.4f} acc@5={final[5]:.4f} | best@1={best_acc1:.4f} (ep {best_ep}) "
          f"| wall time={elapsed:.1f}s")

    # Save metrics and final model.
    with open("metrics.json", "w") as f:
        json.dump({
            "final_acc1": final[1],
            "final_acc5": final[5],
            "best_acc1": best_acc1,
            "best_epoch": best_ep,
            "epochs": args.epochs,
            "batch": args.batch,
            "lr": args.lr,
            "weight_decay": args.wd,
            "arch": args.arch,
            "label_smooth": args.label_smooth,
            "amp": bool(enabled_amp),
            "elapsed_sec": elapsed
        }, f, indent=2)

    torch.save(model.state_dict(), "cifar_fast_final.pth")

if __name__ == "__main__":
    main()

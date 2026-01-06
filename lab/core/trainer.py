# core/trainer.py
import torch

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

class Trainer:
    def __init__(self, model, task, algorithm, device="cpu", input_noise_training=0.0, verbose=False):
        self.model = model.to(device)
        self.task = task
        self.input_noise_training = input_noise_training
        self.algorithm = algorithm
        self.device = device
        self.verbose = verbose

    def fit(self, train_loader, epochs=10, show_progress=True):
        self.algorithm.on_train_start(self.model, self.task, self.device)

        for ep in range(1, epochs + 1):
            # Running stats for epoch summary
            sum_loss = 0.0
            sum_acc = 0.0
            n_batches = 0

            iterator = train_loader
            use_bar = show_progress and (tqdm is not None)

            if use_bar:
                iterator = tqdm(train_loader, desc=f"Epoch {ep}/{epochs}", leave=False)

            for batch in iterator:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)

                if self.input_noise_training > 0.0:
                    x = x + torch.randn_like(x) * self.input_noise_training
                    
                stats = self.algorithm.train_step(self.model, self.task, (x, y), self.device)

                loss = float(stats.get("loss", 0.0))
                acc = float(stats.get("acc", 0.0))

                sum_loss += loss
                sum_acc += acc
                n_batches += 1

                if use_bar:
                    iterator.set_postfix(loss=sum_loss / n_batches, acc=sum_acc / n_batches)

            # Print ONLY end-of-epoch
            mean_loss = sum_loss / max(1, n_batches)
            mean_acc = sum_acc / max(1, n_batches)
            if self.verbose:
                print(f"Epoch {ep}/{epochs} | train_loss={mean_loss:.4f} | train_acc={mean_acc*100:.2f}%")

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        
        if hasattr(self.task, "evaluate"):
            return self.task.evaluate(self.model, loader, self.device)
    
        total_acc = 0.0
        total_n = 0
        total_loss = 0.0

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = self.task.loss(logits, y)
            acc = self.task.metrics(logits, y)["acc"]

            n = x.size(0)
            total_loss += loss.item() * n
            total_acc += acc * n
            total_n += n

        return {"loss": total_loss / total_n, "acc": total_acc / total_n}

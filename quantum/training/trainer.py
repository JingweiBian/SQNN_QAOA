# -*- coding: utf-8 -*-

import os
import sys

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".matplotlib"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..classifiers.networkmodels import (
    DataReuploadingSoftQuantumNeuralNetwork,
    SoftQuantumNeuralNetwork,
    QuantumLoss
)
from tools.utils import (
    EfficientQuantumSampler,
    OptimizerManager,
    plot_training_metrics,
    save_training_history
)
from tools.dataloader import create_mnist_loaders


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


class QuantumTrainer:
    def __init__(self, config):
        self._is_setup = False
        self.config = config

        self.debug_mode = config.get('debug_mode', None)
        self.debug_samples = config.get('debug_samples', 10)
        self.data_dir = config.get('data_dir', './data')
        self.experiment_name = config.get('experiment_name', 'default_experiment')
        self.seed = config.get('seed', 42)

        self.epochs = config.get('epochs', 100)
        self.batch_size = config.get('batch_size', 32)
        self.layer_dims = config.get('layer_dims', None)
        self.layers_size = config.get('layers_size', None)
        self.encoding_config = config.get('encoding_config', ['angle', None])
        self.noise_type = config.get('noise_config', None)
        self.model_type = config.get('model_type', 'original')

        self.optimizer_type = config.get('optimizer_type', 'adam')
        self.scheduler_type = config.get('scheduler_type', None)
        self.optimizer_kwargs = config.get('optimizer_kwargs', {})
        self.scheduler_kwargs = config.get('scheduler_kwargs', {})
        self.evaluate_test_each_epoch = config.get('evaluate_test_each_epoch', False)

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.model = None
        self.optimizer = None
        self.optimizer_manager = None
        self.scheduler = None
        self.loss_fn = None
        self.sampler = None

        self.device = self._resolve_device(config.get('device', 'cpu'))
        self.digits = config.get('digits', [0, 1])
        self.num_classes = len(self.digits)

        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.test_losses = None
        self.results = {}
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': [],
            'test_acc': [],
            'lr': []
        }

    def _resolve_device(self, requested_device):
        device = str(requested_device)
        if not device.startswith('cuda'):
            return device
        if not torch.cuda.is_available():
            print(f"CUDA unavailable, falling back to CPU: {device} -> cpu")
            return 'cpu'
        device_index = int(device.split(':')[1]) if ':' in device else 0
        if device_index >= torch.cuda.device_count():
            print(f"CUDA device {device} unavailable, falling back to cuda:0")
            return 'cuda:0'
        return device

    def setup(self):
        if self._is_setup:
            print("Trainer is already set up; skipping setup.")
            return

        print("=" * 50)
        print("Initializing quantum neural network trainer...")

        print("\n[1/4] Creating data loaders...")
        self._setup_data_loaders()

        print("\n[2/4] Building quantum model...")
        self._setup_quantum_model()

        print("\n[3/4] Creating optimizer and scheduler...")
        self._setup_optimizer()

        print("\n[4/4] Creating loss function...")
        self._setup_loss_function()

        self._is_setup = True
        print("\nAll components initialized.")
        print("=" * 50)

    def _setup_data_loaders(self):
        print(f"Data directory: {self.data_dir}")
        data_exists = self._check_mnist_exists(self.data_dir)
        train_loader, test_loader = create_mnist_loaders(
            self.config,
            download=not data_exists
        )

        train_dataset = train_loader.dataset
        total_numbers = len(train_dataset)
        train_size = int(0.8 * total_numbers)
        val_size = total_numbers - train_size

        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed)
        )

        loader_kwargs = {
            'num_workers': self.config.get('num_workers', 0),
            'pin_memory': str(self.device).startswith('cuda')
        }

        self.train_loader = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            **loader_kwargs
        )
        self.val_loader = DataLoader(
            val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            **loader_kwargs
        )
        self.test_loader = test_loader

        print(f"Original train samples: {total_numbers}")
        print(f"Train samples: {len(train_subset)}")
        print(f"Validation samples: {len(val_subset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")

    def _check_mnist_exists(self, data_dir):
        mnist_path = os.path.join(data_dir, 'MNIST', 'raw')
        required_files = [
            'train-images-idx3-ubyte',
            'train-labels-idx1-ubyte',
            't10k-images-idx3-ubyte',
            't10k-labels-idx1-ubyte'
        ]
        return os.path.exists(mnist_path) and all(
            os.path.exists(os.path.join(mnist_path, file_name))
            for file_name in required_files
        )

    def _setup_quantum_model(self):
        if self.layer_dims is None:
            raise RuntimeError("Missing config: layer_dims")
        if self.layers_size is None:
            raise RuntimeError("Missing config: layers_size")

        model_cls = {
            'original': SoftQuantumNeuralNetwork,
            'data_reuploading': DataReuploadingSoftQuantumNeuralNetwork,
            'encoding_skip': DataReuploadingSoftQuantumNeuralNetwork,
        }.get(self.model_type)
        if model_cls is None:
            raise ValueError(
                f"Unsupported model_type={self.model_type}. "
                "Use 'original' or 'data_reuploading'."
            )

        self.model = model_cls(
            self.layer_dims,
            self.layers_size,
            self.noise_type,
            self.encoding_config
        ).to(self.device)

        print(f"Model type: {self.model_type}")
        print(f"Model created on {self.device}: {self.model}")
        print(f"Layer dims: {self.layer_dims}")

    def _setup_optimizer(self):
        if self.model is None:
            raise RuntimeError("Create the model before the optimizer.")

        self.optimizer_manager = OptimizerManager(
            model_params=self.model.parameters(),
            config=self.config
        )

        optimizer_kwargs = dict(self.optimizer_kwargs or {})
        if 'learning_rate' in self.config and 'lr' not in optimizer_kwargs:
            optimizer_kwargs['lr'] = self.config['learning_rate']

        self.optimizer = self.optimizer_manager.create_optimizer(
            optimizer_type=self.optimizer_type,
            **optimizer_kwargs
        )
        print(f"Optimizer: {self.optimizer}")

        if self.scheduler_type:
            self.scheduler = self.optimizer_manager.create_scheduler(
                scheduler_type=self.scheduler_type,
                **dict(self.scheduler_kwargs or {})
            )
            print(f"Scheduler: {self.scheduler}")
        else:
            self.scheduler = None
            print("Scheduler: none")

    def _setup_loss_function(self):
        self.loss_fn = QuantumLoss()
        print(f"Loss function: {self.loss_fn}")

    def _setup_sampler(self):
        if not self.model:
            raise RuntimeError("Create the quantum model before the sampler.")

        self.sampler = EfficientQuantumSampler(
            self.model,
            self.config.get('training_sample_numbers', 1024),
            self.config.get('evaluation_sample_numbers', 1024),
            self.config.get('chunk_size', 256)
        )

    def train_epoch(self):
        self.model.train()
        if self.sampler:
            self.sampler.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for data, target in self.train_loader:
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            output = self.sampler(data) if self.sampler else self.model(data)
            output = output.to(self.device)
            one_hot_target = F.one_hot(
                target,
                num_classes=self.num_classes
            ).float().to(self.device)

            loss = self.loss_fn(output, one_hot_target)
            loss.backward()
            self.optimizer.step()

            batch_size = data.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            total_correct += output.argmax(dim=1).eq(target).sum().item()

        avg_loss = total_loss / total_samples if total_samples else 0.0
        accuracy = 100.0 * total_correct / total_samples if total_samples else 0.0
        return avg_loss, accuracy

    def validate(self):
        return self._evaluate_loader(self.val_loader)

    def evaluate(self):
        return self._evaluate_loader(self.test_loader or self.val_loader)

    def _evaluate_loader(self, loader):
        self.model.eval()
        if self.sampler:
            self.sampler.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for data, target in loader:
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                output = self.sampler(data) if self.sampler else self.model(data)
                output = output.to(self.device)
                one_hot_target = F.one_hot(
                    target,
                    num_classes=self.num_classes
                ).float().to(self.device)

                loss = self.loss_fn(output, one_hot_target)

                batch_size = data.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                total_correct += output.argmax(dim=1).eq(target).sum().item()

        avg_loss = total_loss / total_samples if total_samples else 0.0
        accuracy = 100.0 * total_correct / total_samples if total_samples else 0.0
        return avg_loss, accuracy

    def _results_dir(self):
        save_dir = os.path.join('checkpoints', self.experiment_name)
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def _save_model(self, epoch, val_acc):
        save_dir = self._results_dir()
        filename = f'epoch{epoch + 1}_acc{val_acc:.2f}.pth'
        save_path = os.path.join(save_dir, filename)

        checkpoint = {
            'epoch': epoch,
            'val_accuracy': val_acc,
            'experiment_name': self.experiment_name,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint: {save_path}")
        return save_path

    def _create_best_model_file(self, best_model_path, best_epoch, best_val_acc):
        checkpoint = torch.load(best_model_path, map_location=self.device)
        checkpoint['is_best'] = True
        checkpoint['best_epoch'] = best_epoch
        checkpoint['best_val_acc'] = best_val_acc
        checkpoint['original_file'] = os.path.basename(best_model_path)

        best_path = os.path.join(os.path.dirname(best_model_path), 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"Best model updated: {best_path}")
        return best_path

    def _record_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc,
                      test_loss=None, test_acc=None):
        lr = self.optimizer.param_groups[0]['lr'] if self.optimizer else None
        epoch_number = epoch + 1

        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_acc)

        self.history['epoch'].append(epoch_number)
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['test_loss'].append(test_loss)
        self.history['test_acc'].append(test_acc)
        self.history['lr'].append(lr)

    def save_training_history(self):
        save_path = os.path.join(self._results_dir(), 'training_history.csv')
        return save_training_history(self.history, save_path)

    def plot_training_curves(self):
        save_path = os.path.join(self._results_dir(), 'training_curves.png')
        return plot_training_metrics(
            self.history,
            save_path,
            title=f'{self.experiment_name} Training Metrics'
        )

    def train(self):
        best_val_acc = -1.0
        best_model_path = None

        print(f"\nStarting training: epochs={self.epochs}, batch_size={self.batch_size}")
        print("=" * 60)

        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            if self.evaluate_test_each_epoch:
                test_loss, test_acc = self.evaluate()
            else:
                test_loss, test_acc = None, None
            self._record_epoch(
                epoch,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                test_loss,
                test_acc
            )

            if self.scheduler:
                self.optimizer_manager.step_scheduler(metrics=val_loss)

            message = (
                f"Epoch {epoch + 1}/{self.epochs}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
            )
            if test_acc is not None:
                message += f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% | "
            message += f"LR: {self.history['lr'][-1]:.6g}"
            print(message)
            sys.stdout.flush()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_path = self._save_model(epoch, val_acc)
                self._create_best_model_file(best_model_path, epoch, best_val_acc)

        test_loss, test_acc = self.evaluate()
        self.results = {
            'best_val_acc': best_val_acc,
            'best_model_path': best_model_path,
            'test_loss': test_loss,
            'test_acc': test_acc
        }

        history_path = self.save_training_history()
        plot_path = self.plot_training_curves()
        print("=" * 60)
        print(f"Training complete. Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        if history_path:
            print(f"Training history saved: {history_path}")
        if plot_path:
            print(f"Training curves saved: {plot_path}")
        return self.results

#!/usr/bin/env python3
"""CIFAR-10 Active Learning Experiments"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import time
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, '/home/mohanganesh/active_learning_coreset')

from active_learning_strategies import RandomSampling, GreedyKCenter, LeaderClustering, AdvancedLeader


class VGG(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Linear(512, num_classes)
    
    def forward(self, x):
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        out = self.classifier(feat)
        return out, feat


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total, total_loss / len(loader)


def test_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total, total_loss / len(loader)


def run_experiment(strategy_name, strategy_class, args, device):
    print(f"\n{'='*80}\nSTRATEGY: {strategy_name}\n{'='*80}")
    
    # Disable cuDNN for compatibility
    torch.backends.cudnn.enabled = False
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)
    
    all_indices = list(range(len(trainset)))
    random.shuffle(all_indices)
    labeled_indices = all_indices[:args.initial_labeled]
    unlabeled_indices = all_indices[args.initial_labeled:]
    
    # V3: Pass total_rounds to strategy if it supports it
    try:
        active_learner = strategy_class(N=len(trainset), budget=args.budget, total_rounds=args.rounds)
    except TypeError:
        # Fallback for strategies that don't support total_rounds parameter
        active_learner = strategy_class(N=len(trainset), budget=args.budget)
    
    results = {
        'strategy': strategy_name, 'rounds': [], 'labeled_sizes': [],
        'test_accuracies': [], 'sampling_times': [], 'training_times': [], 'total_times': [],
    }
    
    for round_num in range(args.rounds):
        print(f"\n{'='*80}\nROUND {round_num+1}/{args.rounds}\nLabeled: {len(labeled_indices)}, Unlabeled: {len(unlabeled_indices)}\n{'='*80}")
        
        round_start = time.time()
        
        # Select new samples FIRST (before training)
        sampling_time = 0
        if round_num > 0 and len(unlabeled_indices) >= args.budget:
            print(f"\nSelecting {args.budget} new samples using {strategy_name}...")
            sampling_start = time.time()
            
            unlabeled_subset = torch.utils.data.Subset(trainset, unlabeled_indices)
            
            # V3: Pass round_num to select_batch if it supports it
            try:
                selected_relative = active_learner.select_batch(model, unlabeled_subset, round_num=round_num)
            except TypeError:
                # Fallback for strategies that don't support round_num parameter
                selected_relative = active_learner.select_batch(model, unlabeled_subset)
            
            selected_global = [unlabeled_indices[i] for i in selected_relative]
            
            sampling_time = time.time() - sampling_start
            print(f"Sampling time: {sampling_time:.2f}s")
            
            labeled_indices.extend(selected_global)
            unlabeled_indices = [idx for idx in unlabeled_indices if idx not in selected_global]
            print(f"Updated - Labeled: {len(labeled_indices)}, Unlabeled: {len(unlabeled_indices)}")
        
        # Now train model with current labeled set
        labeled_subset = torch.utils.data.Subset(trainset, labeled_indices)
        trainloader = torch.utils.data.DataLoader(labeled_subset, batch_size=128, shuffle=True, num_workers=0)
        
        model = VGG(num_classes=10).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[160, 240], gamma=0.1)
        
        print(f"\nTraining for {args.epochs} epochs...")
        training_start = time.time()
        
        for epoch in range(args.epochs):
            train_acc, train_loss = train_epoch(model, trainloader, optimizer, criterion, device)
            scheduler.step()
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{args.epochs}: Train Acc={train_acc:.2f}%")
        
        training_time = time.time() - training_start
        
        print("\nTesting...")
        test_acc, test_loss = test_model(model, testloader, criterion, device)
        print(f"Test Accuracy: {test_acc:.2f}%")
        
        round_time = time.time() - round_start
        
        results['rounds'].append(round_num + 1)
        results['labeled_sizes'].append(len(labeled_indices))
        results['test_accuracies'].append(test_acc)
        results['sampling_times'].append(sampling_time)
        results['training_times'].append(training_time)
        results['total_times'].append(round_time)
        
        print(f"\nRound {round_num+1} timing: Training={training_time:.2f}s, Sampling={sampling_time:.2f}s, Total={round_time:.2f}s")
    
    output_dir = Path('cifar10_results')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / f'{strategy_name}_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n{'='*80}\nEXPERIMENT COMPLETE: {strategy_name}\n{'='*80}")
    print(f"Final test accuracy: {results['test_accuracies'][-1]:.2f}%")
    print(f"Average sampling time: {np.mean(results['sampling_times']):.2f}s per round")
    print(f"Total time: {sum(results['total_times']):.2f}s")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 Active Learning')
    parser.add_argument('--strategy', type=str, required=True, choices=['random', 'greedy', 'leader', 'advanced'])
    parser.add_argument('--initial_labeled', type=int, default=5000)
    parser.add_argument('--budget', type=int, default=2500)
    parser.add_argument('--rounds', type=int, default=9)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    strategies = {
        'random': ('Random', RandomSampling),
        'greedy': ('Greedy_K-Center', GreedyKCenter),
        'leader': ('Leader_Clustering', LeaderClustering),
        'advanced': ('Advanced_Leader', AdvancedLeader),
    }
    
    strategy_name, strategy_class = strategies[args.strategy]
    run_experiment(strategy_name, strategy_class, args, device)


if __name__ == '__main__':
    main()

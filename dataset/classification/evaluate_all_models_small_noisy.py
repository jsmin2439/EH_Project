#!/usr/bin/env python3
"""
experiment_v3_results의 모든 모델에 대해 small/noisy 성능을 평가하고 
기존 분석에 추가하는 스크립트
"""

import sys
import os
sys.path.append('.')

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from dataloader import DigitData
import models
from torchvision import transforms

class ModelEvaluator:
    def __init__(self, data_dir="../digit_data"):
        # 절대 경로로 변환
        if not os.path.isabs(data_dir):
            data_dir = os.path.abspath(data_dir)
        self.data_dir = data_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 작은 글자 및 노이즈 샘플 로드
        self.small_samples = self.load_sample_list(f"{data_dir}/small_valid.txt")
        self.noisy_samples = self.load_sample_list(f"{data_dir}/noisy_valid.txt")
        
        print(f"Loaded {len(self.small_samples)} small samples")
        print(f"Loaded {len(self.noisy_samples)} noisy samples")
        
        # Transform 설정
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.80048384, 0.44734452, 0.50106468], 
                               std=[0.22327253, 0.29523788, 0.24583565])
        ])
    
    def load_sample_list(self, path):
        try:
            with open(path, 'r') as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"Warning: {path} not found")
            return []
    
    def load_model(self, model_name, model_path):
        """모델 로드"""
        try:
            if model_name == 'convnext':
                model = models.convnext_tiny_custom(num_classes=10)
            elif model_name == 'swin':
                model = models.swin_tiny_custom(num_classes=10)
            elif model_name == 'resnet20':
                model = models.resnet20(num_classes=10)
            elif model_name == 'model_resnet32x4_wide':
                model = models.resnet32x4(num_classes=10)  # wide version
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            # 모델 상태 로드
            if model_path.endswith('final_model.pth'):
                checkpoint = torch.load(model_path, map_location=self.device)
                model.load_state_dict(checkpoint)
            else:  # best_model.pth
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'model_state' in checkpoint:
                    model.load_state_dict(checkpoint['model_state'])
                else:
                    model.load_state_dict(checkpoint)
            
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading model {model_name} from {model_path}: {e}")
            return None
    
    def evaluate_model_on_subsets(self, model, model_name):
        """모델을 전체/작은글자/노이즈 서브셋에 대해 평가"""
        
        # 전체 validation dataset
        val_dataset = DigitData(self.data_dir, 64, 'valid', transform=self.transform, return_path=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # 전체 성능 평가
        total_correct = 0
        total_samples = 0
        small_correct = 0
        small_total = 0
        noisy_correct = 0
        noisy_total = 0
        
        small_set = set(self.small_samples)
        noisy_set = set(self.noisy_samples)
        
        with torch.no_grad():
            for batch in val_loader:
                images, labels, paths = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(images)
                _, predicted = outputs.max(1)
                
                correct = predicted.eq(labels)
                
                for i, path in enumerate(paths):
                    is_correct = correct[i].item()
                    
                    total_correct += is_correct
                    total_samples += 1
                    
                    if path in small_set:
                        small_correct += is_correct
                        small_total += 1
                    
                    if path in noisy_set:
                        noisy_correct += is_correct
                        noisy_total += 1
        
        # 결과 계산
        overall_acc = total_correct / total_samples if total_samples > 0 else 0
        small_acc = small_correct / small_total if small_total > 0 else None
        noisy_acc = noisy_correct / noisy_total if noisy_total > 0 else None
        
        return {
            'overall_acc': overall_acc,
            'small_acc': small_acc,
            'noisy_acc': noisy_acc,
            'total_samples': total_samples,
            'small_samples': small_total,
            'noisy_samples': noisy_total
        }
    
    def evaluate_all_models(self):
        """모든 모델 평가"""
        results = {}
        
        models_info = [
            ('resnet20', 'experiment_v3_results/resnet20/best_model.pth'),
            ('convnext', 'experiment_v3_results/convnext/final_model.pth'),
            ('swin', 'experiment_v3_results/swin/final_model.pth'),
            ('model_resnet32x4_wide', 'experiment_v3_results/model_resnet32x4_wide/best_model.pth')
        ]
        
        for model_name, model_path in models_info:
            print(f"\nEvaluating {model_name}...")
            
            if not os.path.exists(model_path):
                print(f"Model file not found: {model_path}")
                continue
            
            model = self.load_model(model_name, model_path)
            if model is None:
                continue
            
            metrics = self.evaluate_model_on_subsets(model, model_name)
            results[model_name] = metrics
            
            print(f"Results for {model_name}:")
            print(f"  Overall Accuracy: {metrics['overall_acc']:.4f}")
            if metrics['small_acc'] is not None:
                print(f"  Small Digit Accuracy: {metrics['small_acc']:.4f}")
            if metrics['noisy_acc'] is not None:
                print(f"  Noisy Accuracy: {metrics['noisy_acc']:.4f}")
            print(f"  Small samples: {metrics['small_samples']}")
            print(f"  Noisy samples: {metrics['noisy_samples']}")
        
        return results
    
    def update_summary_files(self, evaluation_results):
        """summary.json 파일들을 업데이트"""
        
        for model_name, metrics in evaluation_results.items():
            summary_path = f"experiment_v3_results/{model_name}/summary.json"
            
            if os.path.exists(summary_path):
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                
                # final_metrics 섹션 업데이트
                if 'final_metrics' not in summary:
                    summary['final_metrics'] = {}
                
                summary['final_metrics'].update({
                    'overall_acc': metrics['overall_acc'],
                    'small_acc': metrics['small_acc'],
                    'noisy_acc': metrics['noisy_acc'],
                    'total_samples': metrics['total_samples'],
                    'small_samples': metrics['small_samples'],
                    'noisy_samples': metrics['noisy_samples']
                })
                
                # 백업 후 저장
                backup_path = f"{summary_path}.backup"
                os.rename(summary_path, backup_path)
                
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                
                print(f"Updated {summary_path}")
        
        # 전체 결과를 별도 파일로 저장
        with open('evaluation_results_small_noisy.json', 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        print("Evaluation results saved to evaluation_results_small_noisy.json")


def main():
    evaluator = ModelEvaluator()
    
    print("Starting evaluation of all models on small/noisy subsets...")
    results = evaluator.evaluate_all_models()
    
    if results:
        print("\nUpdating summary files...")
        evaluator.update_summary_files(results)
        print("\nEvaluation completed!")
    else:
        print("No models were successfully evaluated.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
experiment_v3_results의 4개 모델에 대한 포괄적인 시각화 및 비교 분석
30 에폭 기준으로 다양한 지표와 시각화를 생성
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class ExperimentV3Analyzer:
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            base_dir = "experiment_v3_results"  # 현재 디렉토리 기준
        self.base_dir = Path(base_dir)
        self.models = ['resnet20', 'convnext', 'swin', 'model_resnet32x4_wide']
        self.model_display_names = {
            'resnet20': 'ResNet-20',
            'convnext': 'ConvNeXt',
            'swin': 'Swin Transformer', 
            'model_resnet32x4_wide': 'ResNet-32x4-Wide'
        }
        self.output_dir = Path("../../visualizations/experiment_v3_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_model_data(self) -> Dict[str, Dict]:
        """모든 모델의 데이터를 로드"""
        model_data = {}
        
        for model in self.models:
            model_path = self.base_dir / model / "summary.json"
            if model_path.exists():
                with open(model_path, 'r') as f:
                    data = json.load(f)
                model_data[model] = data
                print(f"Loaded data for {model}")
            else:
                print(f"Warning: {model_path} not found")
                
        return model_data
    
    def extract_30_epoch_metrics(self, model_data: Dict[str, Dict]) -> Dict[str, Dict]:
        """30 에폭까지의 메트릭 추출"""
        epoch_30_data = {}
        
        for model_name, data in model_data.items():
            if 'history' in data:
                history = data['history'][:30]  # 처음 30 에폭만 사용
                
                # 30 에폭에서의 메트릭
                epoch_30_metrics = {
                    'train_acc_30': history[29]['train_acc'] if len(history) >= 30 else history[-1]['train_acc'],
                    'val_acc_30': history[29]['val_acc'] if len(history) >= 30 else history[-1]['val_acc'],
                    'train_loss_30': history[29]['train_loss'] if len(history) >= 30 else history[-1]['train_loss'],
                    'best_val_acc_30': max([h['val_acc'] for h in history]),
                    'convergence_speed': self.calculate_convergence_speed(history),
                    'stability': self.calculate_stability(history),
                    'history': history
                }
                
                # Small/Noisy 성능이 있으면 추가
                if 'final_metrics' in data:
                    fm = data['final_metrics']
                    epoch_30_metrics.update({
                        'small_acc': fm.get('small_acc'),
                        'noisy_acc': fm.get('noisy_acc'),
                        'overall_acc': fm.get('overall_acc'),
                        'small_samples': fm.get('small_samples', 0),
                        'noisy_samples': fm.get('noisy_samples', 0),
                        'total_samples': fm.get('total_samples', 0)
                    })
                
                epoch_30_data[model_name] = epoch_30_metrics
                
        return epoch_30_data
    
    def calculate_convergence_speed(self, history: List[Dict]) -> float:
        """수렴 속도 계산 (95% 최종 성능에 도달하는 에폭)"""
        val_accs = [h['val_acc'] for h in history]
        final_acc = val_accs[-1]
        target_acc = final_acc * 0.95
        
        for i, acc in enumerate(val_accs):
            if acc >= target_acc:
                return i + 1
        return len(val_accs)
    
    def calculate_stability(self, history: List[Dict]) -> float:
        """안정성 계산 (마지막 10 에폭의 표준편차)"""
        val_accs = [h['val_acc'] for h in history]
        if len(val_accs) >= 10:
            last_10 = val_accs[-10:]
            return float(np.std(last_10))
        return float(np.std(val_accs))
    
    def create_comprehensive_comparison_table(self, epoch_30_data: Dict[str, Dict]) -> pd.DataFrame:
        """포괄적인 비교 테이블 생성"""
        
        metrics_data = []
        
        for model_name, metrics in epoch_30_data.items():
            display_name = self.model_display_names.get(model_name, model_name)
            
            row = {
                'Model': display_name,
                '30-Epoch Val Acc': f"{metrics['val_acc_30']:.4f}",
                'Best Val Acc (30ep)': f"{metrics['best_val_acc_30']:.4f}",
                '30-Epoch Train Acc': f"{metrics['train_acc_30']:.4f}",
                '30-Epoch Train Loss': f"{metrics['train_loss_30']:.4f}",
                'Convergence Speed (epochs)': int(metrics['convergence_speed']),
                'Stability (std)': f"{metrics['stability']:.4f}",
                'Overfitting Gap': f"{metrics['train_acc_30'] - metrics['val_acc_30']:.4f}"
            }
            
            # Small/Noisy 메트릭 추가 (있는 경우)
            if metrics.get('small_acc') is not None:
                row.update({
                    'Small Digit Acc': f"{metrics['small_acc']:.4f}",
                    'Noisy Acc': f"{metrics['noisy_acc']:.4f}",
                    'Small Samples': metrics['small_samples'],
                    'Noisy Samples': metrics['noisy_samples']
                })
            
            metrics_data.append(row)
        
        df = pd.DataFrame(metrics_data)
        
        # 테이블을 CSV와 HTML로 저장
        df.to_csv(self.output_dir / "model_comparison_30epochs.csv", index=False)
        
        # HTML 테이블 생성 (더 예쁘게)
        html_table = df.to_html(index=False, classes='table table-striped', 
                               table_id='comparison-table', escape=False)
        
        with open(self.output_dir / "model_comparison_30epochs.html", 'w') as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Model Comparison - 30 Epochs</title>
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
                <style>
                    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; }}
                    .table {{ margin-top: 20px; }}
                    .table th {{ background-color: #f8f9fa; }}
                    .title {{ color: #2c3e50; margin-bottom: 30px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1 class="title">Model Performance Comparison - 30 Epochs Analysis</h1>
                    <p class="text-muted">Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    {html_table}
                </div>
            </body>
            </html>
            """)
        
        return df
    
    def plot_training_curves(self, epoch_30_data: Dict[str, Dict]):
        """훈련 곡선 시각화"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 색상 팔레트
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        for i, (model_name, metrics) in enumerate(epoch_30_data.items()):
            display_name = self.model_display_names.get(model_name, model_name)
            color = colors[i % len(colors)]
            history = metrics['history']
            
            epochs = [h['epoch'] for h in history]
            train_accs = [h['train_acc'] for h in history]
            val_accs = [h['val_acc'] for h in history]
            train_losses = [h['train_loss'] for h in history]
            
            # Train vs Validation Accuracy
            axes[0, 0].plot(epochs, train_accs, '--', color=color, label=f'{display_name} (Train)', alpha=0.8)
            axes[0, 0].plot(epochs, val_accs, '-', color=color, label=f'{display_name} (Val)', linewidth=2)
            
            # Training Loss
            axes[0, 1].plot(epochs, train_losses, color=color, label=display_name, linewidth=2)
            
            # Accuracy Gap (Overfitting)
            acc_gaps = [t - v for t, v in zip(train_accs, val_accs)]
            axes[1, 0].plot(epochs, acc_gaps, color=color, label=display_name, linewidth=2)
            
            # Validation Accuracy (확대)
            axes[1, 1].plot(epochs, val_accs, color=color, label=display_name, linewidth=2, marker='o', markersize=3)
        
        # 서브플롯 설정
        axes[0, 0].set_title('Training vs Validation Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Training Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('Overfitting Gap (Train - Val Accuracy)', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy Gap')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        axes[1, 1].set_title('Validation Accuracy (Detailed)', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Validation Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_curves_30epochs.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / "training_curves_30epochs.pdf", bbox_inches='tight')
        plt.close()
    
    def plot_performance_radar(self, epoch_30_data: Dict[str, Dict]):
        """레이더 차트로 다차원 성능 비교"""
        
        # 메트릭 정규화를 위한 함수
        def normalize_metric(values, reverse=False):
            min_val, max_val = min(values), max(values)
            if max_val == min_val:
                return [1.0] * len(values)
            if reverse:  # 낮을수록 좋은 메트릭 (loss, convergence_speed, stability)
                return [(max_val - v) / (max_val - min_val) for v in values]
            else:  # 높을수록 좋은 메트릭 (accuracy)
                return [(v - min_val) / (max_val - min_val) for v in values]
        
        # 메트릭 추출
        models = list(epoch_30_data.keys())
        display_names = [self.model_display_names.get(m, m) for m in models]
        
        val_accs = [epoch_30_data[m]['val_acc_30'] for m in models]
        train_losses = [epoch_30_data[m]['train_loss_30'] for m in models]
        conv_speeds = [epoch_30_data[m]['convergence_speed'] for m in models]
        stabilities = [epoch_30_data[m]['stability'] for m in models]
        
        # 정규화
        norm_val_accs = normalize_metric(val_accs)
        norm_losses = normalize_metric(train_losses, reverse=True)
        norm_conv_speeds = normalize_metric(conv_speeds, reverse=True)
        norm_stabilities = normalize_metric(stabilities, reverse=True)
        
        # 레이더 차트
        categories = ['Validation\nAccuracy', 'Low Training\nLoss', 'Fast\nConvergence', 'High\nStability']
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # 원형으로 만들기
        
        for i, model in enumerate(models):
            values = [
                norm_val_accs[i],
                norm_losses[i], 
                norm_conv_speeds[i],
                norm_stabilities[i]
            ]
            values += values[:1]  # 원형으로 만들기
            
            display_name = self.model_display_names.get(model, model)
            ax.plot(angles, values, 'o-', linewidth=2, label=display_name, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Radar Chart (30 Epochs)', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_radar_30epochs.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / "performance_radar_30epochs.pdf", bbox_inches='tight')
        plt.close()
    
    def plot_small_noisy_performance(self, epoch_30_data: Dict[str, Dict]):
        """Small/Noisy 샘플 성능 상세 분석"""
        
        # Small/Noisy 데이터가 있는 모델 필터링
        models_with_small_noisy = {k: v for k, v in epoch_30_data.items() 
                                  if v.get('small_acc') is not None}
        
        if not models_with_small_noisy:
            print("No small/noisy performance data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        models = list(models_with_small_noisy.keys())
        display_names = [self.model_display_names.get(m, m) for m in models]
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'][:len(models)]
        
        overall_accs = [models_with_small_noisy[m]['overall_acc'] for m in models]
        small_accs = [models_with_small_noisy[m]['small_acc'] for m in models]
        noisy_accs = [models_with_small_noisy[m]['noisy_acc'] for m in models]
        
        # 1. 전체 vs Small vs Noisy 정확도 비교
        x = np.arange(len(models))
        width = 0.25
        
        axes[0, 0].bar(x - width, overall_accs, width, label='Overall', color='#2E86AB', alpha=0.8)
        axes[0, 0].bar(x, small_accs, width, label='Small Digits', color='#A23B72', alpha=0.8)
        axes[0, 0].bar(x + width, noisy_accs, width, label='Noisy', color='#F18F01', alpha=0.8)
        
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Overall vs Small vs Noisy Accuracy', fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(display_names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 정확도 값 표시
        for i, (overall, small, noisy) in enumerate(zip(overall_accs, small_accs, noisy_accs)):
            axes[0, 0].text(i - width, overall + 0.001, f'{overall:.3f}', ha='center', va='bottom', fontsize=8)
            axes[0, 0].text(i, small + 0.001, f'{small:.3f}', ha='center', va='bottom', fontsize=8)
            axes[0, 0].text(i + width, noisy + 0.001, f'{noisy:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. 성능 드롭 분석
        small_drops = [(overall - small) * 100 for overall, small in zip(overall_accs, small_accs)]
        noisy_drops = [(overall - noisy) * 100 for overall, noisy in zip(overall_accs, noisy_accs)]
        
        axes[0, 1].bar(x - width/2, small_drops, width, label='Small Digit Drop', color='#A23B72', alpha=0.8)
        axes[0, 1].bar(x + width/2, noisy_drops, width, label='Noisy Drop', color='#F18F01', alpha=0.8)
        
        axes[0, 1].set_xlabel('Models')
        axes[0, 1].set_ylabel('Performance Drop (%)')
        axes[0, 1].set_title('Performance Drop from Overall Accuracy', fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(display_names, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 샘플 수 분포
        small_samples = [models_with_small_noisy[m]['small_samples'] for m in models]
        noisy_samples = [models_with_small_noisy[m]['noisy_samples'] for m in models]
        total_samples = [models_with_small_noisy[m]['total_samples'] for m in models]
        
        small_ratios = [s/t * 100 for s, t in zip(small_samples, total_samples)]
        noisy_ratios = [n/t * 100 for n, t in zip(noisy_samples, total_samples)]
        
        axes[1, 0].bar(display_names, small_ratios, label='Small Digits', color='#A23B72', alpha=0.8)
        axes[1, 0].bar(display_names, noisy_ratios, bottom=small_ratios, label='Noisy', color='#F18F01', alpha=0.8)
        
        axes[1, 0].set_xlabel('Models')
        axes[1, 0].set_ylabel('Percentage of Total Samples (%)')
        axes[1, 0].set_title('Small/Noisy Sample Distribution', fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)
        
        # 4. 정확도 vs 성능 드롭 산점도
        axes[1, 1].scatter(overall_accs, small_drops, s=100, color='#A23B72', alpha=0.7, label='Small Digit Drop')
        axes[1, 1].scatter(overall_accs, noisy_drops, s=100, color='#F18F01', alpha=0.7, label='Noisy Drop')
        
        for i, name in enumerate(display_names):
            axes[1, 1].annotate(name, (overall_accs[i], small_drops[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        axes[1, 1].set_xlabel('Overall Accuracy')
        axes[1, 1].set_ylabel('Performance Drop (%)')
        axes[1, 1].set_title('Overall Accuracy vs Performance Drop', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "small_noisy_analysis_30epochs.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / "small_noisy_analysis_30epochs.pdf", bbox_inches='tight')
        plt.close()
    
    def create_convergence_analysis(self, epoch_30_data: Dict[str, Dict]):
        """수렴 속도 및 안정성 상세 분석"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        models = list(epoch_30_data.keys())
        display_names = [self.model_display_names.get(m, m) for m in models]
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        # 1. 수렴 속도 비교
        conv_speeds = [epoch_30_data[m]['convergence_speed'] for m in models]
        
        bars = axes[0, 0].bar(display_names, conv_speeds, color=colors[:len(models)], alpha=0.8)
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Epochs to 95% Performance')
        axes[0, 0].set_title('Convergence Speed Comparison', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45)
        
        # 값 표시
        for bar, speed in zip(bars, conv_speeds):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                          f'{int(speed)}', ha='center', va='bottom', fontweight='bold')
        
        # 2. 안정성 비교 (낮을수록 좋음)
        stabilities = [epoch_30_data[m]['stability'] for m in models]
        
        bars = axes[0, 1].bar(display_names, stabilities, color=colors[:len(models)], alpha=0.8)
        axes[0, 1].set_xlabel('Models')
        axes[0, 1].set_ylabel('Stability (Lower is Better)')
        axes[0, 1].set_title('Training Stability Comparison', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)
        
        # 값 표시
        for bar, stability in zip(bars, stabilities):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                          f'{stability:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 3. 마지막 10 에폭의 정확도 변화
        for i, (model_name, metrics) in enumerate(epoch_30_data.items()):
            history = metrics['history']
            if len(history) >= 10:
                last_10_epochs = list(range(len(history)-9, len(history)+1))
                last_10_accs = [h['val_acc'] for h in history[-10:]]
                
                display_name = self.model_display_names.get(model_name, model_name)
                axes[1, 0].plot(last_10_epochs, last_10_accs, 'o-', 
                              color=colors[i], label=display_name, linewidth=2, markersize=5)
        
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Validation Accuracy')
        axes[1, 0].set_title('Last 10 Epochs - Stability Analysis', fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 학습률 효율성 (정확도 / 에폭 수)
        val_accs_30 = [epoch_30_data[m]['val_acc_30'] for m in models]
        efficiency = [acc / 30 for acc in val_accs_30]  # 30 에폭 기준
        
        bars = axes[1, 1].bar(display_names, efficiency, color=colors[:len(models)], alpha=0.8)
        axes[1, 1].set_xlabel('Models')
        axes[1, 1].set_ylabel('Learning Efficiency (Acc/Epoch)')
        axes[1, 1].set_title('Learning Efficiency (30 Epochs)', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)
        
        # 값 표시
        for bar, eff in zip(bars, efficiency):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                          f'{eff:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "convergence_analysis_30epochs.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / "convergence_analysis_30epochs.pdf", bbox_inches='tight')
        plt.close()
    
    def generate_detailed_report(self, epoch_30_data: Dict[str, Dict], comparison_df: pd.DataFrame):
        """상세 텍스트 리포트 생성"""
        
        report_path = self.output_dir / "detailed_analysis_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Experiment V3 Results - Comprehensive Analysis (30 Epochs)\n\n")
            f.write(f"**Generated on:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            
            # 최고 성능 모델 찾기
            best_val_model = max(epoch_30_data.items(), key=lambda x: x[1]['val_acc_30'])
            fastest_conv_model = min(epoch_30_data.items(), key=lambda x: x[1]['convergence_speed'])
            most_stable_model = min(epoch_30_data.items(), key=lambda x: x[1]['stability'])
            
            f.write(f"- **Best 30-Epoch Validation Accuracy:** {self.model_display_names[best_val_model[0]]} ({best_val_model[1]['val_acc_30']:.4f})\n")
            f.write(f"- **Fastest Convergence:** {self.model_display_names[fastest_conv_model[0]]} ({fastest_conv_model[1]['convergence_speed']:.1f} epochs)\n")
            f.write(f"- **Most Stable Training:** {self.model_display_names[most_stable_model[0]]} (std: {most_stable_model[1]['stability']:.4f})\n\n")
            
            f.write("## Detailed Model Analysis\n\n")
            
            for model_name, metrics in epoch_30_data.items():
                display_name = self.model_display_names.get(model_name, model_name)
                f.write(f"### {display_name}\n\n")
                
                f.write(f"**Performance Metrics (30 Epochs):**\n")
                f.write(f"- Validation Accuracy: {metrics['val_acc_30']:.4f}\n")
                f.write(f"- Training Accuracy: {metrics['train_acc_30']:.4f}\n")
                f.write(f"- Training Loss: {metrics['train_loss_30']:.4f}\n")
                f.write(f"- Best Validation Accuracy: {metrics['best_val_acc_30']:.4f}\n")
                f.write(f"- Overfitting Gap: {metrics['train_acc_30'] - metrics['val_acc_30']:.4f}\n\n")
                
                f.write(f"**Training Characteristics:**\n")
                f.write(f"- Convergence Speed: {metrics['convergence_speed']:.1f} epochs to 95% performance\n")
                f.write(f"- Training Stability: {metrics['stability']:.4f} (std of last 10 epochs)\n\n")
                
                # Small/Noisy 성능 (있는 경우)
                if metrics.get('small_acc') is not None:
                    f.write(f"**Challenging Sample Performance:**\n")
                    f.write(f"- Small Digit Accuracy: {metrics['small_acc']:.4f}\n")
                    f.write(f"- Noisy Sample Accuracy: {metrics['noisy_acc']:.4f}\n")
                    f.write(f"- Small Digit Performance Drop: {(metrics['overall_acc'] - metrics['small_acc']) * 100:.2f}%\n")
                    f.write(f"- Noisy Sample Performance Drop: {(metrics['overall_acc'] - metrics['noisy_acc']) * 100:.2f}%\n\n")
                
                f.write("---\n\n")
            
            f.write("## Key Insights\n\n")
            
            # 성능 순위
            sorted_models = sorted(epoch_30_data.items(), key=lambda x: x[1]['val_acc_30'], reverse=True)
            f.write("**Performance Ranking (30-Epoch Val Accuracy):**\n")
            for i, (model_name, metrics) in enumerate(sorted_models, 1):
                display_name = self.model_display_names.get(model_name, model_name)
                f.write(f"{i}. {display_name}: {metrics['val_acc_30']:.4f}\n")
            f.write("\n")
            
            # 수렴 속도 순위
            sorted_conv = sorted(epoch_30_data.items(), key=lambda x: x[1]['convergence_speed'])
            f.write("**Convergence Speed Ranking:**\n")
            for i, (model_name, metrics) in enumerate(sorted_conv, 1):
                display_name = self.model_display_names.get(model_name, model_name)
                f.write(f"{i}. {display_name}: {metrics['convergence_speed']:.1f} epochs\n")
            f.write("\n")
            
            # 안정성 순위
            sorted_stability = sorted(epoch_30_data.items(), key=lambda x: x[1]['stability'])
            f.write("**Training Stability Ranking:**\n")
            for i, (model_name, metrics) in enumerate(sorted_stability, 1):
                display_name = self.model_display_names.get(model_name, model_name)
                f.write(f"{i}. {display_name}: {metrics['stability']:.4f} (lower is better)\n")
            f.write("\n")
            
            f.write("## Recommendations\n\n")
            f.write("Based on the 30-epoch analysis:\n\n")
            f.write(f"1. **For Best Accuracy:** Use {self.model_display_names[best_val_model[0]]} - achieved highest validation accuracy\n")
            f.write(f"2. **For Fast Training:** Use {self.model_display_names[fastest_conv_model[0]]} - converges fastest to target performance\n")
            f.write(f"3. **For Stable Training:** Use {self.model_display_names[most_stable_model[0]]} - most consistent training behavior\n\n")
            
            # Small/Noisy 성능이 있는 경우 추가 권장사항
            models_with_small = {k: v for k, v in epoch_30_data.items() if v.get('small_acc') is not None}
            if models_with_small:
                best_small = max(models_with_small.items(), key=lambda x: x[1]['small_acc'])
                f.write(f"4. **For Small Digit Performance:** Use {self.model_display_names[best_small[0]]} - best performance on challenging small digits\n\n")
        
        print(f"Detailed report saved to: {report_path}")
    
    def run_complete_analysis(self):
        """전체 분석 실행"""
        print("Starting comprehensive analysis of experiment_v3_results...")
        
        # 데이터 로드
        model_data = self.load_model_data()
        if not model_data:
            print("No model data found!")
            return
        
        # 30 에폭 메트릭 추출
        epoch_30_data = self.extract_30_epoch_metrics(model_data)
        
        print(f"Analyzing {len(epoch_30_data)} models...")
        
        # 1. 비교 테이블 생성
        print("Creating comparison table...")
        comparison_df = self.create_comprehensive_comparison_table(epoch_30_data)
        
        # 2. 훈련 곡선 시각화
        print("Plotting training curves...")
        self.plot_training_curves(epoch_30_data)
        
        # 3. 레이더 차트
        print("Creating performance radar chart...")
        self.plot_performance_radar(epoch_30_data)
        
        # 4. Small/Noisy 성능 분석
        print("Analyzing small/noisy performance...")
        self.plot_small_noisy_performance(epoch_30_data)
        
        # 5. 수렴 분석
        print("Creating convergence analysis...")
        self.create_convergence_analysis(epoch_30_data)
        
        # 6. 상세 리포트
        print("Generating detailed report...")
        self.generate_detailed_report(epoch_30_data, comparison_df)
        
        print(f"\n✅ Complete analysis finished!")
        print(f"📁 All outputs saved to: {self.output_dir}")
        print(f"📊 Generated files:")
        for file in sorted(self.output_dir.glob("*")):
            print(f"   - {file.name}")


def main():
    analyzer = ExperimentV3Analyzer()
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
종합적인 실험 결과 시각화 도구
- 성능 비교 차트
- 학습 곡선 분석
- 작은 글자 성능 분석
- 오류 분석 및 시각화
- HTML 보고서 생성
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from collections import defaultdict, Counter
import base64
from io import BytesIO

# 스타일 설정
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


class ExperimentVisualizer:
    def __init__(self, experiments_dir: str, output_dir: str):
        self.experiments_dir = Path(experiments_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 실험 데이터 로드
        self.experiments = self._load_experiments()
        
        # 색상 팔레트
        self.colors = sns.color_palette("husl", len(self.experiments))
        self.color_map = {name: color for name, color in zip(self.experiments.keys(), self.colors)}
        
    def _load_experiments(self) -> Dict[str, Dict]:
        """실험 결과 로드"""
        experiments = {}
        
        for exp_dir in self.experiments_dir.iterdir():
            if exp_dir.is_dir():
                summary_path = exp_dir / 'summary.json'
                if summary_path.exists():
                    try:
                        with open(summary_path, 'r') as f:
                            data = json.load(f)
                        experiments[exp_dir.name] = data
                        print(f"Loaded experiment: {exp_dir.name}")
                    except Exception as e:
                        print(f"Error loading {summary_path}: {e}")
        
        return experiments
    
    def plot_performance_comparison(self) -> str:
        """실험별 성능 비교 차트"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        exp_names = []
        best_val_accs = []
        final_overall_accs = []
        small_accs = []
        noisy_accs = []
        
        for name, data in self.experiments.items():
            exp_names.append(name.replace('_', '\n'))
            best_val_accs.append(data['best_val_acc'])
            final_overall_accs.append(data['final_metrics']['overall_acc'])
            
            small_acc = data['final_metrics'].get('small_acc')
            noisy_acc = data['final_metrics'].get('noisy_acc')
            small_accs.append(small_acc if small_acc is not None else 0)
            noisy_accs.append(noisy_acc if noisy_acc is not None else 0)
        
        # Best Validation Accuracy
        bars1 = axes[0, 0].bar(exp_names, best_val_accs, color=self.colors)
        axes[0, 0].set_title('Best Validation Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0.9, 1.0)
        for i, v in enumerate(best_val_accs):
            axes[0, 0].text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Final Overall Accuracy
        bars2 = axes[0, 1].bar(exp_names, final_overall_accs, color=self.colors)
        axes[0, 1].set_title('Final Overall Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_ylim(0.9, 1.0)
        for i, v in enumerate(final_overall_accs):
            axes[0, 1].text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Small Digit Accuracy
        valid_small = [(name, acc) for name, acc in zip(exp_names, small_accs) if acc > 0]
        if valid_small:
            names, accs = zip(*valid_small)
            valid_colors = [self.colors[i] for i, acc in enumerate(small_accs) if acc > 0]
            bars3 = axes[1, 0].bar(names, accs, color=valid_colors)
            axes[1, 0].set_title('Small Digit Accuracy', fontsize=14, fontweight='bold')
            axes[1, 0].set_ylabel('Accuracy')
            for i, v in enumerate(accs):
                axes[1, 0].text(i, v + 0.005, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
        else:
            axes[1, 0].text(0.5, 0.5, 'No small digit data available', 
                          ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=12)
            axes[1, 0].set_title('Small Digit Accuracy', fontsize=14, fontweight='bold')
        
        # Training Configuration Comparison
        configs = []
        for name, data in self.experiments.items():
            config = data.get('config', {})
            configs.append({
                'Experiment': name.replace('_', '\n'),
                'LR': config.get('lr', 'N/A'),
                'Batch Size': config.get('batch_size', 'N/A'),
                'Epochs': config.get('epochs', 'N/A')
            })
        
        axes[1, 1].axis('off')
        config_text = "Training Configurations:\n\n"
        for i, conf in enumerate(configs):
            config_text += f"{conf['Experiment']}: LR={conf['LR']}, BS={conf['Batch Size']}, E={conf['Epochs']}\n"
        axes[1, 1].text(0.1, 0.9, config_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, va='top', ha='left', fontfamily='monospace')
        
        plt.tight_layout()
        save_path = self.output_dir / 'performance_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_training_curves(self) -> str:
        """학습 곡선 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        for name, data in self.experiments.items():
            history = data.get('history', [])
            if not history:
                continue
                
            epochs = [h['epoch'] for h in history]
            train_losses = [h['train_loss'] for h in history]
            train_accs = [h['train_acc'] for h in history]
            val_accs = [h['val_acc'] for h in history]
            
            color = self.color_map[name]
            
            # Training Loss
            axes[0, 0].plot(epochs, train_losses, label=name.replace('_', ' '), 
                          color=color, linewidth=2, marker='o', markersize=3)
            
            # Training Accuracy
            axes[0, 1].plot(epochs, train_accs, label=name.replace('_', ' '), 
                          color=color, linewidth=2, marker='o', markersize=3)
            
            # Validation Accuracy
            axes[1, 0].plot(epochs, val_accs, label=name.replace('_', ' '), 
                          color=color, linewidth=2, marker='o', markersize=3)
        
        # Styling
        axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Training Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate vs Performance
        axes[1, 1].axis('off')
        summary_text = "Key Insights:\n\n"
        
        for name, data in self.experiments.items():
            config = data.get('config', {})
            best_acc = data['best_val_acc']
            lr = config.get('lr', 'N/A')
            summary_text += f"{name}:\n"
            summary_text += f"  Best Val Acc: {best_acc:.4f}\n"
            summary_text += f"  Learning Rate: {lr}\n"
            summary_text += f"  Final Train Acc: {data.get('history', [{}])[-1].get('train_acc', 'N/A'):.4f}\n\n"
        
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes, 
                       fontsize=10, va='top', ha='left', fontfamily='monospace')
        
        plt.tight_layout()
        save_path = self.output_dir / 'training_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_error_analysis(self) -> str:
        """오류 분석 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 실험별 오류 수
        exp_names = []
        error_counts = []
        
        for name, data in self.experiments.items():
            errors = data['final_metrics'].get('error_examples', [])
            exp_names.append(name.replace('_', '\n'))
            error_counts.append(len(errors))
        
        bars = axes[0, 0].bar(exp_names, error_counts, color=self.colors)
        axes[0, 0].set_title('Number of Classification Errors', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Error Count')
        for i, v in enumerate(error_counts):
            axes[0, 0].text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
        
        # 전체 실험의 클래스별 오류 분포
        all_errors = []
        for data in self.experiments.values():
            errors = data['final_metrics'].get('error_examples', [])
            for error in errors:
                all_errors.append((error['target'], error['pred']))
        
        if all_errors:
            # Confusion matrix-style error distribution
            error_matrix = np.zeros((10, 10))
            for target, pred in all_errors:
                error_matrix[target][pred] += 1
            
            im = axes[0, 1].imshow(error_matrix, cmap='Reds', aspect='auto')
            axes[0, 1].set_title('Error Distribution (Target vs Predicted)', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Predicted Class')
            axes[0, 1].set_ylabel('True Class')
            axes[0, 1].set_xticks(range(10))
            axes[0, 1].set_yticks(range(10))
            
            # 값 표시
            for i in range(10):
                for j in range(10):
                    if error_matrix[i][j] > 0:
                        axes[0, 1].text(j, i, f'{int(error_matrix[i][j])}', 
                                       ha='center', va='center', fontweight='bold')
            
            plt.colorbar(im, ax=axes[0, 1])
        
        # 가장 많이 틀린 클래스
        target_errors = Counter([target for target, _ in all_errors])
        pred_errors = Counter([pred for _, pred in all_errors])
        
        if target_errors:
            classes = list(range(10))
            true_counts = [target_errors.get(i, 0) for i in classes]
            pred_counts = [pred_errors.get(i, 0) for i in classes]
            
            x = np.arange(len(classes))
            width = 0.35
            
            bars1 = axes[1, 0].bar(x - width/2, true_counts, width, label='Actual Class Errors', alpha=0.8)
            bars2 = axes[1, 0].bar(x + width/2, pred_counts, width, label='Predicted Class Errors', alpha=0.8)
            
            axes[1, 0].set_title('Error Distribution by Class', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Digit Class')
            axes[1, 0].set_ylabel('Error Count')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(classes)
            axes[1, 0].legend()
        
        # 실험별 성능 요약
        axes[1, 1].axis('off')
        summary_text = "Error Analysis Summary:\n\n"
        
        total_samples = sum(data['final_metrics']['total_samples'] for data in self.experiments.values()) // len(self.experiments)
        total_errors = sum(error_counts)
        
        summary_text += f"Total Samples (avg): {total_samples}\n"
        summary_text += f"Total Errors: {total_errors}\n"
        summary_text += f"Overall Error Rate: {total_errors/total_samples*100:.2f}%\n\n"
        
        if all_errors:
            most_confused = Counter([(t, p) for t, p in all_errors]).most_common(5)
            summary_text += "Most Common Confusions:\n"
            for (true_class, pred_class), count in most_confused:
                summary_text += f"  {true_class} → {pred_class}: {count} times\n"
        
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes, 
                       fontsize=11, va='top', ha='left', fontfamily='monospace')
        
        plt.tight_layout()
        save_path = self.output_dir / 'error_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def create_detailed_comparison_table(self) -> str:
        """상세한 비교 테이블 생성"""
        data = []
        
        for name, exp_data in self.experiments.items():
            config = exp_data.get('config', {})
            metrics = exp_data['final_metrics']
            
            row = {
                'Experiment': name.replace('_', ' ').title(),
                'Best Val Acc': f"{exp_data['best_val_acc']:.4f}",
                'Final Overall Acc': f"{metrics['overall_acc']:.4f}",
                'Small Acc': f"{metrics.get('small_acc', 0):.4f}" if metrics.get('small_acc') else 'N/A',
                'Noisy Acc': f"{metrics.get('noisy_acc', 0):.4f}" if metrics.get('noisy_acc') else 'N/A',
                'Total Samples': metrics['total_samples'],
                'Error Count': len(metrics.get('error_examples', [])),
                'Error Rate': f"{len(metrics.get('error_examples', [])) / metrics['total_samples'] * 100:.2f}%",
                'Learning Rate': config.get('lr', 'N/A'),
                'Batch Size': config.get('batch_size', 'N/A'),
                'Epochs': config.get('epochs', 'N/A'),
                'Optimizer': config.get('optimizer', 'N/A'),
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # HTML 테이블로 저장
        html_path = self.output_dir / 'comparison_table.html'
        with open(html_path, 'w') as f:
            f.write('<html><head><style>')
            f.write('table { border-collapse: collapse; width: 100%; margin: 20px 0; }')
            f.write('th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }')
            f.write('th { background-color: #f2f2f2; font-weight: bold; }')
            f.write('tr:nth-child(even) { background-color: #f9f9f9; }')
            f.write('</style></head><body>')
            f.write('<h2>Experiment Comparison Table</h2>')
            f.write(df.to_html(index=False, escape=False))
            f.write('</body></html>')
        
        # CSV로도 저장
        csv_path = self.output_dir / 'comparison_table.csv'
        df.to_csv(csv_path, index=False)
        
        return str(html_path)
    
    def generate_comprehensive_report(self) -> str:
        """종합 보고서 HTML 생성"""
        print("Generating visualizations...")
        
        # 모든 시각화 생성
        perf_chart = self.plot_performance_comparison()
        training_curves = self.plot_training_curves()
        error_analysis = self.plot_error_analysis()
        comparison_table = self.create_detailed_comparison_table()
        
        # 이미지를 base64로 인코딩
        def img_to_base64(img_path):
            with open(img_path, 'rb') as f:
                return base64.b64encode(f.read()).decode()
        
        perf_b64 = img_to_base64(perf_chart)
        curves_b64 = img_to_base64(training_curves)
        error_b64 = img_to_base64(error_analysis)
        
        # HTML 보고서 생성
        html_content = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Digit Classification Experiments Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; line-height: 1.6; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #2c3e50; text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; border-left: 4px solid #3498db; padding-left: 15px; margin-top: 30px; }}
        .section {{ margin-bottom: 40px; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .img-container {{ text-align: center; margin: 20px 0; }}
        img {{ max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
        .summary {{ background-color: #ecf0f1; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .highlight {{ background-color: #fff3cd; padding: 15px; border-left: 4px solid #ffc107; margin: 15px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: center; }}
        th {{ background-color: #3498db; color: white; font-weight: bold; }}
        tr:nth-child(even) {{ background-color: #f8f9fa; }}
        .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #fff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔢 Digit Classification Experiments Report</h1>
        
        <div class="summary">
            <h2>📊 Executive Summary</h2>
            <p>이 보고서는 다양한 ResNet 아키텍처와 훈련 전략을 사용한 숫자 분류 실험 결과를 종합적으로 분석합니다.</p>
            <div>
                <div class="metric">
                    <strong>총 실험 수:</strong> {len(self.experiments)}
                </div>
                <div class="metric">
                    <strong>최고 성능:</strong> {max(exp['best_val_acc'] for exp in self.experiments.values()):.4f}
                </div>
                <div class="metric">
                    <strong>평균 성능:</strong> {np.mean([exp['best_val_acc'] for exp in self.experiments.values()]):.4f}
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>📈 성능 비교 분석</h2>
            <div class="img-container">
                <img src="data:image/png;base64,{perf_b64}" alt="Performance Comparison">
            </div>
            <div class="highlight">
                <strong>주요 발견사항:</strong> 각 실험의 validation accuracy, overall accuracy를 비교하여 
                가장 효과적인 모델 구성을 식별할 수 있습니다.
            </div>
        </div>
        
        <div class="section">
            <h2>📉 학습 곡선 분석</h2>
            <div class="img-container">
                <img src="data:image/png;base64,{curves_b64}" alt="Training Curves">
            </div>
            <div class="highlight">
                <strong>학습 패턴:</strong> 각 모델의 수렴 속도, 과적합 여부, 최적 에포크 수를 분석할 수 있습니다.
            </div>
        </div>
        
        <div class="section">
            <h2>❌ 오류 분석</h2>
            <div class="img-container">
                <img src="data:image/png;base64,{error_b64}" alt="Error Analysis">
            </div>
            <div class="highlight">
                <strong>오류 패턴:</strong> 어떤 숫자들이 자주 혼동되는지, 각 모델의 약점을 파악할 수 있습니다.
            </div>
        </div>
        
        <div class="section">
            <h2>📋 상세 비교표</h2>
"""
        
        # 비교 테이블 HTML 내용 추가
        with open(comparison_table, 'r') as f:
            table_content = f.read()
            # HTML 헤더/푸터 제거하고 테이블만 추출
            start = table_content.find('<table')
            end = table_content.find('</table>') + 8
            table_only = table_content[start:end]
            html_content += table_only
        
        html_content += """
        </div>
        
        <div class="section">
            <h2>💡 결론 및 권장사항</h2>
            <div class="summary">
"""
        
        # 결론 생성
        best_exp = max(self.experiments.items(), key=lambda x: x[1]['best_val_acc'])
        html_content += f"""
                <h3>🏆 최고 성능 모델</h3>
                <p><strong>{best_exp[0].replace('_', ' ').title()}</strong>이 {best_exp[1]['best_val_acc']:.4f}의 
                최고 validation accuracy를 달성했습니다.</p>
                
                <h3>📌 주요 통찰</h3>
                <ul>
"""
        
        # 실험별 특징 분석
        for name, data in self.experiments.items():
            error_rate = len(data['final_metrics'].get('error_examples', [])) / data['final_metrics']['total_samples'] * 100
            html_content += f"<li><strong>{name.replace('_', ' ').title()}</strong>: "
            html_content += f"Validation Accuracy {data['best_val_acc']:.4f}, "
            html_content += f"Error Rate {error_rate:.2f}%</li>"
        
        html_content += """
                </ul>
                
                <h3>🔮 향후 개선 방향</h3>
                <ul>
                    <li>작은 글자 인식 성능 향상을 위한 특화된 데이터 증강 기법 도입</li>
                    <li>앙상블 방법을 통한 전체적인 성능 개선</li>
                    <li>오류가 집중된 클래스에 대한 추가 데이터 수집 및 학습</li>
                </ul>
            </div>
        </div>
        
        <div style="text-align: center; margin-top: 40px; padding: 20px; background-color: #ecf0f1; border-radius: 8px;">
            <p><em>Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
        </div>
    </div>
</body>
</html>
        """
        
        # 보고서 저장
        report_path = self.output_dir / 'comprehensive_report.html'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Comprehensive report saved to: {report_path}")
        return str(report_path)


def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive experiment visualizations')
    parser.add_argument('--experiments_dir', type=str, default='logs/single_runs',
                       help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    print(f"Loading experiments from: {args.experiments_dir}")
    print(f"Output directory: {args.output_dir}")
    
    visualizer = ExperimentVisualizer(args.experiments_dir, args.output_dir)
    
    if not visualizer.experiments:
        print("No experiments found!")
        return
    
    print(f"Found {len(visualizer.experiments)} experiments:")
    for name in visualizer.experiments:
        print(f"  - {name}")
    
    # 종합 보고서 생성
    report_path = visualizer.generate_comprehensive_report()
    
    print(f"\n✅ All visualizations completed!")
    print(f"📊 Comprehensive report: {report_path}")
    print(f"📁 All files saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
작은 글자 성능 상세 분석 도구
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import pandas as pd
from typing import Dict, List, Any, Optional
import shutil


def analyze_small_digit_performance(experiments_dir: str, data_dir: str, output_dir: str):
    """작은 글자 성능 상세 분석"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 작은 글자 리스트 로드
    small_valid_path = Path(data_dir) / 'small_valid.txt'
    if not small_valid_path.exists():
        print(f"Small validation list not found: {small_valid_path}")
        return
    
    with open(small_valid_path, 'r') as f:
        small_samples = set(line.strip() for line in f if line.strip())
    
    print(f"Found {len(small_samples)} small validation samples")
    
    # 실험별 작은 글자 성능 분석
    results = {}
    
    for exp_dir in Path(experiments_dir).iterdir():
        if not exp_dir.is_dir():
            continue
            
        summary_path = exp_dir / 'summary.json'
        if not summary_path.exists():
            continue
            
        try:
            with open(summary_path, 'r') as f:
                data = json.load(f)
            
            error_examples = data['final_metrics'].get('error_examples', [])
            
            # 작은 글자에서의 오류 분석
            small_errors = [err for err in error_examples if err['path'] in small_samples]
            small_correct = len(small_samples) - len(small_errors)
            small_accuracy = small_correct / len(small_samples) if small_samples else 0
            
            # 전체 성능 대비 작은 글자 성능
            overall_accuracy = data['final_metrics']['overall_acc']
            performance_gap = overall_accuracy - small_accuracy
            
            results[exp_dir.name] = {
                'overall_acc': overall_accuracy,
                'small_acc': small_accuracy,
                'performance_gap': performance_gap,
                'small_errors': small_errors,
                'small_total': len(small_samples),
                'small_correct': small_correct
            }
            
            print(f"{exp_dir.name}: Small digit accuracy = {small_accuracy:.4f} "
                  f"(Gap: {performance_gap:.4f})")
            
        except Exception as e:
            print(f"Error processing {exp_dir.name}: {e}")
    
    if not results:
        print("No valid experiment results found")
        return
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    exp_names = list(results.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(exp_names)))
    
    # 1. 전체 vs 작은 글자 정확도 비교
    overall_accs = [results[name]['overall_acc'] for name in exp_names]
    small_accs = [results[name]['small_acc'] for name in exp_names]
    
    x = np.arange(len(exp_names))
    width = 0.35
    
    bars1 = axes[0, 0].bar(x - width/2, overall_accs, width, 
                          label='Overall Accuracy', alpha=0.8, color=colors)
    bars2 = axes[0, 0].bar(x + width/2, small_accs, width, 
                          label='Small Digit Accuracy', alpha=0.8, color=colors)
    
    axes[0, 0].set_title('Overall vs Small Digit Accuracy', fontweight='bold', fontsize=14)
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([name.replace('_', '\n') for name in exp_names])
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 값 표시
    for i, (overall, small) in enumerate(zip(overall_accs, small_accs)):
        axes[0, 0].text(i - width/2, overall + 0.005, f'{overall:.3f}', 
                       ha='center', va='bottom', fontweight='bold', fontsize=10)
        axes[0, 0].text(i + width/2, small + 0.005, f'{small:.3f}', 
                       ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 2. 성능 격차
    gaps = [results[name]['performance_gap'] for name in exp_names]
    bars = axes[0, 1].bar(exp_names, gaps, color=colors, alpha=0.8)
    axes[0, 1].set_title('Performance Gap (Overall - Small Digit)', fontweight='bold', fontsize=14)
    axes[0, 1].set_ylabel('Accuracy Gap')
    axes[0, 1].set_xticklabels([name.replace('_', '\n') for name in exp_names])
    
    for i, gap in enumerate(gaps):
        axes[0, 1].text(i, gap + 0.001, f'{gap:.3f}', 
                       ha='center', va='bottom', fontweight='bold')
    
    # 3. 작은 글자 오류 수
    error_counts = [len(results[name]['small_errors']) for name in exp_names]
    bars = axes[1, 0].bar(exp_names, error_counts, color=colors, alpha=0.8)
    axes[1, 0].set_title('Small Digit Classification Errors', fontweight='bold', fontsize=14)
    axes[1, 0].set_ylabel('Number of Errors')
    axes[1, 0].set_xticklabels([name.replace('_', '\n') for name in exp_names])
    
    for i, count in enumerate(error_counts):
        axes[1, 0].text(i, count + 0.5, str(count), 
                       ha='center', va='bottom', fontweight='bold')
    
    # 4. 요약 테이블
    axes[1, 1].axis('off')
    
    # 테이블 데이터 준비
    table_data = []
    for name in exp_names:
        res = results[name]
        table_data.append([
            name.replace('_', ' ').title(),
            f"{res['overall_acc']:.4f}",
            f"{res['small_acc']:.4f}",
            f"{res['performance_gap']:.4f}",
            f"{len(res['small_errors'])}/{res['small_total']}"
        ])
    
    table = axes[1, 1].table(cellText=table_data,
                           colLabels=['Experiment', 'Overall Acc', 'Small Acc', 'Gap', 'Errors'],
                           cellLoc='center',
                           loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # 헤더 스타일링
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.tight_layout()
    save_path = output_path / 'small_digit_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 상세 보고서 생성
    report_path = output_path / 'small_digit_report.html'
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Small Digit Performance Analysis</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
            .container {{ max-width: 1000px; margin: 0 auto; }}
            h1, h2 {{ color: #2c3e50; }}
            .highlight {{ background-color: #fff3cd; padding: 15px; border-left: 4px solid #ffc107; margin: 15px 0; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: center; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #f8f9fa; border-radius: 8px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 Small Digit Performance Analysis</h1>
            
            <div class="highlight">
                <strong>분석 개요:</strong> 작은 크기 숫자 이미지({len(small_samples)}개 샘플)에 대한 
                각 모델의 성능을 상세 분석했습니다.
            </div>
            
            <h2>📊 핵심 메트릭</h2>
            <div>
    """
    
    best_small = max(results.items(), key=lambda x: x[1]['small_acc'])
    worst_gap = max(results.items(), key=lambda x: x[1]['performance_gap'])
    
    html_content += f"""
                <div class="metric">
                    <strong>최고 작은글자 성능:</strong><br>
                    {best_small[0].replace('_', ' ').title()}<br>
                    {best_small[1]['small_acc']:.4f}
                </div>
                <div class="metric">
                    <strong>최대 성능 격차:</strong><br>
                    {worst_gap[0].replace('_', ' ').title()}<br>
                    {worst_gap[1]['performance_gap']:.4f}
                </div>
                <div class="metric">
                    <strong>평균 작은글자 성능:</strong><br>
                    {np.mean([res['small_acc'] for res in results.values()]):.4f}
                </div>
            </div>
            
            <h2>📈 상세 결과 테이블</h2>
            <table>
                <tr>
                    <th>실험명</th>
                    <th>전체 정확도</th>
                    <th>작은글자 정확도</th>
                    <th>성능 격차</th>
                    <th>오류 수/전체</th>
                    <th>오류율</th>
                </tr>
    """
    
    for name, res in results.items():
        error_rate = len(res['small_errors']) / res['small_total'] * 100
        html_content += f"""
                <tr>
                    <td>{name.replace('_', ' ').title()}</td>
                    <td>{res['overall_acc']:.4f}</td>
                    <td>{res['small_acc']:.4f}</td>
                    <td>{res['performance_gap']:.4f}</td>
                    <td>{len(res['small_errors'])}/{res['small_total']}</td>
                    <td>{error_rate:.1f}%</td>
                </tr>
        """
    
    html_content += """
            </table>
            
            <h2>💡 분석 결과 및 권장사항</h2>
            <ul>
    """
    
    if best_small[1]['small_acc'] > 0.95:
        html_content += f"<li>✅ <strong>{best_small[0]}</strong> 모델이 작은 글자에서 우수한 성능({best_small[1]['small_acc']:.4f})을 보입니다.</li>"
    
    if worst_gap[1]['performance_gap'] > 0.05:
        html_content += f"<li>⚠️ <strong>{worst_gap[0]}</strong> 모델에서 작은 글자 성능 저하가 큽니다. 추가 개선이 필요합니다.</li>"
    
    html_content += """
                <li>📈 작은 글자 특화 데이터 증강 기법 도입 고려</li>
                <li>🔍 Focus sampling과 weighted loss의 효과 분석 필요</li>
                <li>📊 Multi-scale training의 효과 검증</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✅ Small digit analysis completed!")
    print(f"📊 Chart saved to: {save_path}")
    print(f"📋 Report saved to: {report_path}")
    
    return save_path, report_path


def create_error_sample_visualization(experiments_dir: str, data_dir: str, output_dir: str):
    """오류 샘플 시각화"""
    
    output_path = Path(output_dir)
    error_samples_dir = output_path / 'error_samples'
    error_samples_dir.mkdir(parents=True, exist_ok=True)
    
    # 각 실험의 오류 샘플 수집
    for exp_dir in Path(experiments_dir).iterdir():
        if not exp_dir.is_dir():
            continue
            
        summary_path = exp_dir / 'summary.json'
        if not summary_path.exists():
            continue
            
        try:
            with open(summary_path, 'r') as f:
                data = json.load(f)
            
            error_examples = data['final_metrics'].get('error_examples', [])
            
            if not error_examples:
                continue
                
            exp_error_dir = error_samples_dir / exp_dir.name
            exp_error_dir.mkdir(exist_ok=True)
            
            # 처음 20개 오류 샘플만 복사
            for i, error in enumerate(error_examples[:20]):
                src_path = Path(data_dir) / error['path']
                if src_path.exists():
                    filename = f"error_{i+1:02d}_true_{error['target']}_pred_{error['pred']}.jpg"
                    dst_path = exp_error_dir / filename
                    shutil.copy2(src_path, dst_path)
            
            print(f"Copied {min(20, len(error_examples))} error samples for {exp_dir.name}")
            
        except Exception as e:
            print(f"Error processing {exp_dir.name}: {e}")
    
    print(f"Error samples saved to: {error_samples_dir}")
    return error_samples_dir


def main():
    parser = argparse.ArgumentParser(description='Analyze small digit performance in detail')
    parser.add_argument('--experiments_dir', type=str, default='logs/single_runs',
                       help='Directory containing experiment results')
    parser.add_argument('--data_dir', type=str, default='dataset/digit_data',
                       help='Directory containing the dataset')
    parser.add_argument('--output_dir', type=str, default='visualizations/detailed',
                       help='Output directory for detailed analysis')
    
    args = parser.parse_args()
    
    print(f"Analyzing experiments from: {args.experiments_dir}")
    print(f"Using dataset from: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # 작은 글자 성능 분석
    analyze_small_digit_performance(args.experiments_dir, args.data_dir, args.output_dir)
    
    # 오류 샘플 시각화
    create_error_sample_visualization(args.experiments_dir, args.data_dir, args.output_dir)


if __name__ == '__main__':
    main()
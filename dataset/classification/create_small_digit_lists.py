#!/usr/bin/env python3
"""
작은 크기 숫자 이미지를 식별하고 리스트 파일을 생성하는 스크립트
"""

import os
import argparse
from PIL import Image
import numpy as np
import json
from typing import Dict, Tuple, Iterable


def analyze_image_size(image_path: str) -> Tuple[int, int, int, float, float]:
    """
    이미지 분석: 전체 크기, 텍스트 바운딩 영역과 픽셀 비율, 대비를 반환
    Returns: (width, height, content_area, non_bg_ratio, contrast)
    """
    try:
        with Image.open(image_path) as img:
            # RGB로 변환
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # numpy array로 변환
            img_array = np.array(img)
            
            # 배경이 아닌 픽셀 찾기 (완전히 흰색이 아닌 픽셀)
            # RGB 값이 모두 240 이상인 경우를 배경으로 간주
            non_bg_mask = ~np.all(img_array >= 240, axis=2)
            total_area = img_array.shape[0] * img_array.shape[1]
            non_bg_pixels = int(non_bg_mask.sum())
            non_bg_ratio = non_bg_pixels / total_area if total_area > 0 else 0.0

            # 대비 측정 (그레이스케일 표준편차)
            gray = img_array.mean(axis=2) / 255.0
            contrast = float(np.std(gray))
            
            if np.any(non_bg_mask):
                # 컨텐츠가 있는 영역의 바운딩 박스 계산
                rows = np.any(non_bg_mask, axis=1)
                cols = np.any(non_bg_mask, axis=0)
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                
                content_width = cmax - cmin + 1
                content_height = rmax - rmin + 1
                content_area = content_width * content_height
            else:
                content_area = 0
            
            return img.width, img.height, content_area, non_bg_ratio, contrast
    
    except Exception as e:
        print(f"Error analyzing {image_path}: {e}")
        return 0, 0, 0, 0.0, 0.0


def analyze_dataset(data_dir: str, split: str = 'train') -> Dict[str, Dict]:
    """데이터셋의 모든 이미지를 분석"""
    
    # 이미지 파일 목록 읽기
    list_file = os.path.join(data_dir, f'{split}_data.txt')
    if not os.path.exists(list_file):
        raise FileNotFoundError(f"List file not found: {list_file}")
    
    with open(list_file, 'r') as f:
        image_paths = [line.strip() for line in f if line.strip()]
    
    print(f"Analyzing {len(image_paths)} images in {split} set...")
    
    results = {}
    
    for i, rel_path in enumerate(image_paths):
        if i % 1000 == 0:
            print(f"Progress: {i}/{len(image_paths)}")
        
        full_path = os.path.join(data_dir, rel_path)
        if os.path.exists(full_path):
            width, height, content_area, non_bg_ratio, contrast = analyze_image_size(full_path)
            
            # 클래스 정보 추출 (디렉토리 이름에서)
            digit_class = int(os.path.dirname(rel_path))
            
            results[rel_path] = {
                'width': width,
                'height': height, 
                'content_area': content_area,
                'total_area': width * height,
                'content_ratio': content_area / (width * height) if width * height > 0 else 0,
                'non_bg_ratio': non_bg_ratio,
                'contrast': contrast,
                'digit_class': digit_class
            }
    
    return results


def create_small_digit_lists(analysis_results: Dict[str, Dict], 
                           output_dir: str,
                           small_threshold_percentile: float = 25,
                           noisy_threshold_ratio: float = 0.3,
                           noisy_contrast_threshold: float = 0.08,
                           split: str = 'train') -> None:
    """
    분석 결과를 바탕으로 작은 글자, 노이즈 샘플 리스트 생성
    """
    
    # 컨텐츠 영역 크기 통계 계산
    content_areas = [info['content_area'] for info in analysis_results.values() if info['content_area'] > 0]
    content_ratios = [info['content_ratio'] for info in analysis_results.values()]
    non_bg_ratios = [info['non_bg_ratio'] for info in analysis_results.values()]
    contrast_values = [info['contrast'] for info in analysis_results.values()]

    if not content_areas:
        print("No valid content areas found; skipping list creation.")
        return

    small_threshold = np.percentile(content_areas, small_threshold_percentile)
    
    print(f"Content area statistics:")
    print(f"  Mean: {np.mean(content_areas):.1f}")
    print(f"  Median: {np.median(content_areas):.1f}") 
    print(f"  {small_threshold_percentile}th percentile: {small_threshold:.1f}")
    print(f"  Max: {np.max(content_areas):.1f}")
    
    print(f"\nContent ratio statistics:")
    print(f"  Mean: {np.mean(content_ratios):.3f}")
    print(f"  Median: {np.median(content_ratios):.3f}")
    print(f"\nNon-background pixel ratio statistics:")
    print(f"  Mean: {np.mean(non_bg_ratios):.3f}")
    print(f"  Median: {np.median(non_bg_ratios):.3f}")
    print(f"  Min: {np.min(non_bg_ratios):.3f}")
    print(f"  Max: {np.max(non_bg_ratios):.3f}")
    print(f"\nContrast statistics (std of grayscale):")
    print(f"  Mean: {np.mean(contrast_values):.3f}")
    print(f"  Median: {np.median(contrast_values):.3f}")
    print(f"  Min: {np.min(contrast_values):.3f}")
    print(f"  Max: {np.max(contrast_values):.3f}")
    
    # 작은 글자 샘플 식별 (컨텐츠 영역이 작은 것들)
    small_samples = []
    noisy_samples = []  # 컨텐츠 비율이 낮은 것들 (노이즈가 많거나 흐린 이미지)
    
    for rel_path, info in analysis_results.items():
        if info['content_area'] > 0 and info['content_area'] <= small_threshold:
            small_samples.append(rel_path)
        
        if (info['content_ratio'] < noisy_threshold_ratio) or (info['non_bg_ratio'] < noisy_threshold_ratio) or (info['contrast'] < noisy_contrast_threshold):
            noisy_samples.append(rel_path)
    
    print(f"\nIdentified samples:")
    print(f"  Small samples (content_area <= {small_threshold:.1f}): {len(small_samples)}")
    print(f"  Noisy samples (content_ratio < {noisy_threshold_ratio:.3f}): {len(noisy_samples)}")
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 작은 글자 샘플 리스트 저장
    small_path = os.path.join(output_dir, f'small_{split}.txt')
    with open(small_path, 'w') as f:
        for sample in sorted(small_samples):
            f.write(sample + '\n')
    
    # 노이즈 샘플 리스트 저장  
    noisy_path = os.path.join(output_dir, f'noisy_{split}.txt')
    with open(noisy_path, 'w') as f:
        for sample in sorted(noisy_samples):
            f.write(sample + '\n')
    
    # 하드 샘플 리스트 (작은 글자 + 노이즈)
    hard_samples = list(set(small_samples + noisy_samples))
    hard_path = os.path.join(output_dir, f'hard_{split}.txt')
    with open(hard_path, 'w') as f:
        for sample in sorted(hard_samples):
            f.write(sample + '\n')
    
    # 분석 결과도 저장 (samples 제외 - 너무 큼)
    analysis_path = os.path.join(output_dir, f'size_analysis_{split}.json')
    with open(analysis_path, 'w') as f:
        json.dump({
            'statistics': {
                'total_samples': len(analysis_results),
                'small_threshold': float(small_threshold),
                'noisy_ratio_threshold': float(noisy_threshold_ratio),
                'noisy_contrast_threshold': float(noisy_contrast_threshold),
                'small_count': len(small_samples),
                'noisy_count': len(noisy_samples), 
                'hard_count': len(hard_samples),
                'content_area_stats': {
                    'mean': float(np.mean(content_areas)),
                    'median': float(np.median(content_areas)),
                    'std': float(np.std(content_areas)),
                    'min': float(np.min(content_areas)),
                    'max': float(np.max(content_areas))
                },
                'non_bg_ratio_stats': {
                    'mean': float(np.mean(non_bg_ratios)),
                    'median': float(np.median(non_bg_ratios)),
                    'std': float(np.std(non_bg_ratios)),
                    'min': float(np.min(non_bg_ratios)),
                    'max': float(np.max(non_bg_ratios))
                },
                'contrast_stats': {
                    'mean': float(np.mean(contrast_values)),
                    'median': float(np.median(contrast_values)),
                    'std': float(np.std(contrast_values)),
                    'min': float(np.min(contrast_values)),
                    'max': float(np.max(contrast_values))
                },
            }
        }, f, indent=2)
    
    print(f"\nOutput files created:")
    print(f"  Small {split} samples: {small_path}")
    print(f"  Noisy {split} samples: {noisy_path}")
    print(f"  Hard {split} samples: {hard_path}")
    print(f"  Analysis results: {analysis_path}")


def parse_splits(split_arg: str) -> Iterable[str]:
    split_arg = split_arg.strip()
    if split_arg.lower() in {'all', 'both'}:
        return ['train', 'valid']
    return [token.strip() for token in split_arg.split(',') if token.strip()]


def main():
    parser = argparse.ArgumentParser(description='Create small digit sample lists')
    parser.add_argument('--data_dir', type=str, default='dataset/digit_data',
                       help='Path to digit_data directory')
    parser.add_argument('--output_dir', type=str, default='dataset/digit_data',
                       help='Output directory for list files')
    parser.add_argument('--split', type=str, default='all',
                       help="Dataset split(s) to analyze (comma separated, e.g. 'train,valid' or 'all')")
    parser.add_argument('--small_percentile', type=float, default=25,
                       help='Percentile threshold for small samples')
    parser.add_argument('--noisy_ratio', type=float, default=0.3,
                       help='Content ratio threshold for noisy samples')
    parser.add_argument('--noisy_contrast', type=float, default=0.08,
                       help='Contrast threshold (std of grayscale, 0-1) for noisy samples')
    
    args = parser.parse_args()

    splits = list(parse_splits(args.split))
    if not splits:
        raise ValueError("No valid splits provided. Use 'train', 'valid', or comma-separated values.")

    for split in splits:
        print(f"\nAnalyzing {split} dataset in {args.data_dir}")
        results = analyze_dataset(args.data_dir, split)
        create_small_digit_lists(
            results, 
            args.output_dir,
            args.small_percentile,
            args.noisy_ratio,
            args.noisy_contrast,
            split
        )


if __name__ == '__main__':
    main()

# Experiment V3 Results - Comprehensive Analysis (30 Epochs)

**Generated on:** 2025-11-17 03:51:54

## Executive Summary

- **Best 30-Epoch Validation Accuracy:** ConvNeXt (0.9988)
- **Fastest Convergence:** ConvNeXt (1.0 epochs)
- **Most Stable Training:** ResNet-32x4-Wide (std: 0.0015)

## Detailed Model Analysis

### ResNet-20

**Performance Metrics (30 Epochs):**
- Validation Accuracy: 0.9791
- Training Accuracy: 0.9916
- Training Loss: 0.0392
- Best Validation Accuracy: 0.9892
- Overfitting Gap: 0.0125

**Training Characteristics:**
- Convergence Speed: 2.0 epochs to 95% performance
- Training Stability: 0.0213 (std of last 10 epochs)

**Challenging Sample Performance:**
- Small Digit Accuracy: 0.9838
- Noisy Sample Accuracy: 0.9742
- Small Digit Performance Drop: 0.79%
- Noisy Sample Performance Drop: 1.75%

---

### ConvNeXt

**Performance Metrics (30 Epochs):**
- Validation Accuracy: 0.9988
- Training Accuracy: 0.9959
- Training Loss: 0.0167
- Best Validation Accuracy: 0.9988
- Overfitting Gap: -0.0029

**Training Characteristics:**
- Convergence Speed: 1.0 epochs to 95% performance
- Training Stability: 0.0028 (std of last 10 epochs)

---

### Swin Transformer

**Performance Metrics (30 Epochs):**
- Validation Accuracy: 0.9975
- Training Accuracy: 0.9900
- Training Loss: 0.0313
- Best Validation Accuracy: 0.9975
- Overfitting Gap: -0.0075

**Training Characteristics:**
- Convergence Speed: 1.0 epochs to 95% performance
- Training Stability: 0.0017 (std of last 10 epochs)

---

### ResNet-32x4-Wide

**Performance Metrics (30 Epochs):**
- Validation Accuracy: 0.9905
- Training Accuracy: 0.9960
- Training Loss: 0.0143
- Best Validation Accuracy: 0.9905
- Overfitting Gap: 0.0056

**Training Characteristics:**
- Convergence Speed: 1.0 epochs to 95% performance
- Training Stability: 0.0015 (std of last 10 epochs)

---

## Key Insights

**Performance Ranking (30-Epoch Val Accuracy):**
1. ConvNeXt: 0.9988
2. Swin Transformer: 0.9975
3. ResNet-32x4-Wide: 0.9905
4. ResNet-20: 0.9791

**Convergence Speed Ranking:**
1. ConvNeXt: 1.0 epochs
2. Swin Transformer: 1.0 epochs
3. ResNet-32x4-Wide: 1.0 epochs
4. ResNet-20: 2.0 epochs

**Training Stability Ranking:**
1. ResNet-32x4-Wide: 0.0015 (lower is better)
2. Swin Transformer: 0.0017 (lower is better)
3. ConvNeXt: 0.0028 (lower is better)
4. ResNet-20: 0.0213 (lower is better)

## Recommendations

Based on the 30-epoch analysis:

1. **For Best Accuracy:** Use ConvNeXt - achieved highest validation accuracy
2. **For Fast Training:** Use ConvNeXt - converges fastest to target performance
3. **For Stable Training:** Use ResNet-32x4-Wide - most consistent training behavior

4. **For Small Digit Performance:** Use ResNet-20 - best performance on challenging small digits


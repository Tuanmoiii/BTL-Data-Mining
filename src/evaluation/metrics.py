"""
metrics.py - Tính toán các metrics đánh giá
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             silhouette_score, davies_bouldin_score, calinski_harabasz_score)
import logging
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """
    Class tính toán các metrics cho regression, clustering, association
    """
    
    def __init__(self):
        pass
    
    # ==================== REGRESSION METRICS ====================
    
    def regression_metrics(self, 
                           y_true: np.ndarray, 
                           y_pred: np.ndarray,
                           prefix: str = '') -> Dict[str, float]:
        """
        Tính tất cả metrics cho regression
        
        Args:
            y_true: Giá trị thực
            y_pred: Giá trị dự báo
            prefix: Tiền tố cho tên metric
            
        Returns:
            Dict các metrics
        """
        metrics = {}
        
        # MAE
        metrics[f'{prefix}mae'] = mean_absolute_error(y_true, y_pred)
        
        # MSE
        metrics[f'{prefix}mse'] = mean_squared_error(y_true, y_pred)
        
        # RMSE
        metrics[f'{prefix}rmse'] = np.sqrt(metrics[f'{prefix}mse'])
        
        # R²
        metrics[f'{prefix}r2'] = r2_score(y_true, y_pred)
        
        # MAPE (Mean Absolute Percentage Error)
        mask = y_true != 0
        if mask.sum() > 0:
            metrics[f'{prefix}mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            metrics[f'{prefix}mape'] = np.nan
        
        # SMAPE (Symmetric Mean Absolute Percentage Error)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator != 0
        if mask.sum() > 0:
            metrics[f'{prefix}smape'] = np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
        else:
            metrics[f'{prefix}smape'] = np.nan
        
        # Max Error
        metrics[f'{prefix}max_error'] = np.max(np.abs(y_true - y_pred))
        
        return metrics
    
    def regression_metrics_summary(self, 
                                   y_true: np.ndarray, 
                                   y_pred: np.ndarray) -> pd.DataFrame:
        """
        Tạo bảng tóm tắt các metrics regression
        
        Args:
            y_true: Giá trị thực
            y_pred: Giá trị dự báo
            
        Returns:
            DataFrame tóm tắt
        """
        metrics = self.regression_metrics(y_true, y_pred)
        
        df = pd.DataFrame({
            'Metric': list(metrics.keys()),
            'Value': list(metrics.values())
        })
        
        # Format số
        df['Value'] = df['Value'].apply(lambda x: f"{x:.4f}")
        
        return df
    
    def regression_residuals(self, 
                            y_true: np.ndarray, 
                            y_pred: np.ndarray) -> Dict:
        """
        Phân tích residuals (phần dư)
        
        Args:
            y_true: Giá trị thực
            y_pred: Giá trị dự báo
            
        Returns:
            Dict phân tích residuals
        """
        residuals = y_true - y_pred
        
        analysis = {
            'residuals': residuals,
            'mean_residual': np.mean(residuals),
            'std_residual': np.std(residuals),
            'skewness': pd.Series(residuals).skew(),
            'kurtosis': pd.Series(residuals).kurtosis(),
            'min_residual': np.min(residuals),
            'max_residual': np.max(residuals),
            'percentiles': {
                '25%': np.percentile(residuals, 25),
                '50%': np.percentile(residuals, 50),
                '75%': np.percentile(residuals, 75),
                '90%': np.percentile(residuals, 90),
                '95%': np.percentile(residuals, 95)
            }
        }
        
        # Kiểm tra phân phối chuẩn của residuals
        from scipy import stats
        stat, p_value = stats.shapiro(residuals[:5000] if len(residuals) > 5000 else residuals)
        analysis['normality_test'] = {
            'statistic': stat,
            'p_value': p_value,
            'is_normal': p_value > 0.05
        }
        
        return analysis
    
    # ==================== CLUSTERING METRICS ====================
    
    def clustering_metrics(self, 
                          X: np.ndarray, 
                          labels: np.ndarray,
                          prefix: str = '') -> Dict[str, float]:
        """
        Tính metrics cho clustering
        
        Args:
            X: Dữ liệu đầu vào
            labels: Nhãn cụm
            prefix: Tiền tố cho tên metric
            
        Returns:
            Dict các metrics
        """
        metrics = {}
        
        # Silhouette Score
        if len(set(labels)) > 1:
            if -1 in labels:  # Có nhiễu (DBSCAN)
                mask = labels != -1
                if mask.sum() > 1 and len(set(labels[mask])) > 1:
                    metrics[f'{prefix}silhouette'] = silhouette_score(X[mask], labels[mask])
                else:
                    metrics[f'{prefix}silhouette'] = -1
            else:
                metrics[f'{prefix}silhouette'] = silhouette_score(X, labels)
        else:
            metrics[f'{prefix}silhouette'] = -1
        
        # Davies-Bouldin Index (thấp hơn = tốt hơn)
        if len(set(labels)) > 1 and -1 not in set(labels):
            metrics[f'{prefix}davies_bouldin'] = davies_bouldin_score(X, labels)
        else:
            metrics[f'{prefix}davies_bouldin'] = -1
        
        # Calinski-Harabasz Index (cao hơn = tốt hơn)
        if len(set(labels)) > 1:
            metrics[f'{prefix}calinski_harabasz'] = calinski_harabasz_score(X, labels)
        else:
            metrics[f'{prefix}calinski_harabasz'] = -1
        
        return metrics
    
    def clustering_metrics_summary(self, 
                                   X: np.ndarray, 
                                   labels: np.ndarray) -> pd.DataFrame:
        """
        Tạo bảng tóm tắt các metrics clustering
        
        Args:
            X: Dữ liệu đầu vào
            labels: Nhãn cụm
            
        Returns:
            DataFrame tóm tắt
        """
        metrics = self.clustering_metrics(X, labels)
        
        df = pd.DataFrame({
            'Metric': list(metrics.keys()),
            'Value': list(metrics.values())
        })
        
        # Format số
        df['Value'] = df['Value'].apply(lambda x: f"{x:.4f}" if x != -1 else "N/A")
        
        return df
    
    # ==================== ASSOCIATION RULE METRICS ====================
    
    def association_metrics_summary(self, rules: pd.DataFrame) -> pd.DataFrame:
        """
        Tóm tắt thống kê các luật kết hợp
        
        Args:
            rules: DataFrame các luật
            
        Returns:
            DataFrame thống kê
        """
        if rules is None or len(rules) == 0:
            return pd.DataFrame()
        
        stats = {
            'Metric': ['Số lượng luật', 'Support trung bình', 'Confidence trung bình', 
                      'Lift trung bình', 'Lift max', 'Lift min', 'Leverage trung bình',
                      'Conviction trung bình'],
            'Value': [
                len(rules),
                f"{rules['support'].mean():.4f}",
                f"{rules['confidence'].mean():.4f}",
                f"{rules['lift'].mean():.4f}",
                f"{rules['lift'].max():.4f}",
                f"{rules['lift'].min():.4f}",
                f"{rules['leverage'].mean():.4f}" if 'leverage' in rules.columns else "N/A",
                f"{rules['conviction'].mean():.4f}" if 'conviction' in rules.columns else "N/A"
            ]
        }
        
        return pd.DataFrame(stats)
    
    # ==================== COMPARISON METRICS ====================
    
    def compare_models(self, results_dict: Dict[str, Dict]) -> pd.DataFrame:
        """
        So sánh nhiều mô hình
        
        Args:
            results_dict: Dict {model_name: {metrics_dict}}
            
        Returns:
            DataFrame so sánh
        """
        comparison = []
        
        for model_name, metrics in results_dict.items():
            row = {'Model': model_name}
            row.update(metrics)
            comparison.append(row)
        
        df = pd.DataFrame(comparison)
        
        # Sắp xếp theo MAE (nếu có)
        if 'test_mae' in df.columns:
            df = df.sort_values('test_mae')
        
        return df
    
    def calculate_improvement(self, 
                             baseline_metric: float, 
                             model_metric: float,
                             metric_name: str = 'MAE',
                             higher_is_better: bool = False) -> Dict:
        """
        Tính % cải thiện so với baseline
        
        Args:
            baseline_metric: Giá trị baseline
            model_metric: Giá trị mô hình
            metric_name: Tên metric
            higher_is_better: True nếu metric cao hơn là tốt hơn
            
        Returns:
            Dict thông tin cải thiện
        """
        if higher_is_better:
            improvement = ((model_metric - baseline_metric) / baseline_metric) * 100
        else:
            improvement = ((baseline_metric - model_metric) / baseline_metric) * 100
        
        return {
            'metric': metric_name,
            'baseline': baseline_metric,
            'model': model_metric,
            'improvement': improvement,
            'improvement_pct': f"{improvement:.2f}%",
            'is_better': improvement > 0 if higher_is_better else improvement > 0
        }
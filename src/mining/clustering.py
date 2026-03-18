"""
clustering.py - Phân cụm dữ liệu cây trồng
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import logging
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class CropClusterer:
    """
    Class phân cụm dữ liệu cây trồng
    """
    
    def __init__(self, random_state: int = 42):
        """
        Khởi tạo CropClusterer
        
        Args:
            random_state: Seed cho random
        """
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.cluster_labels = None
        self.cluster_profiles = None
        
    def prepare_clustering_data(self, 
                                df: pd.DataFrame,
                                feature_cols: Optional[List[str]] = None,
                                scale: bool = True) -> Tuple[np.ndarray, List[str]]:
        """
        Chuẩn bị dữ liệu cho phân cụm
        
        Args:
            df: DataFrame đầu vào
            feature_cols: Danh sách cột đặc trưng
            scale: Có chuẩn hóa dữ liệu không
            
        Returns:
            (X_scaled, feature_names)
        """
        if feature_cols is None:
            # Lấy các cột số, loại bỏ cột định danh và target
            exclude_cols = ['Area', 'Item', 'Year', 'hg/ha_yield']
            feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                           if col not in exclude_cols]
        
        X = df[feature_cols].values
        
        if scale:
            X_scaled = self.scaler.fit_transform(X)
            logger.info(f"Đã chuẩn hóa dữ liệu với {len(feature_cols)} features")
        else:
            X_scaled = X
        
        return X_scaled, feature_cols
    
    def find_optimal_k(self, 
                       X: np.ndarray,
                       k_range: range = range(2, 11),
                       method: str = 'silhouette') -> Dict[int, float]:
        """
        Tìm số cụm tối ưu
        
        Args:
            X: Dữ liệu đầu vào
            k_range: Dải số cụm cần thử
            method: Phương pháp đánh giá ('silhouette', 'elbow')
            
        Returns:
            Dict {k: score}
        """
        scores = {}
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            
            if method == 'silhouette':
                if len(set(labels)) > 1:  # Silhouette cần ít nhất 2 cụm
                    score = silhouette_score(X, labels)
                else:
                    score = -1
            elif method == 'elbow':
                score = kmeans.inertia_
            
            scores[k] = score
            logger.info(f"k={k}: {method}={score:.4f}")
        
        return scores
    
    def kmeans_clustering(self, 
                          X: np.ndarray,
                          n_clusters: int = 3,
                          **kwargs) -> np.ndarray:
        """
        Phân cụm bằng K-Means
        
        Args:
            X: Dữ liệu đầu vào
            n_clusters: Số cụm
            **kwargs: Tham số bổ sung cho KMeans
            
        Returns:
            Nhãn cụm
        """
        logger.info(f"Bắt đầu K-Means clustering với n_clusters={n_clusters}")
        
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10,
            **kwargs
        )
        
        self.cluster_labels = self.model.fit_predict(X)
        
        # Đếm số lượng mỗi cụm
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        for cluster, count in zip(unique, counts):
            logger.info(f"  Cụm {cluster}: {count} mẫu ({count/len(X)*100:.1f}%)")
        
        return self.cluster_labels
    
    def hierarchical_clustering(self,
                                X: np.ndarray,
                                n_clusters: int = 3,
                                linkage: str = 'ward',
                                **kwargs) -> np.ndarray:
        """
        Phân cụm phân cấp (HAC)
        
        Args:
            X: Dữ liệu đầu vào
            n_clusters: Số cụm
            linkage: Phương pháp linkage ('ward', 'complete', 'average', 'single')
            **kwargs: Tham số bổ sung
            
        Returns:
            Nhãn cụm
        """
        logger.info(f"Bắt đầu Hierarchical clustering với n_clusters={n_clusters}, linkage={linkage}")
        
        self.model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            **kwargs
        )
        
        self.cluster_labels = self.model.fit_predict(X)
        
        # Đếm số lượng mỗi cụm
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        for cluster, count in zip(unique, counts):
            logger.info(f"  Cụm {cluster}: {count} mẫu ({count/len(X)*100:.1f}%)")
        
        return self.cluster_labels
    
    def dbscan_clustering(self,
                          X: np.ndarray,
                          eps: float = 0.5,
                          min_samples: int = 5,
                          **kwargs) -> np.ndarray:
        """
        Phân cụm DBSCAN
        
        Args:
            X: Dữ liệu đầu vào
            eps: Khoảng cách tối đa giữa các điểm trong cùng cụm
            min_samples: Số mẫu tối thiểu để tạo core point
            **kwargs: Tham số bổ sung
            
        Returns:
            Nhãn cụm ( -1 là nhiễu)
        """
        logger.info(f"Bắt đầu DBSCAN clustering với eps={eps}, min_samples={min_samples}")
        
        self.model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            **kwargs
        )
        
        self.cluster_labels = self.model.fit_predict(X)
        
        # Đếm số lượng mỗi cụm
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        n_noise = np.sum(self.cluster_labels == -1)
        
        for cluster, count in zip(unique, counts):
            if cluster == -1:
                logger.info(f"  Nhiễu: {count} mẫu ({count/len(X)*100:.1f}%)")
            else:
                logger.info(f"  Cụm {cluster}: {count} mẫu ({count/len(X)*100:.1f}%)")
        
        return self.cluster_labels
    
    def evaluate_clustering(self, X: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Đánh giá chất lượng phân cụm
        
        Args:
            X: Dữ liệu đầu vào
            labels: Nhãn cụm
            
        Returns:
            Dict các metrics
        """
        metrics = {}
        
        # Silhouette Score
        if len(set(labels)) > 1 and -1 not in set(labels):  # Không có nhiễu
            metrics['silhouette'] = silhouette_score(X, labels)
        else:
            # Xử lý trường hợp có nhiễu
            mask = labels != -1
            if mask.sum() > 1 and len(set(labels[mask])) > 1:
                metrics['silhouette'] = silhouette_score(X[mask], labels[mask])
            else:
                metrics['silhouette'] = -1
        
        # Davies-Bouldin Index
        if len(set(labels)) > 1 and -1 not in set(labels):
            metrics['davies_bouldin'] = davies_bouldin_score(X, labels)
        else:
            metrics['davies_bouldin'] = -1
        
        # Calinski-Harabasz Index
        if len(set(labels)) > 1:
            metrics['calinski_harabasz'] = calinski_harabasz_score(X, labels)
        else:
            metrics['calinski_harabasz'] = -1
        
        logger.info(f"Đánh giá phân cụm:")
        logger.info(f"  Silhouette Score: {metrics['silhouette']:.4f}")
        logger.info(f"  Davies-Bouldin Index: {metrics['davies_bouldin']:.4f}")
        logger.info(f"  Calinski-Harabasz Index: {metrics['calinski_harabasz']:.4f}")
        
        return metrics
    
    def create_cluster_profiles(self, 
                                df: pd.DataFrame,
                                cluster_col: str = 'cluster') -> pd.DataFrame:
        """
        Tạo hồ sơ (profile) cho từng cụm
        
        Args:
            df: DataFrame gốc có thêm cột cluster
            cluster_col: Tên cột chứa nhãn cụm
            
        Returns:
            DataFrame profile của các cụm
        """
        if cluster_col not in df.columns:
            logger.error(f"Không tìm thấy cột {cluster_col}")
            return pd.DataFrame()
        
        # Lấy các cột số để phân tích
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if cluster_col in numeric_cols:
            numeric_cols.remove(cluster_col)
        
        # Thống kê theo cụm
        profiles = []
        
        for cluster in sorted(df[cluster_col].unique()):
            cluster_data = df[df[cluster_col] == cluster]
            
            profile = {
                'cluster': cluster,
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(df) * 100
            }
            
            # Thống kê cho từng cột số
            for col in numeric_cols:
                profile[f'{col}_mean'] = cluster_data[col].mean()
                profile[f'{col}_std'] = cluster_data[col].std()
                profile[f'{col}_min'] = cluster_data[col].min()
                profile[f'{col}_max'] = cluster_data[col].max()
            
            # Thống kê cho cột phân loại (nếu có)
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()
            for col in cat_cols[:3]:  # Chỉ lấy vài cột để tránh quá nhiều
                if col in cluster_data.columns:
                    top_value = cluster_data[col].mode().iloc[0] if not cluster_data[col].mode().empty else 'N/A'
                    profile[f'{col}_mode'] = top_value
            
            profiles.append(profile)
        
        self.cluster_profiles = pd.DataFrame(profiles)
        logger.info(f"Đã tạo profile cho {len(profiles)} cụm")
        
        return self.cluster_profiles
    
    def compare_cluster_yield(self, 
                              df: pd.DataFrame,
                              cluster_col: str = 'cluster',
                              yield_col: str = 'hg/ha_yield') -> pd.DataFrame:
        """
        So sánh năng suất trung bình giữa các cụm
        
        Args:
            df: DataFrame có cột cluster
            cluster_col: Tên cột cluster
            yield_col: Tên cột năng suất
            
        Returns:
            DataFrame so sánh năng suất
        """
        if cluster_col not in df.columns or yield_col not in df.columns:
            logger.error(f"Thiếu cột {cluster_col} hoặc {yield_col}")
            return pd.DataFrame()
        
        comparison = df.groupby(cluster_col)[yield_col].agg([
            ('mean_yield', 'mean'),
            ('std_yield', 'std'),
            ('min_yield', 'min'),
            ('max_yield', 'max'),
            ('count', 'count')
        ]).reset_index()
        
        # Thêm xếp hạng
        comparison = comparison.sort_values('mean_yield', ascending=False)
        comparison['rank'] = range(1, len(comparison) + 1)
        
        logger.info(f"So sánh năng suất giữa các cụm:")
        for _, row in comparison.iterrows():
            logger.info(f"  Cụm {row[cluster_col]}: {row['mean_yield']:.2f} ± {row['std_yield']:.2f} (rank {row['rank']})")
        
        return comparison
    
    def run_clustering_pipeline(self,
                                df: pd.DataFrame,
                                feature_cols: Optional[List[str]] = None,
                                method: str = 'kmeans',
                                n_clusters: int = 3,
                                **kwargs) -> Dict:
        """
        Chạy toàn bộ pipeline phân cụm
        
        Args:
            df: DataFrame đầu vào
            feature_cols: Danh sách cột đặc trưng
            method: Phương pháp phân cụm ('kmeans', 'hierarchical', 'dbscan')
            n_clusters: Số cụm (cho KMeans và HAC)
            **kwargs: Tham số bổ sung cho phương pháp cụ thể
            
        Returns:
            Dict chứa kết quả
        """
        logger.info("=" * 60)
        logger.info("BẮT ĐẦU PIPELINE PHÂN CỤM")
        logger.info("=" * 60)
        
        # 1. Chuẩn bị dữ liệu
        X, used_features = self.prepare_clustering_data(df, feature_cols, scale=True)
        logger.info(f"Sử dụng {len(used_features)} features: {used_features}")
        
        # 2. Tìm k tối ưu nếu không được chỉ định
        if n_clusters is None and method in ['kmeans', 'hierarchical']:
            logger.info("Tìm số cụm tối ưu...")
            scores = self.find_optimal_k(X, k_range=range(2, 8), method='silhouette')
            n_clusters = max(scores, key=scores.get)
            logger.info(f"Chọn k={n_clusters} dựa trên silhouette score")
        
        # 3. Phân cụm
        if method == 'kmeans':
            labels = self.kmeans_clustering(X, n_clusters=n_clusters, **kwargs)
        elif method == 'hierarchical':
            labels = self.hierarchical_clustering(X, n_clusters=n_clusters, **kwargs)
        elif method == 'dbscan':
            labels = self.dbscan_clustering(X, **kwargs)
        else:
            logger.error(f"Phương pháp {method} không được hỗ trợ")
            return {}
        
        # 4. Đánh giá
        metrics = self.evaluate_clustering(X, labels)
        
        # 5. Thêm nhãn vào dataframe
        df_with_cluster = df.copy()
        df_with_cluster['cluster'] = labels
        
        # 6. Tạo profile
        profiles = self.create_cluster_profiles(df_with_cluster)
        
        # 7. So sánh năng suất
        yield_comparison = self.compare_cluster_yield(df_with_cluster)
        
        logger.info("=" * 60)
        logger.info("HOÀN THÀNH PIPELINE PHÂN CỤM")
        logger.info("=" * 60)
        
        return {
            'labels': labels,
            'model': self.model,
            'metrics': metrics,
            'profiles': profiles,
            'yield_comparison': yield_comparison,
            'df_with_cluster': df_with_cluster,
            'used_features': used_features,
            'n_clusters': len(set(labels)) - (1 if -1 in labels else 0)
        }
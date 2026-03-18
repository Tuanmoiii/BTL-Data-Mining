"""
plots.py - Các hàm vẽ biểu đồ dùng chung
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Cấu hình mặc định cho matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class Plotter:
    """
    Class vẽ các loại biểu đồ
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        """
        Khởi tạo Plotter
        
        Args:
            figsize: Kích thước figure mặc định
            dpi: Độ phân giải
        """
        self.figsize = figsize
        self.dpi = dpi
        self.current_fig = None
        self.current_ax = None
    
    # ==================== EDA PLOTS ====================
    
    def plot_distribution(self,
                         data: pd.Series,
                         title: str = 'Distribution',
                         xlabel: Optional[str] = None,
                         ylabel: str = 'Frequency',
                         bins: int = 30,
                         kde: bool = True) -> plt.Figure:
        """
        Vẽ phân phối của một biến
        
        Args:
            data: Dữ liệu cần vẽ
            title: Tiêu đề
            xlabel: Nhãn trục x
            ylabel: Nhãn trục y
            bins: Số bins
            kde: Vẽ đường KDE
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        sns.histplot(data, kde=kde, bins=bins, ax=ax)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel if xlabel else data.name, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        
        # Thêm statistics
        ax.text(0.02, 0.98, f'Mean: {data.mean():.2f}\nStd: {data.std():.2f}\nSkew: {data.skew():.2f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        self.current_fig = fig
        return fig
    
    def plot_correlation_matrix(self,
                               df: pd.DataFrame,
                               title: str = 'Correlation Matrix',
                               figsize: Optional[Tuple[int, int]] = None,
                               annot: bool = True,
                               cmap: str = 'coolwarm') -> plt.Figure:
        """
        Vẽ ma trận tương quan
        
        Args:
            df: DataFrame chứa các cột số
            title: Tiêu đề
            figsize: Kích thước figure
            annot: Hiển thị giá trị
            cmap: Colormap
            
        Returns:
            Figure object
        """
        # Chỉ lấy cột số
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            logger.warning("Cần ít nhất 2 cột số để vẽ correlation matrix")
            return None
        
        figsize = figsize or (max(10, len(numeric_df.columns)), max(8, len(numeric_df.columns)))
        fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)
        
        corr = numeric_df.corr()
        
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=annot, cmap=cmap, center=0,
                   square=True, linewidths=1, ax=ax, fmt='.2f')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        self.current_fig = fig
        return fig
    
    def plot_boxplots(self,
                     df: pd.DataFrame,
                     columns: Optional[List[str]] = None,
                     title: str = 'Boxplots') -> plt.Figure:
        """
        Vẽ boxplot cho nhiều cột
        
        Args:
            df: DataFrame
            columns: Danh sách cột cần vẽ
            title: Tiêu đề
            
        Returns:
            Figure object
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        n_cols = min(4, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*6, n_rows*4), dpi=self.dpi)
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
        
        for i, col in enumerate(columns):
            if i < len(axes):
                sns.boxplot(y=df[col], ax=axes[i])
                axes[i].set_title(f'{col}', fontsize=12)
                axes[i].set_ylabel('')
        
        # Ẩn các subplot thừa
        for i in range(len(columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        self.current_fig = fig
        return fig
    
    def plot_time_series(self,
                        df: pd.DataFrame,
                        date_col: str,
                        value_cols: List[str],
                        title: str = 'Time Series') -> plt.Figure:
        """
        Vẽ biểu đồ chuỗi thời gian
        
        Args:
            df: DataFrame
            date_col: Cột thời gian
            value_cols: Các cột giá trị
            title: Tiêu đề
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        for col in value_cols:
            if col in df.columns:
                ax.plot(df[date_col], df[col], marker='o', markersize=3, linewidth=1, label=col)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(date_col, fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.current_fig = fig
        return fig
    
    # ==================== REGRESSION PLOTS ====================
    
    def plot_prediction_vs_actual(self,
                                  y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  title: str = 'Predicted vs Actual') -> plt.Figure:
        """
        Vẽ biểu đồ so sánh giá trị dự báo và thực tế
        
        Args:
            y_true: Giá trị thực
            y_pred: Giá trị dự báo
            title: Tiêu đề
            
        Returns:
            Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(self.figsize[0]*2, self.figsize[1]), dpi=self.dpi)
        
        # Scatter plot
        ax = axes[0]
        ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
        
        # Đường y=x
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual', fontsize=12)
        ax.set_ylabel('Predicted', fontsize=12)
        ax.set_title('Scatter Plot', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Residuals plot
        ax = axes[1]
        residuals = y_true - y_pred
        ax.scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Residuals', fontsize=12)
        ax.set_title('Residuals Plot', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        self.current_fig = fig
        return fig
    
    def plot_feature_importance(self,
                               importance_df: pd.DataFrame,
                               title: str = 'Feature Importance',
                               top_n: int = 20) -> plt.Figure:
        """
        Vẽ biểu đồ feature importance
        
        Args:
            importance_df: DataFrame với cột 'feature' và 'importance'
            title: Tiêu đề
            top_n: Số lượng feature hiển thị
            
        Returns:
            Figure object
        """
        df_plot = importance_df.head(top_n).copy()
        df_plot = df_plot.sort_values('importance', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(df_plot)*0.3)), dpi=self.dpi)
        
        ax.barh(df_plot['feature'], df_plot['importance'], color='steelblue')
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Thêm giá trị
        for i, (_, row) in enumerate(df_plot.iterrows()):
            ax.text(row['importance'] + 0.01, i, f'{row["importance"]:.4f}', 
                   va='center', fontsize=9)
        
        plt.tight_layout()
        self.current_fig = fig
        return fig
    
    def plot_model_comparison(self,
                            comparison_df: pd.DataFrame,
                            metric: str = 'MAE',
                            title: str = 'Model Comparison') -> plt.Figure:
        """
        Vẽ biểu đồ so sánh các mô hình
        
        Args:
            comparison_df: DataFrame so sánh (từ MetricsCalculator.compare_models)
            metric: Metric cần so sánh
            title: Tiêu đề
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        models = comparison_df['Model'].tolist()
        values = comparison_df[metric].tolist()
        colors = ['green' if i == 0 else 'steelblue' for i in range(len(models))]
        
        bars = ax.bar(models, values, color=colors, alpha=0.7)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'{title} - {metric}', fontsize=14, fontweight='bold')
        
        # Thêm giá trị trên bar
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=10)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        self.current_fig = fig
        return fig
    
    # ==================== CLUSTERING PLOTS ====================
    
    def plot_clusters(self,
                     X: np.ndarray,
                     labels: np.ndarray,
                     feature_names: Optional[List[str]] = None,
                     title: str = 'Cluster Visualization') -> plt.Figure:
        """
        Vẽ biểu đồ phân cụm (PCA nếu > 2 chiều)
        
        Args:
            X: Dữ liệu
            labels: Nhãn cụm
            feature_names: Tên các feature
            title: Tiêu đề
            
        Returns:
            Figure object
        """
        from sklearn.decomposition import PCA
        
        # Giảm chiều nếu cần
        if X.shape[1] > 2:
            pca = PCA(n_components=2, random_state=42)
            X_plot = pca.fit_transform(X)
            explained_var = pca.explained_variance_ratio_
            xlabel = f'PC1 ({explained_var[0]:.1%})'
            ylabel = f'PC2 ({explained_var[1]:.1%})'
        else:
            X_plot = X
            xlabel = feature_names[0] if feature_names else 'Feature 1'
            ylabel = feature_names[1] if feature_names and len(feature_names) > 1 else 'Feature 2'
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Phân biệt nhiễu (label = -1)
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            if label == -1:
                # Nhiễu - vẽ màu đen với marker khác
                ax.scatter(X_plot[mask, 0], X_plot[mask, 1], 
                          c='black', marker='x', s=50, label='Noise', alpha=0.6)
            else:
                ax.scatter(X_plot[mask, 0], X_plot[mask, 1], 
                          c=[colors[i]], marker='o', s=50, label=f'Cluster {label}', alpha=0.7, edgecolors='k')
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.current_fig = fig
        return fig
    
    def plot_cluster_profiles(self,
                             profiles: pd.DataFrame,
                             feature_cols: List[str],
                             title: str = 'Cluster Profiles') -> plt.Figure:
        """
        Vẽ biểu đồ profiles của các cụm
        
        Args:
            profiles: DataFrame profiles (từ clustering)
            feature_cols: Các cột feature cần vẽ
            title: Tiêu đề
            
        Returns:
            Figure object
        """
        # Lấy các cột mean
        mean_cols = [f for f in feature_cols if f + '_mean' in profiles.columns]
        
        if not mean_cols:
            logger.warning("Không tìm thấy cột mean trong profiles")
            return None
        
        # Chuẩn bị dữ liệu
        plot_data = []
        for _, row in profiles.iterrows():
            cluster = row['cluster']
            for feat in mean_cols:
                orig_feat = feat.replace('_mean', '')
                plot_data.append({
                    'Cluster': f'Cluster {cluster}',
                    'Feature': orig_feat,
                    'Value': row[feat]
                })
        
        df_plot = pd.DataFrame(plot_data)
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        sns.barplot(data=df_plot, x='Feature', y='Value', hue='Cluster', ax=ax)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Feature', fontsize=12)
        ax.set_ylabel('Mean Value', fontsize=12)
        ax.legend(title='Cluster')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        self.current_fig = fig
        return fig
    
    # ==================== ASSOCIATION PLOTS ====================
    
    def plot_top_rules(self,
                      rules: pd.DataFrame,
                      metric: str = 'lift',
                      top_n: int = 10,
                      title: str = 'Top Association Rules') -> plt.Figure:
        """
        Vẽ biểu đồ top luật kết hợp
        
        Args:
            rules: DataFrame rules
            metric: Metric để sắp xếp
            top_n: Số lượng luật hiển thị
            title: Tiêu đề
            
        Returns:
            Figure object
        """
        top_rules = rules.nlargest(top_n, metric).copy()
        
        # Tạo nhãn cho luật
        top_rules['rule'] = top_rules.apply(
            lambda row: f"{', '.join([str(i)[:20] for i in row['antecedents']])} → {', '.join([str(i)[:20] for i in row['consequents']])}",
            axis=1
        )
        
        # Giới hạn độ dài nhãn
        top_rules['rule'] = top_rules['rule'].apply(lambda x: x if len(x) < 50 else x[:47] + '...')
        
        fig, axes = plt.subplots(1, 3, figsize=(self.figsize[0]*2, self.figsize[1]), dpi=self.dpi)
        
        # Lift
        ax = axes[0]
        y_pos = range(len(top_rules))
        ax.barh(y_pos, top_rules['lift'], color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_rules['rule'])
        ax.set_xlabel('Lift', fontsize=12)
        ax.set_title('Top Rules by Lift', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        
        # Confidence
        ax = axes[1]
        ax.barh(y_pos, top_rules['confidence'], color='coral')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([])  # Ẩn nhãn
        ax.set_xlabel('Confidence', fontsize=12)
        ax.set_title('Confidence', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        
        # Support
        ax = axes[2]
        ax.barh(y_pos, top_rules['support'], color='green')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([])  # Ẩn nhãn
        ax.set_xlabel('Support', fontsize=12)
        ax.set_title('Support', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        self.current_fig = fig
        return fig
    
    def plot_rule_metrics_scatter(self,
                                 rules: pd.DataFrame,
                                 title: str = 'Association Rules Metrics') -> plt.Figure:
        """
        Vẽ scatter plot giữa support, confidence, lift
        
        Args:
            rules: DataFrame rules
            title: Tiêu đề
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        scatter = ax.scatter(rules['support'], rules['confidence'], 
                           c=rules['lift'], cmap='viridis', s=50, alpha=0.6, edgecolors='k')
        
        ax.set_xlabel('Support', fontsize=12)
        ax.set_ylabel('Confidence', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Colorbar cho lift
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Lift', fontsize=12)
        
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.current_fig = fig
        return fig
    
    # ==================== SAVE/LOAD ====================
    
    def save_current_figure(self, 
                           filename: str,
                           output_dir: Optional[Path] = None,
                           formats: List[str] = ['png', 'pdf']) -> Dict[str, Path]:
        """
        Lưu figure hiện tại
        
        Args:
            filename: Tên file
            output_dir: Thư mục lưu
            formats: Các định dạng
            
        Returns:
            Dict {format: path}
        """
        if self.current_fig is None:
            logger.error("Không có figure hiện tại để lưu")
            return {}
        
        if output_dir is None:
            output_dir = Path.cwd() / 'figures'
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = {}
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        for fmt in formats:
            path = output_dir / f"{filename}_{timestamp}.{fmt}"
            self.current_fig.savefig(path, dpi=self.dpi, bbox_inches='tight')
            saved_paths[fmt] = path
        
        logger.info(f"Đã lưu figure {filename} tại: {saved_paths}")
        return saved_paths
    
    def close_current_figure(self):
        """Đóng figure hiện tại"""
        if self.current_fig:
            plt.close(self.current_fig)
            self.current_fig = None
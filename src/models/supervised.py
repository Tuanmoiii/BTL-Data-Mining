"""
supervised.py - Mô hình hồi quy dự báo năng suất cây trồng
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class CropYieldPredictor:
    """
    Class dự báo năng suất cây trồng bằng các mô hình hồi quy
    """
    
    # Định nghĩa các mô hình hỗ trợ
    SUPPORTED_MODELS = {
        'linear': LinearRegression,
        'ridge': Ridge,
        'lasso': Lasso,
        'random_forest': RandomForestRegressor,
        'xgboost': xgb.XGBRegressor,
        'gradient_boosting': GradientBoostingRegressor
    }
    
    def __init__(self, random_state: int = 42, model_dir: Optional[Union[str, Path]] = None):
        """
        Khởi tạo CropYieldPredictor
        
        Args:
            random_state: Seed cho random
            model_dir: Thư mục lưu model
        """
        self.random_state = random_state
        self.model_dir = Path(model_dir) if model_dir else None
        self.model = None
        self.model_name = None
        self.feature_importance = None
        self.training_history = {}
        
        if self.model_dir:
            self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_data(self, 
                     df: pd.DataFrame,
                     target_col: str = 'hg/ha_yield',
                     feature_cols: Optional[List[str]] = None,
                     test_size: float = 0.2,
                     validation_size: float = 0.1,
                     time_split: bool = True,
                     year_col: str = 'Year') -> Dict:
        """
        Chuẩn bị dữ liệu cho huấn luyện
        
        Args:
            df: DataFrame đầu vào
            target_col: Tên cột mục tiêu
            feature_cols: Danh sách cột đặc trưng
            test_size: Tỷ lệ tập test
            validation_size: Tỷ lệ tập validation (từ tập train)
            time_split: Chia theo thời gian (True) hay random (False)
            year_col: Tên cột năm (nếu time_split=True)
            
        Returns:
            Dict chứa X_train, X_val, X_test, y_train, y_val, y_test
        """
        if target_col not in df.columns:
            logger.error(f"Không tìm thấy cột target: {target_col}")
            return {}
        
        # Xác định feature columns
        if feature_cols is None:
            # Lấy tất cả cột số trừ target và các cột định danh
            exclude_cols = [target_col, 'Area', 'Item', 'Year', 'cluster']
            feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                           if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Xử lý missing values trong features
        X = X.fillna(X.mean())
        
        logger.info(f"Features: {feature_cols}")
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        
        # Chia dữ liệu
        if time_split and year_col in df.columns:
            # Chia theo thời gian
            years = df[year_col].unique()
            years = np.sort(years)
            
            n_years = len(years)
            n_test_years = max(1, int(n_years * test_size))
            n_val_years = max(1, int(n_years * validation_size))
            
            test_years = years[-n_test_years:]
            val_years = years[-(n_test_years + n_val_years):-n_test_years]
            train_years = years[:-(n_test_years + n_val_years)]
            
            logger.info(f"Train years: {train_years}")
            logger.info(f"Validation years: {val_years}")
            logger.info(f"Test years: {test_years}")
            
            # Tạo indices
            train_idx = df[df[year_col].isin(train_years)].index
            val_idx = df[df[year_col].isin(val_years)].index
            test_idx = df[df[year_col].isin(test_years)].index
            
            X_train, y_train = X.loc[train_idx], y.loc[train_idx]
            X_val, y_val = X.loc[val_idx], y.loc[val_idx]
            X_test, y_test = X.loc[test_idx], y.loc[test_idx]
            
        else:
            # Chia random
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )
            
            # Chia validation từ train
            val_ratio = validation_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_ratio, random_state=self.random_state
            )
        
        data_dict = {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'feature_cols': feature_cols
        }
        
        logger.info(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
        
        return data_dict
    
    def train_model(self,
                    model_type: str,
                    X_train: pd.DataFrame,
                    y_train: pd.Series,
                    X_val: Optional[pd.DataFrame] = None,
                    y_val: Optional[pd.Series] = None,
                    **kwargs) -> Any:
        """
        Huấn luyện mô hình
        
        Args:
            model_type: Loại mô hình ('linear', 'ridge', 'random_forest', 'xgboost')
            X_train: Features train
            y_train: Target train
            X_val: Features validation (optional)
            y_val: Target validation (optional)
            **kwargs: Tham số bổ sung cho mô hình
            
        Returns:
            Mô hình đã huấn luyện
        """
        if model_type not in self.SUPPORTED_MODELS:
            logger.error(f"Model {model_type} không được hỗ trợ. Chọn từ: {list(self.SUPPORTED_MODELS.keys())}")
            return None
        
        model_class = self.SUPPORTED_MODELS[model_type]
        
        # Tham số mặc định theo từng model
        default_params = {
            'linear': {},
            'ridge': {'alpha': 1.0, 'random_state': self.random_state},
            'lasso': {'alpha': 1.0, 'random_state': self.random_state},
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': self.random_state,
                'n_jobs': -1
            },
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state,
                'n_jobs': -1,
                'eval_metric': 'mae'
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'max_depth': 3,
                'learning_rate': 0.1,
                'random_state': self.random_state
            }
        }
        
        # Kết hợp tham số
        params = default_params.get(model_type, {}).copy()
        params.update(kwargs)
        
        logger.info(f"Huấn luyện model {model_type} với params: {params}")
        
        # Tạo và huấn luyện model
        start_time = time.time()
        
        if model_type in ['xgboost'] and X_val is not None and y_val is not None:
            # XGBoost có thể dùng early stopping
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.model = model_class(**params)
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
        else:
            self.model = model_class(**params)
            self.model.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        logger.info(f"Hoàn thành huấn luyện trong {train_time:.2f} giây")
        
        self.model_name = model_type
        self.training_history[model_type] = {
            'train_time': train_time,
            'params': params
        }
        
        # Tính feature importance nếu có
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            logger.info(f"\nTop 5 features quan trọng nhất:\n{self.feature_importance.head(5)}")
        elif model_type in ['linear', 'ridge', 'lasso'] and hasattr(self.model, 'coef_'):
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': np.abs(self.model.coef_)
            }).sort_values('importance', ascending=False)
            logger.info(f"\nTop 5 features quan trọng nhất (|coefficient|):\n{self.feature_importance.head(5)}")
        
        return self.model
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Dự báo với mô hình đã huấn luyện
        
        Args:
            X: Features cần dự báo
            
        Returns:
            Mảng các giá trị dự báo
        """
        if self.model is None:
            logger.error("Chưa có model nào được huấn luyện")
            return None
        
        return self.model.predict(X)
    
    def evaluate(self, 
                 y_true: np.ndarray, 
                 y_pred: np.ndarray,
                 prefix: str = '') -> Dict[str, float]:
        """
        Đánh giá mô hình với các metrics
        
        Args:
            y_true: Giá trị thực tế
            y_pred: Giá trị dự báo
            prefix: Tiền tố cho tên metric
            
        Returns:
            Dict các metrics
        """
        metrics = {}
        
        # MAE - Mean Absolute Error
        metrics[f'{prefix}mae'] = mean_absolute_error(y_true, y_pred)
        
        # RMSE - Root Mean Squared Error
        metrics[f'{prefix}rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # R² - R-squared
        metrics[f'{prefix}r2'] = r2_score(y_true, y_pred)
        
        # MAPE - Mean Absolute Percentage Error (tránh chia cho 0)
        mask = y_true != 0
        if mask.sum() > 0:
            metrics[f'{prefix}mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            metrics[f'{prefix}mape'] = np.nan
        
        # Log metrics
        logger.info(f"{prefix}Metrics:")
        logger.info(f"  MAE: {metrics[f'{prefix}mae']:.4f}")
        logger.info(f"  RMSE: {metrics[f'{prefix}rmse']:.4f}")
        logger.info(f"  R²: {metrics[f'{prefix}r2']:.4f}")
        logger.info(f"  MAPE: {metrics[f'{prefix}mape']:.2f}%")
        
        return metrics
    
    def cross_validate(self, 
                       X: pd.DataFrame, 
                       y: pd.Series,
                       model_type: str,
                       cv: int = 5,
                       **kwargs) -> Dict[str, np.ndarray]:
        """
        Cross-validation
        
        Args:
            X: Features
            y: Target
            model_type: Loại mô hình
            cv: Số folds
            **kwargs: Tham số cho mô hình
            
        Returns:
            Dict kết quả CV
        """
        if model_type not in self.SUPPORTED_MODELS:
            logger.error(f"Model {model_type} không được hỗ trợ")
            return {}
        
        model_class = self.SUPPORTED_MODELS[model_type]
        
        # Tạo model với params
        default_params = {
            'linear': {},
            'ridge': {'alpha': 1.0, 'random_state': self.random_state},
            'lasso': {'alpha': 1.0, 'random_state': self.random_state},
            'random_forest': {'random_state': self.random_state, 'n_jobs': -1},
            'xgboost': {'random_state': self.random_state, 'n_jobs': -1}
        }
        
        params = default_params.get(model_type, {}).copy()
        params.update(kwargs)
        
        model = model_class(**params)
        
        # Thực hiện CV
        logger.info(f"Cross-validation với {cv} folds...")
        
        mae_scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
        rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error'))
        r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        
        results = {
            'mae': mae_scores,
            'mae_mean': mae_scores.mean(),
            'mae_std': mae_scores.std(),
            'rmse': rmse_scores,
            'rmse_mean': rmse_scores.mean(),
            'rmse_std': rmse_scores.std(),
            'r2': r2_scores,
            'r2_mean': r2_scores.mean(),
            'r2_std': r2_scores.std()
        }
        
        logger.info(f"CV Results - MAE: {results['mae_mean']:.4f} ± {results['mae_std']:.4f}")
        logger.info(f"            RMSE: {results['rmse_mean']:.4f} ± {results['rmse_std']:.4f}")
        logger.info(f"            R²: {results['r2_mean']:.4f} ± {results['r2_std']:.4f}")
        
        return results
    
    def grid_search(self,
                    X_train: pd.DataFrame,
                    y_train: pd.Series,
                    model_type: str,
                    param_grid: Dict,
                    cv: int = 5,
                    scoring: str = 'neg_mean_absolute_error') -> Dict:
        """
        Grid Search tìm tham số tối ưu
        
        Args:
            X_train: Features train
            y_train: Target train
            model_type: Loại mô hình
            param_grid: Grid tham số
            cv: Số folds
            scoring: Metric đánh giá
            
        Returns:
            Dict kết quả grid search
        """
        if model_type not in self.SUPPORTED_MODELS:
            logger.error(f"Model {model_type} không được hỗ trợ")
            return {}
        
        model_class = self.SUPPORTED_MODELS[model_type]
        
        logger.info(f"Bắt đầu Grid Search với {len(param_grid)} parameters")
        logger.info(f"Parameter grid: {param_grid}")
        
        grid_search = GridSearchCV(
            model_class(random_state=self.random_state),
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        search_time = time.time() - start_time
        
        logger.info(f"Grid Search hoàn thành trong {search_time:.2f} giây")
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best score: {grid_search.best_score_:.4f}")
        
        return {
            'best_estimator': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_,
            'search_time': search_time
        }
    
    def save_model(self, filename: str) -> bool:
        """
        Lưu mô hình đã huấn luyện
        
        Args:
            filename: Tên file (không cần extension)
            
        Returns:
            True nếu thành công
        """
        if self.model is None:
            logger.error("Không có model để lưu")
            return False
        
        if self.model_dir is None:
            logger.error("Chưa指定 thư mục lưu model")
            return False
        
        # Thêm timestamp
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.model_dir / f"{filename}_{self.model_name}_{timestamp}.pkl"
        
        try:
            joblib.dump({
                'model': self.model,
                'model_name': self.model_name,
                'feature_importance': self.feature_importance,
                'training_history': self.training_history,
                'random_state': self.random_state
            }, model_path)
            
            logger.info(f"Đã lưu model tại: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Lỗi khi lưu model: {e}")
            return False
    
    def load_model(self, filepath: Union[str, Path]) -> bool:
        """
        Tải mô hình đã lưu
        
        Args:
            filepath: Đường dẫn file model
            
        Returns:
            True nếu thành công
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.error(f"Không tìm thấy file: {filepath}")
            return False
        
        try:
            data = joblib.load(filepath)
            self.model = data['model']
            self.model_name = data['model_name']
            self.feature_importance = data.get('feature_importance')
            self.training_history = data.get('training_history', {})
            self.random_state = data.get('random_state', self.random_state)
            
            logger.info(f"Đã tải model từ: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Lỗi khi tải model: {e}")
            return False
    
    def run_modeling_pipeline(self,
                              df: pd.DataFrame,
                              target_col: str = 'hg/ha_yield',
                              feature_cols: Optional[List[str]] = None,
                              models_to_try: List[str] = ['linear', 'ridge', 'random_forest', 'xgboost'],
                              test_size: float = 0.2,
                              time_split: bool = True) -> Dict:
        """
        Chạy toàn bộ pipeline huấn luyện và đánh giá
        
        Args:
            df: DataFrame đầu vào
            target_col: Cột mục tiêu
            feature_cols: Danh sách feature
            models_to_try: Các mô hình cần thử
            test_size: Tỷ lệ test
            time_split: Chia theo thời gian
            
        Returns:
            Dict kết quả
        """
        logger.info("=" * 60)
        logger.info("BẮT ĐẦU PIPELINE MÔ HÌNH HÓA")
        logger.info("=" * 60)
        
        # 1. Chuẩn bị dữ liệu
        data = self.prepare_data(
            df, 
            target_col=target_col, 
            feature_cols=feature_cols,
            test_size=test_size,
            time_split=time_split
        )
        
        if not data:
            logger.error("Không thể chuẩn bị dữ liệu")
            return {}
        
        X_train, y_train = data['X_train'], data['y_train']
        X_val, y_val = data['X_val'], data['y_val']
        X_test, y_test = data['X_test'], data['y_test']
        
        # 2. Thử các mô hình
        results = {}
        best_model = None
        best_score = float('inf')
        
        for model_type in models_to_try:
            if model_type not in self.SUPPORTED_MODELS:
                logger.warning(f"Bỏ qua {model_type} (không được hỗ trợ)")
                continue
            
            logger.info(f"\n{'='*40}")
            logger.info(f"Huấn luyện {model_type}")
            logger.info(f"{'='*40}")
            
            # Train
            self.train_model(model_type, X_train, y_train, X_val, y_val)
            
            # Predict
            y_train_pred = self.predict(X_train)
            y_val_pred = self.predict(X_val)
            y_test_pred = self.predict(X_test)
            
            # Evaluate
            train_metrics = self.evaluate(y_train, y_train_pred, 'train_')
            val_metrics = self.evaluate(y_val, y_val_pred, 'val_')
            test_metrics = self.evaluate(y_test, y_test_pred, 'test_')
            
            # Lưu kết quả
            results[model_type] = {
                'model': self.model,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'feature_importance': self.feature_importance,
                'predictions': {
                    'train': y_train_pred,
                    'val': y_val_pred,
                    'test': y_test_pred
                }
            }
            
            # So sánh
            val_mae = val_metrics['val_mae']
            if val_mae < best_score:
                best_score = val_mae
                best_model = model_type
                self.model = results[model_type]['model']
                self.model_name = model_type
        
        # 3. Tổng hợp kết quả
        comparison = pd.DataFrame({
            model: {
                'Train MAE': results[model]['train_metrics']['train_mae'],
                'Val MAE': results[model]['val_metrics']['val_mae'],
                'Test MAE': results[model]['test_metrics']['test_mae'],
                'Test RMSE': results[model]['test_metrics']['test_rmse'],
                'Test R²': results[model]['test_metrics']['test_r2']
            }
            for model in results
        }).T
        
        logger.info("\n" + "="*60)
        logger.info("SO SÁNH CÁC MÔ HÌNH")
        logger.info("="*60)
        logger.info(f"\n{comparison.to_string()}")
        logger.info(f"\nBest model: {best_model} (Val MAE: {best_score:.4f})")
        
        return {
            'results': results,
            'comparison': comparison,
            'best_model': best_model,
            'best_score': best_score,
            'data': data
        }
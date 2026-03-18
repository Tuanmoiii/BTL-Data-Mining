"""
builder.py - Feature engineering cho dự báo năng suất cây trồng
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import logging
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class FeatureBuilder:
    """
    Class xây dựng và thiết kế đặc trưng
    """
    
    def __init__(self, random_state: int = 42):
        """
        Khởi tạo FeatureBuilder
        
        Args:
            random_state: Seed cho random
        """
        self.random_state = random_state
        self.feature_names = []
        self.poly_features = None
        
    def create_lag_features(self, 
                           df: pd.DataFrame, 
                           group_cols: List[str] = ['Area', 'Item'],
                           target_col: str = 'hg/ha_yield',
                           lags: List[int] = [1, 2, 3]) -> pd.DataFrame:
        """
        Tạo lag features (năng suất năm trước)
        
        Args:
            df: DataFrame đầu vào
            group_cols: Các cột để group (vùng, loại cây)
            target_col: Cột mục tiêu cần tạo lag
            lags: Danh sách các lag (1 = năm trước, 2 = 2 năm trước)
            
        Returns:
            DataFrame với các lag features
        """
        df_feat = df.copy()
        
        # Sắp xếp theo năm
        df_feat = df_feat.sort_values(['Year'] + group_cols)
        
        for lag in lags:
            lag_col = f'{target_col}_lag_{lag}'
            
            # Tạo lag cho từng nhóm
            df_feat[lag_col] = df_feat.groupby(group_cols)[target_col].shift(lag)
            
            logger.info(f"Đã tạo {lag_col}: {df_feat[lag_col].notna().sum()} giá trị")
            self.feature_names.append(lag_col)
        
        return df_feat
    
    def create_rolling_features(self,
                               df: pd.DataFrame,
                               group_cols: List[str] = ['Area', 'Item'],
                               target_col: str = 'hg/ha_yield',
                               windows: List[int] = [2, 3],
                               agg_funcs: List[str] = ['mean', 'std']) -> pd.DataFrame:
        """
        Tạo rolling statistics (trung bình động, độ lệch chuẩn)
        
        Args:
            df: DataFrame đầu vào
            group_cols: Các cột để group
            target_col: Cột mục tiêu
            windows: Kích thước cửa sổ rolling
            agg_funcs: Các hàm tổng hợp
            
        Returns:
            DataFrame với rolling features
        """
        df_feat = df.copy()
        
        # Sắp xếp theo năm
        df_feat = df_feat.sort_values(['Year'] + group_cols)
        
        for window in windows:
            for func in agg_funcs:
                roll_col = f'{target_col}_roll_{window}_{func}'
                
                # Tính rolling statistics cho từng nhóm
                if func == 'mean':
                    df_feat[roll_col] = df_feat.groupby(group_cols)[target_col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )
                elif func == 'std':
                    df_feat[roll_col] = df_feat.groupby(group_cols)[target_col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).std()
                    )
                elif func == 'min':
                    df_feat[roll_col] = df_feat.groupby(group_cols)[target_col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).min()
                    )
                elif func == 'max':
                    df_feat[roll_col] = df_feat.groupby(group_cols)[target_col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).max()
                    )
                
                self.feature_names.append(roll_col)
                logger.info(f"Đã tạo {roll_col}")
        
        return df_feat
    
    def create_interaction_features(self,
                                   df: pd.DataFrame,
                                   feature_pairs: List[Tuple[str, str]],
                                   operation: str = 'multiply') -> pd.DataFrame:
        """
        Tạo interaction features giữa các cặp biến
        
        Args:
            df: DataFrame đầu vào
            feature_pairs: Danh sách cặp (col1, col2)
            operation: Phép toán ('multiply', 'divide', 'add', 'subtract')
            
        Returns:
            DataFrame với interaction features
        """
        df_feat = df.copy()
        
        for col1, col2 in feature_pairs:
            if col1 not in df_feat.columns or col2 not in df_feat.columns:
                logger.warning(f"Bỏ qua cặp ({col1}, {col2}) do thiếu cột")
                continue
            
            if operation == 'multiply':
                new_col = f'{col1}_x_{col2}'
                df_feat[new_col] = df_feat[col1] * df_feat[col2]
            elif operation == 'divide':
                new_col = f'{col1}_div_{col2}'
                # Tránh chia cho 0
                df_feat[new_col] = df_feat[col1] / (df_feat[col2] + 1e-8)
            elif operation == 'add':
                new_col = f'{col1}_plus_{col2}'
                df_feat[new_col] = df_feat[col1] + df_feat[col2]
            elif operation == 'subtract':
                new_col = f'{col1}_minus_{col2}'
                df_feat[new_col] = df_feat[col1] - df_feat[col2]
            
            self.feature_names.append(new_col)
            logger.info(f"Đã tạo {new_col}")
        
        return df_feat
    
    def create_polynomial_features(self,
                                  df: pd.DataFrame,
                                  feature_cols: List[str],
                                  degree: int = 2,
                                  include_bias: bool = False) -> pd.DataFrame:
        """
        Tạo polynomial features
        
        Args:
            df: DataFrame đầu vào
            feature_cols: Danh sách cột để tạo polynomial
            degree: Bậc của đa thức
            include_bias: Có bao gồm bias (cột 1) không
            
        Returns:
            DataFrame với polynomial features
        """
        df_feat = df.copy()
        
        # Lấy dữ liệu
        X = df_feat[feature_cols].values
        
        # Tạo polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
        X_poly = poly.fit_transform(X)
        
        # Tạo tên cột
        poly_names = []
        for i, name in enumerate(feature_cols):
            poly_names.append(name)  # Bậc 1
            for j in range(i, len(feature_cols)):
                if degree >= 2:
                    poly_names.append(f"{name}_x_{feature_cols[j]}")  # Bậc 2
        
        # Thêm vào dataframe
        for i, col_name in enumerate(poly_names):
            if i < X_poly.shape[1]:
                poly_col = f'poly_{col_name}'
                df_feat[poly_col] = X_poly[:, i]
                self.feature_names.append(poly_col)
        
        self.poly_features = poly
        logger.info(f"Đã tạo {X_poly.shape[1]} polynomial features bậc {degree}")
        
        return df_feat
    
    def create_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo các đặc trưng thời tiết đặc thù cho cây trồng
        
        Args:
            df: DataFrame với các cột avg_temp, average_rain_fall_mm_per_year
            
        Returns:
            DataFrame với weather features
        """
        df_feat = df.copy()
        
        # Kiểm tra các cột cần thiết
        has_temp = 'avg_temp' in df_feat.columns
        has_rain = 'average_rain_fall_mm_per_year' in df_feat.columns
        
        if has_temp and has_rain:
            # Nhiệt độ * lượng mưa (tương tác)
            df_feat['temp_rain_interaction'] = df_feat['avg_temp'] * df_feat['average_rain_fall_mm_per_year']
            self.feature_names.append('temp_rain_interaction')
            
            # Nhiệt độ bình phương (phi tuyến)
            df_feat['temp_squared'] = df_feat['avg_temp'] ** 2
            self.feature_names.append('temp_squared')
            
            # Lượng mưa bình phương
            df_feat['rain_squared'] = df_feat['average_rain_fall_mm_per_year'] ** 2
            self.feature_names.append('rain_squared')
            
            logger.info("Đã tạo weather interaction features")
        
        if has_temp:
            # Phân loại nhiệt độ
            df_feat['temp_category'] = pd.cut(
                df_feat['avg_temp'],
                bins=[-float('inf'), 10, 20, 30, float('inf')],
                labels=['cold', 'mild', 'warm', 'hot']
            )
            logger.info("Đã tạo temp_category")
        
        if has_rain:
            # Phân loại lượng mưa
            df_feat['rain_category'] = pd.cut(
                df_feat['average_rain_fall_mm_per_year'],
                bins=[-float('inf'), 500, 1000, 1500, float('inf')],
                labels=['low', 'medium', 'high', 'very_high']
            )
            logger.info("Đã tạo rain_category")
        
        return df_feat
    
    def create_yield_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo các tỷ lệ liên quan đến năng suất
        
        Args:
            df: DataFrame đầu vào
            
        Returns:
            DataFrame với ratio features
        """
        df_feat = df.copy()
        
        # Năng suất trên lượng mưa (hiệu quả sử dụng nước)
        if 'hg/ha_yield' in df_feat.columns and 'average_rain_fall_mm_per_year' in df_feat.columns:
            df_feat['yield_per_rain'] = df_feat['hg/ha_yield'] / (df_feat['average_rain_fall_mm_per_year'] + 1e-8)
            self.feature_names.append('yield_per_rain')
            logger.info("Đã tạo yield_per_rain")
        
        # Năng suất trên thuốc trừ sâu
        if 'hg/ha_yield' in df_feat.columns and 'pesticides_tonnes' in df_feat.columns:
            df_feat['yield_per_pesticide'] = df_feat['hg/ha_yield'] / (df_feat['pesticides_tonnes'] + 1e-8)
            self.feature_names.append('yield_per_pesticide')
            logger.info("Đã tạo yield_per_pesticide")
        
        return df_feat
    
    def create_time_features(self, df: pd.DataFrame, year_col: str = 'Year') -> pd.DataFrame:
        """
        Tạo đặc trưng thời gian
        
        Args:
            df: DataFrame đầu vào
            year_col: Tên cột năm
            
        Returns:
            DataFrame với time features
        """
        df_feat = df.copy()
        
        if year_col in df_feat.columns:
            # Thập kỷ
            df_feat['decade'] = (df_feat[year_col] // 10) * 10
            self.feature_names.append('decade')
            
            # Khoảng cách từ năm hiện tại đến năm gần nhất
            latest_year = df_feat[year_col].max()
            df_feat['years_ago'] = latest_year - df_feat[year_col]
            self.feature_names.append('years_ago')
            
            logger.info("Đã tạo time features")
        
        return df_feat
    
    def build_all_features(self, 
                          df: pd.DataFrame,
                          target_col: str = 'hg/ha_yield',
                          create_lags: bool = True,
                          create_rolling: bool = True,
                          create_interactions: bool = True,
                          create_weather: bool = True,
                          create_ratios: bool = True,
                          create_time: bool = True) -> pd.DataFrame:
        """
        Xây dựng tất cả đặc trưng
        
        Args:
            df: DataFrame đầu vào
            target_col: Cột mục tiêu
            create_lags: Tạo lag features?
            create_rolling: Tạo rolling features?
            create_interactions: Tạo interaction features?
            create_weather: Tạo weather features?
            create_ratios: Tạo ratio features?
            create_time: Tạo time features?
            
        Returns:
            DataFrame với đầy đủ features
        """
        logger.info("=" * 60)
        logger.info("BẮT ĐẦU XÂY DỰNG FEATURES")
        logger.info("=" * 60)
        
        df_feat = df.copy()
        self.feature_names = []
        
        # 1. Lag features
        if create_lags and target_col in df_feat.columns:
            df_feat = self.create_lag_features(df_feat, target_col=target_col, lags=[1, 2, 3])
        
        # 2. Rolling features
        if create_rolling and target_col in df_feat.columns:
            df_feat = self.create_rolling_features(df_feat, target_col=target_col, windows=[2, 3])
        
        # 3. Weather features
        if create_weather:
            df_feat = self.create_weather_features(df_feat)
        
        # 4. Ratio features
        if create_ratios:
            df_feat = self.create_yield_ratio_features(df_feat)
        
        # 5. Interaction features
        if create_interactions:
            # Các cặp tương tác phổ biến
            pairs = []
            if 'avg_temp' in df_feat.columns and 'average_rain_fall_mm_per_year' in df_feat.columns:
                pairs.append(('avg_temp', 'average_rain_fall_mm_per_year'))
            if 'avg_temp' in df_feat.columns and 'pesticides_tonnes' in df_feat.columns:
                pairs.append(('avg_temp', 'pesticides_tonnes'))
            if 'average_rain_fall_mm_per_year' in df_feat.columns and 'pesticides_tonnes' in df_feat.columns:
                pairs.append(('average_rain_fall_mm_per_year', 'pesticides_tonnes'))
            
            if pairs:
                df_feat = self.create_interaction_features(df_feat, pairs, operation='multiply')
        
        # 6. Time features
        if create_time:
            df_feat = self.create_time_features(df_feat)
        
        logger.info("=" * 60)
        logger.info(f"HOÀN THÀNH: Đã tạo {len(self.feature_names)} features mới")
        logger.info(f"Tổng số cột: {df_feat.shape[1]}")
        logger.info("=" * 60)
        
        return df_feat
    
    def get_feature_names(self) -> List[str]:
        """Lấy danh sách tên các features đã tạo"""
        return self.feature_names
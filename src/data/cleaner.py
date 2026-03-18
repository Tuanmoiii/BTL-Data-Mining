"""
cleaner.py - Xử lý dữ liệu thiếu, outlier, encoding, scaling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

logger = logging.getLogger(__name__)

class DataCleaner:
    """
    Class xử lý làm sạch dữ liệu: missing values, outliers, encoding, scaling
    """
    
    def __init__(self, random_state: int = 42):
        """
        Khởi tạo DataCleaner
        
        Args:
            random_state: Seed cho random
        """
        self.random_state = random_state
        self.imputers = {}
        self.scalers = {}
        self.encoders = {}
        self.outlier_thresholds = {}
        
    def detect_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Phát hiện và báo cáo giá trị thiếu
        
        Args:
            df: DataFrame cần kiểm tra
            
        Returns:
            DataFrame báo cáo missing values
        """
        missing_count = df.isnull().sum()
        missing_percent = (missing_count / len(df)) * 100
        
        missing_df = pd.DataFrame({
            'Cột': missing_count.index,
            'Số lượng thiếu': missing_count.values,
            'Tỷ lệ (%)': missing_percent.values
        })
        missing_df = missing_df[missing_df['Số lượng thiếu'] > 0].sort_values('Tỷ lệ (%)', ascending=False)
        
        if len(missing_df) > 0:
            logger.warning(f"Phát hiện {len(missing_df)} cột có giá trị thiếu")
            logger.info(f"\n{missing_df.to_string(index=False)}")
        else:
            logger.info("Không có giá trị thiếu nào")
        
        return missing_df
    
    def handle_missing_values(self, 
                              df: pd.DataFrame, 
                              strategy: str = 'mean',
                              columns: Optional[List[str]] = None,
                              fill_value: Optional[Any] = None) -> pd.DataFrame:
        """
        Xử lý giá trị thiếu
        
        Args:
            df: DataFrame cần xử lý
            strategy: Chiến lược xử lý ('mean', 'median', 'most_frequent', 'constant', 'drop')
            columns: Danh sách cột cần xử lý (None = tất cả)
            fill_value: Giá trị điền nếu strategy='constant'
            
        Returns:
            DataFrame đã xử lý
        """
        df_clean = df.copy()
        
        if columns is None:
            columns = df.columns[df.isnull().any()].tolist()
        
        if not columns:
            logger.info("Không có cột nào cần xử lý missing")
            return df_clean
        
        logger.info(f"Xử lý missing values cho {len(columns)} cột với strategy='{strategy}'")
        
        if strategy == 'drop':
            # Xóa dòng có missing
            before = len(df_clean)
            df_clean = df_clean.dropna(subset=columns)
            after = len(df_clean)
            logger.info(f"Đã xóa {before - after} dòng có giá trị thiếu")
        
        elif strategy in ['mean', 'median', 'most_frequent', 'constant']:
            for col in columns:
                if col not in df_clean.columns:
                    continue
                
                if strategy == 'constant' and fill_value is not None:
                    df_clean[col].fillna(fill_value, inplace=True)
                    logger.info(f"  - Cột {col}: điền giá trị {fill_value}")
                else:
                    # Tạo imputer
                    imputer = SimpleImputer(strategy=strategy)
                    df_clean[col] = imputer.fit_transform(df_clean[[col]]).ravel()
                    self.imputers[col] = imputer
                    logger.info(f"  - Cột {col}: điền bằng {strategy}")
        
        else:
            logger.error(f"Strategy '{strategy}' không được hỗ trợ")
        
        return df_clean
    
    def detect_outliers_iqr(self, df: pd.DataFrame, columns: Optional[List[str]] = None, 
                            multiplier: float = 1.5) -> Dict[str, Dict]:
        """
        Phát hiện outlier bằng phương pháp IQR
        
        Args:
            df: DataFrame cần kiểm tra
            columns: Danh sách cột số cần kiểm tra
            multiplier: Hệ số nhân cho IQR (1.5 cho outlier nhẹ, 3 cho outlier mạnh)
            
        Returns:
            Dict báo cáo outlier cho từng cột
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_report = {}
        
        for col in columns:
            if col not in df.columns:
                continue
            
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
            if len(outliers) > 0:
                outlier_report[col] = {
                    'n_outliers': len(outliers),
                    'percent': (len(outliers) / len(df)) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'outlier_indices': outliers.index.tolist()[:10]  # Lấy 10 index đầu
                }
                
                logger.info(f"Cột {col}: {len(outliers)} outliers ({outlier_report[col]['percent']:.2f}%)")
        
        return outlier_report
    
    def handle_outliers(self, 
                        df: pd.DataFrame, 
                        method: str = 'cap',
                        columns: Optional[List[str]] = None,
                        multiplier: float = 1.5) -> pd.DataFrame:
        """
        Xử lý outlier
        
        Args:
            df: DataFrame cần xử lý
            method: Phương pháp ('cap' - chặn, 'remove' - xóa)
            columns: Danh sách cột cần xử lý
            multiplier: Hệ số IQR
            
        Returns:
            DataFrame đã xử lý
        """
        df_clean = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        logger.info(f"Xử lý outliers với method='{method}'")
        
        for col in columns:
            if col not in df_clean.columns:
                continue
            
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            n_outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
            
            if n_outliers == 0:
                continue
            
            if method == 'cap':
                # Chặn giá trị
                df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
                logger.info(f"  - Cột {col}: đã chặn {n_outliers} outliers")
                self.outlier_thresholds[col] = {'lower': lower_bound, 'upper': upper_bound}
                
            elif method == 'remove':
                # Xóa dòng có outlier
                before = len(df_clean)
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
                after = len(df_clean)
                logger.info(f"  - Cột {col}: đã xóa {before - after} dòng có outlier")
        
        return df_clean
    
    def encode_categorical(self, 
                          df: pd.DataFrame, 
                          columns: Optional[List[str]] = None,
                          method: str = 'label') -> pd.DataFrame:
        """
        Mã hóa biến phân loại
        
        Args:
            df: DataFrame cần mã hóa
            columns: Danh sách cột phân loại
            method: Phương pháp ('label', 'onehot')
            
        Returns:
            DataFrame đã mã hóa
        """
        df_encoded = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=['object']).columns.tolist()
        
        if not columns:
            logger.info("Không có cột phân loại nào để mã hóa")
            return df_encoded
        
        logger.info(f"Mã hóa {len(columns)} cột phân loại với method='{method}'")
        
        if method == 'label':
            for col in columns:
                if col not in df_encoded.columns:
                    continue
                
                encoder = LabelEncoder()
                df_encoded[col + '_encoded'] = encoder.fit_transform(df_encoded[col].astype(str))
                self.encoders[col] = encoder
                
                # Giữ lại cột gốc hoặc xóa đi
                # df_encoded.drop(col, axis=1, inplace=True)
                
                n_unique = len(encoder.classes_)
                logger.info(f"  - Cột {col}: {n_unique} giá trị unique")
        
        elif method == 'onehot':
            for col in columns:
                if col not in df_encoded.columns:
                    continue
                
                # Tạo one-hot encoding
                dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=False)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                
                # Xóa cột gốc
                df_encoded.drop(col, axis=1, inplace=True)
                
                logger.info(f"  - Cột {col}: tạo {dummies.shape[1]} cột one-hot")
        
        return df_encoded
    
    def scale_features(self, 
                       df: pd.DataFrame, 
                       columns: Optional[List[str]] = None,
                       method: str = 'standard') -> pd.DataFrame:
        """
        Chuẩn hóa đặc trưng
        
        Args:
            df: DataFrame cần chuẩn hóa
            columns: Danh sách cột số cần chuẩn hóa
            method: Phương pháp ('standard', 'minmax')
            
        Returns:
            DataFrame đã chuẩn hóa
        """
        df_scaled = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not columns:
            logger.info("Không có cột số nào để chuẩn hóa")
            return df_scaled
        
        logger.info(f"Chuẩn hóa {len(columns)} cột với method='{method}'")
        
        # Tạo tên cột mới
        suffix = '_scaled'
        scaled_columns = [col + suffix for col in columns]
        
        if method == 'standard':
            scaler = StandardScaler()
            df_scaled[scaled_columns] = scaler.fit_transform(df_scaled[columns])
            self.scalers['standard'] = scaler
            
        elif method == 'minmax':
            scaler = MinMaxScaler()
            df_scaled[scaled_columns] = scaler.fit_transform(df_scaled[columns])
            self.scalers['minmax'] = scaler
        
        # Log thông tin
        for i, col in enumerate(columns):
            logger.info(f"  - {col} -> {scaled_columns[i]}")
        
        return df_scaled
    
    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Xóa các dòng trùng lặp
        
        Args:
            df: DataFrame cần xử lý
            subset: Các cột để xét trùng lặp
            
        Returns:
            DataFrame đã xóa duplicate
        """
        before = len(df)
        df_clean = df.drop_duplicates(subset=subset)
        after = len(df_clean)
        
        if before > after:
            logger.info(f"Đã xóa {before - after} dòng trùng lặp")
        else:
            logger.info("Không có dòng trùng lặp")
        
        return df_clean
    
    def clean_pipeline(self, 
                       df: pd.DataFrame,
                       handle_missing: bool = True,
                       missing_strategy: str = 'mean',
                       handle_outliers: bool = True,
                       outlier_method: str = 'cap',
                       encode_cat: bool = True,
                       encode_method: str = 'label',
                       scale: bool = True,
                       scale_method: str = 'standard',
                       remove_dupes: bool = True) -> pd.DataFrame:
        """
        Pipeline làm sạch dữ liệu hoàn chỉnh
        
        Args:
            df: DataFrame đầu vào
            handle_missing: Xử lý missing?
            missing_strategy: Chiến lược xử lý missing
            handle_outliers: Xử lý outlier?
            outlier_method: Phương pháp xử lý outlier
            encode_cat: Mã hóa biến phân loại?
            encode_method: Phương pháp mã hóa
            scale: Chuẩn hóa?
            scale_method: Phương pháp chuẩn hóa
            remove_dupes: Xóa duplicate?
            
        Returns:
            DataFrame đã làm sạch
        """
        logger.info("=" * 60)
        logger.info("BẮT ĐẦU PIPELINE LÀM SẠCH DỮ LIỆU")
        logger.info("=" * 60)
        
        df_clean = df.copy()
        
        # 1. Xóa duplicate
        if remove_dupes:
            df_clean = self.remove_duplicates(df_clean)
        
        # 2. Phát hiện missing
        missing_df = self.detect_missing_values(df_clean)
        
        # 3. Xử lý missing
        if handle_missing and len(missing_df) > 0:
            df_clean = self.handle_missing_values(df_clean, strategy=missing_strategy)
        
        # 4. Phát hiện và xử lý outlier
        if handle_outliers:
            outlier_report = self.detect_outliers_iqr(df_clean)
            if outlier_report:
                df_clean = self.handle_outliers(df_clean, method=outlier_method)
        
        # 5. Mã hóa categorical
        if encode_cat:
            cat_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
            if cat_cols:
                df_clean = self.encode_categorical(df_clean, columns=cat_cols, method=encode_method)
        
        # 6. Chuẩn hóa
        if scale:
            num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
            # Loại bỏ cột target nếu có
            if 'hg/ha_yield' in num_cols:
                num_cols.remove('hg/ha_yield')
            if num_cols:
                df_clean = self.scale_features(df_clean, columns=num_cols, method=scale_method)
        
        logger.info("=" * 60)
        logger.info(f"HOÀN THÀNH: {df.shape} -> {df_clean.shape}")
        logger.info("=" * 60)
        
        return df_clean
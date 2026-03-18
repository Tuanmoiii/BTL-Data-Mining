"""
loader.py - Đọc dữ liệu và kiểm tra schema
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Class đọc dữ liệu từ các file CSV và kiểm tra schema
    """
    
    # Định nghĩa schema cho từng file
    SCHEMA = {
        'yield.csv': {
            'required_columns': ['Area', 'Item', 'Year', 'hg/ha_yield'],
            'dtypes': {
                'Area': 'object',
                'Item': 'object', 
                'Year': 'int64',
                'hg/ha_yield': 'float64'
            }
        },
        'rainfall.csv': {
            'required_columns': ['Area', 'Item', 'Year', 'average_rain_fall_mm_per_year'],
            'dtypes': {
                'Area': 'object',
                'Item': 'object',
                'Year': 'int64',
                'average_rain_fall_mm_per_year': 'float64'
            }
        },
        'pesticides.csv': {
            'required_columns': ['Area', 'Item', 'Year', 'pesticides_tonnes'],
            'dtypes': {
                'Area': 'object',
                'Item': 'object',
                'Year': 'int64',
                'pesticides_tonnes': 'float64'
            }
        },
        'temp.csv': {
            'required_columns': ['Area', 'Item', 'Year', 'avg_temp'],
            'dtypes': {
                'Area': 'object',
                'Item': 'object',
                'Year': 'int64',
                'avg_temp': 'float64'
            }
        },
        'yield_df.csv': {
            'required_columns': ['Area', 'Item', 'Year', 'hg/ha_yield', 
                                  'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp'],
            'dtypes': {
                'Area': 'object',
                'Item': 'object',
                'Year': 'int64',
                'hg/ha_yield': 'float64',
                'average_rain_fall_mm_per_year': 'float64',
                'pesticides_tonnes': 'float64',
                'avg_temp': 'float64'
            }
        }
    }
    
    def __init__(self, data_dir: Union[str, Path], raw_subdir: str = 'raw', processed_subdir: str = 'processed'):
        """
        Khởi tạo DataLoader
        
        Args:
            data_dir: Đường dẫn đến thư mục data
            raw_subdir: Tên thư mục con chứa dữ liệu raw
            processed_subdir: Tên thư mục con chứa dữ liệu đã xử lý
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / raw_subdir
        self.processed_dir = self.data_dir / processed_subdir
        
        # Kiểm tra thư mục tồn tại
        if not self.raw_dir.exists():
            logger.warning(f"Thư mục raw không tồn tại: {self.raw_dir}")
            self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.processed_dir.exists():
            logger.info(f"Tạo thư mục processed: {self.processed_dir}")
            self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def list_raw_files(self) -> List[str]:
        """Liệt kê các file CSV trong thư mục raw"""
        return [f.name for f in self.raw_dir.glob('*.csv')]
    
    def list_processed_files(self) -> List[str]:
        """Liệt kê các file CSV trong thư mục processed"""
        return [f.name for f in self.processed_dir.glob('*.csv')]
    
    def load_csv(self, filename: str, from_raw: bool = True, **kwargs) -> Optional[pd.DataFrame]:
        """
        Đọc file CSV
        
        Args:
            filename: Tên file CSV
            from_raw: True nếu đọc từ raw/, False nếu đọc từ processed/
            **kwargs: Các tham số bổ sung cho pd.read_csv
            
        Returns:
            DataFrame hoặc None nếu lỗi
        """
        file_dir = self.raw_dir if from_raw else self.processed_dir
        file_path = file_dir / filename
        
        if not file_path.exists():
            logger.error(f"File không tồn tại: {file_path}")
            return None
        
        try:
            df = pd.read_csv(file_path, **kwargs)
            logger.info(f"Đã đọc file {filename}: {df.shape[0]} dòng, {df.shape[1]} cột")
            return df
        except Exception as e:
            logger.error(f"Lỗi đọc file {filename}: {str(e)}")
            return None
    
    def save_csv(self, df: pd.DataFrame, filename: str, to_raw: bool = False, **kwargs) -> bool:
        """
        Lưu DataFrame thành file CSV
        
        Args:
            df: DataFrame cần lưu
            filename: Tên file CSV
            to_raw: True nếu lưu vào raw/, False nếu lưu vào processed/
            **kwargs: Các tham số bổ sung cho df.to_csv
            
        Returns:
            True nếu thành công, False nếu lỗi
        """
        file_dir = self.raw_dir if to_raw else self.processed_dir
        file_path = file_dir / filename
        
        try:
            df.to_csv(file_path, index=False, **kwargs)
            logger.info(f"Đã lưu file {filename}: {df.shape[0]} dòng, {df.shape[1]} cột")
            return True
        except Exception as e:
            logger.error(f"Lỗi lưu file {filename}: {str(e)}")
            return False
    
    def validate_schema(self, df: pd.DataFrame, filename: str) -> Tuple[bool, List[str]]:
        """
        Kiểm tra schema của DataFrame có khớp với định nghĩa không
        
        Args:
            df: DataFrame cần kiểm tra
            filename: Tên file để tra schema
            
        Returns:
            (is_valid, errors) - True nếu hợp lệ, danh sách lỗi
        """
        if filename not in self.SCHEMA:
            logger.warning(f"Không có định nghĩa schema cho file {filename}")
            return True, []
        
        schema = self.SCHEMA[filename]
        errors = []
        
        # Kiểm tra các cột bắt buộc
        for col in schema['required_columns']:
            if col not in df.columns:
                errors.append(f"Thiếu cột bắt buộc: {col}")
        
        # Kiểm tra kiểu dữ liệu
        for col, dtype in schema['dtypes'].items():
            if col in df.columns:
                actual_dtype = str(df[col].dtype)
                expected_dtype = dtype
                
                # Kiểm tra kiểu dữ liệu (chấp nhận một số biến thể)
                if 'int' in expected_dtype and 'int' not in actual_dtype:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        errors.append(f"Cột {col} phải là kiểu số, hiện tại: {actual_dtype}")
                elif 'float' in expected_dtype and 'float' not in actual_dtype:
                    if not pd.api.types.is_float_dtype(df[col]):
                        errors.append(f"Cột {col} phải là kiểu float, hiện tại: {actual_dtype}")
        
        is_valid = len(errors) == 0
        if is_valid:
            logger.info(f"Schema hợp lệ cho file {filename}")
        else:
            logger.error(f"Schema không hợp lệ cho file {filename}: {errors}")
        
        return is_valid, errors
    
    def load_all_raw(self) -> Dict[str, pd.DataFrame]:
        """
        Đọc tất cả các file raw và trả về dictionary
        
        Returns:
            Dict với key là tên file, value là DataFrame
        """
        data_dict = {}
        for filename in self.list_raw_files():
            df = self.load_csv(filename, from_raw=True)
            if df is not None:
                data_dict[filename] = df
                
                # Kiểm tra schema
                self.validate_schema(df, filename)
        
        logger.info(f"Đã đọc {len(data_dict)}/{len(self.list_raw_files())} file raw")
        return data_dict
    
    def merge_crop_data(self, 
                        yield_df: pd.DataFrame, 
                        rainfall_df: pd.DataFrame, 
                        pesticides_df: pd.DataFrame, 
                        temp_df: pd.DataFrame) -> pd.DataFrame:
        """
        Gộp các bảng dữ liệu lại với nhau dựa trên (Area, Item, Year)
        
        Args:
            yield_df: DataFrame năng suất
            rainfall_df: DataFrame lượng mưa
            pesticides_df: DataFrame thuốc trừ sâu
            temp_df: DataFrame nhiệt độ
            
        Returns:
            DataFrame đã gộp
        """
        logger.info("Bắt đầu gộp dữ liệu...")
        
        # Chuẩn bị các DataFrame
        dfs = []
        
        # Yield
        if 'hg/ha_yield' in yield_df.columns:
            df_main = yield_df[['Area', 'Item', 'Year', 'hg/ha_yield']].copy()
        else:
            df_main = yield_df.copy()
        
        # Merge với rainfall
        if rainfall_df is not None:
            rain_cols = ['Area', 'Item', 'Year']
            if 'average_rain_fall_mm_per_year' in rainfall_df.columns:
                rain_cols.append('average_rain_fall_mm_per_year')
            df_main = pd.merge(
                df_main, 
                rainfall_df[rain_cols], 
                on=['Area', 'Item', 'Year'], 
                how='left'
            )
            logger.info(f"Đã merge rainfall: {df_main.shape}")
        
        # Merge với pesticides
        if pesticides_df is not None:
            pest_cols = ['Area', 'Item', 'Year']
            if 'pesticides_tonnes' in pesticides_df.columns:
                pest_cols.append('pesticides_tonnes')
            df_main = pd.merge(
                df_main, 
                pesticides_df[pest_cols], 
                on=['Area', 'Item', 'Year'], 
                how='left'
            )
            logger.info(f"Đã merge pesticides: {df_main.shape}")
        
        # Merge với temp
        if temp_df is not None:
            temp_cols = ['Area', 'Item', 'Year']
            if 'avg_temp' in temp_df.columns:
                temp_cols.append('avg_temp')
            df_main = pd.merge(
                df_main, 
                temp_df[temp_cols], 
                on=['Area', 'Item', 'Year'], 
                how='left'
            )
            logger.info(f"Đã merge temp: {df_main.shape}")
        
        logger.info(f"Hoàn thành gộp dữ liệu: {df_main.shape[0]} dòng, {df_main.shape[1]} cột")
        return df_main
"""
association.py - Khai phá luật kết hợp (Association Rule Mining)
"""

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import logging
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class AssociationMiner:
    """
    Class khai phá luật kết hợp từ dữ liệu cây trồng
    """
    
    def __init__(self, random_state: int = 42):
        """
        Khởi tạo AssociationMiner
        
        Args:
            random_state: Seed cho random
        """
        self.random_state = random_state
        self.frequent_itemsets = None
        self.rules = None
        self.transaction_encoder = None
        
    def discretize_numeric(self, 
                           df: pd.DataFrame, 
                           column: str,
                           bins: int = 3,
                           labels: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Rời rạc hóa cột số để sử dụng cho luật kết hợp
        
        Args:
            df: DataFrame đầu vào
            column: Tên cột cần rời rạc hóa
            bins: Số lượng bins hoặc list các ngưỡng
            labels: Nhãn cho các bins
            
        Returns:
            DataFrame với cột đã được rời rạc hóa
        """
        df_disc = df.copy()
        
        if isinstance(bins, int):
            # Tự động chia bins dựa trên quantile
            if labels is None:
                if bins == 2:
                    labels = ['Low', 'High']
                elif bins == 3:
                    labels = ['Low', 'Medium', 'High']
                elif bins == 4:
                    labels = ['Very Low', 'Low', 'High', 'Very High']
                else:
                    labels = [f'Level_{i+1}' for i in range(bins)]
            
            # Dùng quantile để chia đều
            df_disc[column + '_bin'] = pd.qcut(
                df_disc[column], 
                q=bins, 
                labels=labels, 
                duplicates='drop'
            )
        else:
            # Dùng bins cố định
            df_disc[column + '_bin'] = pd.cut(
                df_disc[column], 
                bins=bins, 
                labels=labels, 
                include_lowest=True
            )
        
        # Thêm tiền tố để phân biệt
        df_disc[column + '_bin'] = column + '=' + df_disc[column + '_bin'].astype(str)
        
        n_unique = df_disc[column + '_bin'].nunique()
        logger.info(f"Đã rời rạc hóa cột {column} thành {n_unique} categories")
        
        return df_disc
    
    def prepare_transactions(self, 
                            df: pd.DataFrame,
                            id_columns: List[str] = ['Area', 'Item', 'Year'],
                            feature_columns: Optional[List[str]] = None,
                            target_col: str = 'hg/ha_yield',
                            discretize_target: bool = True,
                            yield_thresholds: Optional[List[float]] = None) -> List[List[str]]:
        """
        Chuẩn bị dữ liệu dạng transactions cho Apriori
        
        Args:
            df: DataFrame đầu vào
            id_columns: Các cột định danh (không dùng làm item)
            feature_columns: Các cột đặc trưng để tạo items
            target_col: Cột mục tiêu (năng suất)
            discretize_target: Có rời rạc hóa target không
            yield_thresholds: Ngưỡng để phân loại năng suất
            
        Returns:
            List các transactions (mỗi transaction là list các items)
        """
        df_trans = df.copy()
        
        # Nếu không chỉ định feature_columns, lấy tất cả cột trừ id_columns và target
        if feature_columns is None:
            feature_columns = [col for col in df_trans.columns 
                              if col not in id_columns + [target_col]]
        
        # Rời rạc hóa target nếu cần
        if discretize_target and target_col in df_trans.columns:
            if yield_thresholds is None:
                # Phân loại năng suất: thấp, trung bình, cao dựa trên quantile
                low_thresh = df_trans[target_col].quantile(0.33)
                high_thresh = df_trans[target_col].quantile(0.67)
                
                df_trans['yield_level'] = pd.cut(
                    df_trans[target_col],
                    bins=[-float('inf'), low_thresh, high_thresh, float('inf')],
                    labels=['Low', 'Medium', 'High']
                )
            else:
                df_trans['yield_level'] = pd.cut(
                    df_trans[target_col],
                    bins=[-float('inf')] + yield_thresholds + [float('inf')],
                    labels=[f'Level_{i+1}' for i in range(len(yield_thresholds)+1)]
                )
            
            df_trans['yield_level'] = 'yield=' + df_trans['yield_level'].astype(str)
            feature_columns.append('yield_level')
        
        # Tạo transactions
        transactions = []
        
        for idx, row in df_trans.iterrows():
            transaction = []
            
            for col in feature_columns:
                if col in df_trans.columns:
                    # Xử lý theo kiểu dữ liệu
                    if pd.api.types.is_numeric_dtype(df_trans[col]):
                        # Với cột số, kiểm tra xem đã được rời rạc hóa chưa
                        if col + '_bin' in df_trans.columns:
                            # Dùng cột đã rời rạc hóa
                            val = row[col + '_bin']
                            if pd.notna(val):
                                transaction.append(str(val))
                        else:
                            # Tự động rời rạc hóa nếu là số
                            logger.warning(f"Cột {col} là số nhưng chưa được rời rạc hóa")
                    else:
                        # Cột phân loại
                        val = row[col]
                        if pd.notna(val):
                            transaction.append(f"{col}={val}")
            
            if transaction:  # Chỉ thêm nếu không rỗng
                transactions.append(transaction)
        
        logger.info(f"Đã tạo {len(transactions)} transactions, trung bình {np.mean([len(t) for t in transactions]):.2f} items/transaction")
        return transactions
    
    def encode_transactions(self, transactions: List[List[str]]) -> pd.DataFrame:
        """
        Mã hóa transactions thành one-hot encoding cho Apriori
        
        Args:
            transactions: List các transactions
            
        Returns:
            DataFrame one-hot encoded
        """
        te = TransactionEncoder()
        te_ary = te.fit_transform(transactions)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        
        self.transaction_encoder = te
        logger.info(f"Đã mã hóa {len(transactions)} transactions thành {df_encoded.shape[1]} items")
        
        return df_encoded
    
    def mine_frequent_itemsets(self,
                               df_encoded: pd.DataFrame,
                               min_support: float = 0.1,
                               use_fpgrowth: bool = False,
                               max_len: Optional[int] = None) -> pd.DataFrame:
        """
        Khai phá tập phổ biến (frequent itemsets)
        
        Args:
            df_encoded: DataFrame one-hot encoded
            min_support: Ngưỡng support tối thiểu
            use_fpgrowth: Dùng FP-Growth thay vì Apriori
            max_len: Độ dài tối đa của itemset
            
        Returns:
            DataFrame chứa frequent itemsets
        """
        logger.info(f"Bắt đầu khai phá frequent itemsets với min_support={min_support}")
        
        if use_fpgrowth:
            self.frequent_itemsets = fpgrowth(
                df_encoded, 
                min_support=min_support, 
                use_colnames=True,
                max_len=max_len
            )
            logger.info("Sử dụng FP-Growth algorithm")
        else:
            self.frequent_itemsets = apriori(
                df_encoded, 
                min_support=min_support, 
                use_colnames=True,
                max_len=max_len
            )
            logger.info("Sử dụng Apriori algorithm")
        
        logger.info(f"Tìm thấy {len(self.frequent_itemsets)} frequent itemsets")
        
        # Thêm cột độ dài itemset
        self.frequent_itemsets['length'] = self.frequent_itemsets['itemsets'].apply(len)
        
        # Top 10 itemsets phổ biến nhất
        top_itemsets = self.frequent_itemsets.nlargest(10, 'support')[['itemsets', 'support', 'length']]
        logger.info(f"\nTop 10 frequent itemsets:\n{top_itemsets.to_string(index=False)}")
        
        return self.frequent_itemsets
    
    def generate_rules(self,
                       metric: str = 'lift',
                       min_threshold: float = 1.0,
                       min_confidence: Optional[float] = None) -> pd.DataFrame:
        """
        Sinh luật kết hợp từ frequent itemsets
        
        Args:
            metric: Metric để đánh giá ('confidence', 'lift', 'leverage', 'conviction')
            min_threshold: Ngưỡng tối thiểu cho metric
            min_confidence: Ngưỡng confidence tối thiểu (nếu có)
            
        Returns:
            DataFrame chứa các luật kết hợp
        """
        if self.frequent_itemsets is None or len(self.frequent_itemsets) == 0:
            logger.error("Chưa có frequent itemsets. Chạy mine_frequent_itemsets trước.")
            return pd.DataFrame()
        
        logger.info(f"Bắt đầu sinh luật kết hợp với metric='{metric}', min_threshold={min_threshold}")
        
        self.rules = association_rules(
            self.frequent_itemsets, 
            metric=metric, 
            min_threshold=min_threshold
        )
        
        # Lọc theo confidence nếu có
        if min_confidence is not None:
            self.rules = self.rules[self.rules['confidence'] >= min_confidence]
        
        # Sắp xếp theo lift
        self.rules = self.rules.sort_values('lift', ascending=False)
        
        logger.info(f"Tìm thấy {len(self.rules)} luật kết hợp")
        
        if len(self.rules) > 0:
            # Top 10 luật tốt nhất
            top_rules = self.rules.head(10)[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
            logger.info(f"\nTop 10 rules by lift:\n{top_rules.to_string(index=False)}")
        
        return self.rules
    
    def filter_rules_by_consequent(self, consequent_pattern: str) -> pd.DataFrame:
        """
        Lọc các luật có consequent chứa pattern nhất định (ví dụ: 'yield=High')
        
        Args:
            consequent_pattern: Pattern cần tìm trong consequent
            
        Returns:
            DataFrame các luật đã lọc
        """
        if self.rules is None or len(self.rules) == 0:
            logger.error("Chưa có rules. Chạy generate_rules trước.")
            return pd.DataFrame()
        
        filtered = []
        for idx, row in self.rules.iterrows():
            conseq = row['consequents']
            # Kiểm tra nếu bất kỳ item nào trong consequents chứa pattern
            if any(consequent_pattern in str(item) for item in conseq):
                filtered.append(row)
        
        result = pd.DataFrame(filtered)
        logger.info(f"Tìm thấy {len(result)} luật với consequent chứa '{consequent_pattern}'")
        
        return result
    
    def interpret_rules(self, rules: Optional[pd.DataFrame] = None, n_top: int = 20) -> pd.DataFrame:
        """
        Diễn giải các luật kết hợp, thêm cột mô tả
        
        Args:
            rules: DataFrame rules (None = dùng self.rules)
            n_top: Số luật top để diễn giải
            
        Returns:
            DataFrame rules với cột interpretation
        """
        if rules is None:
            rules = self.rules
        
        if rules is None or len(rules) == 0:
            logger.error("Không có rules để diễn giải")
            return pd.DataFrame()
        
        # Lấy top rules theo lift
        top_rules = rules.nlargest(n_top, 'lift').copy()
        
        # Thêm cột diễn giải
        interpretations = []
        
        for idx, row in top_rules.iterrows():
            ante = ', '.join([str(item) for item in row['antecedents']])
            cons = ', '.join([str(item) for item in row['consequents']])
            
            # Tạo câu diễn giải
            if 'yield=High' in cons:
                interp = f"Nếu {ante} → Năng suất CAO (lift={row['lift']:.2f}, conf={row['confidence']:.2f})"
            elif 'yield=Low' in cons:
                interp = f"Nếu {ante} → Năng suất THẤP (lift={row['lift']:.2f}, conf={row['confidence']:.2f})"
            else:
                interp = f"Nếu {ante} → {cons} (lift={row['lift']:.2f}, conf={row['confidence']:.2f})"
            
            interpretations.append(interp)
        
        top_rules['interpretation'] = interpretations
        return top_rules
    
    def run_association_pipeline(self,
                                 df: pd.DataFrame,
                                 min_support: float = 0.1,
                                 min_confidence: float = 0.5,
                                 min_lift: float = 1.2,
                                 target_yield: str = 'High') -> Dict:
        """
        Chạy toàn bộ pipeline khai phá luật kết hợp
        
        Args:
            df: DataFrame đầu vào
            min_support: Ngưỡng support tối thiểu
            min_confidence: Ngưỡng confidence tối thiểu
            min_lift: Ngưỡng lift tối thiểu
            target_yield: Mức năng suất quan tâm ('High', 'Low', 'Medium')
            
        Returns:
            Dict chứa kết quả
        """
        logger.info("=" * 60)
        logger.info("BẮT ĐẦU PIPELINE KHAI PHÁ LUẬT KẾT HỢP")
        logger.info("=" * 60)
        
        # 1. Rời rạc hóa các cột số
        df_disc = df.copy()
        numeric_cols = df_disc.select_dtypes(include=[np.number]).columns.tolist()
        
        # Loại bỏ cột Year nếu có (không nên rời rạc hóa năm)
        if 'Year' in numeric_cols:
            numeric_cols.remove('Year')
        
        for col in numeric_cols:
            if col != 'hg/ha_yield':  # Xử lý yield riêng
                df_disc = self.discretize_numeric(df_disc, col, bins=3)
        
        # 2. Chuẩn bị transactions
        transactions = self.prepare_transactions(
            df_disc, 
            discretize_target=True,
            yield_thresholds=None  # Tự động phân loại
        )
        
        # 3. Mã hóa transactions
        df_encoded = self.encode_transactions(transactions)
        
        # 4. Khai phá frequent itemsets
        self.mine_frequent_itemsets(df_encoded, min_support=min_support)
        
        # 5. Sinh luật
        self.generate_rules(metric='lift', min_threshold=min_lift, min_confidence=min_confidence)
        
        # 6. Lọc luật liên quan đến yield
        yield_rules = self.filter_rules_by_consequent(f'yield={target_yield}')
        
        # 7. Diễn giải
        interpreted = self.interpret_rules(yield_rules, n_top=10)
        
        logger.info("=" * 60)
        logger.info("HOÀN THÀNH PIPELINE LUẬT KẾT HỢP")
        logger.info("=" * 60)
        
        return {
            'frequent_itemsets': self.frequent_itemsets,
            'all_rules': self.rules,
            f'yield_{target_yield}_rules': yield_rules,
            'interpreted_rules': interpreted,
            'n_transactions': len(transactions),
            'n_items': df_encoded.shape[1] if hasattr(df_encoded, 'shape') else 0
        }
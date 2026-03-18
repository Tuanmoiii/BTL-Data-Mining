#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import subprocess
import yaml
import logging
from pathlib import Path
from datetime import datetime

CURRENT_FILE = Path(__file__).resolve()  # outputs/run_pipeline.py
OUTPUTS_DIR = CURRENT_FILE.parent        # thư mục outputs/
ROOT_DIR = OUTPUTS_DIR.parent             # thư mục gốc dự án (chứa data/, notebooks/, scripts/, src/, outputs/)

# Thêm thư mục gốc vào sys.path để import được các module từ src/
sys.path.insert(0, str(ROOT_DIR))

# Cấu hình logging - lưu ngay trong outputs/reports/
LOG_DIR = OUTPUTS_DIR / 'reports'
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config():
    """Đọc cấu hình từ config.yaml ở thư mục gốc"""
    config_path = ROOT_DIR / 'config.yaml'
    if not config_path.exists():
        logger.error(f"Không tìm thấy file config: {config_path}")
        # Thử tìm trong configs/ nếu có
        config_path = ROOT_DIR / 'configs' / 'params.yaml'
        if not config_path.exists():
            return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Đã đọc cấu hình từ: {config_path}")
        return config
    except Exception as e:
        logger.error(f"Lỗi đọc config: {e}")
        return {}

def ensure_output_dirs():
    """Đảm bảo các thư mục con trong outputs tồn tại"""
    dirs = {
        'figures': OUTPUTS_DIR / 'figures',
        'tables': OUTPUTS_DIR / 'tables',
        'models': OUTPUTS_DIR / 'models',
        'reports': OUTPUTS_DIR / 'reports',
    }
    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Đảm bảo thư mục: {path}")
    return dirs

def run_notebook(notebook_relative_path, output_dirs):
    """
    Chạy một notebook
    
    Args:
        notebook_relative_path: đường dẫn tương đối từ thư mục gốc, ví dụ: "notebooks/01_EDA.ipynb"
        output_dirs: dict chứa các thư mục output
    """
    notebook_path = ROOT_DIR / notebook_relative_path
    if not notebook_path.exists():
        logger.error(f"❌ Không tìm thấy notebook: {notebook_path}")
        return False
    
    notebook_name = notebook_path.stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # File output
    output_notebook = output_dirs['reports'] / f"{notebook_name}_executed_{timestamp}.ipynb"
    html_output = output_dirs['reports'] / f"{notebook_name}_{timestamp}.html"
    
    # Tạo file tạm để chuyển tham số
    params = {
        'root_dir': str(ROOT_DIR),
        'output_figures_dir': str(output_dirs['figures']),
        'output_tables_dir': str(output_dirs['tables']),
        'output_models_dir': str(output_dirs['models']),
        'output_reports_dir': str(output_dirs['reports']),
        'timestamp': timestamp
    }
    
    # Lưu params tạm thời
    params_file = output_dirs['reports'] / f"{notebook_name}_params_{timestamp}.yaml"
    with open(params_file, 'w', encoding='utf-8') as f:
        yaml.dump(params, f)
    
    # Câu lệnh chạy notebook với tham số (dùng papermill nếu có, fallback sang nbconvert)
    try:
        # Thử dùng papermill trước
        import papermill as pm
        logger.info(f"🚀 Dùng papermill để chạy: {notebook_name}")
        pm.execute_notebook(
            input_path=str(notebook_path),
            output_path=str(output_notebook),
            parameters=params,
            kernel_name='python3',
            progress_bar=True,
            log_output=True
        )
        logger.info(f"✅ Papermill thành công: {notebook_name}")
    except ImportError:
        # Fallback dùng nbconvert nếu không có papermill
        logger.warning("⚠️ Không tìm thấy papermill, dùng nbconvert thay thế")
        cmd = (f"jupyter nbconvert --to notebook --execute {notebook_path} "
               f"--output {output_notebook.name} --output-dir {output_dirs['reports']} "
               f"--ExecutePreprocessor.timeout=600")
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"❌ Lỗi nbconvert: {result.stderr}")
            return False
    
    # Tạo HTML để dễ xem
    os.system(f"jupyter nbconvert --to html {output_notebook} --output {html_output.name} --output-dir {output_dirs['reports']}")
    
    logger.info(f"📊 Kết quả lưu tại:")
    logger.info(f"   - Notebook: {output_notebook}")
    logger.info(f"   - HTML: {html_output}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Chạy pipeline khai phá dữ liệu')
    parser.add_argument('--skip-eda', action='store_true')
    parser.add_argument('--skip-preprocess', action='store_true')
    parser.add_argument('--skip-mining', action='store_true')
    parser.add_argument('--skip-modeling', action='store_true')
    parser.add_argument('--skip-evaluation', action='store_true')
    parser.add_argument('--fast', action='store_true', help='Chạy nhanh (chỉ baseline)')
    
    args = parser.parse_args()
    
    start_time = datetime.now()
    logger.info(f"\n{'🌟'*30}")
    logger.info("🚀 BẮT ĐẦU PIPELINE")
    logger.info(f"📅 Thời gian: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"📁 Log file: {log_file}")
    logger.info(f"📁 Root dir: {ROOT_DIR}")
    logger.info(f"{'🌟'*30}\n")
    
    # Đọc config
    config = load_config()
    
    # Tạo các thư mục output con
    output_dirs = ensure_output_dirs()
    
    # Định nghĩa các bước
    notebooks = [
        ('eda', '📊 EDA', 'notebooks/01_EDA.ipynb'),
        ('preprocess', '🛠️ Tiền xử lý', 'notebooks/02_preprocess_feature.ipynb'),
        ('mining', '⛏️ Khai phá', 'notebooks/03_mining_or_clustering.ipynb'),
        ('modeling', '🤖 Mô hình hóa', 'notebooks/04_Modeling.ipynb'),
        ('evaluation', '📈 Đánh giá', 'notebooks/05_evaluation_report.ipynb')
    ]
    
    # Lọc theo args
    steps_to_run = []
    skip_map = {
        'eda': args.skip_eda,
        'preprocess': args.skip_preprocess,
        'mining': args.skip_mining,
        'modeling': args.skip_modeling,
        'evaluation': args.skip_evaluation
    }
    
    for step_id, desc, path in notebooks:
        if not skip_map.get(step_id, False):
            steps_to_run.append((step_id, desc, path))
    
    # Chạy từng bước
    success = True
    for step_id, desc, path in steps_to_run:
        logger.info(f"\n{'📌'*20}")
        logger.info(f"BƯỚC: {desc}")
        logger.info(f"{'📌'*20}\n")
        
        if not run_notebook(path, output_dirs):
            logger.error(f"❌ Thất bại ở bước: {desc}")
            success = False
            break
    
    # Tổng kết
    end_time = datetime.now()
    duration = end_time - start_time
    
    summary_file = output_dirs['reports'] / 'pipeline_summary.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"PIPELINE EXECUTION SUMMARY\n")
        f.write(f"{'='*60}\n")
        f.write(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duration: {duration.total_seconds():.2f} seconds\n")
        f.write(f"Status: {'SUCCESS' if success else 'FAILED'}\n")
        f.write(f"Steps executed: {len(steps_to_run)}\n")
        f.write(f"Output directories:\n")
        f.write(f"  - Figures: {output_dirs['figures']}\n")
        f.write(f"  - Tables: {output_dirs['tables']}\n")
        f.write(f"  - Models: {output_dirs['models']}\n")
        f.write(f"  - Reports: {output_dirs['reports']}\n")
    
    logger.info(f"\n{'🎉'*30}")
    logger.info(f"{'✅ THÀNH CÔNG' if success else '❌ THẤT BẠI'}")
    logger.info(f"⏱️ Thời gian: {duration.total_seconds():.2f} giây")
    logger.info(f"📄 Báo cáo: {summary_file}")
    logger.info(f"{'🎉'*30}\n")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
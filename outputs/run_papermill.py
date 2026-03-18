#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_papermill.py

Script chạy notebooks bằng papermill.
Vị trí: outputs/run_papermill.py (theo cấu trúc thực tế)
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from datetime import datetime
import logging
import json

# --- CẤU HÌNH ĐƯỜNG DẪN (QUAN TRỌNG) ---
CURRENT_FILE = Path(__file__).resolve()  # outputs/run_papermill.py
OUTPUTS_DIR = CURRENT_FILE.parent        # thư mục outputs/
ROOT_DIR = OUTPUTS_DIR.parent             # thư mục gốc dự án

# Thêm thư mục gốc vào sys.path
sys.path.insert(0, str(ROOT_DIR))

# Cấu hình logging
LOG_DIR = OUTPUTS_DIR / 'reports'
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f'papermill_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    import papermill as pm
    PAPERMILL_AVAILABLE = True
except ImportError:
    PAPERMILL_AVAILABLE = False
    logger.error("❌ papermill chưa được cài đặt. Chạy: pip install papermill")
    sys.exit(1)

def load_config(config_path):
    """Đọc file cấu hình YAML"""
    if not config_path.exists():
        logger.warning(f"⚠️ Không tìm thấy config: {config_path}")
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def prepare_output_dirs():
    """Chuẩn bị các thư mục output con"""
    dirs = {
        'notebooks': OUTPUTS_DIR / 'notebooks',  # lưu notebooks đã chạy
        'figures': OUTPUTS_DIR / 'figures',
        'tables': OUTPUTS_DIR / 'tables',
        'models': OUTPUTS_DIR / 'models',
        'reports': OUTPUTS_DIR / 'reports',
    }
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    return dirs

def run_notebook(notebook_name, params, output_dirs, timestamp):
    """
    Chạy một notebook với papermill
    
    Args:
        notebook_name: tên file trong thư mục notebooks/ (vd: "01_EDA.ipynb")
        params: dict tham số
        output_dirs: dict các thư mục output
        timestamp: timestamp string
    """
    notebook_path = ROOT_DIR / 'notebooks' / notebook_name
    if not notebook_path.exists():
        logger.error(f"❌ Không tìm thấy: {notebook_path}")
        return None
    
    # File output
    base_name = notebook_name.replace('.ipynb', '')
    output_notebook = output_dirs['notebooks'] / f"{base_name}_{timestamp}.ipynb"
    html_output = output_dirs['reports'] / f"{base_name}_{timestamp}.html"
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🚀 Chạy: {notebook_name}")
    logger.info(f"📂 Output notebook: {output_notebook}")
    logger.info(f"{'='*60}")
    
    try:
        # Chạy với papermill
        pm.execute_notebook(
            input_path=str(notebook_path),
            output_path=str(output_notebook),
            parameters=params,
            kernel_name='python3',
            progress_bar=True,
            log_output=True,
            request_save_on_cell_execute=True
        )
        
        # Tạo HTML
        os.system(f"jupyter nbconvert --to html {output_notebook} --output {html_output.name} --output-dir {output_dirs['reports']}")
        
        logger.info(f"✅ Thành công: {notebook_name}")
        logger.info(f"   - Notebook: {output_notebook}")
        logger.info(f"   - HTML: {html_output}")
        
        return str(output_notebook)
        
    except Exception as e:
        logger.error(f"❌ Lỗi {notebook_name}: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Chạy notebooks bằng papermill')
    parser.add_argument('--notebook', '-n', type=str, help='Chỉ chạy một notebook (vd: "01_EDA.ipynb")')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', 
                       help='File cấu hình (mặc định: config.yaml ở thư mục gốc)')
    parser.add_argument('--kernel', type=str, default='python3', help='Tên kernel')
    parser.add_argument('--no-progress', action='store_true', help='Ẩn progress bar')
    
    args = parser.parse_args()
    
    # Chuẩn bị
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dirs = prepare_output_dirs()
    
    # Đọc config
    config_path = ROOT_DIR / args.config
    config = load_config(config_path)
    
    # Tham số mặc định cho tất cả notebooks
    base_params = {
        'root_dir': str(ROOT_DIR),
        'timestamp': timestamp,
        'output_figures_dir': str(output_dirs['figures']),
        'output_tables_dir': str(output_dirs['tables']),
        'output_models_dir': str(output_dirs['models']),
        'output_reports_dir': str(output_dirs['reports']),
        'random_seed': config.get('random_seed', 42)
    }
    
    # Thêm các tham số từ config
    if 'preprocessing' in config:
        base_params.update(config['preprocessing'])
    if 'modeling' in config:
        base_params.update(config['modeling'])
    
    # Lưu params đã dùng
    params_file = output_dirs['reports'] / f"all_params_{timestamp}.yaml"
    with open(params_file, 'w', encoding='utf-8') as f:
        yaml.dump(base_params, f, allow_unicode=True)
    
    # Danh sách notebooks cần chạy
    if args.notebook:
        notebooks_to_run = [args.notebook]
    else:
        notebooks_to_run = [
            '01_EDA.ipynb',
            '02_preprocess_feature.ipynb',
            '03_mining_or_clustering.ipynb',
            '04_Modeling.ipynb',
            '05_evaluation_report.ipynb'
        ]
    
    # Chạy lần lượt
    results = {}
    for i, nb in enumerate(notebooks_to_run, 1):
        logger.info(f"\n📌 [{i}/{len(notebooks_to_run)}] {nb}")
        
        # Notebook-specific params
        params = base_params.copy()
        if 'EDA' in nb:
            params['notebook_type'] = 'eda'
        elif 'preprocess' in nb:
            params['notebook_type'] = 'preprocess'
        elif 'mining' in nb or 'clustering' in nb:
            params['notebook_type'] = 'mining'
        elif 'Modeling' in nb:
            params['notebook_type'] = 'modeling'
        elif 'evaluation' in nb:
            params['notebook_type'] = 'evaluation'
        
        # Thêm thông tin notebook trước đó
        if i > 1:
            params['previous_notebooks'] = notebooks_to_run[:i-1]
        
        output_path = run_notebook(nb, params, output_dirs, timestamp)
        results[nb] = {
            'success': output_path is not None,
            'output': output_path,
            'timestamp': timestamp
        }
        
        # Hỏi nếu thất bại
        if output_path is None and i < len(notebooks_to_run):
            resp = input(f"\n❌ {nb} thất bại. Tiếp tục? (y/n): ")
            if resp.lower() != 'y':
                break
    
    # Tạo báo cáo tổng kết
    report_file = output_dirs['reports'] / f"papermill_report_{timestamp}.md"
    json_file = output_dirs['reports'] / f"papermill_report_{timestamp}.json"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"# Báo cáo Papermill\n\n")
        f.write(f"- **Thời gian:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Timestamp:** {timestamp}\n")
        f.write(f"- **Config:** {args.config}\n\n")
        
        f.write(f"## Kết quả\n\n")
        f.write(f"| Notebook | Trạng thái | Output |\n")
        f.write(f"|----------|------------|--------|\n")
        
        for nb, info in results.items():
            status = "✅" if info['success'] else "❌"
            out = Path(info['output']).name if info['output'] else "N/A"
            f.write(f"| {nb} | {status} | {out} |\n")
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': timestamp,
            'config': str(config_path),
            'results': results,
            'output_dirs': {k: str(v) for k, v in output_dirs.items()}
        }, f, indent=2, ensure_ascii=False)
    
    # Tổng kết
    total = len(results)
    success = sum(1 for r in results.values() if r['success'])
    
    logger.info(f"\n{'📊'*20}")
    logger.info(f"📊 TỔNG KẾT")
    logger.info(f"   - Tổng số: {total}")
    logger.info(f"   - ✅ Thành công: {success}")
    logger.info(f"   - ❌ Thất bại: {total - success}")
    logger.info(f"   - 📁 Outputs: {OUTPUTS_DIR}")
    logger.info(f"   - 📄 Báo cáo: {report_file}")
    logger.info(f"{'📊'*20}\n")

if __name__ == "__main__":
    main()
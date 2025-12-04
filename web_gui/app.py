from flask import Flask, render_template, request, jsonify, send_file, Response
import subprocess
import os
import json
import threading
import time
import queue
import logging
from datetime import datetime
import webbrowser
import platform
import psutil
import GPUtil
import re
import glob

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for tracking operations
active_operations = {}
operation_logs = {}
operation_queue = queue.Queue()

# Global variable to store real-time CLI output
cli_output_buffer = []
cli_output_lock = threading.Lock()

class OperationManager:
    def __init__(self):
        self.operations = {}
        self.logs = {}
    
    def start_operation(self, op_id, operation_type, params):
        """Start a new operation"""
        self.operations[op_id] = {
            'id': op_id,
            'type': operation_type,
            'params': params,
            'status': 'running',
            'start_time': datetime.now(),
            'progress': 0,
            'logs': []
        }
        self.logs[op_id] = []
        return op_id
    
    def update_operation(self, op_id, status, progress=None, log=None):
        """Update operation status and progress"""
        if op_id in self.operations:
            if status:
                self.operations[op_id]['status'] = status
            if progress is not None:
                self.operations[op_id]['progress'] = progress
            if log:
                self.operations[op_id]['logs'].append(log)
                self.logs[op_id].append(log)
    
    def get_operation(self, op_id):
        """Get operation details"""
        return self.operations.get(op_id, {})
    
    def get_all_operations(self):
        """Get all operations"""
        return self.operations

# Initialize operation manager
op_manager = OperationManager()

def get_project_root():
    """Get the project root directory (parent of web_gui)"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def add_cli_output(message, operation_id=None):
    """Add CLI output to the global buffer"""
    with cli_output_lock:
        timestamp = datetime.now().strftime("%H:%M:%S")
        output_entry = {
            'timestamp': timestamp,
            'message': message,
            'operation_id': operation_id
        }
        cli_output_buffer.append(output_entry)
        
        # Keep only last 1000 entries to prevent memory issues
        if len(cli_output_buffer) > 1000:
            cli_output_buffer.pop(0)

def parse_progress_from_line(line):
    """Parse progress percentage from output line (handles tqdm and other progress formats)"""
    import re
    
    # Try to parse tqdm format: "45%|████▌     | 54/120 [00:20<00:25,  2.65it/s]"
    tqdm_match = re.search(r'(\d+)%', line)
    if tqdm_match:
        return int(tqdm_match.group(1))
    
    # Try to parse frame progress: "Frame 54/120" or "Processing 54 of 120"
    frame_match = re.search(r'(\d+)\s*/\s*(\d+)', line)
    if frame_match:
        current = int(frame_match.group(1))
        total = int(frame_match.group(2))
        if total > 0:
            return int((current / total) * 100)
    
    # Try to parse percentage directly: "Progress: 45%" or "45% complete"
    percent_match = re.search(r'(\d+(?:\.\d+)?)\s*%', line)
    if percent_match:
        return int(float(percent_match.group(1)))
    
    return None

def run_subprocess_with_realtime_output(command, cwd, operation_id):
    """Run subprocess and capture real-time output with progress tracking"""
    try:
        command_tokens = command.split()
        script_name = next((token for token in command_tokens if token.endswith('.py')), '')
        add_cli_output(f"🔍 Original command: {command}", operation_id)
        add_cli_output(f"🔍 Original working directory: {cwd}", operation_id)
        
        # For 3D visualization scripts, change to 3D_vis directory to fix relative path issues
        if '3D_vis/' in command or '3D_vis_' in script_name:
            if script_name:
                # Change working directory to 3D_vis folder for relative paths to work
                vis_dir = os.path.join(get_project_root(), '3D_vis')
                cwd = vis_dir
                add_cli_output(f"📁 Changed working directory to: {vis_dir}", operation_id)
                add_cli_output(f"🔍 Script will now look for data files relative to: {vis_dir}", operation_id)
                
                # Test if the relative path will work
                test_data_path = os.path.join(vis_dir, '..', 'data')
                if os.path.exists(test_data_path):
                    add_cli_output(f"✅ Data directory accessible from new working directory: {test_data_path}", operation_id)
                else:
                    add_cli_output(f"⚠️  Data directory not accessible from new working directory: {test_data_path}", operation_id)
                
                add_cli_output(f"🔍 Final working directory: {cwd}", operation_id)
                add_cli_output(f"🔍 Final command: {command}", operation_id)
                
                # Verify that the script exists in the new working directory
                script_path_in_new_dir = os.path.join(vis_dir, os.path.basename(script_name))
                if os.path.exists(script_path_in_new_dir):
                    add_cli_output(f"✅ Script found in new working directory: {script_path_in_new_dir}", operation_id)
                else:
                    add_cli_output(f"❌ Script NOT found in new working directory: {script_path_in_new_dir}", operation_id)
        
        add_cli_output(f"🚀 Executing subprocess with cwd: {cwd}", operation_id)
        
        # Track progress
        last_progress = 0
        status_message = "Processing..."
        
        # Use Popen for real-time output streaming
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Read output line by line in real-time
        while True:
            output_line = process.stdout.readline()
            if output_line == '' and process.poll() is not None:
                break
            
            if output_line:
                line = output_line.strip()
                if line:
                    add_cli_output(line, operation_id)
                    
                    # Try to parse progress from the line
                    progress = parse_progress_from_line(line)
                    if progress is not None and progress > last_progress:
                        last_progress = progress
                        status_message = f"Processing... {progress}%"
                        op_manager.update_operation(operation_id, 'running', progress=progress, log=line)
                    else:
                        # Update status message with meaningful information
                        if 'rendering' in line.lower() or 'frame' in line.lower():
                            status_message = line[:50] + "..." if len(line) > 50 else line
                        op_manager.update_operation(operation_id, 'running', progress=last_progress, log=line)
        
        # Get return code
        return_code = process.poll()
        
        # Final progress update
        if return_code == 0:
            op_manager.update_operation(operation_id, 'completed', progress=100, log="Operation completed successfully")
        else:
            op_manager.update_operation(operation_id, 'failed', progress=last_progress, log=f"Process exited with code {return_code}")
        
        return return_code if return_code is not None else -1
        
    except subprocess.TimeoutExpired:
        error_msg = "Subprocess timed out after 5 minutes"
        add_cli_output(error_msg, operation_id)
        op_manager.update_operation(operation_id, 'failed', progress=last_progress, log=error_msg)
        return -1
    except Exception as e:
        error_msg = f"Subprocess error: {str(e)}"
        add_cli_output(error_msg, operation_id)
        op_manager.update_operation(operation_id, 'failed', progress=last_progress, log=error_msg)
        return -1

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/categories')
def get_categories():
    """Get available categories (only those with data files)"""
    all_categories = [
        'chest', 'foot', 'head'
    ]
    
    # Filter to only include categories with available data files
    available_categories = []
    for category in all_categories:
        data_file = os.path.join(get_project_root(), 'data', f'{category}_50.pickle')
        if os.path.exists(data_file):
            available_categories.append(category)
    
    return jsonify(available_categories)

@app.route('/api/methods')
def get_methods():
    """Get available methods"""
    methods = ['Lineformer', 'nerf', 'tensorf', 'naf', 'SART', 'ASD_POCS', 'FDK', 'intratomo']
    return jsonify(methods)

@app.route('/api/gpu-info')
def get_gpu_info():
    """Get GPU information - always return GPU 1 (NVIDIA RTX 3050)"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpus = []
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpus.append({
                    'id': i,
                    'name': gpu_name,
                    'memory': torch.cuda.get_device_properties(i).total_memory / 1024**3
                })
            return jsonify({'available': True, 'gpus': gpus, 'default_gpu': 1})
        else:
            return jsonify({'available': False, 'message': 'CUDA not available', 'default_gpu': 1})
    except Exception as e:
        return jsonify({'available': False, 'message': str(e), 'default_gpu': 1})

@app.route('/api/system-info')
def get_system_info():
    """Get system information"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        system_info = {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'cpu_count': psutil.cpu_count(),
            'cpu_percent': cpu_percent,
            'memory_total': memory.total / (1024**3),
            'memory_available': memory.available / (1024**3),
            'memory_percent': memory.percent,
            'disk_total': disk.total / (1024**3),
            'disk_free': disk.free / (1024**3),
            'disk_percent': (disk.used / disk.total) * 100
        }
        return jsonify(system_info)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/cli-output-stream')
def cli_output_stream():
    """Server-Sent Events endpoint for real-time CLI output"""
    def generate():
        last_index = 0
        
        while True:
            with cli_output_lock:
                current_buffer = cli_output_buffer.copy()
            
            if len(current_buffer) > last_index:
                new_entries = current_buffer[last_index:]
                for entry in new_entries:
                    data = f"data: {json.dumps(entry)}\n\n"
                    yield data
                last_index = len(current_buffer)
            
            time.sleep(0.1)  # Small delay to prevent excessive CPU usage
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/cli-output')
def get_cli_output():
    """Get all CLI output (for initial load)"""
    with cli_output_lock:
        return jsonify(cli_output_buffer)

@app.route('/api/clear-cli-output')
def clear_cli_output():
    """Clear CLI output buffer"""
    global cli_output_buffer
    with cli_output_lock:
        cli_output_buffer = []
    return jsonify({'success': True, 'message': 'CLI output cleared'})

@app.route('/api/run-lineformer-test', methods=['POST'])
def run_lineformer_test():
    """Run Lineformer test on specific category"""
    data = request.json
    category = data.get('category')
    output_path = data.get('output_path', 'output')
    
    op_id = f"lineformer_test_{category}_{int(time.time())}"
    
    def run_test_thread():
        try:
            config_path = f"config/Lineformer/{category}_50.yaml"
            weights_path = f"models/{category}/"
            
            command = f"python test.py --method Lineformer --category {category} --config {config_path} --weights {weights_path} --output_path {output_path} --gpu_id 1"
            
            op_manager.start_operation(op_id, 'lineformer_test', {
                'category': category,
                'method': 'Lineformer',
                'output_path': output_path,
                'command': command
            })
            
            add_cli_output(f"🚀 Starting Lineformer test for {category}", op_id)
            op_manager.update_operation(op_id, 'running', 0, f"Starting Lineformer test for {category}")
            
            return_code = run_subprocess_with_realtime_output(
                command, 
                get_project_root(), 
                op_id
            )
            
            if return_code == 0:
                success_msg = f"✅ Lineformer test completed successfully for {category}"
                add_cli_output(success_msg, op_id)
                op_manager.update_operation(op_id, 'completed', 100, success_msg)
            else:
                error_msg = f"❌ Lineformer test failed for {category} (return code: {return_code})"
                add_cli_output(error_msg, op_id)
                op_manager.update_operation(op_id, 'failed', 0, error_msg)
                
        except Exception as e:
            error_msg = f"💥 Exception in Lineformer test: {str(e)}"
            add_cli_output(error_msg, op_id)
            op_manager.update_operation(op_id, 'failed', 0, error_msg)
    
    threading.Thread(target=run_test_thread, daemon=True).start()
    
    return jsonify({'operation_id': op_id, 'status': 'started'})

@app.route('/api/run-3d-static', methods=['POST'])
def run_3d_static():
    """Run 3D static visualization"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided', 'status': 'error'}), 400
        
        category = data.get('category')
        if not category:
            return jsonify({'error': 'Category is required', 'status': 'error'}), 400
        
        # Validate category
        valid_categories = ['chest', 'foot', 'head']
        if category not in valid_categories:
            return jsonify({'error': f'Invalid category. Must be one of: {", ".join(valid_categories)}', 'status': 'error'}), 400
        
        # Check if script exists
        vis_script = f"3D_vis/3D_vis_{category}.py"
        full_script_path = os.path.join(get_project_root(), vis_script)
        if not os.path.exists(full_script_path):
            return jsonify({'error': f'Visualization script not found: {vis_script}', 'status': 'error'}), 404
        
        # Check if required data file exists
        data_file = os.path.join(get_project_root(), 'data', f'{category}_50.pickle')
        if not os.path.exists(data_file):
            return jsonify({'error': f'Required data file not found: data/{category}_50.pickle', 'status': 'error'}), 404
        
        op_id = f"3d_static_{category}_{int(time.time())}"
        
        def run_vis_thread():
            try:
                command = f"python 3D_vis_{category}.py --gpu_id 1"
                
                op_manager.start_operation(op_id, '3d_static', {
                    'category': category,
                    'gpu_id': 1,
                    'command': command
                })
                
                add_cli_output(f"🎨 Starting 3D static visualization for {category}", op_id)
                add_cli_output(f"📁 Using data file: {data_file}", op_id)
                add_cli_output(f"🔧 Command: {command}", op_id)
                add_cli_output(f"💡 Note: Working directory will be changed to 3D_vis/ for relative paths", op_id)
                op_manager.update_operation(op_id, 'running', 0, f"Starting 3D static visualization for {category}")
                
                return_code = run_subprocess_with_realtime_output(
                    command, 
                    get_project_root(), 
                    op_id
                )
                
                if return_code == 0:
                    success_msg = f"✅ 3D static visualization completed for {category}"
                    add_cli_output(success_msg, op_id)
                    op_manager.update_operation(op_id, 'completed', 100, success_msg)
                else:
                    error_msg = f"❌ 3D static visualization failed for {category}"
                    add_cli_output(error_msg, op_id)
                    op_manager.update_operation(op_id, 'failed', 0, error_msg)
                    
            except Exception as e:
                error_msg = f"💥 Exception in 3D static visualization: {str(e)}"
                add_cli_output(error_msg, op_id)
                op_manager.update_operation(op_id, 'failed', 0, error_msg)
        
        threading.Thread(target=run_vis_thread, daemon=True).start()
        
        return jsonify({'operation_id': op_id, 'status': 'started'})
        
    except Exception as e:
        logger.error(f"Error in run_3d_static: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}', 'status': 'error'}), 500

@app.route('/api/run-3d-gif', methods=['POST'])
def run_3d_gif():
    """Run 3D GIF visualization"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided', 'status': 'error'}), 400
        
        category = data.get('category')
        if not category:
            return jsonify({'error': 'Category is required', 'status': 'error'}), 400
        
        # Validate category
        valid_categories = ['chest', 'foot', 'head']
        if category not in valid_categories:
            return jsonify({'error': f'Invalid category. Must be one of: {", ".join(valid_categories)}', 'status': 'error'}), 400
        
        # Check if script exists
        vis_script = f"3D_vis/3D_vis_{category}_gif.py"
        full_script_path = os.path.join(get_project_root(), vis_script)
        if not os.path.exists(full_script_path):
            return jsonify({'error': f'GIF visualization script not found: {vis_script}', 'status': 'error'}), 404
        
        # Check if required data file exists
        data_file = os.path.join(get_project_root(), 'data', f'{category}_50.pickle')
        if not os.path.exists(data_file):
            return jsonify({'error': f'Required data file not found: data/{category}_50.pickle', 'status': 'error'}), 404
        
        op_id = f"3d_gif_{category}_{int(time.time())}"
        
        def run_vis_thread():
            try:
                command = f"python 3D_vis_{category}_gif.py --gpu_id 1"
                
                op_manager.start_operation(op_id, '3d_gif', {
                    'category': category,
                    'gpu_id': 1,
                    'command': command
                })
                
                add_cli_output(f"🎬 Starting 3D GIF visualization for {category}", op_id)
                add_cli_output(f"📁 Using data file: {data_file}", op_id)
                add_cli_output(f"🔧 Command: {command}", op_id)
                add_cli_output(f"💡 Note: Working directory will be changed to 3D_vis/ for relative paths", op_id)
                op_manager.update_operation(op_id, 'running', 0, f"Starting 3D GIF visualization for {category}")
                
                return_code = run_subprocess_with_realtime_output(
                    command, 
                    get_project_root(), 
                    op_id
                )
                
                if return_code == 0:
                    success_msg = f"✅ 3D GIF visualization completed for {category}"
                    add_cli_output(success_msg, op_id)
                    op_manager.update_operation(op_id, 'completed', 100, success_msg)
                else:
                    error_msg = f"❌ 3D GIF visualization failed for {category}"
                    add_cli_output(error_msg, op_id)
                    op_manager.update_operation(op_id, 'failed', 0, error_msg)
                    
            except Exception as e:
                error_msg = f"💥 Exception in 3D GIF visualization: {str(e)}"
                add_cli_output(error_msg, op_id)
                op_manager.update_operation(op_id, 'failed', 0, error_msg)
        
        threading.Thread(target=run_vis_thread, daemon=True).start()
        
        return jsonify({'operation_id': op_id, 'status': 'started'})
        
    except Exception as e:
        logger.error(f"Error in run_3d_gif: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}', 'status': 'error'}), 500

@app.route('/api/run-batch-lineformer', methods=['POST'])
def run_batch_lineformer():
    """Run Lineformer test for all categories"""
    data = request.json
    output_path = data.get('output_path', 'output')
    
    op_id = f"batch_lineformer_{int(time.time())}"
    
    def run_batch_thread():
        try:
            categories = [
                'chest', 'foot', 'head'
            ]
            
            op_manager.start_operation(op_id, 'batch_lineformer', {
                'method': 'Lineformer',
                'output_path': output_path,
                'total_categories': len(categories)
            })
            
            add_cli_output(f"🚀 Starting batch Lineformer test for all categories", op_id)
            
            success_count = 0
            total_categories = len(categories)
            
            for i, category in enumerate(categories):
                progress = (i / total_categories) * 100
                progress_msg = f"📊 Testing {category} ({i+1}/{total_categories}) - Progress: {progress:.1f}%"
                add_cli_output(progress_msg, op_id)
                op_manager.update_operation(op_id, 'running', progress, f"Testing {category} ({i+1}/{total_categories})")
                
                config_path = f"config/Lineformer/{category}_50.yaml"
                weights_path = f"models/{category}/"
                
                command = f"python test.py --method Lineformer --category {category} --config {config_path} --weights {weights_path} --output_path {output_path} --gpu_id 1"
                
                add_cli_output(f"🔧 Running: {command}", op_id)
                
                return_code = run_subprocess_with_realtime_output(
                    command, 
                    get_project_root(), 
                    op_id
                )
                
                if return_code == 0:
                    success_count += 1
                    add_cli_output(f"✅ {category} completed successfully", op_id)
                else:
                    add_cli_output(f"❌ {category} failed", op_id)
                
                time.sleep(1)  # Small delay between tests
            
            final_msg = f"🎉 Batch Lineformer test completed: {success_count}/{total_categories} successful"
            add_cli_output(final_msg, op_id)
            op_manager.update_operation(op_id, 'completed', 100, final_msg)
            
        except Exception as e:
            error_msg = f"💥 Exception in batch Lineformer test: {str(e)}"
            add_cli_output(error_msg, op_id)
            op_manager.update_operation(op_id, 'failed', 0, error_msg)
    
    threading.Thread(target=run_batch_thread, daemon=True).start()
    
    return jsonify({'operation_id': op_id, 'status': 'started'})

@app.route('/api/run-batch-3d-visualization', methods=['POST'])
def run_batch_3d_visualization():
    """Run batch 3D visualization for all categories"""
    data = request.json
    
    op_id = f"batch_3d_vis_{int(time.time())}"
    
    def run_batch_thread():
        try:
            categories = [
                'chest', 'foot', 'head'
            ]
            
            op_manager.start_operation(op_id, 'batch_3d_visualization', {
                'gpu_id': 1,
                'total_categories': len(categories)
            })
            
            add_cli_output(f"🎨 Starting batch 3D visualization for all categories", op_id)
            
            success_count = 0
            total_operations = len(categories) * 2  # Static + GIF for each category
            
            for i, category in enumerate(categories):
                # Static visualization
                vis_script = f"3D_vis/3D_vis_{category}.py"
                full_script_path = os.path.join(get_project_root(), vis_script)
                if os.path.exists(full_script_path):
                    # Check if required data file exists
                    data_file = os.path.join(get_project_root(), 'data', f'{category}_50.pickle')
                    if not os.path.exists(data_file):
                        add_cli_output(f"⚠️  Skipping {category} static visualization - data file not found: data/{category}_50.pickle", op_id)
                        continue
                    
                    progress = (i * 2 / total_operations) * 100
                    progress_msg = f"📊 Static visualization for {category} - Progress: {progress:.1f}%"
                    add_cli_output(progress_msg, op_id)
                    op_manager.update_operation(op_id, 'running', progress, f"Static visualization for {category}")
                    
                    command = f"python 3D_vis_{category}.py --gpu_id 1"
                    add_cli_output(f"🔧 Running: {command}", op_id)
                    add_cli_output(f"💡 Note: Working directory will be changed to 3D_vis/ for relative paths", op_id)
                    
                    return_code = run_subprocess_with_realtime_output(
                        command, 
                        get_project_root(), 
                        op_id
                    )
                    
                    if return_code == 0:
                        success_count += 1
                        add_cli_output(f"✅ {category} static visualization completed", op_id)
                    else:
                        add_cli_output(f"❌ {category} static visualization failed", op_id)
                
                # GIF visualization
                vis_gif_script = f"3D_vis/3D_vis_{category}_gif.py"
                full_gif_script_path = os.path.join(get_project_root(), vis_gif_script)
                if os.path.exists(full_gif_script_path):
                    # Check if required data file exists
                    data_file = os.path.join(get_project_root(), 'data', f'{category}_50.pickle')
                    if not os.path.exists(data_file):
                        add_cli_output(f"⚠️  Skipping {category} GIF visualization - data file not found: data/{category}_50.pickle", op_id)
                        continue
                    
                    progress = ((i * 2 + 1) / total_operations) * 100
                    progress_msg = f"📊 GIF visualization for {category} - Progress: {progress:.1f}%"
                    add_cli_output(progress_msg, op_id)
                    op_manager.update_operation(op_id, 'running', progress, f"GIF visualization for {category}")
                    
                    command = f"python 3D_vis_{category}_gif.py --gpu_id 1"
                    add_cli_output(f"🔧 Running: {command}", op_id)
                    add_cli_output(f"💡 Note: Working directory will be changed to 3D_vis/ for relative paths", op_id)
                    
                    return_code = run_subprocess_with_realtime_output(
                        command, 
                        get_project_root(), 
                        op_id
                    )
                    
                    if return_code == 0:
                        success_count += 1
                        add_cli_output(f"✅ {category} GIF visualization completed", op_id)
                    else:
                        add_cli_output(f"❌ {category} GIF visualization failed", op_id)
                
                time.sleep(1)  # Small delay between visualizations
            
            final_msg = f"🎉 Batch 3D visualization completed: {success_count}/{total_operations} successful"
            add_cli_output(final_msg, op_id)
            op_manager.update_operation(op_id, 'completed', 100, final_msg)
            
        except Exception as e:
            error_msg = f"💥 Exception in batch 3D visualization: {str(e)}"
            add_cli_output(error_msg, op_id)
            op_manager.update_operation(op_id, 'failed', 0, error_msg)
    
    threading.Thread(target=run_batch_thread, daemon=True).start()
    
    return jsonify({'operation_id': op_id, 'status': 'started'})

@app.route('/api/get-output-files')
def get_output_files():
    """Get list of output files for viewing"""
    try:
        category = request.args.get('category')
        output_type = request.args.get('type', 'static')  # 'static' or 'gif'
        
        if not category:
            return jsonify({'error': 'Category is required', 'files': []}), 400
        
        files = []
        
        # Priority 1: Look in 3D_vis/{category}/ directory (where outputs are actually generated)
        vis_category_dir = os.path.join(get_project_root(), '3D_vis', category)
        if os.path.exists(vis_category_dir):
            # Look for static images (PNG) - only main output files, exclude intermediate angle images
            if output_type == 'static':
                # First, check for main output files directly in the category folder
                main_static_pattern = os.path.join(vis_category_dir, "*.png")
                main_static_files = glob.glob(main_static_pattern)
                for file_path in main_static_files:
                    rel_path = os.path.relpath(file_path, get_project_root())
                    files.append({
                        'name': os.path.basename(file_path),
                        'path': rel_path,
                        'type': 'static',
                        'size': os.path.getsize(file_path),
                        'location': f'3D_vis/{category}'
                    })
            
            # Look for GIF files - only main GIF files, exclude intermediate frames
            if output_type == 'gif':
                # Only look for GIF files directly in the category folder or one level deep
                gif_pattern = os.path.join(vis_category_dir, "*.gif")
                gif_files = glob.glob(gif_pattern)
                # Also check one level deep for GIF files in subdirectories
                gif_pattern_subdir = os.path.join(vis_category_dir, "*/*.gif")
                gif_files.extend(glob.glob(gif_pattern_subdir))
                
                for file_path in gif_files:
                    rel_path = os.path.relpath(file_path, get_project_root())
                    files.append({
                        'name': os.path.basename(file_path),
                        'path': rel_path,
                        'type': 'gif',
                        'size': os.path.getsize(file_path),
                        'location': f'3D_vis/{category}'
                    })
        
        # Priority 2: Look in the category-specific directory at root level (e.g., chest/)
        category_dir = os.path.join(get_project_root(), category)
        if os.path.exists(category_dir):
            # Look for static images (PNG) - only main output files, exclude intermediate angle images
            if output_type == 'static':
                # Only files directly in category folder (main outputs)
                static_pattern = os.path.join(category_dir, "*.png")
                static_files = glob.glob(static_pattern)
                for file_path in static_files:
                    rel_path = os.path.relpath(file_path, get_project_root())
                    files.append({
                        'name': os.path.basename(file_path),
                        'path': rel_path,
                        'type': 'static',
                        'size': os.path.getsize(file_path),
                        'location': category
                    })
            
            # Look for GIF files - only main GIF files
            if output_type == 'gif':
                # Only GIF files directly in category folder or one level deep
                gif_pattern = os.path.join(category_dir, "*.gif")
                gif_files = glob.glob(gif_pattern)
                # Also check one level deep for GIF files
                gif_pattern_subdir = os.path.join(category_dir, "*/*.gif")
                gif_files.extend(glob.glob(gif_pattern_subdir))
                
                for file_path in gif_files:
                    rel_path = os.path.relpath(file_path, get_project_root())
                    files.append({
                        'name': os.path.basename(file_path),
                        'path': rel_path,
                        'type': 'gif',
                        'size': os.path.getsize(file_path),
                        'location': category
                    })
        
        # Priority 3: Look in the main output directory
        output_dir = os.path.join(get_project_root(), 'output')
        if os.path.exists(output_dir):
            # Look for static images (PNG)
            if output_type == 'static':
                static_pattern = os.path.join(output_dir, f"*{category}*.png")
                static_files = glob.glob(static_pattern)
                for file_path in static_files:
                    rel_path = os.path.relpath(file_path, get_project_root())
                    files.append({
                        'name': os.path.basename(file_path),
                        'path': rel_path,
                        'type': 'static',
                        'size': os.path.getsize(file_path),
                        'location': 'output'
                    })
            
            # Look for GIF files
            if output_type == 'gif':
                gif_pattern = os.path.join(output_dir, f"*{category}*.gif")
                gif_files = glob.glob(gif_pattern)
                for file_path in gif_files:
                    rel_path = os.path.relpath(file_path, get_project_root())
                    files.append({
                        'name': os.path.basename(file_path),
                        'path': rel_path,
                        'type': 'gif',
                        'size': os.path.getsize(file_path),
                        'location': 'output'
                    })
        
        # Sort files by modification time (newest first)
        files.sort(key=lambda x: os.path.getmtime(os.path.join(get_project_root(), x['path'])), reverse=True)
        
        return jsonify({'files': files, 'category': category, 'type': output_type})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/view-file/<path:file_path>')
def view_file(file_path):
    """Serve output files for viewing"""
    try:
        full_path = os.path.join(get_project_root(), file_path)
        if os.path.exists(full_path):
            return send_file(full_path)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/operation-status/<op_id>')
def get_operation_status(op_id):
    """Get operation status and logs"""
    operation = op_manager.get_operation(op_id)
    if operation:
        return jsonify(operation)
    else:
        return jsonify({'error': 'Operation not found'}), 404

@app.route('/api/operations')
def get_all_operations():
    """Get all operations"""
    return jsonify(op_manager.get_all_operations())

@app.route('/api/open-output-folder')
def open_output_folder():
    """Open output folder in file explorer"""
    try:
        category = request.args.get('category')
        custom_path = request.args.get('path')
        
        # If a custom path is provided, use it directly
        if custom_path:
            folder_to_open = os.path.join(get_project_root(), custom_path)
            if not os.path.exists(folder_to_open):
                return jsonify({'success': False, 'message': f'Folder does not exist: {custom_path}'})
        elif category:
            # Priority 1: Check if 3D_vis/{category}/ directory exists (where outputs are actually generated)
            vis_category_dir = os.path.join(get_project_root(), '3D_vis', category)
            if os.path.exists(vis_category_dir):
                # Check if there are any output files in 3D_vis/{category}/
                gif_files = glob.glob(os.path.join(vis_category_dir, "**/*.gif"), recursive=True)
                png_files = glob.glob(os.path.join(vis_category_dir, "**/*.png"), recursive=True)
                
                if gif_files or png_files:
                    # Open 3D_vis/{category}/ directory if it contains output files
                    folder_to_open = vis_category_dir
                else:
                    # Check category-specific directory at root level
                    category_dir = os.path.join(get_project_root(), category)
                    if os.path.exists(category_dir):
                        gif_files = glob.glob(os.path.join(category_dir, "**/*.gif"), recursive=True)
                        png_files = glob.glob(os.path.join(category_dir, "**/*.png"), recursive=True)
                        if gif_files or png_files:
                            folder_to_open = category_dir
                        else:
                            folder_to_open = os.path.join(get_project_root(), 'output')
                    else:
                        folder_to_open = os.path.join(get_project_root(), 'output')
            else:
                # Check category-specific directory at root level
                category_dir = os.path.join(get_project_root(), category)
                if os.path.exists(category_dir):
                    gif_files = glob.glob(os.path.join(category_dir, "**/*.gif"), recursive=True)
                    png_files = glob.glob(os.path.join(category_dir, "**/*.png"), recursive=True)
                    if gif_files or png_files:
                        folder_to_open = category_dir
                    else:
                        folder_to_open = os.path.join(get_project_root(), 'output')
                else:
                    folder_to_open = os.path.join(get_project_root(), 'output')
        else:
            # No category specified, use main output directory
            folder_to_open = os.path.join(get_project_root(), 'output')
        
        if os.path.exists(folder_to_open):
            if platform.system() == "Windows":
                os.startfile(folder_to_open)
            elif platform.system() == "Darwin":
                subprocess.run(["open", folder_to_open])
            else:
                subprocess.run(["xdg-open", folder_to_open])
            
            # Return the path that was opened
            rel_path = os.path.relpath(folder_to_open, get_project_root())
            return jsonify({'success': True, 'message': f'Output folder opened: {rel_path}'})
        else:
            return jsonify({'success': False, 'message': f'Output folder does not exist: {folder_to_open}'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/data-info')
def get_data_info():
    """Get information about available data files"""
    all_categories = [
        'chest', 'foot', 'head'
    ]
    
    data_info = {}
    total_size = 0
    
    for category in all_categories:
        data_file = os.path.join(get_project_root(), 'data', f'{category}_50.pickle')
        if os.path.exists(data_file):
            file_size = os.path.getsize(data_file)
            file_size_mb = file_size / (1024 * 1024)
            data_info[category] = {
                'available': True,
                'size_mb': round(file_size_mb, 1),
                'path': f'data/{category}_50.pickle'
            }
            total_size += file_size_mb
        else:
            data_info[category] = {
                'available': False,
                'size_mb': 0,
                'path': None
            }
    
    return jsonify({
        'categories': data_info,
        'total_available': len([c for c in data_info.values() if c['available']]),
        'total_size_mb': round(total_size, 1)
    })

if __name__ == '__main__':
    debug_mode = True
    should_run_startup_tasks = (os.environ.get('WERKZEUG_RUN_MAIN') == 'true') or (not debug_mode)
    
    if should_run_startup_tasks:
        print("🚀 Starting Major Project Web GUI...")
        print("📱 Open your browser and go to: http://localhost:5000")
        print("🎨 Beautiful, modern interface with real-time monitoring")
        print("📺 Real-time CLI output streaming enabled")
        print("🎯 GPU 1 (NVIDIA RTX 3050) will be used by default")
        
        # Auto-open browser only once to avoid duplicate windows when the Flask
        # reloader spawns a second process.
        try:
            webbrowser.open('http://localhost:5000')
        except Exception as exc:
            logger.warning("Failed to auto-open browser: %s", exc)
    
    app.run(debug=debug_mode, host='0.0.0.0', port=5000)

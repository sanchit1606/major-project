# CT-Former Web GUI - Technical Documentation

A comprehensive web-based interface for managing CT-Former reconstruction tasks, monitoring progress, and visualizing results. Built with Flask backend and modern JavaScript frontend.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation & Setup](#installation--setup)
- [API Reference](#api-reference)
- [Frontend Implementation](#frontend-implementation)
- [Backend Implementation](#backend-implementation)
- [Real-time Features](#real-time-features)
- [File Management](#file-management)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

The Web GUI provides a user-friendly interface for:
- **3D CT Reconstruction**: Generate static PNG images and animated GIF visualizations
- **Progress Monitoring**: Real-time progress tracking with tqdm integration
- **Output Management**: Browse, preview, and access generated outputs
- **System Monitoring**: GPU, CPU, memory, and storage information
- **Operation Management**: Track and manage multiple concurrent operations

### Supported Operations
- **Static Reconstruction (PNG)**: Generate single-frame 3D volume renderings
- **Dynamic Reconstruction (GIF)**: Create 360-degree rotating animations (120 frames)
- **Category Support**: Chest, Foot, Head anatomical regions

## 🏗️ Architecture

### System Architecture

```
┌─────────────────┐
│   Web Browser   │
│  (Frontend UI)  │
└────────┬────────┘
         │ HTTP/SSE
         │
┌────────▼────────┐
│  Flask Server   │
│   (Backend)     │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼────┐
│Python │ │ File │
│Scripts│ │System│
└───────┘ └──────┘
```

### Technology Stack

#### Backend
- **Framework**: Flask 2.0+
- **Python**: 3.8+
- **Subprocess Management**: `subprocess.Popen` for async execution
- **Real-time Streaming**: Server-Sent Events (SSE)
- **System Monitoring**: `psutil`, `GPUtil`
- **Progress Parsing**: Regex-based tqdm output parsing

#### Frontend
- **HTML5**: Semantic markup with modern features
- **CSS3**: Flexbox, Grid, animations, responsive design
- **JavaScript (ES6+)**: Async/await, Fetch API, IntersectionObserver
- **No Frameworks**: Vanilla JavaScript for lightweight performance

## 📦 Installation & Setup

### Prerequisites
```bash
# Python environment
conda activate ct-former

# Install Flask dependencies
pip install flask psutil GPUtil
```

### Running the GUI
```bash
cd web_gui
python app.py
```

The server will:
1. Start Flask on `http://localhost:5000`
2. Automatically open browser (if not in debug reloader)
3. Display system information and GPU status

### Port Configuration
- **Default Port**: 5000
- **Host**: 0.0.0.0 (accessible from network)
- **Debug Mode**: Enabled for development

## 🔌 API Reference

### Endpoints

#### `GET /`
Serves the main HTML interface.

**Response**: HTML page with GUI

---

#### `GET /api/categories`
Get list of available anatomical categories.

**Response**:
```json
{
  "categories": ["chest", "foot", "head"]
}
```

---

#### `GET /api/data-info/<category>`
Get data availability information for a category.

**Parameters**:
- `category` (path): Anatomical category name

**Response**:
```json
{
  "available": true,
  "file_path": "data/chest_50.pickle",
  "file_size_mb": 125.5,
  "exists": true
}
```

---

#### `POST /api/run-3d-static`
Start static 3D visualization (PNG generation).

**Request Body**:
```json
{
  "category": "chest"
}
```

**Response**:
```json
{
  "operation_id": "3d_static_chest_1234567890",
  "status": "started",
  "message": "Starting 3D static visualization for chest"
}
```

**Command Executed**:
```bash
python 3D_vis/3D_vis_chest.py --gpu_id 1
```

---

#### `POST /api/run-3d-gif`
Start dynamic 3D visualization (GIF animation).

**Request Body**:
```json
{
  "category": "chest"
}
```

**Response**:
```json
{
  "operation_id": "3d_gif_chest_1234567890",
  "status": "started",
  "message": "Starting 3D GIF animation for chest"
}
```

**Command Executed**:
```bash
python 3D_vis/3D_vis_chest_gif.py --gpu_id 1
```

**Progress Tracking**: Parses tqdm output for frame rendering progress

---

#### `GET /api/operation-status/<operation_id>`
Get status of a running operation.

**Parameters**:
- `operation_id` (path): Unique operation identifier

**Response**:
```json
{
  "status": "running",
  "progress": 45,
  "message": "Rendering frame 54/120"
}
```

---

#### `GET /api/operation-output/<operation_id>`
Stream real-time CLI output via Server-Sent Events.

**Parameters**:
- `operation_id` (path): Unique operation identifier

**Response**: SSE stream with format:
```
data: {"type": "output", "content": "Rendering frame 1/120..."}\n\n
data: {"type": "progress", "progress": 1}\n\n
```

---

#### `GET /api/get-output-files`
List available output files for a category.

**Query Parameters**:
- `category`: Anatomical category
- `output_type`: "static" or "gif" (optional)

**Response**:
```json
{
  "files": [
    {
      "name": "3d_chest_reconstructed.png",
      "path": "3D_vis/chest/3d_chest_reconstructed.png",
      "type": "image/png",
      "size": 245678,
      "size_mb": 0.23
    }
  ]
}
```

**File Discovery Logic**:
1. Search `3D_vis/{category}/` directory
2. Filter out intermediate frames (`angle_*.png`)
3. Prioritize final outputs
4. Limit GIF search depth to avoid nested directories

---

#### `GET /api/view-file/<path:file_path>`
Serve output files for viewing/downloading.

**Parameters**:
- `file_path` (path): Relative path to file

**Response**: File content with appropriate MIME type

**Security**: Validates file paths to prevent directory traversal

---

#### `POST /api/open-output-folder`
Open output folder in system file explorer.

**Request Body**:
```json
{
  "category": "chest",
  "file_path": "3D_vis/chest/3d_chest_reconstructed.png"
}
```

**Response**:
```json
{
  "success": true,
  "folder_path": "3D_vis/chest"
}
```

**Platform Support**:
- Windows: `explorer`
- macOS: `open`
- Linux: `xdg-open`

---

#### `GET /api/system-info`
Get system information (CPU, GPU, memory, storage).

**Response**:
```json
{
  "cpu": {
    "count": 8,
    "usage_percent": 25.5
  },
  "memory": {
    "total_gb": 16.0,
    "available_gb": 8.5,
    "used_percent": 46.9
  },
  "gpu": [
    {
      "id": 1,
      "name": "NVIDIA GeForce RTX 3050 Laptop GPU",
      "memory_total_mb": 4096,
      "memory_used_mb": 512,
      "utilization_percent": 15.0
    }
  ],
  "storage": {
    "total_gb": 500.0,
    "used_gb": 250.0,
    "free_gb": 250.0
  }
}
```

## 🎨 Frontend Implementation

### HTML Structure

#### Main Sections
1. **Header**: Title, subtitle, system info
2. **Configuration Card**: Category selection, output type filter
3. **Actions Card**: Operation buttons
4. **Progress & Status Card**: Real-time progress display
5. **Data Availability Panel**: Category data status
6. **Output Viewer**: Grid-based file display
7. **Authors & Developers**: Team information cards

### CSS Architecture

#### Color Scheme
- **Background**: `#AEDEFC` (light blue)
- **Primary**: `#4A90E2` (blue)
- **Success**: `#5CB85C` (green)
- **Warning**: `#F0AD4E` (orange)
- **Danger**: `#D9534F` (red)

#### Layout System
- **Grid Layout**: For output viewer and author cards
- **Flexbox**: For card layouts and button groups
- **Responsive**: Media queries for mobile devices

#### Animations
- **Hover Effects**: Card elevation, button scaling
- **Loading Spinners**: CSS keyframe animations
- **Progress Bars**: Smooth transitions

### JavaScript Architecture

#### Core Modules

**1. Operation Management**
```javascript
// Handles operation lifecycle
- startOperation(type, category)
- monitorOperation(operationId)
- updateProgress(operationId, progress)
- handleOperationComplete(operationId)
```

**2. Real-time Updates**
```javascript
// Server-Sent Events integration
- connectToSSE(operationId)
- parseSSEMessage(message)
- updateCLIOutput(content)
- updateProgressBar(progress)
```

**3. Output Management**
```javascript
// File browsing and display
- loadOutputFiles(category, outputType)
- displayOutputs(files)
- lazyLoadImages() // IntersectionObserver
- viewInFolder(filePath)
```

**4. UI State Management**
```javascript
// Button states and UI updates
- disableButtons()
- restoreButtonContent()
- updateCategoryInfo(category)
- refreshOutputs()
```

#### Key Features

**Lazy Loading**
- Uses `IntersectionObserver` API
- Loads images only when visible in viewport
- Limits displayed files to 20 most recent
- Reduces initial page load time

**Progress Parsing**
- Regex patterns for tqdm output:
  - `(\d+)%\|` - Percentage
  - `(\d+)/(\d+)` - Current/Total items
  - `\[(\d+):(\d+)<(\d+):(\d+),` - Time estimates
- Updates progress bar in real-time
- Calculates ETA from time estimates

**Error Handling**
- Try-catch blocks for async operations
- User-friendly error messages
- Automatic retry for failed operations
- Detailed error logging

## ⚙️ Backend Implementation

### Flask Application Structure

#### Operation Manager
```python
class OperationManager:
    - start_operation(op_id, type, params)
    - update_operation(op_id, status, progress)
    - get_operation(op_id)
    - list_operations()
```

**Operation States**:
- `pending`: Queued but not started
- `running`: Currently executing
- `completed`: Successfully finished
- `failed`: Error occurred
- `cancelled`: User cancelled

#### Subprocess Management

**Real-time Output Streaming**
```python
def run_subprocess_with_realtime_output(command, operation_id):
    # Uses subprocess.Popen with pipes
    # Streams stdout/stderr line-by-line
    # Parses tqdm progress output
    # Updates operation status in real-time
```

**Progress Parsing**
- Detects tqdm progress bars: `2%|1 | 2/120 [00:20<20:42, 10.53s/it]`
- Extracts percentage, current/total, time estimates
- Updates operation progress percentage
- Handles multiple progress formats

**Working Directory Management**
- Switches to `3D_vis/` directory for visualization scripts
- Handles relative vs. absolute paths
- Preserves original working directory

#### File System Operations

**Output File Discovery**
```python
def get_output_files(category, output_type=None):
    # Priority search order:
    # 1. 3D_vis/{category}/
    # 2. {category}/ (root)
    # 3. output/
    # Filters intermediate frames (angle_*.png)
    # Limits GIF search depth
    # Returns sorted file list
```

**Path Validation**
- Prevents directory traversal attacks
- Validates file paths before serving
- Checks file existence and permissions
- Sanitizes user inputs

### Server-Sent Events (SSE)

**Implementation**
```python
@app.route('/api/operation-output/<operation_id>')
def stream_operation_output(operation_id):
    def generate():
        # Stream output lines as SSE events
        # Format: data: {"type": "output", "content": "..."}\n\n
        # Include progress updates
        # Handle connection errors gracefully
    return Response(generate(), mimetype='text/event-stream')
```

**Event Types**:
- `output`: CLI output line
- `progress`: Progress update
- `error`: Error message
- `complete`: Operation finished

## 📊 Real-time Features

### Progress Tracking

#### Static Reconstruction
- **Progress Source**: Script completion status
- **Updates**: Start, running, complete states
- **Display**: Progress bar with percentage

#### GIF Animation
- **Progress Source**: tqdm output parsing
- **Format**: `X%|Y | Z/120 [time<time, rate]`
- **Updates**: Real-time frame rendering progress
- **Display**: Progress bar + frame counter

### CLI Output Streaming

**Features**:
- Real-time line-by-line output
- Color-preserving terminal output
- Scrollable output panel
- Auto-scroll to latest output
- Error highlighting

**Performance**:
- Buffered output (prevents overwhelming client)
- Efficient SSE encoding
- Connection management
- Automatic reconnection on disconnect

## 📁 File Management

### Output Directory Structure

```
3D_vis/
├── chest/
│   ├── 3d_chest_reconstructed.png          # Static output
│   └── elevation_20_sigma_0.55_alpha_0.3_axisoff/
│       ├── angle_0.png                     # Intermediate frames
│       ├── angle_1.png
│       ├── ...
│       └── animation.gif                   # Final GIF
├── foot/
│   └── ...
└── head/
    └── ...
```

### File Filtering

**Excluded Files**:
- Intermediate frames: `angle_*.png`
- Temporary files: `*.tmp`, `*.bak`
- Hidden files: `.*`

**Included Files**:
- Final outputs: `3d_*_reconstructed.png`
- GIF animations: `*.gif`
- Other visualization outputs

### File Serving

**MIME Types**:
- PNG: `image/png`
- GIF: `image/gif`
- Default: `application/octet-stream`

**Caching**:
- Browser caching for static assets
- No caching for dynamic outputs
- ETag support for efficient updates

## 🔧 Configuration

### GPU Configuration

**Default Settings**:
```python
DEFAULT_GPU_ID = "1"  # NVIDIA RTX 3050
```

**GPU Detection**:
- Uses `GPUtil` library
- Lists all available GPUs
- Displays GPU information in UI
- Validates GPU availability before operations

### Output Configuration

**Default Paths**:
- Static outputs: `3D_vis/{category}/`
- GIF outputs: `3D_vis/{category}/elevation_*/`
- Test outputs: `output/{method}/{category}/`

**File Naming**:
- Static: `3d_{category}_reconstructed.png`
- GIF: `animation.gif` (in elevation directory)
- Frames: `angle_{angle}.png`

### Performance Settings

**Concurrent Operations**:
- Maximum: 1 operation at a time (prevents GPU conflicts)
- Queue: Operations queued if one is running
- Timeout: 30 minutes per operation

**Memory Management**:
- Log buffer: 1000 lines per operation
- Output cache: 20 most recent files
- SSE buffer: 100 lines

## 🐛 Troubleshooting

### Common Issues

#### 1. "Failed to start visualization"
**Causes**:
- Script file not found
- GPU not available
- Permission errors

**Solutions**:
- Verify script exists: `ls 3D_vis/3D_vis_chest.py`
- Check GPU: `nvidia-smi`
- Verify permissions: `chmod +x 3D_vis/*.py`

#### 2. Progress Not Updating
**Causes**:
- SSE connection lost
- tqdm output format changed
- Script not producing output

**Solutions**:
- Check browser console for SSE errors
- Verify script produces tqdm output
- Check network connectivity

#### 3. Output Files Not Showing
**Causes**:
- Files in unexpected location
- File filtering too aggressive
- Permission issues

**Solutions**:
- Check actual file locations
- Verify file naming matches patterns
- Check file permissions

#### 4. GPU Not Utilized
**Causes**:
- CUDA not available
- Wrong GPU ID
- Script not GPU-enabled

**Solutions**:
- Verify CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- Check GPU ID matches system
- Verify scripts use GPU (check for `.cuda()` calls)

### Debug Mode

**Enable Debug Logging**:
```python
# In app.py
app.config['DEBUG'] = True
logging.basicConfig(level=logging.DEBUG)
```

**Check Operation Logs**:
- View `/api/operation-status/<id>` for detailed logs
- Check Flask console output
- Review browser developer console

### Performance Optimization

**For Large Outputs**:
- Enable lazy loading (already implemented)
- Limit displayed files (20 most recent)
- Use thumbnail previews
- Implement pagination for many files

**For Slow Operations**:
- Increase operation timeout
- Optimize script execution
- Use GPU acceleration
- Reduce output resolution

## 📈 Technical Specifications

### Browser Compatibility
- **Chrome/Edge**: Full support (recommended)
- **Firefox**: Full support
- **Safari**: Full support (SSE may have limitations)
- **Mobile**: Responsive design, limited functionality

### Performance Metrics
- **Initial Load**: < 2 seconds
- **Operation Start**: < 1 second
- **SSE Latency**: < 100ms
- **File Discovery**: < 500ms for 100 files

### Scalability
- **Concurrent Users**: Limited by Flask (single-threaded)
- **File Count**: Handles 1000+ files efficiently
- **Operation Queue**: Unlimited queued operations
- **Memory Usage**: ~100MB base + 10MB per operation

## 🔐 Security Considerations

### Input Validation
- Sanitize all user inputs
- Validate file paths
- Prevent directory traversal
- Check file extensions

### File Access
- Restrict to output directories only
- Validate file existence
- Check permissions
- Log file access

### Error Handling
- Don't expose internal paths
- Generic error messages for users
- Detailed logs for debugging
- Prevent information leakage

---

**For main project documentation, see [README.md](README.md)**

@echo off
echo Creating Volume Rendering Videos for All Available Body Parts...
echo.

echo Creating Chest Volume Rendering...
python volume_render_chest.py --category chest --data_type gt --fps 30 --rotation_angles 360

echo Creating Foot Volume Rendering...
python volume_render_chest.py --category foot --data_type gt --fps 30 --rotation_angles 360

echo Creating Head Volume Rendering...
python volume_render_chest.py --category head --data_type gt --fps 30 --rotation_angles 360

echo.
echo All Volume Rendering Videos Created Successfully!
echo Videos saved in respective category folders.
pause

@echo off
title Stop IndexTTS2 server
set PORT=7860
echo Stopping IndexTTS2 server on port %PORT%...

powershell -NoProfile -Command "$c = Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue; if ($c) { $ds = $c | Select-Object -ExpandProperty OwningProcess -Unique; foreach ($p in $ds) { $proc = Get-Process -Id $p -ErrorAction SilentlyContinue; if ($proc) { Write-Host ('  Stopping PID ' + $p + ' (' + $proc.ProcessName + ')'); Stop-Process -Id $p -Force } }; Start-Sleep -Seconds 1; Write-Host 'Server stopped.' } else { Write-Host 'No server listening on port %PORT% (already stopped?).' }"

pause
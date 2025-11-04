@echo off
REM 强制 Windows 终端使用 UTF-8 编码
chcp 65001 > nul

ECHO =======================================================
ECHO             AI4MW Web 统一开发服务器
ECHO =======================================================
ECHO.
ECHO 正在激活 Python 虚拟环境 (.venv)...
ECHO.

CALL .\.venv\Scripts\activate

ECHO.
ECHO 虚拟环境已激活。
ECHO 正在使用 Honcho 启动 Django 和 5 个 Celery 进程...
ECHO (按 Ctrl+C 停止所有进程)
ECHO.

rem honcho -f Procfile.windows start
rem ^^^ 上面的命令可能会因为 honcho 自身的 gbk bug 而失败

rem 我们使用一个 Python 包装器来强制 honcho 使用 UTF-8
python -c "import os; os.environ['PYTHONUTF8'] = '1'; import honcho.command; honcho.command.main()" -f Procfile.windows start
@echo off
set ext='.tif'

for /f %%i in ('dir /a:d /b') do (
    cd %%i
    dir *.tif /b>>tag.txt
    cd ..\)
@echo off
setlocal enabledelayedexpansion

echo Activating conda environment: torch_env
REM call conda activate torch_env
call conda activate soilcores

echo Changing to project directory
cd /d D:\monailabel\soilcores3dsegmentation

echo Starting threshold experiments...
echo.

REM Define experiments as variables
set "exp1=python iterate_thresholds.py --model "segresnet_dataset_2_default" --lower-values "0" --upper-values "20,40,60,80" --view "vertical" --no-confirm"
set "exp2=python iterate_thresholds.py --model "dynunet_dataset_2_100k" --lower-values "0" --upper-values "20,40,60,80" --view "vertical" --no-confirm"
set "exp3=python iterate_thresholds.py --model "dynunet_dataset_2_100k" --lower-values "0,20,40,60,80" --upper-values "100" --view "vertical" --no-confirm"
set "exp4=python iterate_thresholds.py --model "dynunet_dataset_2_100k" --lower-values "0" --upper-values "20,40,60,80" --view "horizontal" --no-confirm"
set "exp5=python iterate_thresholds.py --model "dynunet_dataset_2_100k" --lower-values "0,20,40,60,80" --upper-values "100" --view "horizontal" --no-confirm"
set "exp6=python iterate_thresholds.py --model "unet_dataset_2_100k" --lower-values "0" --upper-values "20,40,60,80" --view "vertical" --no-confirm"
set "exp7=python iterate_thresholds.py --model "unet_dataset_2_100k" --lower-values "0,20,40,60,80" --upper-values "100" --view "vertical" --no-confirm"
set "exp8=python iterate_thresholds.py --model "unet_dataset_2_100k" --lower-values "0" --upper-values "20,40,60,80" --view "horizontal" --no-confirm"
set "exp9=python iterate_thresholds.py --model "unet_dataset_2_100k" --lower-values "0,20,40,60,80" --upper-values "100" --view "horizontal" --no-confirm"
set "exp10=python iterate_thresholds.py --model "unet_dataset_2_default" --lower-values "0" --upper-values "20,40,60,80" --view "vertical" --no-confirm"
set "exp11=python iterate_thresholds.py --model "unet_dataset_2_default" --lower-values "0,20,40,60,80" --upper-values "100" --view "vertical" --no-confirm"
set "exp12=python iterate_thresholds.py --model "unet_dataset_2_default" --lower-values "0" --upper-values "20,40,60,80" --view "horizontal" --no-confirm"
set "exp13=python iterate_thresholds.py --model "unet_dataset_2_default" --lower-values "0,20,40,60,80" --upper-values "100" --view "horizontal" --no-confirm"
set "exp14=python iterate_thresholds.py --model "dataset_2adamw_100k_num_heads_2" --lower-values "0" --upper-values "20,40,60,80" --view "vertical" --no-confirm"
set "exp15=python iterate_thresholds.py --model "dataset_2adamw_100k_num_heads_2" --lower-values "0,20,40,60,80" --upper-values "100" --view "vertical" --no-confirm"
set "exp16=python iterate_thresholds.py --model "dataset_2adamw_100k_num_heads_2" --lower-values "0" --upper-values "20,40,60,80" --view "horizontal" --no-confirm"
set "exp17=python iterate_thresholds.py --model "dataset_2adamw_100k_num_heads_2" --lower-values "0,20,40,60,80" --upper-values "100" --view "horizontal" --no-confirm"
set "exp18=python iterate_thresholds.py --model "segresnet_dataset_2_default" --lower-values "0" --upper-values "20" --view "horizontal" --no-confirm"

REM Counter for experiments
set /a exp_count=0

REM Run experiment 1 (completed - removed)
REM set /a exp_count+=1
REM echo ============================================================
REM echo Running Experiment !exp_count!: SegResNet with varying upper thresholds and vertical view
REM echo ============================================================
REM !exp1!

REM Run experiment 2 (DynUNet with varying upper thresholds and vertical view) - COMPLETED
REM set /a exp_count+=1
REM echo ============================================================
REM echo Running Experiment !exp_count!: DynUNet with varying upper thresholds and vertical view
REM echo ============================================================
REM !exp2!

REM Run experiment 3 (DynUNet with varying lower thresholds and vertical view)
set /a exp_count+=1
echo.
echo ============================================================
echo Running Experiment !exp_count!: DynUNet with varying lower thresholds and vertical view
echo ============================================================
REM !exp3!

REM Run experiment 4 (DynUNet with varying upper thresholds and horizontal view)
set /a exp_count+=1
echo.
echo ============================================================
echo Running Experiment !exp_count!: DynUNet with varying upper thresholds and horizontal view
echo ============================================================
REM!exp4!

REM Run experiment 5 (DynUNet with varying lower thresholds and horizontal view)
set /a exp_count+=1
echo.
echo ============================================================
echo Running Experiment !exp_count!: DynUNet with varying lower thresholds and horizontal view
echo ============================================================
REM !exp5!

REM Run experiment 6 (UNet 100k with varying lower thresholds and vertical view)
set /a exp_count+=1
echo.
echo ============================================================
echo Running Experiment !exp_count!: UNet 100k with varying lower thresholds and vertical view
echo ============================================================
REM !exp6!

REM Run experiment 7 (UNet 100k with varying upper thresholds and vertical view)
set /a exp_count+=1
echo.
echo ============================================================
echo Running Experiment !exp_count!: UNet 100k with varying upper thresholds and vertical view
echo ============================================================
REM !exp7!

REM Run experiment 8 (UNet 100k with varying lower thresholds and horizontal view)
set /a exp_count+=1
echo.
echo ============================================================
echo Running Experiment !exp_count!: UNet 100k with varying lower thresholds and horizontal view
echo ============================================================
REM !exp8!

REM Run experiment 9 (UNet 100k with varying upper thresholds and horizontal view)
set /a exp_count+=1
echo.
echo ============================================================
echo Running Experiment !exp_count!: UNet 100k with varying upper thresholds and horizontal view
echo ============================================================
REM !exp9!

REM Run experiment 10 (UNet default with varying lower thresholds and vertical view)
set /a exp_count+=1
echo.
echo ============================================================
echo Running Experiment !exp_count!: UNet default with varying lower thresholds and vertical view
echo ============================================================
REM !exp10!

REM Run experiment 11 (UNet default with varying upper thresholds and vertical view)
set /a exp_count+=1
echo.
echo ============================================================
echo Running Experiment !exp_count!: UNet default with varying upper thresholds and vertical view
echo ============================================================
REM !exp11!

REM Run experiment 12 (UNet default with varying lower thresholds and horizontal view)
set /a exp_count+=1
echo.
echo ============================================================
echo Running Experiment !exp_count!: UNet default with varying lower thresholds and horizontal view
echo ============================================================
REM !exp12!

REM Run experiment 13 (UNet default with varying upper thresholds and horizontal view)
set /a exp_count+=1
echo.
echo ============================================================
echo Running Experiment !exp_count!: UNet default with varying upper thresholds and horizontal view
echo ============================================================
REM !exp13!

REM Run experiment 14 (UNETR with varying lower thresholds and vertical view)
set /a exp_count+=1
echo.
echo ============================================================
echo Running Experiment !exp_count!: UNETR with varying lower thresholds and vertical view
echo ============================================================
REM !exp14!

REM Run experiment 15 (UNETR with varying upper thresholds and vertical view)
set /a exp_count+=1
echo.
echo ============================================================
echo Running Experiment !exp_count!: UNETR with varying upper thresholds and vertical view
echo ============================================================
REM !exp15!

REM Run experiment 16 (UNETR with varying lower thresholds and horizontal view)
set /a exp_count+=1
echo.
echo ============================================================
echo Running Experiment !exp_count!: UNETR with varying lower thresholds and horizontal view
echo ============================================================
REM !exp16!

REM Run experiment 17 (UNETR with varying upper thresholds and horizontal view)
set /a exp_count+=1
echo.
echo ============================================================
echo Running Experiment !exp_count!: UNETR with varying upper thresholds and horizontal view
echo ============================================================
REM !exp17!

REM Run experiment 18 (SegResNet with varying lower thresholds and horizontal view)
set /a exp_count+=1
echo.
echo ============================================================
echo Running Experiment !exp_count!: SegResNet with varying lower thresholds and horizontal view
echo ============================================================
!exp18!

echo.
echo ============================================================
echo All !exp_count! experiments completed!
echo ============================================================
REM pause
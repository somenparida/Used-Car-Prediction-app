<#
Helper PowerShell script for common tasks in this project.

Usage:
  .\run.ps1 -Task train
  .\run.ps1 -Task streamlit
  .\run.ps1 -Task test
#>

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("train","streamlit","test")]
    [string]$Task
)

Set-Location -Path $PSScriptRoot

switch ($Task) {
    'train' {
        python train.py
    }
    'streamlit' {
        streamlit run app.py
    }
    'test' {
        pytest -q
    }
}

<#
PowerShell helper: download GitHub Actions artifact, extract, and print CSV head
Usage:
  1) Create a short-lived PAT with Actions read / repo read permissions.
  2) Run in PowerShell (from repo root):
       pwsh scripts\download_demo_artifact.ps1 -ArtifactId 5211661711 -Lines 20
  3) Revoke the PAT immediately after use.

Notes:
 - Default artifact id is set to the demo artifact we observed.
 - The script will prompt for the PAT (not echoed).
#>
param(
    [string]$ArtifactId = "5211661711",
    [int]$Lines = 20,
    [string]$OutZip = "artifact.zip",
    [string]$OutDir = "artifact_zip"
)

function Write-ErrAndExit([string]$msg){
    Write-Host "ERROR: $msg" -ForegroundColor Red
    exit 1
}

try{
    $secure = Read-Host -Prompt "Enter GitHub PAT (Actions read)" -AsSecureString
    $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    $token = [Runtime.InteropServices.Marshal]::PtrToStringAuto($ptr)
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr)
}catch{
    Write-ErrAndExit "Could not read token."
}

if(-not $token){ Write-ErrAndExit "Empty token provided. Aborting." }

$artifactUrl = "https://api.github.com/repos/Solventerritory/FarmFederate-Advisor/actions/artifacts/$ArtifactId/zip"
Write-Host "Downloading artifact $ArtifactId from GitHub..."

try{
    Invoke-WebRequest -Uri $artifactUrl -Headers @{ Authorization = "token $token"; 'User-Agent' = 'GH-Artifact-Downloader' } -OutFile $OutZip -UseBasicParsing -ErrorAction Stop
    Write-Host "Downloaded to $OutZip"
}catch{
    Write-ErrAndExit "Download failed: $($_.Exception.Message)"
}

# Extract
if(Test-Path $OutDir){ Remove-Item -Recurse -Force $OutDir }
try{
    Expand-Archive -LiteralPath $OutZip -DestinationPath $OutDir -Force
    Write-Host "Extracted to $OutDir"
}catch{
    Write-ErrAndExit "Failed to extract zip: $($_.Exception.Message)"
}

# Search for CSV(s)
$csvFiles = Get-ChildItem -Path $OutDir -Recurse -File -Include "*demo_farm*history*.csv","*demo_farm*history.csv","*.csv" 2>$null
if(-not $csvFiles){
    Write-Host "No CSV files found in artifact. Listing extracted files:"
    Get-ChildItem -Path $OutDir -Recurse | ForEach-Object { Write-Host $_.FullName }
    exit 0
}

# If multiple, pick the one with demo_farm in name first
$selected = $csvFiles | Where-Object { $_.Name -match 'demo_farm' } | Select-Object -First 1
if(-not $selected){ $selected = $csvFiles | Select-Object -First 1 }

Write-Host "Found CSV: $($selected.FullName)" -ForegroundColor Green
Write-Host "--- First $Lines lines ---"
Get-Content -Path $selected.FullName -Encoding UTF8 | Select-Object -First $Lines | ForEach-Object { Write-Host $_ }

Write-Host "--- End ---"
Write-Host "Tip: Revoke the PAT you used immediately after this operation (GitHub -> Settings -> Developer settings -> Personal access tokens)." -ForegroundColor Yellow

# clear memory of token variable
$token = $null
[GC]::Collect(); [GC]::WaitForPendingFinalizers()

exit 0

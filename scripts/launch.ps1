# Start a long job detached from the Claude Code session so a restart doesn't kill it.
# usage: powershell -File scripts/launch.ps1 <name> <args passed to python...>
param([Parameter(Mandatory=$true)][string]$Name, [Parameter(ValueFromRemainingArguments=$true)][string[]]$PyArgs)
$repo = Split-Path -Parent $PSScriptRoot
$log = Join-Path $repo "runs\logs\$Name.log"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$argline = "run --no-capture-output -n minute-agent python " + ($PyArgs -join " ")
$p = Start-Process -FilePath "conda.exe" -ArgumentList $argline -WorkingDirectory $repo -WindowStyle Hidden `
     -RedirectStandardOutput $log -RedirectStandardError "$log.err" -PassThru
"started $Name pid=$($p.Id) log=$log"

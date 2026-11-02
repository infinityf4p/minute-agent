# Start a long job detached from the Claude Code session so a restart doesn't kill it.
# usage: powershell -File scripts/launch.ps1 <job_name> <script.py> [script args...]
# (no declared parameters, so script options such as --name are passed through untouched)
$job = $args[0]
$rest = $args[1..($args.Count - 1)]
$repo = Split-Path -Parent $PSScriptRoot
$log = Join-Path $repo "runs\logs\$job.log"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$argline = "run --no-capture-output -n minute-agent python " + ($rest -join " ")
$p = Start-Process -FilePath "conda.exe" -ArgumentList $argline -WorkingDirectory $repo -WindowStyle Hidden `
     -RedirectStandardOutput $log -RedirectStandardError "$log.err" -PassThru
"started $job pid=$($p.Id) log=$log"

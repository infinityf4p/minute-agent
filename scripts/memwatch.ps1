# Sample python process memory every 3 s until the given log shows a result; print the peaks.
param([string]$Log)
$peak = @{}; $minAvail = 999999
while ($true) {
  Get-Process python -ErrorAction SilentlyContinue | ForEach-Object {
    $gb = [math]::Round($_.WorkingSet64 / 1GB, 2); $priv = [math]::Round($_.PrivateMemorySize64 / 1GB, 2)
    $k = "$($_.Id)"; if (-not $peak.ContainsKey($k) -or $peak[$k][0] -lt $priv) { $peak[$k] = @($priv, $gb, (Get-Date -Format HH:mm:ss)) }
  }
  $a = [int](Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue; if ($a -lt $minAvail) { $minAvail = $a; $at = Get-Date -Format HH:mm:ss }
  if ((Test-Path $Log) -and (Select-String -Path $Log -Pattern 'obj=|Traceback' -Quiet)) { break }
  Start-Sleep 3
}
"min available $minAvail MB at $at"
$peak.GetEnumerator() | Sort-Object { $_.Value[0] } -Descending | Select-Object -First 6 | ForEach-Object { "pid $($_.Key): private peak $($_.Value[0]) GB (ws $($_.Value[1]) GB) at $($_.Value[2])" }

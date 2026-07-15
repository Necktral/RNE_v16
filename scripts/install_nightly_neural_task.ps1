[CmdletBinding()]
param(
    [string]$TaskName = "RNFE-Neural-Nightly",
    [string]$WslDistribution = "Ubuntu-24.04",
    [string]$RunAt = "22:00",
    [string]$Launcher = "/home/wis/Desarrollo/RNE_v16_worktrees/neural-agent-suite/scripts/run_nightly_neural_supervisor.sh"
)

$ErrorActionPreference = "Stop"
$at = [DateTime]::ParseExact($RunAt, "HH:mm", [Globalization.CultureInfo]::InvariantCulture)
$action = New-ScheduledTaskAction `
    -Execute "$env:SystemRoot\System32\wsl.exe" `
    -Argument "-d $WslDistribution -- $Launcher"
$trigger = New-ScheduledTaskTrigger -Daily -At $at
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -WakeToRun `
    -ExecutionTimeLimit (New-TimeSpan -Hours 10) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 5) `
    -MultipleInstances IgnoreNew `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "RNFE neural rehearsal and overnight campaign supervisor in native WSL ext4" `
    -Force | Out-Null

Get-ScheduledTask -TaskName $TaskName |
    Select-Object TaskName, State, Author

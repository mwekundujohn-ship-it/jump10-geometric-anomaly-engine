param(
    [Parameter(Mandatory=$true)][string]$Job,
    [Parameter(Mandatory=$true)][string]$Log
)
$err = $Log + ".err"
Start-Process -FilePath "python" `
    -ArgumentList @("C:\cluster\phase1\phase1_worker.py", "--job", $Job) `
    -RedirectStandardOutput $Log `
    -RedirectStandardError $err `
    -WindowStyle Hidden

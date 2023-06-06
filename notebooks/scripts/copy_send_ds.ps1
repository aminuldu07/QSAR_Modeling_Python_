# PW script to copy all SEND folders (i.e, those that contain define.xml) from the FDA's EDR folder to a local computer 
# Script takes two paramters:
#     -edrDir: The directory mapping of the EDR on the local computer (e.g., Z:\)
#     -targetDir: location on the local computer to copy the SEND datasets
#
#
# Send folders will be copied to $targetDir in a folder called SEND_DATASETS.
# Will also create an ISO 8601 formatted timestamp in a file '.lastupdate.log'.
# This file is used when the script is re-run to only look at the most recently 
# modified folder from the previous run.


# param collection
 param (
    [Parameter(Mandatory=$true)][string]$edrDir,
    [Parameter(Mandatory=$true)][string]$targetDir
 )

# string together folders and files
$targetDir = $targetDir + "\SEND_DATASETS\"
$updateFile = $targetDir + "\.lastupdate.log"

# check if the SEND_DATASETS folder has been created, 
# if so assume the script has been run before and 
# read the '.lastupdate.log' file to avoid redundant 
# folder copying
if(-not (Test-Path $targetDir)) {

	New-Item -Path $targetDir -ItemType "directory" -Force | Out-Null
	New-Item -Path $updateFile -ItemType File -Force | Out-Null
	$currentDate = Get-Date -Format "o"
	[datetime]$startDate = "12/01/2017"

	
	
} else {
	[datetime]$startDate = Get-Content -Path $updateFile -Tail 1
    
    # write new start date
    $currentDate = Get-Date -Format "o"
    
} 	

# Sort folders by modified data
$folders = Get-ChildItem $edrDir  | Where-Object { $_.LastWriteTime -ge $startDate } | sort LastWriteTime

# keep counters for printing and updating 
$year = "2017"
$month - "01"

# count the total number of SEND datasets
$sendCounter = 0

foreach ($folder in $folders) {

    # just some logic to check what year/month is currently being copied 
	$newYear = ($folder.LastWriteTime).Year
	if ($newYear -ne $year) {
		$year = $newYear
		$month= ($folder.LastWriteTime).Month
		Write-host "adding folders from " $month"/"$year
	}

	$newMonth = ($folder.LastWriteTime).Month
	if ($newMonth -ne $month) {
		$month = $newMonth
		Write-host "adding folders from " $month"/"$year 
	}

    # Within each app folder there are several number folders (e.g., 001, 002) which can all
    # potentially contain send data.  Restrict searching these folders by modified dates as 
    # wll.
    # Then, for folders meeting this exclusion, create a recursive search for 'define.xml', 
    # but restrict this search to the \m4\datasets\ subdirectory 
	$appNumber = $folder
	$searchPath = "$($edrDir)" + "$($folder)"
	$newfolders = Get-ChildItem $searchPath | Where-Object {($_.LastWriteTime -ge $startDate ) -and ($_.PSIsContainer)} 
	foreach ($newfolder in $newfolders) {
		$m4dir = "$($searchPath)" + "\" + "$($newfolder)"  + "\m4\datasets"
		if (Test-Path $m4dir) {

            # another counter for SEND datasets in a given application number
            # this is how the new studies will be named in the copied directory 
            # e.g., 
            # -NDA/
            #     /0/$sendxptfiles
            #     /1/$sendxptfiles
            # etc. 
			$DScounter = 0
			Get-ChildItem -Path $m4dir -Filter define.xml -Recurse -ErrorAction SilentlyContinue -Force | % {
     			$defineXMLdir =  Split-Path -Path $_.FullName
			
            
			$entireFolder = $defineXMLdir + "\*"
			$targetDSDir = $targetDir + $appNumber + "\" + $DScounter

            # get the time this particular SEND folder was last modified 
            $sendFolderLastModifiedTime = [datetime]$folderLastWriteTime = (Get-ItemProperty -Path  $defineXMLdir -Name LastWriteTime).lastwritetime
            $sendFolderLastModifiedTime = $sendFolderLastModifiedTime.ToString('o')
            $sendFolderLastModifiedFile = $targetDSDir + "\.lastupdate_EDR.log"

			New-Item -Path $targetDSDir -ItemType "directory" -Force | Out-Null
			Copy-Item $entireFolder $targetDSDir

            
            # write last modified date from EDR
            Set-Content -Path $sendFolderLastModifiedFile -Value $sendFolderLastModifiedTime  

			$DScounter++
			$sendCounter++
			Write-host "Copied " $sendCounter " folders"
				
			}
		}
	}
    
    $EDRSEARCHFOLDER = "$($edrDir)" + "$($folder)"
    [datetime]$folderLastWriteTime = (Get-ItemProperty -Path  $EDRSEARCHFOLDER -Name LastWriteTime).lastwritetime
    Write-host $EDRSEARCHFOLDER $folderLastWriteTime
    Add-Content -Path $updateFile -Value $folderLastWriteTime.ToString('o')

}


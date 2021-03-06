# Quick and dirty program to pull down operational 
# NAM nest 6-hour forecasts for Puerto Rico

# Logan Karsten
# National Center for Atmospheric Research
# Research Applications Laboratory

import datetime
import urllib
from urllib import request
import http
from http import cookiejar
import os
import sys
import smtplib
from email.mime.text import MIMEText
import shutil

def errOut(msgContent,emailTitle,emailRec,lockFile):
	msg = MIMEText(msgContent)
	msg['Subject'] = emailTitle
	msg['From'] = emailRec
	msg['To'] = emailRec
	s = smtplib.SMTP('localhost')
	s.sendmail(emailRec,[emailRec],msg.as_string())
	s.quit()
	# Remove lock file
	os.remove(lockFile)
	sys.exit(1)

def warningOut(msgContent,emailTitle,emailRec,lockFile):
	msg = MIMEText(msgContent)
	msg['Subject'] = emailTitle
	msg['From'] = emailRec
	msg['To'] = emailRec
	s = smtplib.SMTP('localhost')
	s.sendmail(emailRec,[emailRec],msg.as_string())
	s.quit()
	sys.exit(1)

def msgUser(msgContent,msgFlag):
	if msgFlag == 1:
		print(msgContent)

outDir = "/glade/p/cisl/nwc/nwm_forcings/Forcing_Inputs/NAM_Nest_Puerto_Rico"
tmpDir = "/glade/scratch/karsten"
lookBackHours = 72 # How many hours to look for data.....
cleanBackHours = 240 # Period between this time and the beginning of the lookback period to cleanout old data.  
lagBackHours = 6 # Wait at least this long back before searching for files. 
dNowUTC = datetime.datetime.utcnow()
dNow = datetime.datetime(dNowUTC.year,dNowUTC.month,dNowUTC.day,dNowUTC.hour)
ncepHTTP = "https://ftp.ncep.noaa.gov/data/nccf/com/nam/prod"

# Define communication of issues.
emailAddy = 'jon.doe@youremail.com'
errTitle = 'Error_get_NAM_Nest_PR'
warningTitle = 'Warning_get_NAM_Nest_PR'

pid = os.getpid()
lockFile = tmpDir + "/GET_NAM_Nest_PR.lock"

# First check to see if lock file exists, if it does, throw error message as
# another pull program is running. If lock file not found, create one with PID.
if os.path.isfile(lockFile):
	fileLock = open(lockFile,'r')
	pid = fileLock.readline()
	warningMsg =  "WARNING: Another NAM Nest Puerto Rico Fetch Program Running. PID: " + pid
	warningOut(warningMsg,warningTitle,emailAddy,lockFile)
else:
	fileLock = open(lockFile,'w')
	fileLock.write(str(os.getpid()))
	fileLock.close()

for hour in range(cleanBackHours,lookBackHours,-1):
	# Calculate current hour.
	dCurrent = dNow - datetime.timedelta(seconds=3600*hour)

	# Go back in time and clean out any old data to conserve disk space. 
	if dCurrent.hour != 0 and dCurrent.hour != 12 and dCurrent.hour != 6 and dCurrent.hour != 18:
		continue # NAM nest data only available every six hours. 
	else:
		# Compose path to directory containing data. 
		cleanDir = outDir + "/nam." + dCurrent.strftime('%Y%m%d')

		# Check to see if directory exists. If it does, remove it. 
		if os.path.isdir(cleanDir):
			print("Removing old data from: " + cleanDir)
			shutil.rmtree(cleanDir)

# Now that cleaning is done, download files within the download window. 
for hour in range(lookBackHours,lagBackHours,-1):
	# Calculate current hour.
	dCurrent = dNow - datetime.timedelta(seconds=3600*hour)

	if dCurrent.hour != 0 and dCurrent.hour != 12 and dCurrent.hour != 6 and dCurrent.hour != 18:
		continue # NAM nest data only available every six hours. 
	else:
		namOutDir = outDir + "/nam." + dCurrent.strftime('%Y%m%d')
		httpDownloadDir = ncepHTTP + "/nam." + dCurrent.strftime('%Y%m%d')
		if not os.path.isdir(namOutDir):
			os.mkdir(namOutDir)

		nFcstHrs = 60
		for hrDownload in range(1,nFcstHrs + 1):
			fileDownload = "nam.t" + dCurrent.strftime('%H') + \
						   "z.priconest.hiresf" + str(hrDownload).zfill(2) + \
						   ".tm00.grib2"
			url = httpDownloadDir + "/" + fileDownload
			outFile = namOutDir + "/" + fileDownload
			if os.path.isfile(outFile):
				continue
			try:
				request.urlretrieve(url,outFile)
			except:
				print("Unable to retrieve: " + url)
				print("Data may not available yet...")
				continue

# Remove the LOCK file.
os.remove(lockFile)


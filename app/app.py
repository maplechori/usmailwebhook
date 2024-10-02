import tempfile

import torch
import cv2
import requests
import os
import logging
import pathlib
import platform
import shutil

from huggingface_hub import hf_hub_download
from flask import Flask, request, Response, json

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger('USMailDetect')
logger.setLevel(logging.INFO)


@app.route('/', methods=['POST'] )
def detect_usmail():

	logger.info("DETECT USMAIL")
	url = request.json.get('url', None)

	if not url:
		return "No URL"

	plt = platform.system()
	if plt == 'Windows':
		pathlib.PosixPath = pathlib.WindowsPath
	else:
		pathlib.WindowsPath = pathlib.PosixPath

	logger.info(f"Downloading image {url}")
	response = requests.get(url, stream=True)

	if not url.endswith('jpg'):
		return Response(
			response=json.dumps({
				'message': 'This service only accepts JPEG images'
			}),
			status=200,
			mimetype='application/json'
		)

	temporary_file = tempfile.NamedTemporaryFile(delete=False)

	with open(temporary_file.name, "wb") as jpg:
		logger.info(f"Writing temporary file: {jpg.name}")
		shutil.copyfileobj(response.raw, jpg)
		temporary_file.close()
		del response

	us_mail_path = hf_hub_download(repo_id="maplechori/usmaildetection", filename="usmail.pt")
	model = torch.hub.load('ultralytics/yolov5', 'custom', path=us_mail_path)

	img = cv2.imread(temporary_file.name)
	os.unlink(temporary_file.name)

	if img is not None and img.any():
		results = model(img)

		detections = results.pandas().xyxy[0]

		if not detections.empty:
			df = detections.loc[detections["name"] == "us_mail_symbol"]
			if not df.empty and df["confidence"].iloc[0] > 0.20:
				logger.info("Found US Mail symbol!")
				confidence = df["confidence"].iloc[0]

				return Response(response=json.dumps({
					'message': f'US mail symbol identified on image with {confidence} confidence'
				}),
				status=200,
				mimetype='application/json')

	return Response(response=json.dumps({
			'message': 'Unable to identify object in file'
		}),
		status=200,
		mimetype='application/json'
	)


if __name__ == '__main__':
	app.run(host='0.0.0.0', port=4343)
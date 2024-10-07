
import requests
import os
import logging
import warnings

from urllib3.exceptions import InsecureRequestWarning

synology_ip = os.getenv('SYNOLOGY_IP')
synology_port = os.getenv('SYNOLOGY_PORT')
synology_event_token = os.getenv('SYNOLOGY_WEBHOOK_TOKEN')

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger('USMailDetect')
logger.setLevel(logging.INFO)

warnings.simplefilter(action='ignore', category=InsecureRequestWarning)


def push_mobile_message():
    """Push mobile message to app"""
    notification_url = f'http://{synology_ip}:{synology_port}/webapi/SurveillanceStation/Webhook/Incoming/v1'
    notification_payload = {
        'token': synology_event_token,
        'text1': 'USPS Mail detection'
    }

    notification_response = requests.get(notification_url, params=notification_payload, verify=False)

    if notification_response.status_code == 200:
        logging.info('Notification sent successfully')
    else:
        logging.error('Failed to send notification')
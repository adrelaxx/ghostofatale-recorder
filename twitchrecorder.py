import datetime
import enum
import getopt
import logging
import os
import subprocess
import sys
import shutil
import time
import requests
import config


class TwitchResponseStatus(enum.Enum):
    ONLINE = 0
    OFFLINE = 1
    NOT_FOUND = 2
    UNAUTHORIZED = 3
    ERROR = 4


class TwitchRecorder:
    def __init__(self):
        self.ffmpeg_path = "ffmpeg"
        self.refresh = 15
        self.root_path = config.root_path

        self.username = config.username
        self.quality = "best"

        self.client_id = config.client_id
        self.client_secret = config.client_secret
        self.token_url = (
            "https://id.twitch.tv/oauth2/token"
            "?client_id=" + self.client_id +
            "&client_secret=" + self.client_secret +
            "&grant_type=client_credentials"
        )
        self.url = "https://api.twitch.tv/helix/streams"
        self.access_token = self.fetch_access_token()

    def fetch_access_token(self):
        token_response = requests.post(self.token_url, timeout=15)
        token_response.raise_for_status()
        return token_response.json()["access_token"]

    def run(self):
        recorded_path = os.path.join(self.root_path, "recorded", self.username)
        processed_path = os.path.join(self.root_path, "processed", self.username)

        os.makedirs(recorded_path, exist_ok=True)
        os.makedirs(processed_path, exist_ok=True)

        logging.info("Monitoring %s every %s seconds", self.username, self.refresh)

        self.loop_check(recorded_path, processed_path)

    def compress_video(self, input_file, output_file):
        try:
            logging.info("Starting 480p compression...")

            subprocess.call([
                self.ffmpeg_path,
                "-y",
                "-i", input_file,
                "-vf", "scale=-2:480",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "30",
                "-profile:v", "high",
                "-level", "3.1",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-c:a", "aac",
                "-b:a", "64k",
                output_file
            ])

            logging.info("Compression finished. Removing original file.")
            os.remove(input_file)

        except Exception as e:
            logging.error("Compression error: %s", e)

    def check_user(self):
        try:
            headers = {
                "Client-ID": self.client_id,
                "Authorization": "Bearer " + self.access_token
            }
            r = requests.get(
                self.url + "?user_login=" + self.username,
                headers=headers,
                timeout=15
            )
            r.raise_for_status()
            info = r.json()

            if not info["data"]:
                return TwitchResponseStatus.OFFLINE, None
            else:
                return TwitchResponseStatus.ONLINE, info

        except requests.exceptions.RequestException as e:
            if e.response and e.response.status_code == 401:
                return TwitchResponseStatus.UNAUTHORIZED, None
            return TwitchResponseStatus.ERROR, None

    def loop_check(self, recorded_path, processed_path):
        while True:
            status, info = self.check_user()

            if status == TwitchResponseStatus.OFFLINE:
                logging.info("User offline...")
                time.sleep(self.refresh)

            elif status == TwitchResponseStatus.UNAUTHORIZED:
                logging.info("Refreshing Twitch token...")
                self.access_token = self.fetch_access_token()

            elif status == TwitchResponseStatus.ERROR:
                logging.error("API error, retrying in 60 seconds")
                time.sleep(60)

            elif status == TwitchResponseStatus.ONLINE:
                logging.info("User ONLINE. Recording...")

                channel = info["data"][0]
                filename = (
                    self.username + " - " +
                    datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") +
                    ".mp4"
                )

                recorded_file = os.path.join(recorded_path, filename)
                processed_file = os.path.join(processed_path, filename)

                subprocess.call([
                    "streamlink",
                    "--twitch-disable-ads",
                    "twitch.tv/" + self.username,
                    self.quality,
                    "-o", recorded_file
                ])

                logging.info("Live ended. Starting compression...")

                if os.path.exists(recorded_file):
                    self.compress_video(recorded_file, processed_file)

                logging.info("Back to monitoring...")
                time.sleep(self.refresh)


def main(argv):
    logging.basicConfig(level=logging.INFO)
    recorder = TwitchRecorder()
    recorder.run()


if __name__ == "__main__":
    main(sys.argv[1:])

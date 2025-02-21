import requests

url = "https://drive.google.com/file/d/1l91mBAJeu7IDWbBg1taKNIZglJ9yMstE/view?usp=drive_link/detection_model_video.h5"
r = requests.get(url, allow_redirects=True)
open("detection_model_video.h5", "wb").write(r.content)

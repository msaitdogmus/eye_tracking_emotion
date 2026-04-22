# main.py
import argparse
import gaze_core as gc
from gaze_live import live_mode

#PS C:\Users\VIP SAID\PycharmProjects\LST_Software\Day3\sistem_mp_test> python gaze_video.py --video "C:\Users\VIP SAID\PycharmProjects\LST_Software\videos\Erkam_1.mp4"

def parse_args():
    ap = argparse.ArgumentParser(description="Gaze Tracker — Live")
    ap.add_argument("--cam",        type=int,  default=gc.CAM_INDEX,
                    help="camera index (default: 0)")
    ap.add_argument("--em-model",   type=str,  default=gc.EM_MODEL_NAME,
                    help=f"EmotiEffLib model name (default: {gc.EM_MODEL_NAME})")
    ap.add_argument("--no-emotion", action="store_true",
                    help="disable emotion detection entirely")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    gc.EM_MODEL_NAME = args.em_model
    live_mode(cam_index=args.cam,
              em_model=args.em_model,
              no_emotion=args.no_emotion)
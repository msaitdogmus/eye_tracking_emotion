# gaze_live.py
import os
import time
import cv2
import mediapipe as mp
import numpy as np

import gaze_core as gc


def live_mode(cam_index: int = 0, em_model: str = None, no_emotion: bool = False):
    gc.ensure_model()
    detector = gc.build_detector()

    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  gc.CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, gc.CAM_H)
    cap.set(cv2.CAP_PROP_FPS,          gc.CAM_FPS)

    light = gc.LightAdaptor()
    kf    = gc.KF2DAdaptive(fps=gc.CAM_FPS)

    # Tek profil: calib_data.npz varsa yükle, yoksa kalibrasyon yap
    mapper = gc.AffineMapper()
    mapper.load_single()
    mode = "KALIB" if not mapper.ready else "RUN"

    calib_ui = gc.CalibratorUI(label="Kalibrasyon (noktalara bak)")

    head   = gc.HeadCenterTracker(init_frames=60, alpha=0.01)
    fusion = gc.HeadEyeFusion(head)
    voter  = gc.DirectionVoter()
    offmon = gc.OffscreenMonitor(hold=gc.OFFSCREEN_HOLD)

    if mode == "RUN":
        fusion.reset_to(0.5, 0.5)
        voter.seed("CENTER")

    # Emotion
    em_enabled      = not no_emotion
    emotion_tracker = gc.EmotionTracker(em_model or gc.EM_MODEL_NAME) if em_enabled else None
    em_dominant     = "neutral"
    _em_feed_cnt    = 0

    fps    = 0.0
    t_fps  = time.time()
    fcount = 0

    win = "Gaze Live (N:kalib  R:sifirla  ESC:cikis)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def _enter_calib():
        nonlocal mode, calib_ui
        mode     = "KALIB"
        calib_ui = gc.CalibratorUI(label="Kalibrasyon (noktalara bak)")
        mapper.reset()
        kf.reset(); fusion.reset(); voter.reset(); head.reset()
        offmon.force_close()
        if emotion_tracker:
            emotion_tracker.reset()

    def _reset_calib():
        if os.path.exists(gc.CALIB_FILE):
            os.remove(gc.CALIB_FILE)
        _enter_calib()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (gc.INFER_W, gc.INFER_H), interpolation=cv2.INTER_AREA)
            now   = time.time()

            fcount += 1
            if fcount % 10 == 0:
                fps   = 10.0 / max(now - t_fps, 1e-6)
                t_fps = now

            frame     = light.process(frame)
            light_str = light.status()

            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = detector.detect_for_video(mp_image, int(now * 1000))
            face_ok  = bool(result.face_landmarks)

            stable_dir = "NO_FACE"
            conf = 0.0; gp = None; fusion_w = 0.0

            if not face_ok:
                gc.draw_hud(frame, stable_dir, conf, gp, fps,
                            off_active=False, off_dur=0.0, fusion_w=fusion_w,
                            light_str=light_str, mode=mode,
                            emotion_label=em_dominant,
                            emotion_available=(emotion_tracker.available if emotion_tracker else False))
                cv2.imshow(win, frame)
                k = cv2.waitKey(1) & 0xFF
                if k == 27: break
                if k in (ord('n'), ord('N')): _enter_calib()
                if k in (ord('r'), ord('R')): _reset_calib()
                continue

            face = result.face_landmarks[0]

            # Emotion
            if emotion_tracker:
                _em_feed_cnt += 1
                if _em_feed_cnt % gc.EM_FEED_EVERY == 0:
                    emotion_tracker.feed(gc.extract_face_crop(frame, face, align=True))
                _, em_dominant = emotion_tracker.tick()

            # Gaze pipeline
            ear_l     = gc.compute_ear_6pt(face, gc.L_EAR_PTS)
            ear_r     = gc.compute_ear_6pt(face, gc.R_EAR_PTS)
            blink_bad = (0.5 * (ear_l + ear_r) < gc.EAR_BLINK_TH)

            iris_l  = gc.iris_in_eyebox(face, gc.L_EYE, gc.L_IRIS_IDX)
            iris_r  = gc.iris_in_eyebox(face, gc.R_EYE, gc.R_IRIS_IDX)
            if iris_l and iris_r:
                iris_xy = ((iris_l[0]+iris_r[0])*0.5, (iris_l[1]+iris_r[1])*0.5)
            else:
                iris_xy = iris_l or iris_r

            hp          = gc.head_pose_yaw_pitch(result)
            head_center = head.center()
            conf        = gc.confidence_score(face, ear_l, ear_r, hp, head_center,
                                              frame_quality=light.frame_quality)
            gp          = gc.gaze_pose_degrees(face, iris_l=iris_l, iris_r=iris_r)

            x, y, _ = kf.step(iris_xy, valid=(not blink_bad and iris_xy is not None))
            sm_iris  = (x, y) if iris_xy is not None else None

            head.update(hp, iris_xy=sm_iris, conf=conf, blink_bad=blink_bad)
            head_center = head.center()

            # Calibration mode
            if mode == "KALIB":
                done = calib_ui.update(sm_iris, conf, frame, face_ok=True)
                if done:
                    new_m = calib_ui.get_mapper()
                    new_m.save_single()
                    mapper.A = new_m.A; mapper.th_h = new_m.th_h
                    mapper.th_v = new_m.th_v; mapper.ready = True
                    mode = "RUN"
                    kf.reset(); fusion.reset_to(0.5, 0.5); voter.seed("CENTER")

            # Run mode
            else:
                th_h = mapper.th_h if mapper.ready else gc.ZONE_INNER_H_DEF
                th_v = mapper.th_v if mapper.ready else gc.ZONE_INNER_V_DEF

                fused_raw = None
                if mapper.ready and sm_iris and conf >= gc.CONF_MIN:
                    gaze_raw       = mapper.map(sm_iris, clip=False)
                    fused_raw, fusion_w = fusion.fuse(gaze_raw, hp)

                if fused_raw is None:
                    is_off     = gc.head_offscreen_fallback(hp, head_center)
                    stable_dir = "LOW_CONF" if not is_off else "CENTER"
                    off_active, just_returned = offmon.update(is_off, stable_dir, conf)
                else:
                    direction, is_off = gc.zone_from_xy(
                        fused_raw[0], fused_raw[1], th_h, th_v,
                        edge_margin=gc.EDGE_MARGIN)
                    stable_dir = voter.push(direction)
                    off_active, just_returned = offmon.update(is_off, stable_dir, conf)

                if just_returned:
                    kf.reset(); fusion.reset_to(0.5, 0.5); voter.seed("CENTER")

                gc.draw_hud(frame, stable_dir, conf, gp, fps,
                            off_active=off_active,
                            off_dur=offmon.dur if off_active else 0.0,
                            fusion_w=fusion_w, light_str=light_str, mode=mode,
                            emotion_label=em_dominant,
                            emotion_available=(emotion_tracker.available if emotion_tracker else False))

            cv2.imshow(win, frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27: break
            if k in (ord('n'), ord('N')): _enter_calib()
            if k in (ord('r'), ord('R')): _reset_calib()

    finally:
        cap.release()
        cv2.destroyAllWindows()
        try: detector.close()
        except Exception: pass
        try: offmon.force_close()
        except Exception: pass
        if emotion_tracker:
            try: emotion_tracker.stop()
            except Exception: pass


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam",        type=int,  default=gc.CAM_INDEX)
    ap.add_argument("--em-model",   type=str,  default=gc.EM_MODEL_NAME)
    ap.add_argument("--no-emotion", action="store_true")
    args = ap.parse_args()
    gc.EM_MODEL_NAME = args.em_model
    live_mode(cam_index=args.cam, em_model=args.em_model, no_emotion=args.no_emotion)